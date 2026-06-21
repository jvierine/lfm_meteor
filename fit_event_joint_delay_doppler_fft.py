import argparse
import hashlib
import os

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import scipy.signal as sig

import fit_all_ballistic_snr_weighted as base
import fit_all_ceplecha_snr_weighted as cepl
import fit_gcrs_trajectories_lfm_ambiguity as gfit
import sanya_opts as sc
import test_rank02_range_interpolation as interp
from grid_search_delays_beam_axis import DAN_PATTERN, SAN_PATTERN, WEN_PATTERN, load_events, pair_tristatic_events


SCRIPT_VERSION = "v20260618b"
DEFAULT_OUTPUT_BASE = os.path.join("results", f"joint_uncorrected_delay_dechirped_fft_event_{SCRIPT_VERSION}")
DEFAULT_EVENT_ID = "tri_0093_1713816477464351654"
SITE_ORDER = ("sanya", "danzhou", "wenchang")
SITE_LABELS = ("Sanya", "Danzhou", "Wenchang")


def choose_triplet(event_id, triplets, ref_fits):
    if event_id.startswith("tri_"):
        try:
            idx = int(event_id.split("_")[1])
        except Exception:
            idx = None
        if idx is not None and 0 <= idx < len(triplets):
            return idx, triplets[idx]
    for idx, triplet in enumerate(triplets):
        fit0 = base.match_reference_fit(triplet[0], ref_fits)
        if fit0 and fit0["event_id"] == event_id:
            return idx, triplet
    raise ValueError(f"Could not find triplet for event_id={event_id}")


def dechirped_fft_offset_hz(
    row,
    gate,
    sr_mhz,
    bw_mhz,
    pulse_length_us,
    zero_pad_factor,
    gate_upsample_factor=8,
    center_offset_samples=0.0,
):
    if gate_upsample_factor > 1:
        row_work = sig.resample_poly(row, gate_upsample_factor, 1).astype(np.complex128)
        sr_work_mhz = float(sr_mhz) * float(gate_upsample_factor)
        center = int(round((float(gate) + float(center_offset_samples)) * float(gate_upsample_factor)))
    else:
        row_work = np.asarray(row, dtype=np.complex128)
        sr_work_mhz = float(sr_mhz)
        center = int(round(float(gate) + float(center_offset_samples)))
    code, t_s = interp.lfm(
        length_us=float(pulse_length_us),
        sr_mhz=sr_work_mhz,
        bandwidth_hz=float(bw_mhz) * 1e6,
    )
    n_code = len(code)
    start = center - n_code // 2
    stop = start + n_code
    if start < 0 or stop > len(row_work):
        return np.nan, np.nan, np.nan, np.nan
    segment = np.asarray(row_work[start:stop], dtype=np.complex128)
    deramped = segment * np.conj(code.astype(np.complex128))
    y = deramped * np.hanning(n_code)
    n_fft = 1
    while n_fft < int(zero_pad_factor) * n_code:
        n_fft *= 2
    sr_hz = float(sr_work_mhz) * 1e6
    spectrum = np.fft.fftshift(np.fft.fft(y, n=n_fft))
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sr_hz))
    power_db = 10.0 * np.log10(np.maximum(np.abs(spectrum) ** 2.0, 1e-300))
    peak_idx = int(np.nanargmax(power_db))
    peak_hz = float(freq_hz[peak_idx])
    if 0 < peak_idx < len(power_db) - 1:
        ym1, y0, yp1 = map(float, power_db[peak_idx - 1 : peak_idx + 2])
        denom = ym1 - 2.0 * y0 + yp1
        if np.isfinite(denom) and abs(denom) > 1e-30:
            delta = float(np.clip(0.5 * (ym1 - yp1) / denom, -1.0, 1.0))
            peak_hz += delta * float(freq_hz[1] - freq_hz[0])
    prominence_db = float(power_db[peak_idx] - np.nanmedian(power_db))
    return peak_hz, float(sr_hz / n_fft), float(1.0 / (float(pulse_length_us) * 1e-6)), prominence_db


def estimate_fft_observations(
    site_data,
    refined,
    source_indices,
    zero_pad_factor,
    snr_min_db,
    prominence_min_db,
    gate_upsample_factor=8,
    center_offset_samples=0.0,
):
    n = source_indices.shape[0]
    obs_hz = np.full((n, 3), np.nan, dtype=np.float64)
    bin_hz = np.full((n, 3), np.nan, dtype=np.float64)
    resolution_hz = np.full((n, 3), np.nan, dtype=np.float64)
    prominence_db = np.full((n, 3), np.nan, dtype=np.float64)
    keep = np.zeros((n, 3), dtype=bool)
    for site_col, site in enumerate(SITE_ORDER):
        data = site_data[site]
        gates = refined[f"{site}_gate"]
        pulse_length_us = float(data.get("pulse_length_us", 199.0))
        for row_idx, src_idx in enumerate(source_indices[:, site_col]):
            if src_idx < 0 or src_idx >= data["raw"].shape[0]:
                continue
            if float(data["snr_peak_db"][src_idx]) < snr_min_db:
                continue
            peak, fft_bin, fourier_res, prom = dechirped_fft_offset_hz(
                data["raw"][src_idx],
                float(gates[src_idx]),
                data["sr_mhz"],
                data["bw_mhz"],
                pulse_length_us,
                zero_pad_factor,
                gate_upsample_factor=gate_upsample_factor,
                center_offset_samples=center_offset_samples,
            )
            obs_hz[row_idx, site_col] = peak
            bin_hz[row_idx, site_col] = fft_bin
            resolution_hz[row_idx, site_col] = fourier_res
            prominence_db[row_idx, site_col] = prom
            keep[row_idx, site_col] = np.isfinite(peak) and prom >= prominence_min_db
    return {
        "fft_offset_hz": obs_hz,
        "fft_bin_hz": bin_hz,
        "fft_resolution_hz": resolution_hz,
        "fft_prominence_db": prominence_db,
        "fft_keep": keep,
    }


def load_site_h5_with_pulse(path, fit, site):
    data = base.load_site_h5(path, fit, site)
    with h5py.File(path, "r") as h:
        if "pulse_length_us" in h:
            data["pulse_length_us"] = float(h["pulse_length_us"][()])
    return data


def refine_site_without_doppler(site_data, upsample_factor=32, same_mode_offset_samples=0.0):
    coarse = []
    for row in site_data["raw"]:
        gate, _power_db = interp.doppler_matched_filter_peak(
            row,
            0.0,
            site_data["sr_mhz"],
            site_data["bw_mhz"],
            upsample_factor=1,
            coarse_center_gate=None,
        )
        coarse.append(gate)
    coarse = np.asarray(coarse, dtype=np.float64)

    fine = []
    power = []
    for row, center_gate in zip(site_data["raw"], coarse):
        gate, power_db = interp.doppler_matched_filter_peak(
            row,
            0.0,
            site_data["sr_mhz"],
            site_data["bw_mhz"],
            upsample_factor=upsample_factor,
            coarse_center_gate=float(center_gate),
        )
        fine.append(gate)
        power.append(power_db)
    fine = np.asarray(fine, dtype=np.float64) + float(same_mode_offset_samples)
    dr_km = gfit.C / (site_data["sr_mhz"] * 1e6) / 2.0 / 1e3
    return fine, site_data["r0_km"] + dr_km * fine, np.asarray(power, dtype=np.float64)


def forward_model_link_observables(params, t_rel_s, times_ns, rho_of_alt_m):
    """Evaluate per-link path observables for a Ceplecha state.

    Returns one row per pulse and one column per tx-target-rx link in
    SITE_ORDER.  The primary forward-model quantities are the geometric
    tx-target-rx path length L and its time derivative dL/dt.  Doppler and
    LFM-apparent path are derived from those two arrays.
    """

    x_gcrs, v_gcrs, radius_m, mass_kg, success, message = cepl.propagate_ceplecha(params, t_rel_s, rho_of_alt_m)
    x_itrs, v_itrs = base.gcrs_state_samples_to_itrs(x_gcrs, v_gcrs, times_ns)
    path_length_m, path_rate_mps = gfit.link_total_paths_and_rates_m(
        x_itrs,
        v_itrs,
        gfit.LINK_TX_POSITIONS_M,
        gfit.LINK_RX_POSITIONS_M,
    )
    doppler_hz = gfit.doppler_from_path_length_rate_hz(path_rate_mps)
    apparent_path_length_m = path_length_m + gfit.lfm_total_path_bias_m(path_rate_mps)
    return {
        "apparent_path_length_m": apparent_path_length_m,
        "path_length_m": path_length_m,
        "path_rate_mps": path_rate_mps,
        "doppler_hz": doppler_hz,
        "x_gcrs_m": x_gcrs,
        "v_gcrs_mps": v_gcrs,
        "x_itrs_m": x_itrs,
        "v_itrs_mps": v_itrs,
        "radius_m": radius_m,
        "mass_kg": mass_kg,
        "ceplecha_success": success,
        "ceplecha_message": message,
    }


def covariance_from_least_squares(result, n_residuals):
    n_params = int(result.x.size)
    dof = int(max(0, n_residuals - n_params))
    if dof <= 0 or result.jac is None:
        return {
            "parameter_covariance": np.full((n_params, n_params), np.nan, dtype=np.float64),
            "parameter_std": np.full(n_params, np.nan, dtype=np.float64),
            "covariance_available": False,
            "covariance_degrees_of_freedom": dof,
            "covariance_residual_variance": np.nan,
        }
    jac = np.asarray(result.jac, dtype=np.float64)
    residual_variance = float(2.0 * result.cost / dof)
    try:
        cov = np.linalg.pinv(jac.T @ jac) * residual_variance
        cov = 0.5 * (cov + cov.T)
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        available = bool(np.all(np.isfinite(cov)))
    except Exception:
        cov = np.full((n_params, n_params), np.nan, dtype=np.float64)
        std = np.full(n_params, np.nan, dtype=np.float64)
        available = False
    return {
        "parameter_covariance": cov,
        "parameter_std": std,
        "covariance_available": available,
        "covariance_degrees_of_freedom": dof,
        "covariance_residual_variance": residual_variance,
    }


def fit_joint_delay_doppler(
    measured_paths_m,
    times_ns,
    rho_of_alt_m,
    p0,
    sigma_m,
    fft_offset_hz,
    fft_keep,
    sigma_fft_hz,
    keep_rows=None,
    epoch_time_ns=None,
    fit_station_bias=False,
    fft_model="range_offset_corrected_beat",
):
    measured = np.asarray(measured_paths_m, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.int64)
    if keep_rows is None:
        keep_rows = np.ones(len(times), dtype=bool)
    if epoch_time_ns is None:
        epoch_time_ns = int(times[0])
    row_keep = np.asarray(keep_rows, dtype=bool)
    measured_fit = measured[row_keep]
    times_fit = times[row_keep]
    sigma_fit = np.asarray(sigma_m, dtype=np.float64)[row_keep]
    fft_fit = np.asarray(fft_offset_hz, dtype=np.float64)[row_keep]
    fft_keep_fit = np.asarray(fft_keep, dtype=bool)[row_keep] & np.isfinite(fft_fit)
    t_rel_s = (times_fit.astype(np.float64) - float(epoch_time_ns)) / 1e9

    def split_params(x):
        if fit_station_bias:
            return x[:7], x[7:10]
        return x[:7], np.zeros(3, dtype=np.float64)

    def residual(x):
        dyn_params, station_bias_hz = split_params(x)
        model = forward_model_link_observables(dyn_params, t_rel_s, times_fit, rho_of_alt_m)
        apparent = model["apparent_path_length_m"]
        geo = model["path_length_m"]
        doppler = model["doppler_hz"]
        path_resid = ((apparent - measured_fit) / sigma_fit).ravel()
        if fft_model in {"zero_beat", "signed_doppler"}:
            beat_model_hz = -doppler + station_bias_hz[None, :]
        elif fft_model in {"ambiguity_residual", "range_offset_corrected_beat"}:
            # The FFT beat is measured at the fixed matched-filter delay.  A
            # wrong model range leaves a beat term, so the Doppler prediction
            # must be corrected by the model range offset relative to L_meas.
            beat_model_hz = doppler - (gfit.CHIRP_RATE_HZ_PER_S / gfit.C) * (measured_fit - geo) + station_bias_hz[None, :]
        else:
            raise ValueError(f"unknown fft_model={fft_model!r}")
        beat_resid = ((beat_model_hz - fft_fit) / float(sigma_fft_hz))[fft_keep_fit]
        return np.concatenate([path_resid, beat_resid])

    if fit_station_bias:
        model0 = forward_model_link_observables(p0, t_rel_s, times_fit, rho_of_alt_m)
        geo0 = model0["path_length_m"]
        doppler0 = model0["doppler_hz"]
        if fft_model in {"zero_beat", "signed_doppler"}:
            beat0 = -doppler0
        elif fft_model in {"ambiguity_residual", "range_offset_corrected_beat"}:
            beat0 = doppler0 - (gfit.CHIRP_RATE_HZ_PER_S / gfit.C) * (measured_fit - geo0)
        else:
            raise ValueError(f"unknown fft_model={fft_model!r}")
        bias0 = np.zeros(3, dtype=np.float64)
        for col in range(3):
            good = fft_keep_fit[:, col]
            if np.any(good):
                bias0[col] = float(np.nanmedian(fft_fit[good, col] - beat0[good, col]))
        x0 = np.concatenate([p0, bias0])
        lower = np.concatenate(
            [
                np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(cepl.MIN_RADIUS_M)]),
                np.full(3, -500e3),
            ]
        )
        upper = np.concatenate(
            [
                np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(cepl.MAX_RADIUS_M)]),
                np.full(3, 500e3),
            ]
        )
        x_scale = np.concatenate([np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0]), np.full(3, 5e4)])
    else:
        x0 = p0
        lower = np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(cepl.MIN_RADIUS_M)])
        upper = np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(cepl.MAX_RADIUS_M)])
        x_scale = np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0])

    result = so.least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        x_scale=x_scale,
        loss=cepl.ROBUST_LOSS,
        f_scale=cepl.ROBUST_F_SCALE,
        max_nfev=360,
    )
    fit_residual = residual(result.x)
    covariance = covariance_from_least_squares(result, len(fit_residual))
    dyn_params, station_bias_hz = split_params(result.x)
    model = forward_model_link_observables(
        dyn_params,
        t_rel_s,
        times_fit,
        rho_of_alt_m,
    )
    apparent = model["apparent_path_length_m"]
    geo = model["path_length_m"]
    path_rate = model["path_rate_mps"]
    doppler = model["doppler_hz"]
    if fft_model in {"zero_beat", "signed_doppler"}:
        beat_model_hz = -doppler + station_bias_hz[None, :]
    elif fft_model in {"ambiguity_residual", "range_offset_corrected_beat"}:
        beat_model_hz = doppler - (gfit.CHIRP_RATE_HZ_PER_S / gfit.C) * (measured_fit - geo) + station_bias_hz[None, :]
    else:
        raise ValueError(f"unknown fft_model={fft_model!r}")
    path_resid_m = apparent - measured_fit
    beat_resid_hz = beat_model_hz - fft_fit
    if fft_model in {"zero_beat", "signed_doppler"}:
        fft_doppler_hz = -fft_fit
    elif fft_model in {"ambiguity_residual", "range_offset_corrected_beat"}:
        range_offset_frequency_hz = (gfit.CHIRP_RATE_HZ_PER_S / gfit.C) * (measured_fit - geo)
        fft_doppler_hz = fft_fit + range_offset_frequency_hz
    else:
        raise ValueError(f"unknown fft_model={fft_model!r}")
    fft_path_rate_mps = -gfit.RADAR_WAVELENGTH_M * fft_doppler_hz
    path_rate_resid_mps = path_rate - fft_path_rate_mps
    normalized_path = path_resid_m / sigma_fit
    normalized_beat = np.full_like(beat_resid_hz, np.nan)
    normalized_beat[fft_keep_fit] = beat_resid_hz[fft_keep_fit] / float(sigma_fft_hz)
    llh = np.asarray([jcoord.ecef2geodetic(x[0], x[1], x[2]) for x in model["x_itrs_m"]], dtype=np.float64)
    return {
        "params": dyn_params,
        "full_params": result.x,
        "station_fft_bias_hz": station_bias_hz,
        "keep_rows": row_keep,
        "time_ns": times_fit,
        "fit_epoch_time_ns": int(epoch_time_ns),
        "t_rel_s": t_rel_s,
        "measured_total_paths_m": measured_fit,
        "predicted_total_paths_m": apparent,
        "apparent_path_length_m": apparent,
        "geometric_total_paths_m": geo,
        "path_length_m": geo,
        "model_path_rate_mps": path_rate,
        "path_rate_mps": path_rate,
        "fft_path_rate_mps": fft_path_rate_mps,
        "fft_doppler_hz": fft_doppler_hz,
        "model_doppler_hz": doppler,
        "model_fft_peak_hz": beat_model_hz,
        "observed_fft_beat_hz": fft_fit,
        "fft_keep": fft_keep_fit,
        "path_residuals_m": path_resid_m,
        "fft_residuals_hz": beat_resid_hz,
        "path_rate_residuals_mps": path_rate_resid_mps,
        "normalized_path_residuals": normalized_path,
        "normalized_fft_residuals": normalized_beat,
        "x_gcrs_m": model["x_gcrs_m"],
        "v_gcrs_mps": model["v_gcrs_mps"],
        "x_itrs_m": model["x_itrs_m"],
        "v_itrs_mps": model["v_itrs_mps"],
        "lat_deg": llh[:, 0],
        "lon_deg": llh[:, 1],
        "alt_km": llh[:, 2] / 1e3,
        "speed_km_s": np.linalg.norm(model["v_gcrs_mps"], axis=1) / 1e3,
        "radius_m": model["radius_m"],
        "mass_kg": model["mass_kg"],
        "initial_radius_m": float(model["radius_m"][0]),
        "initial_mass_kg": float(model["mass_kg"][0]),
        "rms_total_path_residual_m": float(np.sqrt(np.nanmean(path_resid_m**2.0))),
        "rms_fft_residual_hz": float(np.sqrt(np.nanmean(beat_resid_hz[fft_keep_fit] ** 2.0))) if np.any(fft_keep_fit) else np.nan,
        "rms_path_rate_residual_mps": float(np.sqrt(np.nanmean(path_rate_resid_mps[fft_keep_fit] ** 2.0))) if np.any(fft_keep_fit) else np.nan,
        "weighted_rms": float(np.sqrt(np.nanmean(residual(result.x) ** 2.0))),
        "parameter_covariance": covariance["parameter_covariance"],
        "parameter_std": covariance["parameter_std"],
        "log10_radius_std": float(covariance["parameter_std"][6]) if len(covariance["parameter_std"]) > 6 else np.nan,
        "covariance_available": bool(covariance["covariance_available"]),
        "covariance_degrees_of_freedom": int(covariance["covariance_degrees_of_freedom"]),
        "covariance_residual_variance": float(covariance["covariance_residual_variance"]),
        "n_points": int(len(times_fit)),
        "n_fft_observations": int(np.count_nonzero(fft_keep_fit)),
        "fit_station_bias": bool(fit_station_bias),
        "fft_model": "zero_beat" if fft_model == "signed_doppler" else str(fft_model),
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "optimizer_cost": float(result.cost),
        "ceplecha_success": bool(model["ceplecha_success"]),
        "ceplecha_message": str(model["ceplecha_message"]),
    }


def mass_from_radius(radius_m):
    return (4.0 / 3.0) * np.pi * cepl.METEOROID_DENSITY_KG_M3 * float(radius_m) ** 3.0


def sci_text(value, unit=""):
    if not np.isfinite(value) or value == 0.0:
        return f"{value:.2g}{unit}"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**exponent
    if -2 <= exponent <= 3:
        return f"{value:.3g}{unit}"
    return rf"{mantissa:.2f}\times10^{{{exponent}}}{unit}"


def compact_sci(value):
    if not np.isfinite(value):
        return "nan"
    return f"{value:.2e}"


def radius_mass_interval_text(joint_fit):
    radius_m = float(joint_fit["initial_radius_m"])
    mass_kg = float(joint_fit["initial_mass_kg"])
    log10_radius = np.log10(radius_m)
    log10_radius_std = float(joint_fit.get("log10_radius_std", np.nan))
    if not np.isfinite(log10_radius_std):
        return (
            f"r0 = {radius_m * 1e6:.3g} um\n"
            f"m0 = {compact_sci(mass_kg)} kg"
        )
    lo_r = 10.0 ** (log10_radius - 1.96 * log10_radius_std)
    hi_r = 10.0 ** (log10_radius + 1.96 * log10_radius_std)
    lo_r = float(np.clip(lo_r, cepl.MIN_RADIUS_M, cepl.MAX_RADIUS_M))
    hi_r = float(np.clip(hi_r, cepl.MIN_RADIUS_M, cepl.MAX_RADIUS_M))
    lo_m = mass_from_radius(lo_r)
    hi_m = mass_from_radius(hi_r)
    return (
        f"r0 = {radius_m * 1e6:.3g} um\n"
        f"95% r0: {lo_r * 1e6:.3g} - {hi_r * 1e6:.3g} um\n"
        f"m0 = {compact_sci(mass_kg)} kg\n"
        f"95% m0: {compact_sci(lo_m)} - {compact_sci(hi_m)} kg"
    )


def initial_speed_uncertainty_km_s(joint_fit):
    params = np.asarray(joint_fit.get("params", np.nan), dtype=np.float64)
    cov = np.asarray(joint_fit.get("parameter_covariance", np.nan), dtype=np.float64)
    if params.shape[0] < 6 or cov.shape[0] < 6 or cov.shape[1] < 6:
        return np.nan, np.nan
    v0_mps = params[3:6]
    v0_norm = float(np.linalg.norm(v0_mps))
    if not np.isfinite(v0_norm) or v0_norm <= 0.0:
        return np.nan, np.nan
    v_cov = cov[3:6, 3:6]
    variance = float(v0_mps @ v_cov @ v0_mps) / (v0_norm * v0_norm)
    sigma_mps = np.sqrt(max(variance, 0.0))
    return v0_norm / 1e3, sigma_mps / 1e3


def fit_quality_annotation(joint_fit):
    v0_km_s, v0_sigma_km_s = initial_speed_uncertainty_km_s(joint_fit)
    return (
        f"RMS delay residual = {joint_fit['rms_total_path_residual_m']:.1f} m\n"
        f"RMS Doppler residual = {joint_fit['rms_path_rate_residual_mps']:.0f} m/s\n"
        f"v0 = {v0_km_s:.2f} +/- {v0_sigma_km_s:.2f} km/s"
    )


def deterministic_rng(event_id):
    digest = hashlib.sha256(event_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False) % (2**32)
    return np.random.default_rng(seed)


def model_uncertainty_bands(event_id, joint_fit, rho_of_alt_m, n_draws=96):
    cov = np.asarray(joint_fit.get("parameter_covariance", np.nan), dtype=np.float64)
    params = np.asarray(joint_fit["params"], dtype=np.float64)
    if cov.shape[0] < len(params) or not np.all(np.isfinite(cov[: len(params), : len(params)])):
        return None
    cov = cov[: len(params), : len(params)]
    try:
        eigval, eigvec = np.linalg.eigh(0.5 * (cov + cov.T))
    except Exception:
        return None
    eigval = np.clip(eigval, 0.0, np.inf)
    if not np.any(eigval > 0.0):
        return None
    transform = eigvec @ np.diag(np.sqrt(eigval))
    along_axis = np.nanmean(np.asarray(joint_fit["v_gcrs_mps"], dtype=np.float64), axis=0)
    along_axis = along_axis / max(float(np.linalg.norm(along_axis)), 1e-30)
    rng = deterministic_rng(event_id)
    path_samples = []
    along_velocity_samples = []
    n_try = 0
    while len(path_samples) < n_draws and n_try < n_draws * 8:
        n_try += 1
        trial = params + transform @ rng.standard_normal(len(params))
        trial[6] = np.clip(trial[6], np.log10(cepl.MIN_RADIUS_M), np.log10(cepl.MAX_RADIUS_M))
        model = forward_model_link_observables(
            trial,
            joint_fit["t_rel_s"],
            joint_fit["time_ns"],
            rho_of_alt_m,
        )
        if not model["ceplecha_success"]:
            continue
        path = np.asarray(model["apparent_path_length_m"], dtype=np.float64)
        along_velocity = (np.asarray(model["v_gcrs_mps"], dtype=np.float64) @ along_axis) / 1e3
        if np.all(np.isfinite(path)) and np.all(np.isfinite(along_velocity)):
            path_samples.append(path)
            along_velocity_samples.append(along_velocity)
    if len(path_samples) < max(12, n_draws // 4):
        return None
    path_samples = np.asarray(path_samples, dtype=np.float64)
    along_velocity_samples = np.asarray(along_velocity_samples, dtype=np.float64)
    return {
        "path_lo_m": np.nanpercentile(path_samples, 2.5, axis=0),
        "path_hi_m": np.nanpercentile(path_samples, 97.5, axis=0),
        "along_velocity_lo_km_s": np.nanpercentile(along_velocity_samples, 2.5, axis=0),
        "along_velocity_hi_km_s": np.nanpercentile(along_velocity_samples, 97.5, axis=0),
        "n_draws": int(len(path_samples)),
    }


def plot_joint_fit(event_id, delay_fit, joint_fit, output_base, rho_of_alt_m):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), constrained_layout=True, sharex=True)
    t = joint_fit["t_rel_s"]
    colors = {"sanya": "#4c78a8", "danzhou": "#f58518", "wenchang": "#54a24b"}
    bands = model_uncertainty_bands(event_id, joint_fit, rho_of_alt_m)
    path_origin_m = np.nanmedian(joint_fit["measured_total_paths_m"], axis=0)

    ax = axes[0, 0]
    for col, site in enumerate(SITE_ORDER):
        color = colors[site]
        label = SITE_LABELS[col]
        if bands is not None:
            ax.fill_between(
                t,
                bands["path_lo_m"][:, col] - path_origin_m[col],
                bands["path_hi_m"][:, col] - path_origin_m[col],
                color=color,
                alpha=0.14,
                lw=0,
            )
        ax.plot(
            t,
            joint_fit["predicted_total_paths_m"][:, col] - path_origin_m[col],
            color=color,
            lw=1.7,
            label=f"{label} fit",
        )
        ax.scatter(
            t,
            joint_fit["measured_total_paths_m"][:, col] - path_origin_m[col],
            s=16,
            facecolors="white",
            edgecolors=color,
            linewidths=0.8,
            alpha=0.9,
        )
    ax.set_ylabel("Path offset (m)")
    ax.set_title("Delay measurements and fit")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncols=1)
    ax.text(
        0.04,
        0.05,
        fit_quality_annotation(joint_fit),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
    )

    ax = axes[0, 1]
    along_axis = np.nanmean(np.asarray(joint_fit["v_gcrs_mps"], dtype=np.float64), axis=0)
    along_axis = along_axis / max(float(np.linalg.norm(along_axis)), 1e-30)
    along_velocity_km_s = (np.asarray(joint_fit["v_gcrs_mps"], dtype=np.float64) @ along_axis) / 1e3
    if bands is not None:
        ax.fill_between(
            t,
            bands["along_velocity_lo_km_s"],
            bands["along_velocity_hi_km_s"],
            color="#a6dba0",
            alpha=0.55,
            lw=0,
            label="95% fit band",
        )
    ax.plot(t, along_velocity_km_s, color="#1b7837", lw=1.9, label="joint fit")
    ax.set_ylabel("Along-track velocity (km/s)")
    ax.set_title("Model along-track velocity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.text(
        0.04,
        0.05,
        radius_mass_interval_text(joint_fit),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
    )

    ax = axes[1, 0]
    for col, site in enumerate(SITE_ORDER):
        ax.scatter(
            t,
            joint_fit["path_residuals_m"][:, col],
            s=22,
            color=colors[site],
            label=SITE_LABELS[col],
            alpha=0.85,
        )
    ax.axhline(0, color="0.25", lw=1.0)
    ax.set_xlabel("Time since fit epoch (s)")
    ax.set_ylabel("Delay residual (m)")
    ax.set_title("Delay residuals")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    for col, site in enumerate(SITE_ORDER):
        keep = joint_fit["fft_keep"][:, col]
        ax.scatter(
            t[keep],
            joint_fit["fft_residuals_hz"][keep, col] / 1e3,
            s=22,
            color=colors[site],
            label=SITE_LABELS[col],
            alpha=0.85,
        )
    ax.axhline(0, color="0.25", lw=1.0)
    ax.set_xlabel("Time since fit epoch (s)")
    ax.set_ylabel("Beat residual (kHz)")
    ax.set_title("Doppler residuals")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.savefig(f"{output_base}.png", dpi=220)
    fig.savefig(f"{output_base}.pdf")
    plt.close(fig)


def write_h5(
    output_base,
    event_id,
    delay_fit,
    joint_fit,
    fft_obs,
    sigma_fft_hz,
    zero_pad_factor,
    range_measurement,
    gate_upsample_factor,
    fft_center_offset_samples,
):
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    with h5py.File(f"{output_base}.h5", "w") as h:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["event_id"] = event_id
        h.attrs["sigma_fft_hz"] = float(sigma_fft_hz)
        h.attrs["zero_pad_factor"] = int(zero_pad_factor)
        h.attrs["range_measurement"] = str(range_measurement)
        h.attrs["fft_gate_upsample_factor"] = int(gate_upsample_factor)
        h.attrs["fft_center_offset_samples"] = float(fft_center_offset_samples)
        h.attrs["joint_frequency_model"] = (
            "Least-squares residual has two measurement blocks. Range block: "
            "ordinary matched-filter path measurements without Doppler correction "
            "are fixed measurements and are fit with "
            "measured_path = geometric_path + c*f_D_model/chirp_rate. "
            "Frequency block: dechirped single-pulse FFT beat frequencies are "
            "fixed measurements and are fit with "
            "f_beat = f_D_model - (chirp_rate/c)*(measured_path - geometric_path). "
            "Thus the Doppler prediction includes the model range-offset correction."
        )
        dg = h.create_group("delay_only_fit")
        for key in ("params", "residuals_m", "normalized_residuals", "predicted_total_paths_m", "measured_total_paths_m"):
            if key in delay_fit:
                dg[key] = delay_fit[key]
        for key in ("rms_total_path_residual_m", "weighted_rms", "initial_radius_m", "initial_mass_kg"):
            if key in delay_fit:
                dg.attrs[key] = delay_fit[key]
        jg = h.create_group("joint_fit")
        jg.create_dataset("link_names", data=np.asarray(SITE_ORDER, dtype=object), dtype=string_dtype)
        for key, value in joint_fit.items():
            if np.isscalar(value) or isinstance(value, (str, bytes, bool)):
                jg.attrs[key] = value
            else:
                jg[key] = value
        og = h.create_group("fft_observations")
        og.create_dataset("link_names", data=np.asarray(SITE_ORDER, dtype=object), dtype=string_dtype)
        for key, value in fft_obs.items():
            og[key] = value


def main():
    parser = argparse.ArgumentParser(description="Adopted joint uncorrected-delay + dechirped FFT beat-frequency fit.")
    parser.add_argument("--event-id", default=DEFAULT_EVENT_ID)
    parser.add_argument("--zero-pad-factor", type=int, default=64)
    parser.add_argument("--fft-gate-upsample-factor", type=int, default=32)
    parser.add_argument("--fft-center-offset-samples", type=float, default=0.0)
    parser.add_argument("--range-upsample-factor", type=int, default=32)
    parser.add_argument("--matched-filter-same-offset-samples", type=float, default=0.0)
    parser.add_argument("--snr-min-db", type=float, default=15.0)
    parser.add_argument("--prominence-min-db", type=float, default=8.0)
    parser.add_argument("--sigma-fft-hz", type=float, default=5000.0)
    parser.add_argument("--clip-fft-residual-khz", type=float, default=2.0)
    parser.add_argument(
        "--fft-model",
        choices=("range_offset_corrected_beat", "zero_beat", "signed_doppler", "ambiguity_residual"),
        default="range_offset_corrected_beat",
    )
    parser.add_argument("--fit-station-bias", action="store_true")
    parser.add_argument(
        "--range-measurement",
        choices=("uncorrected", "reference-doppler"),
        default="uncorrected",
        help="Use fd=0 matched-filter delays, or the older reference-Doppler-refined gates.",
    )
    parser.add_argument("--output-base", default=None)
    args = parser.parse_args()

    ref_fits = base.load_reference_fits()
    triplets = pair_tristatic_events(load_events(SAN_PATTERN), load_events(DAN_PATTERN), load_events(WEN_PATTERN))
    idx, triplet = choose_triplet(args.event_id, triplets, ref_fits)
    san_event, dan_event, wen_event = triplet
    fit0 = base.match_reference_fit(san_event, ref_fits)
    if fit0 is None:
        raise RuntimeError(f"No reference fit for triplet {idx}")
    event_id = fit0["event_id"]
    site_data = {
        "sanya": load_site_h5_with_pulse(san_event.path, fit0, "sanya"),
        "danzhou": load_site_h5_with_pulse(dan_event.path, fit0, "danzhou"),
        "wenchang": load_site_h5_with_pulse(wen_event.path, fit0, "wenchang"),
    }
    refined = {}
    for site in SITE_ORDER:
        if args.range_measurement == "uncorrected":
            gate, range_km, _power_db = refine_site_without_doppler(
                site_data[site],
                upsample_factor=args.range_upsample_factor,
                same_mode_offset_samples=args.matched_filter_same_offset_samples,
            )
        else:
            gate, range_km, _power_db = base.refine_site(site_data[site])
        refined[f"{site}_gate"] = gate
        refined[f"{site}_range_km"] = range_km
    refined["sanya_range_km"] = refined["sanya_range_km"] + sc.SANYA_RANGE_CORRECTION_KM
    measured, times_ns, _beijing_ns, snr_db, source_indices = base.matched_measurements_from_sites(
        san_event,
        dan_event,
        wen_event,
        site_data,
        refined,
    )
    order = np.argsort(times_ns)
    measured = measured[order]
    times_ns = times_ns[order]
    snr_db = snr_db[order]
    source_indices = source_indices[order]
    points, keep_geo = base.triangulate_points(measured, san_event.az_deg, san_event.el_deg)
    measured = measured[keep_geo]
    times_ns = times_ns[keep_geo]
    snr_db = snr_db[keep_geo]
    source_indices = source_indices[keep_geo]
    if len(times_ns) < base.MIN_POINTS:
        raise RuntimeError("Too few geometric points after filtering")

    sigma_model = {"sigma_floor_m": 33.39, "sigma_0_m": 236.9}
    sigma_m = base.sigma_from_snr_db(snr_db, sigma_model["sigma_floor_m"], sigma_model["sigma_0_m"])
    rho_of_alt_m, _msis_meta = base.density_interpolator(times_ns, points)
    guesses = cepl.unique_initial_guesses(points, times_ns, reference_fit=fit0)
    delay_fit = cepl.fit_ceplecha_multistart(
        measured,
        times_ns,
        rho_of_alt_m,
        guesses,
        sigma_m=sigma_m,
        keep_rows=np.ones(len(times_ns), dtype=bool),
        epoch_time_ns=int(times_ns[0]),
    )
    fft_obs = estimate_fft_observations(
        site_data,
        refined,
        source_indices,
        args.zero_pad_factor,
        args.snr_min_db,
        args.prominence_min_db,
        gate_upsample_factor=args.fft_gate_upsample_factor,
        center_offset_samples=args.fft_center_offset_samples,
    )
    joint_fit = fit_joint_delay_doppler(
        measured,
        times_ns,
        rho_of_alt_m,
        delay_fit["params"],
        sigma_m,
        fft_obs["fft_offset_hz"],
        fft_obs["fft_keep"],
        args.sigma_fft_hz,
        keep_rows=np.ones(len(times_ns), dtype=bool),
        epoch_time_ns=int(times_ns[0]),
        fit_station_bias=bool(args.fit_station_bias),
        fft_model=args.fft_model,
    )
    if np.isfinite(args.clip_fft_residual_khz) and args.clip_fft_residual_khz > 0.0:
        clip_limit_hz = float(args.clip_fft_residual_khz) * 1e3
        clipped_fft_keep = fft_obs["fft_keep"] & (np.abs(joint_fit["fft_residuals_hz"]) <= clip_limit_hz)
        if np.count_nonzero(clipped_fft_keep) >= base.MIN_POINTS:
            joint_fit = fit_joint_delay_doppler(
                measured,
                times_ns,
                rho_of_alt_m,
                joint_fit["params"],
                sigma_m,
                fft_obs["fft_offset_hz"],
                clipped_fft_keep,
                args.sigma_fft_hz,
                keep_rows=np.ones(len(times_ns), dtype=bool),
                epoch_time_ns=int(times_ns[0]),
                fit_station_bias=bool(args.fit_station_bias),
                fft_model=args.fft_model,
            )
            joint_fit["fft_clip_limit_hz"] = float(clip_limit_hz)
            joint_fit["n_fft_clipped_observations"] = int(np.count_nonzero(fft_obs["fft_keep"]) - np.count_nonzero(clipped_fft_keep))
        else:
            joint_fit["fft_clip_limit_hz"] = float(clip_limit_hz)
            joint_fit["n_fft_clipped_observations"] = 0
    output_base = args.output_base or f"{DEFAULT_OUTPUT_BASE}_{event_id}"
    write_h5(
        output_base,
        event_id,
        delay_fit,
        joint_fit,
        fft_obs,
        args.sigma_fft_hz,
        args.zero_pad_factor,
        args.range_measurement,
        args.fft_gate_upsample_factor,
        args.fft_center_offset_samples,
    )
    plot_joint_fit(event_id, delay_fit, joint_fit, output_base, rho_of_alt_m)
    print(f"event_id={event_id}")
    print(f"n_points={joint_fit['n_points']}")
    print(f"n_fft_observations={joint_fit['n_fft_observations']}")
    print(f"delay_only_path_rms_m={delay_fit['rms_total_path_residual_m']:.3f}")
    print(f"joint_path_rms_m={joint_fit['rms_total_path_residual_m']:.3f}")
    print(f"joint_fft_rms_hz={joint_fit['rms_fft_residual_hz']:.3f}")
    print(f"joint_path_rate_rms_mps={joint_fit['rms_path_rate_residual_mps']:.3f}")
    print(f"delay_only_radius_um={delay_fit['initial_radius_m'] * 1e6:.3f}")
    print(f"joint_radius_um={joint_fit['initial_radius_m'] * 1e6:.3f}")
    print(f"joint_initial_mass_kg={joint_fit['initial_mass_kg']:.6e}")
    print(f"joint_final_radius_um={joint_fit['radius_m'][-1] * 1e6:.3f}")
    print(f"joint_final_mass_kg={joint_fit['mass_kg'][-1]:.6e}")
    print(f"output_h5={output_base}.h5")
    print(f"output_png={output_base}.png")


if __name__ == "__main__":
    main()
