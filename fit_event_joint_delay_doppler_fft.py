import argparse
import hashlib
import os

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import scipy.signal as sig
from matplotlib.lines import Line2D

import fit_all_ballistic_snr_weighted as base
import fit_all_ceplecha_snr_weighted as cepl
import fit_gcrs_trajectories_lfm_ambiguity as gfit
import sanya_opts as sc
import test_rank02_range_interpolation as interp
import plot_sanya_beam_position_histogram as beam_hist
from grid_search_delays_beam_axis import DAN_PATTERN, SAN_PATTERN, WEN_PATTERN, load_events, pair_tristatic_events


SCRIPT_VERSION = "v20260618b"
DEFAULT_OUTPUT_BASE = os.path.join("results", f"joint_uncorrected_delay_dechirped_fft_event_{SCRIPT_VERSION}")
DEFAULT_EVENT_ID = "tri_0093_1713816477464351654"
SITE_ORDER = ("sanya", "danzhou", "wenchang")
SITE_LABELS = ("Sanya", "Danzhou", "Wenchang")
DEFAULT_FFT_TIME_PAD_US = 50.0
DEFAULT_REFERENCE_CHIRP_RATE_SCALE = gfit.REFERENCE_CHIRP_RATE_SCALE
CANONICAL_PULSE_LENGTH_US = gfit.LFM_DURATION_S * 1e6
DEFAULT_FINAL_DELAY_RESIDUAL_CLIP_M = 50.0
DEFAULT_MAX_LOG10_RADIUS_STD = 1.0
DEFAULT_SYSTEM_NOISE_H5 = os.path.join("results", "sanya_4mhz_system_noise_power_100pulse.h5")
DEFAULT_BAD_FIT_RETAINED_PATH_RMS_M = 100.0
DEFAULT_BAD_FIT_RETAINED_FFT_RMS_HZ = 1500.0
DEFAULT_BAD_FIT_MAX_RETRY = 3
DEFAULT_COINCIDENT_DELAY_WEIGHT = 5.0


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


def lfm_reference_for_offsets(
    sample_offsets,
    sr_mhz,
    bandwidth_hz,
    pulse_length_us,
    chirp_rate_scale=DEFAULT_REFERENCE_CHIRP_RATE_SCALE,
):
    t_s = np.asarray(sample_offsets, dtype=np.float64) / (float(sr_mhz) * 1e6)
    sweep_rate = (
        float(bandwidth_hz)
        * 1e6
        / float(pulse_length_us)
        / 2.0
        * float(chirp_rate_scale)
    )
    code = np.exp(
        1j
        * 2.0
        * np.pi
        * (t_s * float(bandwidth_hz) / 2.0 - sweep_rate * t_s**2.0)
    )
    return code.astype(np.complex128)


class RawVoltageNoisePower:
    def __init__(self, path):
        self.path = path
        self.time_ns_by_site = {}
        self.power_by_site = {}
        with h5py.File(path, "r") as h:
            names = [
                value.decode("utf-8").lower() if isinstance(value, bytes) else str(value).lower()
                for value in h["site_names"][:]
            ]
            station_id = np.asarray(h["bins/station_id"][:], dtype=np.int64)
            time_ns = np.asarray(h["bins/time_utc_mid_ns"][:], dtype=np.int64)
            power = np.asarray(h["bins/noise_power_mean_raw_voltage"][:], dtype=np.float64)
        for sid, name in enumerate(names):
            good = (station_id == sid) & np.isfinite(time_ns) & np.isfinite(power) & (power > 0.0)
            order = np.argsort(time_ns[good])
            self.time_ns_by_site[name] = time_ns[good][order].astype(np.float64)
            self.power_by_site[name] = power[good][order].astype(np.float64)

    def power(self, site, time_ns):
        site = site.lower()
        if site not in self.time_ns_by_site:
            raise KeyError(f"No raw-voltage noise power for site {site!r}")
        return np.interp(
            np.asarray(time_ns, dtype=np.float64),
            self.time_ns_by_site[site],
            self.power_by_site[site],
            left=np.nan,
            right=np.nan,
        )


def normalized_matched_filter_snr_db(site_data, refined_gate, site, noise_power):
    pulse_length_us = CANONICAL_PULSE_LENGTH_US
    sr_mhz = float(site_data["sr_mhz"])
    bw_mhz = float(site_data["bw_mhz"])
    code, _t_s = interp.lfm(
        length_us=pulse_length_us,
        sr_mhz=sr_mhz,
        bandwidth_hz=bw_mhz * 1e6,
    )
    code = np.asarray(code, dtype=np.complex128)
    code_energy = float(np.sum(np.abs(code) ** 2.0))
    raw = np.asarray(site_data["raw"], dtype=np.complex128)
    gates = np.asarray(refined_gate, dtype=np.float64)
    noise = noise_power.power(site, site_data["times_ns"])
    snr_db = np.full(raw.shape[0], np.nan, dtype=np.float64)
    half = int(np.ceil(interp.SEARCH_HALF_WIDTH_GATES))
    for idx, row in enumerate(raw):
        center = int(round(gates[idx]))
        corr = sig.fftconvolve(row, np.conj(code), mode="same")
        lo = max(0, center - half)
        hi = min(len(corr), center + half + 1)
        if hi <= lo or not np.isfinite(noise[idx]) or noise[idx] <= 0.0:
            continue
        peak_power = float(np.nanmax(np.abs(corr[lo:hi]) ** 2.0))
        snr_linear = peak_power / max(code_energy * float(noise[idx]), 1e-300)
        snr_db[idx] = 10.0 * np.log10(max(snr_linear, 1e-300))
    return snr_db


def site_delay_to_total_path_m(site, gate, sr_mhz):
    if site == "sanya":
        delay_us = sc.SANYA_CORRECTED_TXRX_DELAY_US + float(gate) / float(sr_mhz)
    elif site == "danzhou":
        delay_us = sc.DANZHOU_FIRST_SAMPLE_DELAY_US + float(gate) / float(sr_mhz)
    elif site == "wenchang":
        delay_us = sc.WENCHANG_FIRST_SAMPLE_DELAY_US + float(gate) / float(sr_mhz)
    else:
        raise ValueError(f"unknown site {site!r}")
    return float(gfit.delay_us_to_total_path_m(delay_us))


def shift_constant_velocity_epoch(guesses, old_epoch_time_ns, new_epoch_time_ns):
    dt_s = (float(new_epoch_time_ns) - float(old_epoch_time_ns)) / 1e9
    shifted = []
    for guess in guesses:
        value = np.asarray(guess, dtype=np.float64).copy()
        if value.size >= 6 and np.all(np.isfinite(value[:6])):
            value[:3] = value[:3] + dt_s * value[3:6]
        shifted.append(value)
    return shifted


def add_randomized_initial_guesses(
    guesses,
    n_random,
    seed=None,
    position_sigma_m=1500.0,
    velocity_sigma_mps=800.0,
    log10_radius_sigma=0.6,
):
    guesses = list(guesses)
    if int(n_random) <= 0 or not guesses:
        return guesses
    rng = np.random.default_rng(seed)
    base_guesses = [np.asarray(guess, dtype=np.float64) for guess in guesses if len(guess) >= 7 and np.all(np.isfinite(guess[:7]))]
    if not base_guesses:
        return guesses
    for _idx in range(int(n_random)):
        guess = np.asarray(base_guesses[int(rng.integers(0, len(base_guesses)))], dtype=np.float64).copy()
        guess[:3] += rng.normal(0.0, float(position_sigma_m), size=3)
        guess[3:6] += rng.normal(0.0, float(velocity_sigma_mps), size=3)
        guess[6] = np.clip(
            guess[6] + rng.normal(0.0, float(log10_radius_sigma)),
            np.log10(cepl.MIN_RADIUS_M),
            np.log10(cepl.MAX_RADIUS_M),
        )
        guesses.append(guess)
    return guesses


def reference_points_itrs(reference_fit, times_ns):
    times = np.asarray(times_ns, dtype=np.int64)
    t_rel_s = (times.astype(np.float64) - float(reference_fit["t0_ns"])) / 1e9
    positions_itrs_m, _velocities_itrs_mps = gfit.gcrs_state_to_itrs(
        np.asarray(reference_fit["r0_gcrs_m"], dtype=np.float64),
        np.asarray(reference_fit["v0_gcrs_mps"], dtype=np.float64),
        t_rel_s,
        times,
    )
    return positions_itrs_m


def finite_sigma_from_snr_db(snr_db, sigma_floor_m, sigma_0_m, fallback_snr_db=0.0):
    snr = np.asarray(snr_db, dtype=np.float64)
    safe_snr = np.where(np.isfinite(snr), snr, float(fallback_snr_db))
    sigma = base.sigma_from_snr_db(safe_snr, sigma_floor_m, sigma_0_m)
    return np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, float(sigma_0_m))


def load_manual_outlier_masks(path, event_id, times_ns, n_sites=3):
    n = len(times_ns)
    path_mask = np.zeros((n, n_sites), dtype=bool)
    fft_mask = np.zeros((n, n_sites), dtype=bool)
    if not path:
        return path_mask, fft_mask
    if not os.path.exists(path):
        return path_mask, fft_mask
    try:
        with h5py.File(path, "r") as h:
            if event_id not in h:
                return path_mask, fft_mask
            g = h[event_id]
            stored_times = np.asarray(g.get("time_ns", []), dtype=np.int64)
            if stored_times.size == 0:
                return path_mask, fft_mask
            index_by_time = {int(t): idx for idx, t in enumerate(np.asarray(times_ns, dtype=np.int64))}
            stored_to_current = [index_by_time.get(int(t), -1) for t in stored_times]
            for name, target in (("delay_outlier", path_mask), ("fft_outlier", fft_mask)):
                if name not in g:
                    continue
                stored = np.asarray(g[name][:], dtype=bool)
                if stored.ndim != 2 or stored.shape[1] != n_sites:
                    continue
                for stored_idx, current_idx in enumerate(stored_to_current[: stored.shape[0]]):
                    if current_idx >= 0:
                        target[current_idx, :] = stored[stored_idx, :]
    except Exception as exc:
        print(f"manual_outlier_warning={exc}")
    return path_mask, fft_mask


def seed_params_from_existing_h5(path, fit_epoch_time_ns, default_radius_m=20e-6):
    if not path or not os.path.exists(path):
        return None, None
    try:
        with h5py.File(path, "r") as h:
            if "joint_fit" not in h or "params" not in h["joint_fit"]:
                return None, None
            group = h["joint_fit"]
            params = np.asarray(group["params"][:], dtype=np.float64)
            model_kind = group.attrs.get("dynamical_model", "ceplecha")
            if isinstance(model_kind, bytes):
                model_kind = model_kind.decode("utf-8")
            old_epoch_time_ns = int(group.attrs.get("fit_epoch_time_ns", fit_epoch_time_ns))
    except Exception:
        return None, None
    if params.size < 6 or not np.all(np.isfinite(params[:6])):
        return None, None
    shifted = shift_constant_velocity_epoch([params[:6]], old_epoch_time_ns, fit_epoch_time_ns)[0]
    if str(model_kind) == "ceplecha" and params.size >= 7 and np.isfinite(params[6]):
        seed = np.asarray(params[:7], dtype=np.float64).copy()
        seed[:6] = shifted[:6]
        return seed, "existing_ceplecha"
    seed = np.concatenate([shifted[:6], [np.log10(float(default_radius_m))]])
    return seed, "existing_constant_velocity_promoted"


def assemble_union_measurements_from_sites(events_by_site, site_data, refined, snr_by_site):
    input_times_are_utc = gfit.event_times_are_utc(
        events_by_site["sanya"],
        events_by_site["danzhou"],
        events_by_site["wenchang"],
    )
    observations = []
    for site_col, site in enumerate(SITE_ORDER):
        event = events_by_site[site]
        for src_idx, raw_time_ns in enumerate(event.times_ns):
            time_ns = int(raw_time_ns) if input_times_are_utc else int(raw_time_ns) - base.SOURCE_TIMEZONE_OFFSET_NS
            observations.append((time_ns, site_col, site, int(src_idx)))
    observations.sort(key=lambda item: item[0])
    tolerance_ns = int(gfit.MATCH_TOLERANCE_MS * 1e6)
    clusters = []
    for obs in observations:
        obs_time, site_col = obs[0], obs[1]
        best_idx = None
        best_dt = None
        for cluster_idx, cluster in enumerate(clusters):
            if site_col in cluster["site_cols"]:
                continue
            dt = abs(obs_time - cluster["time_ref_ns"])
            if dt > tolerance_ns:
                continue
            if best_dt is None or dt < best_dt:
                best_idx = cluster_idx
                best_dt = dt
        if best_idx is None:
            clusters.append({"time_ref_ns": obs_time, "site_cols": {site_col}, "observations": [obs]})
        else:
            clusters[best_idx]["site_cols"].add(site_col)
            clusters[best_idx]["observations"].append(obs)

    measured = []
    times_ns = []
    beijing_local_times_ns = []
    snr_db = []
    source_indices = []
    for cluster in clusters:
        observations_in_cluster = cluster["observations"]
        t_cluster = int(round(np.mean([obs[0] for obs in observations_in_cluster])))
        row_measured = np.full(3, np.nan, dtype=np.float64)
        row_snr = np.full(3, np.nan, dtype=np.float64)
        row_sources = np.full(3, -1, dtype=np.int32)
        for site_col, site in enumerate(SITE_ORDER):
            site_obs = [obs for obs in observations_in_cluster if obs[1] == site_col]
            if not site_obs:
                continue
            obs = min(site_obs, key=lambda item: abs(item[0] - t_cluster))
            src_idx = obs[3]
            row_sources[site_col] = src_idx
            row_measured[site_col] = site_delay_to_total_path_m(
                site,
                refined[f"{site}_gate"][src_idx],
                site_data[site]["sr_mhz"],
            )
            row_snr[site_col] = snr_by_site[site][src_idx]
        if np.count_nonzero(np.isfinite(row_measured)) == 0:
            continue
        measured.append(row_measured)
        times_ns.append(t_cluster)
        beijing_local_times_ns.append(t_cluster + base.SOURCE_TIMEZONE_OFFSET_NS)
        snr_db.append(row_snr)
        source_indices.append(row_sources)
    return (
        np.asarray(measured, dtype=np.float64),
        np.asarray(times_ns, dtype=np.int64),
        np.asarray(beijing_local_times_ns, dtype=np.int64),
        np.asarray(snr_db, dtype=np.float64),
        np.asarray(source_indices, dtype=np.int32),
    )


def dechirped_fft_offset_hz(
    row,
    gate,
    sr_mhz,
    bw_mhz,
    pulse_length_us,
    zero_pad_factor,
    gate_upsample_factor=8,
    center_offset_samples=0.0,
    time_pad_us=DEFAULT_FFT_TIME_PAD_US,
    chirp_rate_scale=DEFAULT_REFERENCE_CHIRP_RATE_SCALE,
):
    if gate_upsample_factor > 1:
        row_work = sig.resample_poly(row, gate_upsample_factor, 1).astype(np.complex128)
        sr_work_mhz = float(sr_mhz) * float(gate_upsample_factor)
        center = int(round((float(gate) + float(center_offset_samples)) * float(gate_upsample_factor)))
    else:
        row_work = np.asarray(row, dtype=np.complex128)
        sr_work_mhz = float(sr_mhz)
        center = int(round(float(gate) + float(center_offset_samples)))
    code, _t_s = interp.lfm(
        length_us=float(pulse_length_us),
        sr_mhz=sr_work_mhz,
        bandwidth_hz=float(bw_mhz) * 1e6,
    )
    n_code = len(code)
    pulse_start = center - n_code // 2
    pulse_stop = pulse_start + n_code
    pad_samples = int(round(float(time_pad_us) * sr_work_mhz))
    start = max(0, pulse_start - pad_samples)
    stop = min(len(row_work), pulse_stop + pad_samples)
    if pulse_start < 0 or pulse_stop > len(row_work):
        return np.nan, np.nan, np.nan, np.nan
    segment = np.asarray(row_work[start:stop], dtype=np.complex128)
    sample_offsets = np.arange(start, stop, dtype=np.float64) - float(pulse_start)
    reference = lfm_reference_for_offsets(
        sample_offsets,
        sr_work_mhz,
        float(bw_mhz) * 1e6,
        float(pulse_length_us),
        chirp_rate_scale=chirp_rate_scale,
    )
    deramped = segment * np.conj(reference)
    n_analysis = len(deramped)
    y = deramped * np.hanning(n_analysis)
    n_fft = 1
    while n_fft < int(zero_pad_factor) * n_analysis:
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
    return peak_hz, float(sr_hz / n_fft), float(1.0 / (n_analysis / sr_hz)), prominence_db


def estimate_fft_observations(
    site_data,
    refined,
    source_indices,
    zero_pad_factor,
    snr_min_db,
    prominence_min_db,
    gate_upsample_factor=8,
    center_offset_samples=0.0,
    time_pad_us=DEFAULT_FFT_TIME_PAD_US,
    chirp_rate_scale=DEFAULT_REFERENCE_CHIRP_RATE_SCALE,
):
    n = source_indices.shape[0]
    obs_hz = np.full((n, 3), np.nan, dtype=np.float64)
    bin_hz = np.full((n, 3), np.nan, dtype=np.float64)
    resolution_hz = np.full((n, 3), np.nan, dtype=np.float64)
    prominence_db = np.full((n, 3), np.nan, dtype=np.float64)
    snr_db = np.full((n, 3), np.nan, dtype=np.float64)
    keep = np.zeros((n, 3), dtype=bool)
    for site_col, site in enumerate(SITE_ORDER):
        data = site_data[site]
        gates = refined[f"{site}_gate"]
        pulse_length_us = CANONICAL_PULSE_LENGTH_US
        for row_idx, src_idx in enumerate(source_indices[:, site_col]):
            if src_idx < 0 or src_idx >= data["raw"].shape[0]:
                continue
            snr_db[row_idx, site_col] = float(data["snr_peak_db"][src_idx])
            if snr_db[row_idx, site_col] < snr_min_db:
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
                time_pad_us=time_pad_us,
                chirp_rate_scale=chirp_rate_scale,
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
        "fft_snr_db": snr_db,
        "fft_keep": keep,
    }


def load_site_h5_with_pulse(path, fit, site):
    data = base.load_site_h5(path, fit, site)
    with h5py.File(path, "r") as h:
        if "pulse_length_us" in h:
            data["pulse_length_us"] = float(h["pulse_length_us"][()])
    data["source_pulse_length_us"] = float(data.get("pulse_length_us", np.nan))
    data["pulse_length_us"] = CANONICAL_PULSE_LENGTH_US
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


def forward_model_constant_velocity_link_observables(params, t_rel_s, times_ns, rho_of_alt_m=None):
    """Evaluate per-link observables for a constant-GCRS-velocity trajectory."""

    params = np.asarray(params, dtype=np.float64)
    t_rel_s = np.asarray(t_rel_s, dtype=np.float64)
    x_gcrs = params[:3][None, :] + t_rel_s[:, None] * params[3:6][None, :]
    v_gcrs = np.repeat(params[3:6][None, :], len(t_rel_s), axis=0)
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
        "radius_m": np.full(len(t_rel_s), np.nan, dtype=np.float64),
        "mass_kg": np.full(len(t_rel_s), np.nan, dtype=np.float64),
        "ceplecha_success": True,
        "ceplecha_message": "constant_velocity",
    }


def forward_model_for_kind(params, t_rel_s, times_ns, rho_of_alt_m, model_kind):
    if model_kind == "ceplecha":
        return forward_model_link_observables(params, t_rel_s, times_ns, rho_of_alt_m)
    if model_kind == "constant_velocity":
        return forward_model_constant_velocity_link_observables(params, t_rel_s, times_ns, rho_of_alt_m)
    raise ValueError(f"unknown model_kind={model_kind!r}")


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
    reference_chirp_rate_scale=DEFAULT_REFERENCE_CHIRP_RATE_SCALE,
    path_keep=None,
    model_kind="ceplecha",
):
    chirp_rate_hz_per_s = gfit.NOMINAL_CHIRP_RATE_HZ_PER_S * float(reference_chirp_rate_scale)
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
    if path_keep is None:
        path_keep_fit = np.ones_like(measured_fit, dtype=bool)
    else:
        path_keep_fit = np.asarray(path_keep, dtype=bool)[row_keep] & np.isfinite(measured_fit)
    fft_fit = np.asarray(fft_offset_hz, dtype=np.float64)[row_keep]
    fft_keep_fit = np.asarray(fft_keep, dtype=bool)[row_keep] & np.isfinite(fft_fit)
    t_rel_s = (times_fit.astype(np.float64) - float(epoch_time_ns)) / 1e9
    n_dyn = 7 if model_kind == "ceplecha" else 6

    def split_params(x):
        if fit_station_bias:
            return x[:n_dyn], x[n_dyn : n_dyn + 3]
        return x[:n_dyn], np.zeros(3, dtype=np.float64)

    def residual(x):
        dyn_params, station_bias_hz = split_params(x)
        model = forward_model_for_kind(dyn_params, t_rel_s, times_fit, rho_of_alt_m, model_kind)
        apparent = model["apparent_path_length_m"]
        geo = model["path_length_m"]
        doppler = model["doppler_hz"]
        path_resid = ((apparent - measured_fit) / sigma_fit)[path_keep_fit]
        if fft_model in {"zero_beat", "signed_doppler"}:
            beat_model_hz = -doppler + station_bias_hz[None, :]
        elif fft_model in {"ambiguity_residual", "range_offset_corrected_beat"}:
            # The FFT beat is measured at the fixed matched-filter delay.  A
            # wrong model range leaves a beat term, so the Doppler prediction
            # must be corrected by the model range offset relative to L_meas.
            beat_model_hz = doppler - (chirp_rate_hz_per_s / gfit.C) * (measured_fit - geo) + station_bias_hz[None, :]
        else:
            raise ValueError(f"unknown fft_model={fft_model!r}")
        beat_resid = ((beat_model_hz - fft_fit) / float(sigma_fft_hz))[fft_keep_fit]
        return np.concatenate([path_resid, beat_resid])

    if fit_station_bias:
        p0_dyn = np.asarray(p0, dtype=np.float64)[:n_dyn]
        model0 = forward_model_for_kind(p0_dyn, t_rel_s, times_fit, rho_of_alt_m, model_kind)
        geo0 = model0["path_length_m"]
        doppler0 = model0["doppler_hz"]
        if fft_model in {"zero_beat", "signed_doppler"}:
            beat0 = -doppler0
        elif fft_model in {"ambiguity_residual", "range_offset_corrected_beat"}:
            beat0 = doppler0 - (chirp_rate_hz_per_s / gfit.C) * (measured_fit - geo0)
        else:
            raise ValueError(f"unknown fft_model={fft_model!r}")
        bias0 = np.zeros(3, dtype=np.float64)
        for col in range(3):
            good = fft_keep_fit[:, col]
            if np.any(good):
                bias0[col] = float(np.nanmedian(fft_fit[good, col] - beat0[good, col]))
        x0 = np.concatenate([p0_dyn, bias0])
        if model_kind == "ceplecha":
            dyn_lower = np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(cepl.MIN_RADIUS_M)])
            dyn_upper = np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(cepl.MAX_RADIUS_M)])
            dyn_scale = np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0])
        else:
            dyn_lower = np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4])
            dyn_upper = np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4])
            dyn_scale = np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4])
        lower = np.concatenate([dyn_lower, np.full(3, -500e3)])
        upper = np.concatenate([dyn_upper, np.full(3, 500e3)])
        x_scale = np.concatenate([dyn_scale, np.full(3, 5e4)])
    else:
        x0 = np.asarray(p0, dtype=np.float64)[:n_dyn]
        if model_kind == "ceplecha":
            lower = np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(cepl.MIN_RADIUS_M)])
            upper = np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(cepl.MAX_RADIUS_M)])
            x_scale = np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0])
        else:
            lower = np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4])
            upper = np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4])
            x_scale = np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4])

    x0 = np.minimum(np.maximum(x0, lower), upper)
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
    model = forward_model_for_kind(
        dyn_params,
        t_rel_s,
        times_fit,
        rho_of_alt_m,
        model_kind,
    )
    apparent = model["apparent_path_length_m"]
    geo = model["path_length_m"]
    path_rate = model["path_rate_mps"]
    doppler = model["doppler_hz"]
    if fft_model in {"zero_beat", "signed_doppler"}:
        beat_model_hz = -doppler + station_bias_hz[None, :]
    elif fft_model in {"ambiguity_residual", "range_offset_corrected_beat"}:
        beat_model_hz = doppler - (chirp_rate_hz_per_s / gfit.C) * (measured_fit - geo) + station_bias_hz[None, :]
    else:
        raise ValueError(f"unknown fft_model={fft_model!r}")
    path_resid_m = apparent - measured_fit
    beat_resid_hz = beat_model_hz - fft_fit
    if fft_model in {"zero_beat", "signed_doppler"}:
        fft_doppler_hz = -fft_fit
    elif fft_model in {"ambiguity_residual", "range_offset_corrected_beat"}:
        range_offset_frequency_hz = (chirp_rate_hz_per_s / gfit.C) * (measured_fit - geo)
        fft_doppler_hz = fft_fit + range_offset_frequency_hz
    else:
        raise ValueError(f"unknown fft_model={fft_model!r}")
    fft_path_rate_mps = -gfit.RADAR_WAVELENGTH_M * fft_doppler_hz
    path_rate_resid_mps = path_rate - fft_path_rate_mps
    normalized_path = path_resid_m / sigma_fit
    normalized_beat = np.full_like(beat_resid_hz, np.nan)
    normalized_beat[fft_keep_fit] = beat_resid_hz[fft_keep_fit] / float(sigma_fft_hz)
    retained_path_resid_m = path_resid_m[path_keep_fit]
    all_finite_path_resid_m = path_resid_m[np.isfinite(path_resid_m)]
    retained_fft_resid_hz = beat_resid_hz[fft_keep_fit]
    retained_path_rate_resid_mps = path_rate_resid_mps[fft_keep_fit]
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
        "path_keep": path_keep_fit,
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
        "rms_total_path_residual_m": float(np.sqrt(np.nanmean(retained_path_resid_m**2.0))) if retained_path_resid_m.size else np.nan,
        "all_finite_path_residual_rms_m": float(np.sqrt(np.nanmean(all_finite_path_resid_m**2.0))) if all_finite_path_resid_m.size else np.nan,
        "rms_fft_residual_hz": float(np.sqrt(np.nanmean(retained_fft_resid_hz**2.0))) if retained_fft_resid_hz.size else np.nan,
        "rms_path_rate_residual_mps": float(np.sqrt(np.nanmean(retained_path_rate_resid_mps**2.0))) if retained_path_rate_resid_mps.size else np.nan,
        "mean_abs_total_path_residual_m": finite_mean_abs(retained_path_resid_m),
        "mean_abs_fft_residual_hz": finite_mean_abs(retained_fft_resid_hz),
        "mean_abs_path_rate_residual_mps": finite_mean_abs(retained_path_rate_resid_mps),
        "weighted_rms": float(np.sqrt(np.nanmean(residual(result.x) ** 2.0))),
        "parameter_covariance": covariance["parameter_covariance"],
        "parameter_std": covariance["parameter_std"],
        "log10_radius_std": float(covariance["parameter_std"][6]) if model_kind == "ceplecha" and len(covariance["parameter_std"]) > 6 else np.nan,
        "covariance_available": bool(covariance["covariance_available"]),
        "covariance_degrees_of_freedom": int(covariance["covariance_degrees_of_freedom"]),
        "covariance_residual_variance": float(covariance["covariance_residual_variance"]),
        "n_points": int(len(times_fit)),
        "n_path_observations": int(np.count_nonzero(path_keep_fit)),
        "n_fft_observations": int(np.count_nonzero(fft_keep_fit)),
        "dynamical_model": str(model_kind),
        "fit_station_bias": bool(fit_station_bias),
        "fft_model": "zero_beat" if fft_model == "signed_doppler" else str(fft_model),
        "reference_chirp_rate_scale": float(reference_chirp_rate_scale),
        "reference_chirp_rate_hz_per_s": float(chirp_rate_hz_per_s),
        "nominal_chirp_rate_hz_per_s": float(gfit.NOMINAL_CHIRP_RATE_HZ_PER_S),
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "optimizer_cost": float(result.cost),
        "ceplecha_success": bool(model["ceplecha_success"]),
        "ceplecha_message": str(model["ceplecha_message"]),
    }


def mass_from_radius(radius_m):
    return (4.0 / 3.0) * np.pi * cepl.METEOROID_DENSITY_KG_M3 * float(radius_m) ** 3.0


def finite_mean_abs(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.mean(np.abs(finite)))


def constant_velocity_initial_params(fit):
    params = np.asarray(fit["params"], dtype=np.float64)
    return params[:6].copy()


def radius_uncertainty_is_large(joint_fit, max_log10_radius_std=DEFAULT_MAX_LOG10_RADIUS_STD):
    if str(joint_fit.get("dynamical_model", "ceplecha")) != "ceplecha":
        return False
    log10_radius_std = float(joint_fit.get("log10_radius_std", np.nan))
    if not np.isfinite(log10_radius_std):
        return True
    if log10_radius_std > float(max_log10_radius_std):
        return True
    radius_m = float(joint_fit.get("initial_radius_m", np.nan))
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        return True
    lo_log10 = np.log10(radius_m) - 1.96 * log10_radius_std
    hi_log10 = np.log10(radius_m) + 1.96 * log10_radius_std
    return bool(
        lo_log10 <= np.log10(cepl.MIN_RADIUS_M) + 1e-6
        or hi_log10 >= np.log10(cepl.MAX_RADIUS_M) - 1e-6
    )


def delay_clip_mask_from_fit(joint_fit, clip_m):
    if not (np.isfinite(clip_m) and clip_m > 0.0):
        return np.asarray(joint_fit["path_keep"], dtype=bool)
    return np.asarray(joint_fit["path_keep"], dtype=bool) & (np.abs(joint_fit["path_residuals_m"]) <= float(clip_m))


def delay_clip_is_fit_usable(path_keep, fft_keep, n_dyn, fit_station_bias):
    n_params = int(n_dyn) + (3 if fit_station_bias else 0)
    n_residuals = int(np.count_nonzero(path_keep) + np.count_nonzero(fft_keep))
    n_rows_with_delay = int(np.count_nonzero(np.any(path_keep, axis=1)))
    return bool(n_residuals > n_params and n_rows_with_delay >= base.MIN_POINTS)


def bad_fit_reasons(joint_fit, path_rms_limit_m, fft_rms_limit_hz):
    reasons = []
    path_rms = float(joint_fit.get("rms_total_path_residual_m", np.nan))
    fft_rms = float(joint_fit.get("rms_fft_residual_hz", np.nan))
    if not np.isfinite(path_rms) or path_rms > float(path_rms_limit_m):
        reasons.append(f"path_rms={path_rms:.3g}m")
    if np.any(joint_fit.get("fft_keep", np.zeros((0, 3), dtype=bool))):
        if not np.isfinite(fft_rms) or fft_rms > float(fft_rms_limit_hz):
            reasons.append(f"fft_rms={fft_rms:.3g}Hz")
    if not bool(joint_fit.get("optimizer_success", False)):
        reasons.append("optimizer_unsuccessful")
    return reasons


def fit_quality_score(joint_fit, path_rms_scale_m=50.0, fft_rms_scale_hz=500.0):
    path_rms = float(joint_fit.get("rms_total_path_residual_m", np.inf))
    fft_rms = float(joint_fit.get("rms_fft_residual_hz", np.inf))
    if not np.isfinite(path_rms):
        path_rms = np.inf
    if not np.isfinite(fft_rms):
        fft_rms = 0.0
    return float(path_rms / float(path_rms_scale_m) + fft_rms / float(fft_rms_scale_hz))


def refit_with_masks(
    measured,
    times_ns,
    rho_of_alt_m,
    p0,
    sigma_m,
    fft_obs,
    fft_keep,
    sigma_fft_hz,
    epoch_time_ns,
    fit_station_bias,
    fft_model,
    reference_chirp_rate_scale,
    path_keep,
    model_kind,
):
    return fit_joint_delay_doppler(
        measured,
        times_ns,
        rho_of_alt_m,
        p0,
        sigma_m,
        fft_obs["fft_offset_hz"],
        fft_keep,
        sigma_fft_hz,
        keep_rows=np.ones(len(times_ns), dtype=bool),
        epoch_time_ns=epoch_time_ns,
        fit_station_bias=fit_station_bias,
        fft_model=fft_model,
        reference_chirp_rate_scale=reference_chirp_rate_scale,
        path_keep=path_keep,
        model_kind=model_kind,
    )


def try_recover_bad_fit(
    joint_fit,
    delay_seed_params,
    measured,
    times_ns,
    rho_of_alt_m,
    sigma_m,
    fft_obs,
    sigma_fft_hz,
    epoch_time_ns,
    fit_station_bias,
    fft_model,
    reference_chirp_rate_scale,
    delay_clip_m,
    fft_clip_hz,
    path_rms_limit_m,
    fft_rms_limit_hz,
    max_retry=DEFAULT_BAD_FIT_MAX_RETRY,
):
    best = joint_fit
    best_score = fit_quality_score(best)
    recovery_notes = []
    seed_path_keep = np.isfinite(np.asarray(measured, dtype=np.float64))
    seed_fft_keep = np.asarray(fft_obs["fft_keep"], dtype=bool)
    if delay_clip_is_fit_usable(seed_path_keep, seed_fft_keep, n_dyn=7, fit_station_bias=fit_station_bias):
        try:
            seeded = refit_with_masks(
                measured,
                times_ns,
                rho_of_alt_m,
                np.asarray(delay_seed_params, dtype=np.float64),
                sigma_m,
                fft_obs,
                seed_fft_keep,
                sigma_fft_hz,
                epoch_time_ns,
                fit_station_bias,
                fft_model,
                reference_chirp_rate_scale,
                seed_path_keep,
                "ceplecha",
            )
            seeded["bad_fit_recovery_step"] = "delay_seed_ceplecha"
            score = fit_quality_score(seeded)
            if score < best_score:
                best = seeded
                best_score = score
        except Exception as exc:
            recovery_notes.append(f"delay_seed_ceplecha_failed:{exc}")
    if delay_clip_is_fit_usable(seed_path_keep, seed_fft_keep, n_dyn=6, fit_station_bias=fit_station_bias):
        try:
            seeded_constant = refit_with_masks(
                measured,
                times_ns,
                rho_of_alt_m,
                np.asarray(delay_seed_params, dtype=np.float64)[:6],
                sigma_m,
                fft_obs,
                seed_fft_keep,
                sigma_fft_hz,
                epoch_time_ns,
                fit_station_bias,
                fft_model,
                reference_chirp_rate_scale,
                seed_path_keep,
                "constant_velocity",
            )
            seeded_constant["fallback_from_ceplecha"] = True
            seeded_constant["fallback_reason"] = "bad_retained_residual"
            seeded_constant["bad_fit_recovery_step"] = "delay_seed_constant_velocity"
            score = fit_quality_score(seeded_constant)
            if score < best_score:
                best = seeded_constant
                best_score = score
        except Exception as exc:
            recovery_notes.append(f"delay_seed_constant_velocity_failed:{exc}")
    current = joint_fit
    for _idx in range(int(max_retry)):
        next_path_keep = delay_clip_mask_from_fit(current, delay_clip_m)
        next_fft_keep = np.asarray(current["fft_keep"], dtype=bool)
        if np.isfinite(fft_clip_hz) and fft_clip_hz > 0.0:
            next_fft_keep = next_fft_keep & (np.abs(current["fft_residuals_hz"]) <= float(fft_clip_hz))
        if (
            np.array_equal(next_path_keep, np.asarray(current["path_keep"], dtype=bool))
            and np.array_equal(next_fft_keep, np.asarray(current["fft_keep"], dtype=bool))
        ):
            break
        n_dyn = 7 if str(current.get("dynamical_model", "ceplecha")) == "ceplecha" else 6
        if not delay_clip_is_fit_usable(next_path_keep, next_fft_keep, n_dyn=n_dyn, fit_station_bias=fit_station_bias):
            break
        try:
            current = refit_with_masks(
                measured,
                times_ns,
                rho_of_alt_m,
                current["params"],
                sigma_m,
                fft_obs,
                next_fft_keep,
                sigma_fft_hz,
                epoch_time_ns,
                fit_station_bias,
                fft_model,
                reference_chirp_rate_scale,
                next_path_keep,
                str(current.get("dynamical_model", "ceplecha")),
            )
        except Exception as exc:
            recovery_notes.append(f"iterative_clip_failed:{exc}")
            break
        current["bad_fit_recovery_step"] = "iterative_clip"
        score = fit_quality_score(current)
        if score < best_score:
            best = current
            best_score = score

    reasons = bad_fit_reasons(best, path_rms_limit_m, fft_rms_limit_hz)
    if reasons:
        final_path_keep = delay_clip_mask_from_fit(best, delay_clip_m)
        final_fft_keep = np.asarray(best["fft_keep"], dtype=bool)
        if np.isfinite(fft_clip_hz) and fft_clip_hz > 0.0:
            final_fft_keep = final_fft_keep & (np.abs(best["fft_residuals_hz"]) <= float(fft_clip_hz))
        if delay_clip_is_fit_usable(final_path_keep, final_fft_keep, n_dyn=6, fit_station_bias=fit_station_bias):
            try:
                constant_fit = refit_with_masks(
                    measured,
                    times_ns,
                    rho_of_alt_m,
                    constant_velocity_initial_params(best),
                    sigma_m,
                    fft_obs,
                    final_fft_keep,
                    sigma_fft_hz,
                    epoch_time_ns,
                    fit_station_bias,
                    fft_model,
                    reference_chirp_rate_scale,
                    final_path_keep,
                    "constant_velocity",
                )
                constant_fit["fallback_from_ceplecha"] = str(best.get("dynamical_model", "ceplecha")) == "ceplecha"
                constant_fit["fallback_reason"] = "bad_retained_residual"
                constant_fit["bad_fit_recovery_step"] = "constant_velocity"
                score = fit_quality_score(constant_fit)
                if score < best_score:
                    best = constant_fit
                    best_score = score
            except Exception as exc:
                recovery_notes.append(f"constant_velocity_failed:{exc}")

    final_reasons = bad_fit_reasons(best, path_rms_limit_m, fft_rms_limit_hz)
    best["bad_fit_detected"] = bool(final_reasons)
    best["bad_fit_reasons"] = ";".join(final_reasons)
    best["bad_fit_recovery_notes"] = ";".join(recovery_notes)
    return best


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
    if str(joint_fit.get("dynamical_model", "ceplecha")) == "constant_velocity":
        return "constant velocity model\nr0, m0 not fitted"
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


def beat_residual_hz_to_total_path_rate_mps(freq_hz):
    return -gfit.RADAR_WAVELENGTH_M * np.asarray(freq_hz, dtype=np.float64)


def beat_residual_khz_to_total_path_rate_mps(freq_khz):
    return beat_residual_hz_to_total_path_rate_mps(np.asarray(freq_khz, dtype=np.float64) * 1e3)


def total_path_rate_mps_to_beat_residual_khz(path_rate_mps):
    return -np.asarray(path_rate_mps, dtype=np.float64) / gfit.RADAR_WAVELENGTH_M / 1e3


def sanya_beam_offsets_deg(positions_itrs_m):
    tx = np.asarray(gfit.LINK_TX_POSITIONS_M[0], dtype=np.float64)
    lat_deg, lon_deg, _alt_m = jcoord.ecef2geodetic(*tx)
    los_ecef = np.asarray(positions_itrs_m, dtype=np.float64) - tx[None, :]
    los_ecef = los_ecef / np.maximum(np.linalg.norm(los_ecef, axis=1, keepdims=True), 1e-30)
    los_enu = beam_hist.ecef_to_enu_vectors(los_ecef, lat_deg, lon_deg)
    site = beam_hist.gain_model.SITES[0]
    pointing = beam_hist.gain_model.unit(beam_hist.gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    east_axis, north_axis = beam_hist.local_sky_axes(pointing)
    return beam_hist.angular_offsets_deg(los_enu, pointing, east_axis, north_axis)


def solve_measured_positions_itrs(measured_total_paths_m, fitted_positions_itrs_m):
    measured = np.asarray(measured_total_paths_m, dtype=np.float64)
    fitted = np.asarray(fitted_positions_itrs_m, dtype=np.float64)
    positions = np.full_like(fitted, np.nan, dtype=np.float64)
    for idx, paths in enumerate(measured):
        if not np.all(np.isfinite(paths)):
            continue
        x0 = fitted[idx] if idx < len(fitted) and np.all(np.isfinite(fitted[idx])) else gfit.LINK_TX_POSITIONS_M[0]
        try:
            positions[idx] = gfit.solve_position_from_total_paths_m(paths, x0)
        except Exception:
            continue
    return positions


def fit_quality_annotation(joint_fit):
    v0_km_s, v0_sigma_km_s = initial_speed_uncertainty_km_s(joint_fit)
    delay_keep = np.asarray(joint_fit["path_keep"], dtype=bool)
    delay_abs_m = np.abs(np.asarray(joint_fit["path_residuals_m"], dtype=np.float64)[delay_keep])
    delay_mean_abs_m = (
        float(np.nanmean(delay_abs_m[np.isfinite(delay_abs_m)]))
        if np.any(np.isfinite(delay_abs_m))
        else np.nan
    )
    fft_keep = np.asarray(joint_fit["fft_keep"], dtype=bool)
    doppler_abs_mps = np.abs(
        beat_residual_hz_to_total_path_rate_mps(
            np.asarray(joint_fit["fft_residuals_hz"], dtype=np.float64)[fft_keep]
        )
    )
    doppler_mean_abs_mps = (
        float(np.nanmean(doppler_abs_mps[np.isfinite(doppler_abs_mps)]))
        if np.any(np.isfinite(doppler_abs_mps))
        else np.nan
    )
    return (
        f"mean |delay residual| = {delay_mean_abs_m:.1f} m\n"
        f"mean |Doppler residual| = {doppler_mean_abs_mps:.0f} m/s\n"
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
    beam_east_samples = []
    beam_north_samples = []
    n_try = 0
    model_kind = str(joint_fit.get("dynamical_model", "ceplecha"))
    while len(path_samples) < n_draws and n_try < n_draws * 8:
        n_try += 1
        trial = params + transform @ rng.standard_normal(len(params))
        if model_kind == "ceplecha":
            trial[6] = np.clip(trial[6], np.log10(cepl.MIN_RADIUS_M), np.log10(cepl.MAX_RADIUS_M))
        try:
            model = forward_model_for_kind(
                trial,
                joint_fit["t_rel_s"],
                joint_fit["time_ns"],
                rho_of_alt_m,
                model_kind,
            )
        except Exception:
            continue
        if not model["ceplecha_success"]:
            continue
        path = np.asarray(model["apparent_path_length_m"], dtype=np.float64)
        along_velocity = (np.asarray(model["v_gcrs_mps"], dtype=np.float64) @ along_axis) / 1e3
        beam_east_deg, beam_north_deg = sanya_beam_offsets_deg(model["x_itrs_m"])
        if np.all(np.isfinite(path)) and np.all(np.isfinite(along_velocity)):
            path_samples.append(path)
            along_velocity_samples.append(along_velocity)
            beam_east_samples.append(beam_east_deg)
            beam_north_samples.append(beam_north_deg)
    if len(path_samples) < max(12, n_draws // 4):
        return None
    path_samples = np.asarray(path_samples, dtype=np.float64)
    along_velocity_samples = np.asarray(along_velocity_samples, dtype=np.float64)
    beam_east_samples = np.asarray(beam_east_samples, dtype=np.float64)
    beam_north_samples = np.asarray(beam_north_samples, dtype=np.float64)
    return {
        "path_lo_m": np.nanpercentile(path_samples, 2.5, axis=0),
        "path_hi_m": np.nanpercentile(path_samples, 97.5, axis=0),
        "along_velocity_lo_km_s": np.nanpercentile(along_velocity_samples, 2.5, axis=0),
        "along_velocity_hi_km_s": np.nanpercentile(along_velocity_samples, 97.5, axis=0),
        "beam_east_lo_deg": np.nanpercentile(beam_east_samples, 2.5, axis=0),
        "beam_east_hi_deg": np.nanpercentile(beam_east_samples, 97.5, axis=0),
        "beam_north_lo_deg": np.nanpercentile(beam_north_samples, 2.5, axis=0),
        "beam_north_hi_deg": np.nanpercentile(beam_north_samples, 97.5, axis=0),
        "n_draws": int(len(path_samples)),
    }


def plot_joint_fit(event_id, delay_fit, joint_fit, output_base, rho_of_alt_m, snr_db=None):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), constrained_layout=True)
    t = joint_fit["t_rel_s"]
    colors = {"sanya": "#4c78a8", "danzhou": "#f58518", "wenchang": "#54a24b"}
    bands = model_uncertainty_bands(event_id, joint_fit, rho_of_alt_m)

    ax = axes[0, 0]
    fit_east_deg, fit_north_deg = sanya_beam_offsets_deg(joint_fit["x_itrs_m"])
    path_keep = np.asarray(joint_fit.get("path_keep", np.ones_like(joint_fit["measured_total_paths_m"], dtype=bool)), dtype=bool)
    sanya_snr = None
    if snr_db is not None:
        snr_arr = np.asarray(snr_db, dtype=np.float64)
        if snr_arr.ndim == 2 and snr_arr.shape[0] == len(fit_east_deg):
            sanya_snr = snr_arr[:, 0]
    if sanya_snr is None:
        position_keep = path_keep[:, 0] & np.isfinite(fit_east_deg) & np.isfinite(fit_north_deg)
        position_excluded = (~path_keep[:, 0]) & np.isfinite(fit_east_deg) & np.isfinite(fit_north_deg)
    else:
        position_keep = path_keep[:, 0] & np.isfinite(sanya_snr) & np.isfinite(fit_east_deg) & np.isfinite(fit_north_deg)
        position_excluded = (~path_keep[:, 0]) & np.isfinite(sanya_snr) & np.isfinite(fit_east_deg) & np.isfinite(fit_north_deg)
    grid_limit = 2.6
    grid = np.linspace(-grid_limit, grid_limit, 240)
    east_grid, north_grid = np.meshgrid(grid, grid)
    gain_grid = beam_hist.sanya_beam_relative_gain_db(east_grid, north_grid)
    ax.contour(
        east_grid,
        north_grid,
        gain_grid,
        levels=[-30.0, -20.0, -13.0, -10.0, -3.0],
        colors=["0.65", "0.50", "0.35", "0.25", "0.05"],
        linewidths=[0.6, 0.7, 0.8, 0.9, 1.1],
        linestyles=[":", "--", "-.", "-", "-"],
    )
    if bands is not None:
        step = max(1, len(fit_east_deg) // 12)
        idx = np.arange(0, len(fit_east_deg), step)
        ax.errorbar(
            fit_east_deg[idx],
            fit_north_deg[idx],
            xerr=np.vstack([
                fit_east_deg[idx] - bands["beam_east_lo_deg"][idx],
                bands["beam_east_hi_deg"][idx] - fit_east_deg[idx],
            ]),
            yerr=np.vstack([
                fit_north_deg[idx] - bands["beam_north_lo_deg"][idx],
                bands["beam_north_hi_deg"][idx] - fit_north_deg[idx],
            ]),
            fmt="none",
            ecolor="#1b7837",
            elinewidth=0.9,
            alpha=0.35,
            capsize=0,
            zorder=1,
        )
    ax.plot(fit_east_deg, fit_north_deg, color="#1b7837", lw=1.8, label="fit", zorder=2)
    snr_color = sanya_snr
    if np.any(position_excluded):
        ax.scatter(
            fit_east_deg[position_excluded],
            fit_north_deg[position_excluded],
            s=22,
            facecolors="none",
            edgecolors="0.45",
            linewidths=0.8,
            alpha=0.35,
            label="excluded",
            zorder=2,
        )
    if snr_color is None:
        sc_meas = ax.scatter(
            fit_east_deg[position_keep],
            fit_north_deg[position_keep],
            s=22,
            color="#4c78a8",
            alpha=0.85,
            label="Sanya echo",
            zorder=3,
        )
    else:
        finite_snr = snr_color[position_keep & np.isfinite(snr_color)]
        if finite_snr.size:
            snr_vmin = float(np.nanmin(finite_snr))
            snr_vmax = float(np.nanmax(finite_snr))
            if snr_vmax <= snr_vmin:
                snr_vmin -= 0.5
                snr_vmax += 0.5
        else:
            snr_vmin = None
            snr_vmax = None
        sc_meas = ax.scatter(
            fit_east_deg[position_keep],
            fit_north_deg[position_keep],
            c=snr_color[position_keep],
            s=24,
            cmap="viridis",
            vmin=snr_vmin,
            vmax=snr_vmax,
            edgecolors="none",
            linewidths=0.0,
            alpha=0.9,
            label="Sanya echo",
            zorder=3,
        )
        cb = fig.colorbar(sc_meas, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Sanya matched-filter SNR (dB)")
    ax.axhline(0.0, color="0.35", lw=0.7, alpha=0.6)
    ax.axvline(0.0, color="0.35", lw=0.7, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-grid_limit, grid_limit)
    ax.set_ylim(-grid_limit, grid_limit)
    ax.set_xlabel("Sanya beam east offset (deg)")
    ax.set_ylabel("Sanya beam north offset (deg)")
    ax.set_title("Head positions in Sanya TX beam")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
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
    ax.legend(loc="upper right", fontsize=8)
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
        keep = np.asarray(joint_fit["path_keep"], dtype=bool)[:, col]
        finite = np.isfinite(joint_fit["path_residuals_m"][:, col])
        excluded = finite & ~keep
        if np.any(excluded):
            ax.scatter(
                t[excluded],
                joint_fit["path_residuals_m"][excluded, col],
                s=20,
                facecolors="none",
                edgecolors=colors[site],
                linewidths=0.8,
                alpha=0.35,
            )
        ax.scatter(
            t[keep],
            joint_fit["path_residuals_m"][keep, col],
            s=22,
            color=colors[site],
            label=SITE_LABELS[col],
            alpha=0.85,
        )
    ax.axhline(0, color="0.25", lw=1.0)
    ax.set_xlabel("Time since fit epoch (s)")
    ax.set_ylabel("Delay residual (m)")
    ax.set_title("Delay residuals")
    ax.set_ylim(-100.0, 100.0)
    ax.grid(True, alpha=0.25)
    if snr_db is not None:
        snr_arr = np.asarray(snr_db, dtype=np.float64)
        if snr_arr.ndim == 2 and snr_arr.shape[0] == len(t):
            if np.any(np.isfinite(snr_arr)):
                ax_snr = ax.twinx()
                finite_snr = snr_arr[np.isfinite(snr_arr)]
                lo = float(np.nanmin(finite_snr))
                hi = float(np.nanmax(finite_snr))
                if hi <= lo:
                    lo -= 0.5
                    hi += 0.5
                pad = max(0.5, 0.06 * (hi - lo))
                for col, site in enumerate(SITE_ORDER):
                    good = np.isfinite(snr_arr[:, col])
                    if not np.any(good):
                        continue
                    (snr_handle,) = ax_snr.plot(
                        t[good],
                        snr_arr[good, col],
                        color=colors[site],
                        lw=1.1,
                        alpha=0.75,
                        label=f"{SITE_LABELS[col]} SNR",
                        zorder=0,
                    )
                ax_snr.set_ylabel("Matched-filter SNR (dB)")
                ax_snr.set_ylim(lo - pad, hi + pad)
                ax_snr.grid(False)

    ax = axes[1, 1]
    for col, site in enumerate(SITE_ORDER):
        keep = joint_fit["fft_keep"][:, col]
        finite = np.isfinite(joint_fit["fft_residuals_hz"][:, col])
        excluded = finite & ~keep
        if np.any(excluded):
            ax.scatter(
                t[excluded],
                joint_fit["fft_residuals_hz"][excluded, col] / 1e3,
                s=20,
                facecolors="none",
                edgecolors=colors[site],
                linewidths=0.8,
                alpha=0.35,
            )
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
    ax.set_title("Beat-frequency residuals")
    ax.set_ylim(
        total_path_rate_mps_to_beat_residual_khz(1000.0),
        total_path_rate_mps_to_beat_residual_khz(-1000.0),
    )
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="0.35", label="excluded"))
    labels.append("excluded")
    ax.legend(handles, labels, loc="best", fontsize=8)
    secax = ax.secondary_yaxis(
        "right",
        functions=(beat_residual_khz_to_total_path_rate_mps, total_path_rate_mps_to_beat_residual_khz),
    )
    secax.set_ylabel("Equivalent total-path-rate residual (m/s)")

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
    fft_time_pad_us,
    reference_chirp_rate_scale,
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
        h.attrs["fft_time_pad_us"] = float(fft_time_pad_us)
        h.attrs["reference_chirp_rate_scale"] = float(reference_chirp_rate_scale)
        h.attrs["reference_chirp_rate_hz_per_s"] = float(gfit.NOMINAL_CHIRP_RATE_HZ_PER_S * float(reference_chirp_rate_scale))
        h.attrs["nominal_chirp_rate_hz_per_s"] = float(gfit.NOMINAL_CHIRP_RATE_HZ_PER_S)
        h.attrs["joint_frequency_model"] = (
            "Least-squares residual has two measurement blocks. Range block: "
            "ordinary matched-filter path measurements without Doppler correction "
            "are fixed measurements and are fit with "
            "measured_path = geometric_path + c*f_D_model/chirp_rate. "
            "Frequency block: dechirped single-pulse FFT beat frequencies are "
            "fixed measurements and are fit with "
            "f_beat = f_D_model - (chirp_rate/c)*(measured_path - geometric_path). "
            "Thus the Doppler prediction includes the model range-offset correction. "
            "The chirp_rate is the calibrated reference chirp rate."
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
    parser.add_argument("--fft-time-pad-us", type=float, default=DEFAULT_FFT_TIME_PAD_US)
    parser.add_argument("--reference-chirp-rate-scale", type=float, default=DEFAULT_REFERENCE_CHIRP_RATE_SCALE)
    parser.add_argument("--range-upsample-factor", type=int, default=32)
    parser.add_argument("--matched-filter-same-offset-samples", type=float, default=0.0)
    parser.add_argument("--snr-min-db", type=float, default=15.0)
    parser.add_argument("--prominence-min-db", type=float, default=8.0)
    parser.add_argument("--sigma-fft-hz", type=float, default=5000.0)
    parser.add_argument("--clip-fft-residual-khz", type=float, default=1.5)
    parser.add_argument("--final-delay-residual-clip-m", type=float, default=DEFAULT_FINAL_DELAY_RESIDUAL_CLIP_M)
    parser.add_argument("--max-log10-radius-std-before-constant-velocity", type=float, default=DEFAULT_MAX_LOG10_RADIUS_STD)
    parser.add_argument("--bad-fit-retained-path-rms-m", type=float, default=DEFAULT_BAD_FIT_RETAINED_PATH_RMS_M)
    parser.add_argument("--bad-fit-retained-fft-rms-hz", type=float, default=DEFAULT_BAD_FIT_RETAINED_FFT_RMS_HZ)
    parser.add_argument("--bad-fit-max-retry", type=int, default=DEFAULT_BAD_FIT_MAX_RETRY)
    parser.add_argument("--min-geometric-points", type=int, default=base.MIN_POINTS)
    parser.add_argument("--system-noise-h5", default=DEFAULT_SYSTEM_NOISE_H5)
    parser.add_argument("--manual-outlier-h5", default=None)
    parser.add_argument("--random-initial-guesses", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--force-model-reevaluation", action="store_true")
    parser.add_argument("--coincident-delay-weight", type=float, default=DEFAULT_COINCIDENT_DELAY_WEIGHT)
    parser.add_argument(
        "--fft-model",
        choices=("range_offset_corrected_beat", "zero_beat", "signed_doppler", "ambiguity_residual"),
        default="range_offset_corrected_beat",
    )
    parser.add_argument("--fit-station-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--range-measurement",
        choices=("uncorrected", "reference-doppler"),
        default="uncorrected",
        help="Use fd=0 matched-filter delays, or the older reference-Doppler-refined gates.",
    )
    parser.add_argument("--output-base", default=None)
    parser.add_argument("--seed-from-existing-h5", default=None)
    parser.add_argument("--fit-mode", choices=("joint", "delay-only"), default="joint")
    args = parser.parse_args()
    min_geometric_points = int(args.min_geometric_points)
    if min_geometric_points < 3:
        raise ValueError("--min-geometric-points must be at least 3")
    original_min_points = int(base.MIN_POINTS)
    base.MIN_POINTS = min_geometric_points

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
    noise_power = RawVoltageNoisePower(args.system_noise_h5)
    snr_by_site = {
        site: normalized_matched_filter_snr_db(site_data[site], refined[f"{site}_gate"], site, noise_power)
        for site in SITE_ORDER
    }
    for site in SITE_ORDER:
        site_data[site]["snr_peak_db_rti_median"] = np.asarray(site_data[site]["snr_peak_db"], dtype=np.float64)
        site_data[site]["snr_peak_db"] = snr_by_site[site]

    seed_measured, seed_times_ns, _seed_beijing_ns, seed_snr_db, _seed_source_indices = base.matched_measurements_from_sites(
        san_event,
        dan_event,
        wen_event,
        site_data,
        refined,
    )
    seed_order = np.argsort(seed_times_ns)
    seed_measured = seed_measured[seed_order]
    seed_times_ns = seed_times_ns[seed_order]
    seed_snr_db = seed_snr_db[seed_order]
    points, seed_keep_geo = base.triangulate_points(seed_measured, san_event.az_deg, san_event.el_deg)
    seed_measured = seed_measured[seed_keep_geo]
    seed_times_ns = seed_times_ns[seed_keep_geo]
    seed_snr_db = seed_snr_db[seed_keep_geo]
    use_coincident_seed = len(seed_times_ns) >= min_geometric_points

    measured, times_ns, _beijing_ns, snr_db, source_indices = assemble_union_measurements_from_sites(
        {"sanya": san_event, "danzhou": dan_event, "wenchang": wen_event},
        site_data,
        refined,
        snr_by_site,
    )
    order = np.argsort(times_ns)
    measured = measured[order]
    times_ns = times_ns[order]
    snr_db = snr_db[order]
    source_indices = source_indices[order]
    path_keep_initial = np.isfinite(measured)
    manual_path_outlier, manual_fft_outlier = load_manual_outlier_masks(args.manual_outlier_h5, event_id, times_ns)
    path_keep_initial &= ~manual_path_outlier
    if np.count_nonzero(path_keep_initial) < min_geometric_points * 3:
        raise RuntimeError("Too few union delay observations after filtering")
    fit_epoch_time_ns = int(times_ns[0])

    sigma_model = {"sigma_floor_m": 33.39, "sigma_0_m": 236.9}
    sigma_m = finite_sigma_from_snr_db(snr_db, sigma_model["sigma_floor_m"], sigma_model["sigma_0_m"])
    coincident_delay_rows = np.all(path_keep_initial, axis=1)
    if np.isfinite(args.coincident_delay_weight) and args.coincident_delay_weight > 1.0:
        sigma_m[coincident_delay_rows, :] = sigma_m[coincident_delay_rows, :] / float(args.coincident_delay_weight)
    existing_seed, existing_seed_source = seed_params_from_existing_h5(args.seed_from_existing_h5, fit_epoch_time_ns)
    if existing_seed is not None:
        reference_points = reference_points_itrs(fit0, times_ns)
        rho_of_alt_m, _msis_meta = base.density_interpolator(times_ns, reference_points)
        delay_seed_params = existing_seed
        delay_seed_source = existing_seed_source
        guesses = [existing_seed]
        delay_fit = {
            "params": delay_seed_params,
            "rms_total_path_residual_m": np.nan,
            "weighted_rms": np.nan,
            "initial_radius_m": float(10.0 ** delay_seed_params[6]),
            "initial_mass_kg": mass_from_radius(float(10.0 ** delay_seed_params[6])),
            "seed_source": delay_seed_source,
        }
    elif use_coincident_seed:
        seed_sigma_m = finite_sigma_from_snr_db(seed_snr_db, sigma_model["sigma_floor_m"], sigma_model["sigma_0_m"])
        rho_of_alt_m, _msis_meta = base.density_interpolator(seed_times_ns, points)
        guesses = cepl.unique_initial_guesses(points, seed_times_ns, reference_fit=fit0)
        guesses = shift_constant_velocity_epoch(guesses, int(seed_times_ns[0]), fit_epoch_time_ns)
        delay_fit = cepl.fit_ceplecha_multistart(
            seed_measured,
            seed_times_ns,
            rho_of_alt_m,
            guesses,
            sigma_m=seed_sigma_m,
            keep_rows=np.ones(len(seed_times_ns), dtype=bool),
            epoch_time_ns=fit_epoch_time_ns,
        )
        delay_seed_source = "coincident_triangulation"
        delay_seed_params = delay_fit["params"]
    else:
        reference_points = reference_points_itrs(fit0, times_ns)
        rho_of_alt_m, _msis_meta = base.density_interpolator(times_ns, reference_points)
        guesses = [cepl.reference_state_guess(fit0, fit_epoch_time_ns)]
        guesses = [guess for guess in guesses if guess is not None and np.all(np.isfinite(guess[:7]))]
        delay_seed_source = "reference_union_times"
        if not guesses:
            raise RuntimeError("No finite reference seed for sparse-coincident event")
        delay_seed_params = guesses[0]
        delay_seed_radius_m = float(10.0 ** delay_seed_params[6])
        delay_fit = {
            "params": delay_seed_params,
            "rms_total_path_residual_m": np.nan,
            "weighted_rms": np.nan,
            "initial_radius_m": delay_seed_radius_m,
            "initial_mass_kg": float(
                (4.0 / 3.0)
                * np.pi
                * delay_seed_radius_m**3.0
                * cepl.METEOROID_DENSITY_KG_M3
            ),
            "seed_source": delay_seed_source,
        }
    guesses = add_randomized_initial_guesses(guesses, args.random_initial_guesses, seed=args.random_seed)
    fft_obs = estimate_fft_observations(
        site_data,
        refined,
        source_indices,
        args.zero_pad_factor,
        args.snr_min_db,
        args.prominence_min_db,
        gate_upsample_factor=args.fft_gate_upsample_factor,
        center_offset_samples=args.fft_center_offset_samples,
        time_pad_us=args.fft_time_pad_us,
        chirp_rate_scale=args.reference_chirp_rate_scale,
    )
    fft_obs["fft_keep"] = np.asarray(fft_obs["fft_keep"], dtype=bool) & ~manual_fft_outlier
    fit_station_bias = bool(args.fit_station_bias)
    if args.fit_mode == "delay-only":
        fft_obs["fft_keep"] = np.zeros_like(fft_obs["fft_keep"], dtype=bool)
        fit_station_bias = False
    joint_fit = fit_joint_delay_doppler(
        measured,
        times_ns,
        rho_of_alt_m,
        delay_seed_params,
        sigma_m,
        fft_obs["fft_offset_hz"],
        fft_obs["fft_keep"],
        args.sigma_fft_hz,
        keep_rows=np.ones(len(times_ns), dtype=bool),
        epoch_time_ns=fit_epoch_time_ns,
        fit_station_bias=fit_station_bias,
        fft_model=args.fft_model,
        reference_chirp_rate_scale=args.reference_chirp_rate_scale,
        path_keep=path_keep_initial,
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
                epoch_time_ns=fit_epoch_time_ns,
                fit_station_bias=fit_station_bias,
                fft_model=args.fft_model,
                reference_chirp_rate_scale=args.reference_chirp_rate_scale,
                path_keep=path_keep_initial,
            )
            joint_fit["fft_clip_limit_hz"] = float(clip_limit_hz)
            joint_fit["n_fft_clipped_observations"] = int(np.count_nonzero(fft_obs["fft_keep"]) - np.count_nonzero(clipped_fft_keep))
        else:
            joint_fit["fft_clip_limit_hz"] = float(clip_limit_hz)
            joint_fit["n_fft_clipped_observations"] = 0
    final_fft_keep = np.asarray(joint_fit["fft_keep"], dtype=bool)
    final_path_keep = delay_clip_mask_from_fit(joint_fit, args.final_delay_residual_clip_m)
    n_delay_clipped = int(np.count_nonzero(joint_fit["path_keep"]) - np.count_nonzero(final_path_keep))
    if n_delay_clipped > 0 and delay_clip_is_fit_usable(
        final_path_keep,
        final_fft_keep,
        n_dyn=7,
        fit_station_bias=fit_station_bias,
    ):
        joint_fit = fit_joint_delay_doppler(
            measured,
            times_ns,
            rho_of_alt_m,
            joint_fit["params"],
            sigma_m,
            fft_obs["fft_offset_hz"],
            final_fft_keep,
            args.sigma_fft_hz,
            keep_rows=np.ones(len(times_ns), dtype=bool),
            epoch_time_ns=fit_epoch_time_ns,
            fit_station_bias=fit_station_bias,
            fft_model=args.fft_model,
            reference_chirp_rate_scale=args.reference_chirp_rate_scale,
            path_keep=final_path_keep,
            model_kind="ceplecha",
        )
        joint_fit["delay_clip_limit_m"] = float(args.final_delay_residual_clip_m)
        joint_fit["n_delay_clipped_observations"] = n_delay_clipped
    else:
        joint_fit["delay_clip_limit_m"] = float(args.final_delay_residual_clip_m)
        joint_fit["n_delay_clipped_observations"] = 0
        final_path_keep = np.asarray(joint_fit["path_keep"], dtype=bool)
    if radius_uncertainty_is_large(
        joint_fit,
        max_log10_radius_std=args.max_log10_radius_std_before_constant_velocity,
    ):
        inherited_n_delay_clipped = int(joint_fit.get("n_delay_clipped_observations", 0))
        constant_fit = fit_joint_delay_doppler(
            measured,
            times_ns,
            rho_of_alt_m,
            constant_velocity_initial_params(joint_fit),
            sigma_m,
            fft_obs["fft_offset_hz"],
            final_fft_keep,
            args.sigma_fft_hz,
            keep_rows=np.ones(len(times_ns), dtype=bool),
            epoch_time_ns=fit_epoch_time_ns,
            fit_station_bias=fit_station_bias,
            fft_model=args.fft_model,
            reference_chirp_rate_scale=args.reference_chirp_rate_scale,
            path_keep=final_path_keep,
            model_kind="constant_velocity",
        )
        constant_fit["delay_clip_limit_m"] = float(args.final_delay_residual_clip_m)
        constant_fit["n_delay_clipped_observations"] = inherited_n_delay_clipped
        constant_fit["fallback_from_ceplecha"] = True
        constant_fit["fallback_reason"] = "large_radius_uncertainty"
        constant_fit["fallback_log10_radius_std"] = float(joint_fit.get("log10_radius_std", np.nan))
        constant_fit["max_log10_radius_std_before_constant_velocity"] = float(args.max_log10_radius_std_before_constant_velocity)
        joint_fit = constant_fit
    else:
        joint_fit["fallback_from_ceplecha"] = False
        joint_fit["fallback_reason"] = ""
        joint_fit["fallback_log10_radius_std"] = float(joint_fit.get("log10_radius_std", np.nan))
        joint_fit["max_log10_radius_std_before_constant_velocity"] = float(args.max_log10_radius_std_before_constant_velocity)
    if args.force_model_reevaluation and str(joint_fit.get("dynamical_model", "ceplecha")) == "ceplecha":
        try:
            constant_candidate = fit_joint_delay_doppler(
                measured,
                times_ns,
                rho_of_alt_m,
                constant_velocity_initial_params(joint_fit),
                sigma_m,
                fft_obs["fft_offset_hz"],
                np.asarray(joint_fit["fft_keep"], dtype=bool),
                args.sigma_fft_hz,
                keep_rows=np.ones(len(times_ns), dtype=bool),
                epoch_time_ns=fit_epoch_time_ns,
                fit_station_bias=fit_station_bias,
                fft_model=args.fft_model,
                reference_chirp_rate_scale=args.reference_chirp_rate_scale,
                path_keep=np.asarray(joint_fit["path_keep"], dtype=bool),
                model_kind="constant_velocity",
            )
            constant_candidate["fallback_from_ceplecha"] = True
            constant_candidate["fallback_reason"] = "force_model_reevaluation_constant_velocity_preferred"
            constant_candidate["fallback_log10_radius_std"] = float(joint_fit.get("log10_radius_std", np.nan))
            constant_candidate["max_log10_radius_std_before_constant_velocity"] = float(args.max_log10_radius_std_before_constant_velocity)
            if fit_quality_score(constant_candidate) < fit_quality_score(joint_fit):
                joint_fit = constant_candidate
        except Exception as exc:
            joint_fit["force_model_reevaluation_warning"] = str(exc)
    joint_fit["force_model_reevaluation"] = bool(args.force_model_reevaluation)
    pre_recovery_reasons = bad_fit_reasons(
        joint_fit,
        args.bad_fit_retained_path_rms_m,
        args.bad_fit_retained_fft_rms_hz,
    )
    if pre_recovery_reasons:
        inherited_n_delay_clipped = int(joint_fit.get("n_delay_clipped_observations", 0))
        joint_fit = try_recover_bad_fit(
            joint_fit,
            delay_fit["params"],
            measured,
            times_ns,
            rho_of_alt_m,
            sigma_m,
            fft_obs,
            args.sigma_fft_hz,
            fit_epoch_time_ns,
            fit_station_bias,
            args.fft_model,
            args.reference_chirp_rate_scale,
            args.final_delay_residual_clip_m,
            float(args.clip_fft_residual_khz) * 1e3,
            args.bad_fit_retained_path_rms_m,
            args.bad_fit_retained_fft_rms_hz,
            max_retry=args.bad_fit_max_retry,
        )
        joint_fit["n_delay_clipped_observations"] = int(
            max(
                inherited_n_delay_clipped,
                np.count_nonzero(path_keep_initial) - np.count_nonzero(np.asarray(joint_fit["path_keep"], dtype=bool)),
            )
        )
        joint_fit["delay_clip_limit_m"] = float(args.final_delay_residual_clip_m)
        joint_fit["pre_recovery_bad_fit_reasons"] = ";".join(pre_recovery_reasons)
    else:
        joint_fit["bad_fit_detected"] = False
        joint_fit["bad_fit_reasons"] = ""
        joint_fit["bad_fit_recovery_notes"] = ""
        joint_fit["pre_recovery_bad_fit_reasons"] = ""
    joint_fit["bad_fit_retained_path_rms_limit_m"] = float(args.bad_fit_retained_path_rms_m)
    joint_fit["bad_fit_retained_fft_rms_limit_hz"] = float(args.bad_fit_retained_fft_rms_hz)
    joint_fit["bad_fit_max_retry"] = int(args.bad_fit_max_retry)
    joint_fit["coincident_delay_weight"] = float(args.coincident_delay_weight)
    joint_fit["n_coincident_delay_constraint_rows"] = int(np.count_nonzero(coincident_delay_rows))
    joint_fit["min_geometric_points"] = int(min_geometric_points)
    joint_fit["default_min_geometric_points"] = int(original_min_points)
    joint_fit["delay_seed_source"] = str(delay_seed_source)
    joint_fit["used_coincident_delay_seed"] = bool(use_coincident_seed and existing_seed is None)
    joint_fit["fit_mode"] = str(args.fit_mode)
    if args.seed_from_existing_h5:
        joint_fit["seed_from_existing_h5"] = os.path.abspath(args.seed_from_existing_h5)
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
        args.fft_time_pad_us,
        args.reference_chirp_rate_scale,
    )
    plot_joint_fit(event_id, delay_fit, joint_fit, output_base, rho_of_alt_m, snr_db=snr_db)
    print(f"event_id={event_id}")
    print(f"n_points={joint_fit['n_points']}")
    print(f"n_path_observations={joint_fit['n_path_observations']}")
    print(f"n_fft_observations={joint_fit['n_fft_observations']}")
    print(f"n_delay_clipped_observations={joint_fit.get('n_delay_clipped_observations', 0)}")
    print(f"n_coincident_delay_constraint_rows={joint_fit.get('n_coincident_delay_constraint_rows', 0)}")
    print(f"coincident_delay_weight={joint_fit.get('coincident_delay_weight', np.nan)}")
    print(f"dynamical_model={joint_fit.get('dynamical_model', 'ceplecha')}")
    print(f"fallback_reason={joint_fit.get('fallback_reason', '')}")
    print(f"fallback_log10_radius_std={joint_fit.get('fallback_log10_radius_std', np.nan)}")
    print(f"bad_fit_detected={joint_fit.get('bad_fit_detected', False)}")
    print(f"pre_recovery_bad_fit_reasons={joint_fit.get('pre_recovery_bad_fit_reasons', '')}")
    print(f"bad_fit_reasons={joint_fit.get('bad_fit_reasons', '')}")
    print(f"bad_fit_recovery_step={joint_fit.get('bad_fit_recovery_step', '')}")
    print(f"delay_only_path_rms_m={delay_fit['rms_total_path_residual_m']:.3f}")
    print(f"joint_path_rms_m={joint_fit['rms_total_path_residual_m']:.3f}")
    print(f"joint_path_mean_abs_m={joint_fit['mean_abs_total_path_residual_m']:.3f}")
    print(f"joint_fft_rms_hz={joint_fit['rms_fft_residual_hz']:.3f}")
    print(f"joint_fft_mean_abs_hz={joint_fit['mean_abs_fft_residual_hz']:.3f}")
    print(f"joint_path_rate_rms_mps={joint_fit['rms_path_rate_residual_mps']:.3f}")
    print(f"joint_path_rate_mean_abs_mps={joint_fit['mean_abs_path_rate_residual_mps']:.3f}")
    print(f"delay_only_radius_um={delay_fit['initial_radius_m'] * 1e6:.3f}")
    print(f"joint_radius_um={joint_fit['initial_radius_m'] * 1e6:.3f}")
    print(f"joint_initial_mass_kg={joint_fit['initial_mass_kg']:.6e}")
    print(f"joint_final_radius_um={joint_fit['radius_m'][-1] * 1e6:.3f}")
    print(f"joint_final_mass_kg={joint_fit['mass_kg'][-1]:.6e}")
    print(f"reference_chirp_rate_scale={args.reference_chirp_rate_scale:.9f}")
    print(f"output_h5={output_base}.h5")
    print(f"output_png={output_base}.png")


if __name__ == "__main__":
    main()
