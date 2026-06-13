import json
import os

import h5py
import jcoord
import numpy as np
import scipy.optimize as so

import fit_rank02_ballistic_snr as ballistic
import measure_rank02_single_pulse_acf_doppler as acf
import test_rank02_range_interpolation as interp


SCRIPT_VERSION = "v20260611a"
UPSAMPLE_FACTOR = 4
OUTPUT_BASE = os.path.join("results", f"rank02_ballistic_with_acf_doppler_{SCRIPT_VERSION}")
MIN_DOPPLER_SIGMA_HZ = 1500.0


def build_range_and_doppler_measurements():
    fit = interp.load_reference_fit()
    site_data = {site: interp.load_site(site, fit) for site in interp.SITE_ORDER}
    coarse_gates = interp.precompute_coarse_gates(site_data)

    refined = {}
    for site in interp.SITE_ORDER:
        fine_gate, fine_range_km, _power_db = interp.refine_site_ranges(site_data[site], UPSAMPLE_FACTOR, coarse_gates[site])
        refined[f"{site}_gate"] = fine_gate
        refined[site] = fine_range_km

    acf_results = {
        site: acf.measure_site(site, site_data[site], refined[f"{site}_gate"], upsample_factor=UPSAMPLE_FACTOR)
        for site in interp.SITE_ORDER
    }

    measured, times_ns, source_indices = interp.matched_measurements(site_data, refined)
    snr_db = np.column_stack(
        [
            site_data["sanya"]["snr_peak_db"][source_indices[:, 0]],
            site_data["danzhou"]["snr_peak_db"][source_indices[:, 1]],
            site_data["wenchang"]["snr_peak_db"][source_indices[:, 2]],
        ]
    )
    doppler_hz = np.column_stack(
        [
            acf_results["sanya"]["acf_measured_hz"][source_indices[:, 0]],
            acf_results["danzhou"]["acf_measured_hz"][source_indices[:, 1]],
            acf_results["wenchang"]["acf_measured_hz"][source_indices[:, 2]],
        ]
    )
    doppler_mask = np.isfinite(doppler_hz)
    return site_data, measured, times_ns, snr_db, doppler_hz, doppler_mask, acf_results


def doppler_sigma_by_station(acf_results):
    sigma = {}
    summary = {}
    for site, data in acf_results.items():
        good = np.isfinite(data["acf_residual_hz"])
        residual = data["acf_residual_hz"][good]
        if len(residual) >= 3:
            med = float(np.nanmedian(residual))
            robust = float(1.4826 * np.nanmedian(np.abs(residual - med)))
        else:
            med = np.nan
            robust = np.nan
        sig = max(robust if np.isfinite(robust) else MIN_DOPPLER_SIGMA_HZ, MIN_DOPPLER_SIGMA_HZ)
        sigma[site] = sig
        summary[site] = {
            "n_high_coherence": int(np.count_nonzero(good)),
            "n_total": int(len(good)),
            "median_residual_hz": med if np.isfinite(med) else None,
            "robust_sigma_hz": robust if np.isfinite(robust) else None,
            "sigma_used_hz": float(sig),
            "median_coherence": float(np.nanmedian(data["coherence"][good])) if np.any(good) else None,
        }
    return np.asarray([sigma["sanya"], sigma["danzhou"], sigma["wenchang"]], dtype=np.float64), summary


def predict_ballistic_paths_and_omega(params, t_rel_s, rho_of_alt_m):
    x_itrs, v_itrs, b_drag = ballistic.propagate_ballistic_itrs(params, t_rel_s, rho_of_alt_m)
    total_paths_m, path_rates_mps = interp.total_paths_and_rates(x_itrs, v_itrs)
    corrected_paths_m = total_paths_m + interp.lfm_total_path_bias_m(path_rates_mps)
    bragg_k_rad_m = bragg_vectors_rad_m(x_itrs)
    omega_rad_s = np.sum(bragg_k_rad_m * v_itrs[:, None, :], axis=2)
    return corrected_paths_m, omega_rad_s, bragg_k_rad_m, x_itrs, v_itrs, b_drag


def bragg_vectors_rad_m(x_itrs_m):
    tx_vectors = x_itrs_m[:, None, :] - interp.LINK_TX_POSITIONS_M[None, :, :]
    rx_vectors = x_itrs_m[:, None, :] - interp.LINK_RX_POSITIONS_M[None, :, :]
    tx_unit = tx_vectors / np.linalg.norm(tx_vectors, axis=2)[:, :, None]
    rx_unit = rx_vectors / np.linalg.norm(rx_vectors, axis=2)[:, :, None]
    # With the Doppler convention used here, f_D = -dL/dt/lambda, so
    # omega_D = k_B dot v with k_B = -(2 pi/lambda)(u_tx + u_rx).
    return -(2.0 * np.pi / interp.RADAR_WAVELENGTH_M) * (tx_unit + rx_unit)


def linearized_uncertainty(result, n_residuals):
    n_params = int(result.x.size)
    dof = int(n_residuals - n_params)
    if dof <= 0:
        return {"degrees_of_freedom": dof, "covariance_available": False}
    s_sq = float(2.0 * result.cost / dof)
    try:
        cov = np.linalg.pinv(result.jac.T @ result.jac) * s_sq
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        return {"degrees_of_freedom": dof, "residual_variance": s_sq, "covariance_available": False}
    return {
        "degrees_of_freedom": dof,
        "residual_variance": s_sq,
        "covariance_available": True,
        "parameter_std": [float(x) for x in std],
        "position_std_m": [float(x) for x in std[:3]],
        "velocity_std_mps": [float(x) for x in std[3:6]],
        "log10_b_std": float(std[6]),
    }


def fit_ballistic_joint(
    measured_total_paths_m,
    times_ns,
    rho_of_alt_m,
    p0_params,
    sigma_path_m,
    measured_doppler_hz=None,
    doppler_mask=None,
    sigma_doppler_hz=None,
):
    t_rel_s = (np.asarray(times_ns, dtype=np.float64) - float(times_ns[0])) / 1e9
    measured_paths = np.asarray(measured_total_paths_m, dtype=np.float64)
    sigma_path = np.asarray(sigma_path_m, dtype=np.float64)
    use_doppler = measured_doppler_hz is not None and doppler_mask is not None and np.any(doppler_mask)
    if use_doppler:
        measured_omega = 2.0 * np.pi * np.asarray(measured_doppler_hz, dtype=np.float64)
        doppler_mask = np.asarray(doppler_mask, dtype=bool)
        sigma_omega = (2.0 * np.pi * np.asarray(sigma_doppler_hz, dtype=np.float64))[None, :]

    def residual(x):
        pred_paths, pred_omega, _bragg_k, _x_itrs, _v_itrs, _b = predict_ballistic_paths_and_omega(x, t_rel_s, rho_of_alt_m)
        pieces = [((pred_paths - measured_paths) / sigma_path).ravel()]
        if use_doppler:
            dop = ((pred_omega - measured_omega) / sigma_omega)[doppler_mask]
            pieces.append(dop)
        return np.concatenate(pieces)

    lower = np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(ballistic.MIN_B)])
    upper = np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(ballistic.MAX_B)])
    result = so.least_squares(
        residual,
        p0_params,
        bounds=(lower, upper),
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0]),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=300,
    )
    pred_paths, pred_omega, bragg_k_rad_m, x_itrs, v_itrs, b_drag = predict_ballistic_paths_and_omega(result.x, t_rel_s, rho_of_alt_m)
    path_resid = pred_paths - measured_paths
    omega_resid = pred_omega - 2.0 * np.pi * measured_doppler_hz if measured_doppler_hz is not None else None
    llh = np.asarray([jcoord.ecef2geodetic(x[0], x[1], x[2]) for x in x_itrs], dtype=np.float64)
    out = {
        "params": result.x,
        "b_drag_m2_per_kg": float(b_drag),
        "path_residuals_m": path_resid,
        "weighted_path_residuals": path_resid / sigma_path,
        "omega_residuals_rad_s": omega_resid,
        "bragg_k_rad_m": bragg_k_rad_m,
        "x_itrs_m": x_itrs,
        "v_itrs_mps": v_itrs,
        "alt_km": llh[:, 2] / 1e3,
        "speed_km_s": np.linalg.norm(v_itrs, axis=1) / 1e3,
        "rms_total_path_residual_m": float(np.sqrt(np.mean(path_resid**2.0))),
        "median_abs_total_path_residual_m": float(np.median(np.abs(path_resid))),
        "weighted_path_rms": float(np.sqrt(np.mean((path_resid / sigma_path) ** 2.0))),
        "n_path_residuals": int(path_resid.size),
        "n_doppler_residuals": int(np.count_nonzero(doppler_mask)) if use_doppler else 0,
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "optimizer_cost": float(result.cost),
        "linearized_uncertainty": linearized_uncertainty(result, len(residual(result.x))),
    }
    if use_doppler:
        used = omega_resid[doppler_mask]
        out["omega_rms_rad_s"] = float(np.sqrt(np.mean(used**2.0)))
        out["omega_median_abs_rad_s"] = float(np.median(np.abs(used)))
        out["doppler_rms_hz_equivalent"] = float(np.sqrt(np.mean((used / (2.0 * np.pi)) ** 2.0)))
        out["doppler_median_abs_hz_equivalent"] = float(np.median(np.abs(used / (2.0 * np.pi))))
        out["weighted_omega_rms"] = float(np.sqrt(np.mean(((omega_resid / sigma_omega)[doppler_mask]) ** 2.0)))
    return out


def json_summary(fit):
    keys = [
        "rms_total_path_residual_m",
        "median_abs_total_path_residual_m",
        "weighted_path_rms",
        "omega_rms_rad_s",
        "omega_median_abs_rad_s",
        "doppler_rms_hz_equivalent",
        "doppler_median_abs_hz_equivalent",
        "weighted_omega_rms",
        "n_path_residuals",
        "n_doppler_residuals",
        "b_drag_m2_per_kg",
        "optimizer_success",
        "optimizer_nfev",
        "optimizer_cost",
        "linearized_uncertainty",
    ]
    out = {key: fit[key] for key in keys if key in fit}
    out["start_speed_km_s"] = float(fit["speed_km_s"][0])
    out["end_speed_km_s"] = float(fit["speed_km_s"][-1])
    out["start_alt_km"] = float(fit["alt_km"][0])
    out["end_alt_km"] = float(fit["alt_km"][-1])
    return out


def main():
    os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)
    site_data, measured, times_ns, snr_db, doppler_hz, doppler_mask, acf_results = build_range_and_doppler_measurements()
    rho_of_alt_m, msis_meta = ballistic.make_density_interpolator(times_ns, measured)
    p0 = ballistic.initial_ballistic_guess(measured, times_ns, log10_b=np.log10(40.0))

    first_pass = fit_ballistic_joint(measured, times_ns, rho_of_alt_m, p0, np.ones_like(measured))
    sigma_model = ballistic.fit_sigma_model(first_pass["path_residuals_m"], snr_db)
    sigma_path = ballistic.sigma_from_snr_db(snr_db, sigma_model["sigma_floor_m"], sigma_model["sigma_0_m"])
    sigma_doppler, acf_summary = doppler_sigma_by_station(acf_results)

    path_only = fit_ballistic_joint(measured, times_ns, rho_of_alt_m, first_pass["params"], sigma_path)
    path_plus_doppler = fit_ballistic_joint(
        measured,
        times_ns,
        rho_of_alt_m,
        path_only["params"],
        sigma_path,
        measured_doppler_hz=doppler_hz,
        doppler_mask=doppler_mask,
        sigma_doppler_hz=sigma_doppler,
    )

    out = {
        "script": os.path.basename(__file__),
        "script_version": SCRIPT_VERSION,
        "event_id_local": interp.EVENT_ID_LOCAL,
        "event_id_utc": interp.EVENT_ID_UTC,
        "upsample_factor": UPSAMPLE_FACTOR,
        "acf_settings": {
            "max_lag_us": acf.MAX_LAG_US,
            "min_lag_samples": acf.MIN_LAG_SAMPLES,
            "snr_min_db": acf.SNR_MIN_DB,
            "coherence_min": acf.COHERENCE_MIN,
        },
        "msis": msis_meta,
        "sigma_path_model": sigma_model,
        "sigma_doppler_hz_by_station": {
            "sanya": float(sigma_doppler[0]),
            "danzhou": float(sigma_doppler[1]),
            "wenchang": float(sigma_doppler[2]),
        },
        "acf_summary": acf_summary,
        "path_only_weighted_ballistic": json_summary(path_only),
        "path_plus_acf_doppler_weighted_ballistic": json_summary(path_plus_doppler),
    }
    with open(f"{OUTPUT_BASE}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"wrote {OUTPUT_BASE}.json")
    print(
        "path only: "
        f"path RMS={path_only['rms_total_path_residual_m']:.2f} m, "
        f"B={path_only['b_drag_m2_per_kg']:.3g}, "
        f"n_path={path_only['n_path_residuals']}"
    )
    print(
        "path+doppler: "
        f"path RMS={path_plus_doppler['rms_total_path_residual_m']:.2f} m, "
        f"omega RMS={path_plus_doppler['omega_rms_rad_s']:.1f} rad/s "
        f"({path_plus_doppler['doppler_rms_hz_equivalent'] / 1e3:.2f} kHz), "
        f"B={path_plus_doppler['b_drag_m2_per_kg']:.3g}, "
        f"n_dop={path_plus_doppler['n_doppler_residuals']}"
    )


if __name__ == "__main__":
    main()
