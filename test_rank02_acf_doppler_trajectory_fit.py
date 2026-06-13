import json
import os

import jcoord
import numpy as np
import scipy.optimize as so

import measure_rank02_single_pulse_acf_doppler as acf
import test_rank02_range_interpolation as interp


SCRIPT_VERSION = "v20260611b"
UPSAMPLE_FACTOR = 4
OUTPUT_BASE = os.path.join("results", f"rank02_acf_doppler_trajectory_fit_{SCRIPT_VERSION}")


def copy_site_data_with_doppler(site_data, doppler_hz):
    out = dict(site_data)
    doppler = np.asarray(doppler_hz, dtype=np.float64).copy()
    fallback = np.asarray(site_data["doppler_hz"], dtype=np.float64)
    doppler[~np.isfinite(doppler)] = fallback[~np.isfinite(doppler)]
    out["doppler_hz"] = doppler
    return out


def compute_acf_doppler(site_data, coarse_gates):
    return {site: acf.measure_site(site, site_data[site], coarse_gates[site]) for site in interp.SITE_ORDER}


def refine_all_sites(site_data, coarse_gates, upsample_factor):
    refined = {}
    powers = {}
    for site in interp.SITE_ORDER:
        fine_gate, fine_range_km, power_db = interp.refine_site_ranges(site_data[site], upsample_factor, coarse_gates[site])
        refined[f"{site}_gate"] = fine_gate
        refined[site] = fine_range_km
        powers[site] = power_db
    return refined, powers


def original_refined(site_data):
    return {
        "sanya": site_data["sanya"]["range_km"],
        "sanya_gate": site_data["sanya"]["range_gate"].astype(np.float64),
        "danzhou": site_data["danzhou"]["range_km"],
        "danzhou_gate": site_data["danzhou"]["range_gate"].astype(np.float64),
        "wenchang": site_data["wenchang"]["range_km"],
        "wenchang_gate": site_data["wenchang"]["range_gate"].astype(np.float64),
    }


def nearest_index(values, target):
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return None
    return int(np.argmin(np.abs(values - int(target))))


def matched_measurements_with_masks(site_data, refined_ranges, valid_masks):
    san = site_data["sanya"]
    dan = site_data["danzhou"]
    wen = site_data["wenchang"]
    tolerance_ns = int(7.5e6)
    measured_total_paths = []
    residual_masks = []
    times_ns = []
    source_indices = []
    for san_idx, san_t in enumerate(san["times_ns"]):
        dan_idx = nearest_index(dan["times_ns"], san_t)
        wen_idx = nearest_index(wen["times_ns"], san_t)
        if dan_idx is None or wen_idx is None:
            continue
        dan_t = int(dan["times_ns"][dan_idx])
        wen_t = int(wen["times_ns"][wen_idx])
        if abs(dan_t - int(san_t)) > tolerance_ns or abs(wen_t - int(san_t)) > tolerance_ns:
            continue
        residual_mask = np.asarray(
            [
                bool(valid_masks["sanya"][san_idx]),
                bool(valid_masks["danzhou"][dan_idx]),
                bool(valid_masks["wenchang"][wen_idx]),
            ],
            dtype=bool,
        )
        if not np.any(residual_mask):
            continue
        san_total = 2.0 * refined_ranges["sanya"][san_idx] * 1e3
        dan_total = interp.delay_us_to_total_path_m(
            interp.SITE_DELAY_US["danzhou"] + refined_ranges["danzhou_gate"][dan_idx] / dan["sr_mhz"]
        )
        wen_total = interp.delay_us_to_total_path_m(
            interp.SITE_DELAY_US["wenchang"] + refined_ranges["wenchang_gate"][wen_idx] / wen["sr_mhz"]
        )
        measured_total_paths.append([san_total, float(dan_total), float(wen_total)])
        residual_masks.append(residual_mask)
        times_ns.append(int(round((int(san_t) + dan_t + wen_t) / 3.0)))
        source_indices.append([san_idx, dan_idx, wen_idx])
    return (
        np.asarray(measured_total_paths, dtype=np.float64),
        np.asarray(residual_masks, dtype=bool),
        np.asarray(times_ns, dtype=np.int64),
        np.asarray(source_indices, dtype=np.int32),
    )


def covariance_summary(result, n_residuals):
    n_params = int(result.x.size)
    dof = int(n_residuals - n_params)
    if dof <= 0:
        return {
            "degrees_of_freedom": dof,
            "parameter_std": [None] * n_params,
            "residual_variance_m2": None,
            "covariance_available": False,
        }
    jac = np.asarray(result.jac, dtype=np.float64)
    s_sq = float(2.0 * result.cost / dof)
    try:
        cov = np.linalg.pinv(jac.T @ jac) * s_sq
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        return {
            "degrees_of_freedom": dof,
            "parameter_std": [None] * n_params,
            "residual_variance_m2": s_sq,
            "covariance_available": False,
        }
    return {
        "degrees_of_freedom": dof,
        "parameter_std": [float(x) for x in std],
        "position_std_m": [float(x) for x in std[:3]],
        "velocity_std_mps": [float(x) for x in std[3:6]],
        "acceleration_std_mps2": [float(x) for x in std[6:9]] if n_params >= 9 else None,
        "residual_variance_m2": s_sq,
        "covariance_available": True,
    }


def fit_trajectory_masked(measured_total_paths_m, residual_mask, times_ns, san_az_deg, san_el_deg, san_median_range_km, acceleration=False):
    x0 = interp.initial_guess(san_az_deg, san_el_deg, san_median_range_km)
    points = []
    valid_measurements = []
    valid_masks = []
    valid_times = []
    for measured, mask, t_ns in zip(measured_total_paths_m, residual_mask, times_ns):
        point = interp.solve_position_from_total_paths_m(measured, x0)
        x0 = point
        llh = jcoord.ecef2geodetic(point[0], point[1], point[2])
        if not np.all(np.isfinite(llh)) or float(llh[0]) > interp.MAX_LAT_DEG:
            continue
        points.append(point)
        valid_measurements.append(measured)
        valid_masks.append(mask)
        valid_times.append(t_ns)
    if len(points) < interp.MIN_POINTS:
        raise RuntimeError("Too few valid points for masked trajectory fit")

    points = np.asarray(points, dtype=np.float64)
    valid_measurements = np.asarray(valid_measurements, dtype=np.float64)
    valid_masks = np.asarray(valid_masks, dtype=bool)
    valid_times = np.asarray(valid_times, dtype=np.int64)
    order = np.argsort(valid_times)
    points = points[order]
    valid_measurements = valid_measurements[order]
    valid_masks = valid_masks[order]
    valid_times = valid_times[order]
    t_rel_s = (valid_times.astype(np.float64) - float(valid_times[0])) / 1e9
    r0, v0 = interp.linear_initial_state(points, valid_times)
    if acceleration:
        p0 = np.concatenate([r0, v0, np.zeros(3, dtype=np.float64)])
        x_scale = np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1e4, 1e4, 1e4])
    else:
        p0 = np.concatenate([r0, v0])
        x_scale = np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4])

    def residual(params):
        predicted, _ = interp.predict_paths(params, t_rel_s, valid_times, acceleration=acceleration)
        return (predicted - valid_measurements)[valid_masks]

    result = so.least_squares(residual, p0, method="trf", x_scale=x_scale, max_nfev=250)
    predicted, x_itrs = interp.predict_paths(result.x, t_rel_s, valid_times, acceleration=acceleration)
    residuals_all = predicted - valid_measurements
    residuals_used = residuals_all[valid_masks]
    llh = np.asarray([jcoord.ecef2geodetic(x[0], x[1], x[2]) for x in x_itrs], dtype=np.float64)
    summary = {
        "params": result.x,
        "rms_total_path_residual_m": float(np.sqrt(np.mean(residuals_used**2.0))),
        "median_abs_total_path_residual_m": float(np.median(np.abs(residuals_used))),
        "n_points": int(len(valid_times)),
        "n_residuals_used": int(np.count_nonzero(valid_masks)),
        "n_residuals_by_station": {
            "sanya": int(np.count_nonzero(valid_masks[:, 0])),
            "danzhou": int(np.count_nonzero(valid_masks[:, 1])),
            "wenchang": int(np.count_nonzero(valid_masks[:, 2])),
        },
        "duration_s": float(t_rel_s[-1] - t_rel_s[0]),
        "speed_km_s": float(np.linalg.norm(result.x[3:6]) / 1e3),
        "start_alt_km": float(llh[0, 2] / 1e3),
        "end_alt_km": float(llh[-1, 2] / 1e3),
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "optimizer_cost": float(result.cost),
        "linearized_uncertainty": covariance_summary(result, int(np.count_nonzero(valid_masks))),
    }
    if acceleration:
        summary["accel_mps2"] = float(np.linalg.norm(result.x[6:9]))
        summary["along_track_accel_mps2"] = float(np.dot(result.x[6:9], result.x[3:6]) / np.linalg.norm(result.x[3:6]))
    return summary


def serializable_fit_summary(fit):
    out = interp.json_fit_summary(fit)
    out["n_residuals_used"] = fit["n_residuals_used"]
    out["n_residuals_by_station"] = fit["n_residuals_by_station"]
    out["linearized_uncertainty"] = fit["linearized_uncertainty"]
    return out


def fit_case(case_name, site_data, refined, valid_masks):
    measured, residual_mask, times_ns, source_indices = matched_measurements_with_masks(site_data, refined, valid_masks)
    if len(times_ns) < interp.MIN_POINTS:
        raise RuntimeError(f"{case_name}: too few high-coherence points")
    const_fit = fit_trajectory_masked(
        measured,
        residual_mask,
        times_ns,
        site_data["sanya"]["az_deg"],
        site_data["sanya"]["el_deg"],
        float(np.median(refined["sanya"][source_indices[:, 0]])),
        acceleration=False,
    )
    accel_fit = fit_trajectory_masked(
        measured,
        residual_mask,
        times_ns,
        site_data["sanya"]["az_deg"],
        site_data["sanya"]["el_deg"],
        float(np.median(refined["sanya"][source_indices[:, 0]])),
        acceleration=True,
    )
    return {
        "n_time_points": int(len(times_ns)),
        "n_residuals_used": int(np.count_nonzero(residual_mask)),
        "n_residuals_by_station": {
            "sanya": int(np.count_nonzero(residual_mask[:, 0])),
            "danzhou": int(np.count_nonzero(residual_mask[:, 1])),
            "wenchang": int(np.count_nonzero(residual_mask[:, 2])),
        },
        "constant_velocity": serializable_fit_summary(const_fit),
        "constant_acceleration": serializable_fit_summary(accel_fit),
    }


def summarize_acf(acf_results):
    out = {}
    for site, data in acf_results.items():
        good = np.isfinite(data["acf_residual_hz"])
        residual_khz = data["acf_residual_hz"][good] / 1e3
        out[site] = {
            "valid_high_coherence": int(np.count_nonzero(good)),
            "total_pulses": int(len(good)),
            "median_residual_khz": float(np.nanmedian(residual_khz)) if len(residual_khz) else None,
            "robust_sigma_residual_khz": float(1.4826 * np.nanmedian(np.abs(residual_khz - np.nanmedian(residual_khz))))
            if len(residual_khz)
            else None,
            "median_coherence": float(np.nanmedian(data["coherence"][good])) if len(residual_khz) else None,
        }
    return out


def main():
    os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)
    fit = interp.load_reference_fit()
    site_data_fitted = {site: interp.load_site(site, fit) for site in interp.SITE_ORDER}
    coarse_gates = interp.precompute_coarse_gates(site_data_fitted)
    acf_results = compute_acf_doppler(site_data_fitted, coarse_gates)
    valid_masks = {site: np.isfinite(acf_results[site]["acf_measured_hz"]) for site in interp.SITE_ORDER}

    fitted_refined, fitted_powers = refine_all_sites(site_data_fitted, coarse_gates, UPSAMPLE_FACTOR)
    acf_site_data = {
        site: copy_site_data_with_doppler(site_data_fitted[site], acf_results[site]["acf_measured_hz"])
        for site in interp.SITE_ORDER
    }
    acf_refined, acf_powers = refine_all_sites(acf_site_data, coarse_gates, UPSAMPLE_FACTOR)
    original = original_refined(site_data_fitted)

    cases = {
        "original_stored_peaks_no_doppler_rematch": fit_case("original", site_data_fitted, original, valid_masks),
        "fitted_doppler_4x_rematch": fit_case("fitted_doppler", site_data_fitted, fitted_refined, valid_masks),
        "acf_measured_doppler_4x_rematch": fit_case("acf_doppler", acf_site_data, acf_refined, valid_masks),
    }

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
        "acf_summary": summarize_acf(acf_results),
        "cases": cases,
    }
    with open(f"{OUTPUT_BASE}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"wrote {OUTPUT_BASE}.json")
    for name, case in cases.items():
        cv = case["constant_velocity"]
        ca = case["constant_acceleration"]
        print(
            f"{name}: points={case['n_time_points']} residuals={case['n_residuals_used']} "
            f"cv_rms={cv['rms_total_path_residual_m']:.2f} m "
            f"ca_rms={ca['rms_total_path_residual_m']:.2f} m "
            f"ca_median_abs={ca['median_abs_total_path_residual_m']:.2f} m"
        )


if __name__ == "__main__":
    main()
