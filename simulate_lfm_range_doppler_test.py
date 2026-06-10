import os

import h5py
import jcoord
import numpy as np
import scipy.optimize as so
from astropy.time import Time

from fit_gcrs_trajectories_lfm_ambiguity import (
    C,
    ecef_to_gcrs,
    half_path_from_total_path_m,
    gcrs_state_to_itrs,
    linear_initial_state_from_uncorrected_positions,
    predict_total_paths_m,
    solve_position_from_total_paths_m,
)


OUTPUT_H5 = os.path.join("results", "lfm_range_doppler_simulation_test_v20260610.h5")
N_PULSES = 21
IPP_S = 0.005
T0_ISOT = "2024-04-22T16:00:00"
START_LAT_DEG = 18.60
START_LON_DEG = 109.69
START_ALT_KM = 110.0
VELOCITY_ENU_MPS = np.array([8000.0, -18000.0, -38000.0], dtype=np.float64)


def enu_basis(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float64)
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], dtype=np.float64)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=np.float64)
    return east, north, up


def make_times():
    t0_ns = int(round(Time(T0_ISOT, scale="utc").unix * 1e9))
    t_rel_s = IPP_S * np.arange(N_PULSES, dtype=np.float64)
    times_ns = t0_ns + np.round(t_rel_s * 1e9).astype(np.int64)
    return times_ns, t_rel_s


def make_truth_state(times_ns):
    start_itrs = np.asarray(
        jcoord.geodetic2ecef(START_LAT_DEG, START_LON_DEG, START_ALT_KM * 1e3),
        dtype=np.float64,
    )
    east, north, up = enu_basis(START_LAT_DEG, START_LON_DEG)
    velocity_itrs = (
        VELOCITY_ENU_MPS[0] * east
        + VELOCITY_ENU_MPS[1] * north
        + VELOCITY_ENU_MPS[2] * up
    )

    one_second_ns = np.asarray([times_ns[0], times_ns[0] + int(1e9)], dtype=np.int64)
    two_itrs_points = np.vstack([start_itrs, start_itrs + velocity_itrs])
    two_gcrs_points = ecef_to_gcrs(two_itrs_points, one_second_ns)
    r0_gcrs_m = two_gcrs_points[0]
    v0_gcrs_mps = two_gcrs_points[1] - two_gcrs_points[0]
    return np.concatenate([r0_gcrs_m, v0_gcrs_mps]), start_itrs, velocity_itrs


def solve_uncorrected_positions(measured_total_paths_m):
    points = []
    x0 = np.asarray(jcoord.geodetic2ecef(START_LAT_DEG, START_LON_DEG, START_ALT_KM * 1e3), dtype=np.float64)
    for measured in measured_total_paths_m:
        point = solve_position_from_total_paths_m(measured, x0)
        points.append(point)
        x0 = point
    return np.asarray(points, dtype=np.float64)


def fit_state(measured_total_paths_m, p0, t_rel_s, times_ns, include_lfm):
    def residual(params):
        predicted = predict_total_paths_m(params, t_rel_s, times_ns, include_lfm=include_lfm)
        return (predicted - measured_total_paths_m).ravel()

    return so.least_squares(
        residual,
        p0,
        method="trf",
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4]),
        max_nfev=120,
    )


def trajectory_errors(truth_params, fit_params, t_rel_s, times_ns):
    true_itrs, true_vel_itrs = gcrs_state_to_itrs(truth_params[:3], truth_params[3:], t_rel_s, times_ns)
    fit_itrs, fit_vel_itrs = gcrs_state_to_itrs(fit_params[:3], fit_params[3:], t_rel_s, times_ns)
    position_error_m = np.linalg.norm(fit_itrs - true_itrs, axis=1)
    velocity_error_mps = np.linalg.norm(fit_vel_itrs - true_vel_itrs, axis=1)
    return true_itrs, fit_itrs, position_error_m, velocity_error_mps


def summarize_fit(name, result, measured_total_paths_m, t_rel_s, times_ns, truth_params):
    predicted = predict_total_paths_m(result.x, t_rel_s, times_ns, include_lfm=(name == "lfm_corrected"))
    total_path_residuals_m = predicted - measured_total_paths_m
    half_path_diagnostic_residuals_m = half_path_from_total_path_m(total_path_residuals_m)
    _, _, position_error_m, velocity_error_mps = trajectory_errors(truth_params, result.x, t_rel_s, times_ns)
    return {
        "name": name,
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "rms_total_path_residual_m": float(np.sqrt(np.mean(total_path_residuals_m**2))),
        "max_abs_total_path_residual_m": float(np.max(np.abs(total_path_residuals_m))),
        "rms_half_path_diagnostic_residual_m": float(np.sqrt(np.mean(half_path_diagnostic_residuals_m**2))),
        "max_abs_half_path_diagnostic_residual_m": float(np.max(np.abs(half_path_diagnostic_residuals_m))),
        "median_position_error_m": float(np.median(position_error_m)),
        "max_position_error_m": float(np.max(position_error_m)),
        "median_velocity_error_mps": float(np.median(velocity_error_mps)),
        "max_velocity_error_mps": float(np.max(velocity_error_mps)),
        "params": result.x,
        "total_path_residuals_m": total_path_residuals_m,
        "half_path_diagnostic_residuals_m": half_path_diagnostic_residuals_m,
        "position_error_m": position_error_m,
        "velocity_error_mps": velocity_error_mps,
    }


def write_h5(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["purpose"] = "Synthetic LFM range-Doppler ambiguity recovery test"
        h.attrs["fit_residual_coordinate"] = "total propagation path length"
        h.attrs["half_path_diagnostic_coordinate_m"] = "diagnostic half-path coordinate only; not used for fitting"
        h.attrs["t0_isot"] = T0_ISOT
        h.attrs["n_pulses"] = N_PULSES
        h.attrs["ipp_s"] = IPP_S
        h.attrs["start_lat_deg"] = START_LAT_DEG
        h.attrs["start_lon_deg"] = START_LON_DEG
        h.attrs["start_alt_km"] = START_ALT_KM
        h.attrs["velocity_enu_mps"] = VELOCITY_ENU_MPS
        h["time_ns"] = data["times_ns"]
        h["t_rel_s"] = data["t_rel_s"]
        h["truth_params_gcrs"] = data["truth_params"]
        h["prior_params_gcrs"] = data["prior_params"]
        h["measured_total_paths_m"] = data["measured_total_paths_m"]
        h["measured_half_path_diagnostic_m"] = data["measured_half_path_diagnostic_m"]
        h["uncorrected_positions_itrs_m"] = data["uncorrected_positions_itrs_m"]
        h["true_positions_itrs_m"] = data["true_positions_itrs_m"]
        for summary in data["summaries"]:
            group = h.create_group(summary["name"])
            group.attrs["success"] = summary["success"]
            group.attrs["nfev"] = summary["nfev"]
            group.attrs["rms_total_path_residual_m"] = summary["rms_total_path_residual_m"]
            group.attrs["max_abs_total_path_residual_m"] = summary["max_abs_total_path_residual_m"]
            group.attrs["rms_half_path_diagnostic_residual_m"] = summary["rms_half_path_diagnostic_residual_m"]
            group.attrs["max_abs_half_path_diagnostic_residual_m"] = summary["max_abs_half_path_diagnostic_residual_m"]
            group.attrs["median_position_error_m"] = summary["median_position_error_m"]
            group.attrs["max_position_error_m"] = summary["max_position_error_m"]
            group.attrs["median_velocity_error_mps"] = summary["median_velocity_error_mps"]
            group.attrs["max_velocity_error_mps"] = summary["max_velocity_error_mps"]
            group["params_gcrs"] = summary["params"]
            group["total_path_residuals_m"] = summary["total_path_residuals_m"]
            group["half_path_diagnostic_residuals_m"] = summary["half_path_diagnostic_residuals_m"]
            group["position_error_m"] = summary["position_error_m"]
            group["velocity_error_mps"] = summary["velocity_error_mps"]


def main():
    times_ns, t_rel_s = make_times()
    truth_params, _, _ = make_truth_state(times_ns)
    measured_total_paths_m = predict_total_paths_m(truth_params, t_rel_s, times_ns, include_lfm=True)
    measured_half_path_diagnostic_m = half_path_from_total_path_m(measured_total_paths_m)
    uncorrected_positions_itrs_m = solve_uncorrected_positions(measured_total_paths_m)
    r0_prior, v0_prior, _ = linear_initial_state_from_uncorrected_positions(uncorrected_positions_itrs_m, times_ns)
    prior_params = np.concatenate([r0_prior, v0_prior])

    no_lfm_result = fit_state(measured_total_paths_m, prior_params, t_rel_s, times_ns, include_lfm=False)
    lfm_result = fit_state(measured_total_paths_m, prior_params, t_rel_s, times_ns, include_lfm=True)
    true_positions_itrs_m, _, _, _ = trajectory_errors(truth_params, truth_params, t_rel_s, times_ns)

    summaries = [
        summarize_fit("no_lfm_correction", no_lfm_result, measured_total_paths_m, t_rel_s, times_ns, truth_params),
        summarize_fit("lfm_corrected", lfm_result, measured_total_paths_m, t_rel_s, times_ns, truth_params),
    ]
    data = {
        "times_ns": times_ns,
        "t_rel_s": t_rel_s,
        "truth_params": truth_params,
        "prior_params": prior_params,
        "measured_total_paths_m": measured_total_paths_m,
        "measured_half_path_diagnostic_m": measured_half_path_diagnostic_m,
        "uncorrected_positions_itrs_m": uncorrected_positions_itrs_m,
        "true_positions_itrs_m": true_positions_itrs_m,
        "summaries": summaries,
    }
    write_h5(OUTPUT_H5, data)

    print(f"wrote {OUTPUT_H5}")
    for summary in summaries:
        print(summary["name"])
        print(f"  success: {summary['success']} nfev={summary['nfev']}")
        print(f"  total-path RMS residual: {summary['rms_total_path_residual_m']:.6f} m")
        print(f"  half-path diagnostic RMS residual: {summary['rms_half_path_diagnostic_residual_m']:.6f} m")
        print(f"  max abs total-path residual: {summary['max_abs_total_path_residual_m']:.6f} m")
        print(f"  median/max position error: {summary['median_position_error_m']:.6f} / {summary['max_position_error_m']:.6f} m")
        print(f"  median/max velocity error: {summary['median_velocity_error_mps']:.6f} / {summary['max_velocity_error_mps']:.6f} m/s")


if __name__ == "__main__":
    main()
