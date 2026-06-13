import os

import astropy.units as u
import h5py
import jcoord
import numpy as np
import scipy.optimize as so
import stuffr
from astropy.coordinates import GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time

import sanya_opts as sc
from grid_search_delays_beam_axis import (
    C,
    DAN_CENTER_US,
    DAN_PATTERN,
    MAX_LAT_DEG,
    SAN_PATTERN,
    WEN_CENTER_US,
    WEN_PATTERN,
    gate_to_delay_us,
    initial_guess,
    load_events,
    nearest_index,
    pair_tristatic_events,
    range_gates_to_km,
)


OUTPUT_H5 = os.path.join("results", "gcrs_trajectory_fits_lfm_ambiguity_v20260610.h5")
MIN_POINTS = 3
MATCH_TOLERANCE_MS = 7.5
SOURCE_TIMEZONE_OFFSET_HOURS = 8.0
SOURCE_TIMEZONE_OFFSET_NS = int(SOURCE_TIMEZONE_OFFSET_HOURS * 3600.0 * 1e9)
RADAR_FREQUENCY_HZ = 440e6
BANDWIDTH_HZ = 4e6
LFM_DURATION_S = 199e-6
CHIRP_RATE_HZ_PER_S = BANDWIDTH_HZ / LFM_DURATION_S
RADAR_WAVELENGTH_M = C / RADAR_FREQUENCY_HZ

LINK_NAMES = np.array(["sanya", "danzhou", "wenchang"])
LINK_TX_POSITIONS_M = np.asarray([sc.p_san, sc.p_san, sc.p_san], dtype=np.float64)
LINK_RX_POSITIONS_M = np.asarray([sc.p_san, sc.p_dan, sc.p_wen], dtype=np.float64)


def match_pulses_with_time(san_event, dan_event, wen_event, tolerance_ms=MATCH_TOLERANCE_MS):
    tolerance_ns = int(tolerance_ms * 1e6)
    matches = []
    for san_idx, san_t in enumerate(san_event.times_ns):
        dan_idx = nearest_index(dan_event.times_ns, san_t)
        wen_idx = nearest_index(wen_event.times_ns, san_t)
        if dan_idx is None or wen_idx is None:
            continue
        dan_t = int(dan_event.times_ns[dan_idx])
        wen_t = int(wen_event.times_ns[wen_idx])
        if abs(dan_t - int(san_t)) > tolerance_ns:
            continue
        if abs(wen_t - int(san_t)) > tolerance_ns:
            continue
        matches.append(
            {
                "san_idx": int(san_idx),
                "dan_idx": int(dan_idx),
                "wen_idx": int(wen_idx),
                "time_ns": int(round((int(san_t) + dan_t + wen_t) / 3.0)),
            }
        )
    return matches


def beijing_local_ns_to_utc_ns(times_ns):
    """Raw MATLAB time fields are Beijing local time (UTC+8), stored as naive ns."""
    return np.asarray(times_ns, dtype=np.int64) - SOURCE_TIMEZONE_OFFSET_NS


def event_times_are_utc(*events):
    return all(bool(getattr(event, "times_ns_are_utc", False)) for event in events)


def ecef_to_gcrs(points_ecef_m, times_ns):
    obstime = Time(np.asarray(times_ns, dtype=np.float64) / 1e9, format="unix", scale="utc")
    representation = CartesianRepresentation(
        points_ecef_m[:, 0] * u.m,
        points_ecef_m[:, 1] * u.m,
        points_ecef_m[:, 2] * u.m,
    )
    itrs = ITRS(representation, obstime=obstime)
    gcrs = itrs.transform_to(GCRS(obstime=obstime))
    return gcrs.cartesian.xyz.to_value(u.m).T


def gcrs_state_to_itrs(r0_gcrs_m, v0_gcrs_mps, t_rel_s, times_ns):
    obstime = Time(np.asarray(times_ns, dtype=np.float64) / 1e9, format="unix", scale="utc")
    positions = r0_gcrs_m[None, :] + t_rel_s[:, None] * v0_gcrs_mps[None, :]
    representation = CartesianRepresentation(
        positions[:, 0] * u.m,
        positions[:, 1] * u.m,
        positions[:, 2] * u.m,
        differentials=CartesianDifferential(
            np.repeat(v0_gcrs_mps[0], len(t_rel_s)) * u.m / u.s,
            np.repeat(v0_gcrs_mps[1], len(t_rel_s)) * u.m / u.s,
            np.repeat(v0_gcrs_mps[2], len(t_rel_s)) * u.m / u.s,
        ),
    )
    gcrs = GCRS(representation, obstime=obstime)
    itrs = gcrs.transform_to(ITRS(obstime=obstime))
    positions_itrs = itrs.cartesian.without_differentials().xyz.to_value(u.m).T
    velocities_itrs = itrs.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s).T
    return positions_itrs, velocities_itrs


def doppler_from_path_length_rate_hz(path_length_rate_mps):
    return -path_length_rate_mps / RADAR_WAVELENGTH_M


def lfm_total_path_bias_m(path_length_rate_mps):
    """Apparent total-path bias from the Sanya LFM range-Doppler coupling.

    Satellite passes validate the convention that the measured apparent path is
    geometric_path + (f0/gamma) * path_length_rate.  Callers therefore add this
    helper to the geometric tx-target-rx path when predicting measurements.
    """
    doppler_hz = doppler_from_path_length_rate_hz(path_length_rate_mps)
    return -C * doppler_hz / CHIRP_RATE_HZ_PER_S


def half_path_from_total_path_m(total_path_m):
    """Diagnostic half-path coordinate, useful only for comparison with old plots."""
    return 0.5 * np.asarray(total_path_m, dtype=np.float64)


def delay_us_to_total_path_m(delay_us):
    return C * np.asarray(delay_us, dtype=np.float64) * 1e-6


def link_total_paths_and_rates_m(positions_itrs_m, velocities_itrs_mps, tx_positions_m, rx_positions_m):
    """Return total tx-target-rx path lengths and path-length rates for each link."""
    positions = np.asarray(positions_itrs_m, dtype=np.float64)
    velocities = np.asarray(velocities_itrs_mps, dtype=np.float64)
    tx_positions = np.asarray(tx_positions_m, dtype=np.float64)
    rx_positions = np.asarray(rx_positions_m, dtype=np.float64)

    tx_vectors = positions[:, None, :] - tx_positions[None, :, :]
    rx_vectors = positions[:, None, :] - rx_positions[None, :, :]
    tx_distances = np.linalg.norm(tx_vectors, axis=2)
    rx_distances = np.linalg.norm(rx_vectors, axis=2)
    tx_unit = tx_vectors / tx_distances[:, :, None]
    rx_unit = rx_vectors / rx_distances[:, :, None]
    total_paths_m = tx_distances + rx_distances
    path_rates_mps = np.sum((tx_unit + rx_unit) * velocities[:, None, :], axis=2)
    return total_paths_m, path_rates_mps


def predict_total_paths_m(params, t_rel_s, times_ns, include_lfm=True, tx_positions_m=None, rx_positions_m=None):
    r0_gcrs_m = np.asarray(params[:3], dtype=np.float64)
    v0_gcrs_mps = np.asarray(params[3:], dtype=np.float64)
    x_itrs, v_itrs = gcrs_state_to_itrs(r0_gcrs_m, v0_gcrs_mps, t_rel_s, times_ns)
    if tx_positions_m is None:
        tx_positions_m = LINK_TX_POSITIONS_M
    if rx_positions_m is None:
        rx_positions_m = LINK_RX_POSITIONS_M
    total_paths_m, path_rates_mps = link_total_paths_and_rates_m(
        x_itrs,
        v_itrs,
        tx_positions_m,
        rx_positions_m,
    )
    if include_lfm:
        total_paths_m = total_paths_m + lfm_total_path_bias_m(path_rates_mps)
    return total_paths_m


def linear_initial_state_from_uncorrected_positions(points_ecef_m, times_ns):
    """Use the uncorrected tri-static positions only as the optimizer seed."""
    points_gcrs_m = ecef_to_gcrs(points_ecef_m, times_ns)
    t_rel_s = (np.asarray(times_ns, dtype=np.float64) - float(times_ns[0])) / 1e9
    design = np.column_stack([np.ones_like(t_rel_s), t_rel_s])
    coeffs = np.linalg.lstsq(design, points_gcrs_m, rcond=None)[0]
    return coeffs[0, :], coeffs[1, :], points_gcrs_m


def solve_position_from_total_paths_m(measured_total_paths_m, x0, tx_positions_m=None, rx_positions_m=None):
    if tx_positions_m is None:
        tx_positions_m = LINK_TX_POSITIONS_M
    if rx_positions_m is None:
        rx_positions_m = LINK_RX_POSITIONS_M
    measured_total_paths_m = np.asarray(measured_total_paths_m, dtype=np.float64)
    tx_positions_m = np.asarray(tx_positions_m, dtype=np.float64)
    rx_positions_m = np.asarray(rx_positions_m, dtype=np.float64)

    def residual(x):
        tx_distances = np.linalg.norm(x[None, :] - tx_positions_m, axis=1)
        rx_distances = np.linalg.norm(x[None, :] - rx_positions_m, axis=1)
        return tx_distances + rx_distances - measured_total_paths_m

    return so.least_squares(residual, x0=x0, method="lm").x


def solve_triplet(event_id, san_event, dan_event, wen_event):
    matches = match_pulses_with_time(san_event, dan_event, wen_event)
    if len(matches) < MIN_POINTS:
        return None
    input_times_are_utc = event_times_are_utc(san_event, dan_event, wen_event)

    san_one_way_ranges_km = range_gates_to_km(san_event.range_gate, san_event.r0_km, san_event.sr_mhz)
    san_total_paths_m = 2.0 * san_one_way_ranges_km * 1e3
    dan_total_paths_m = delay_us_to_total_path_m(DAN_CENTER_US + gate_to_delay_us(dan_event.range_gate, dan_event.sr_mhz))
    wen_total_paths_m = delay_us_to_total_path_m(WEN_CENTER_US + gate_to_delay_us(wen_event.range_gate, wen_event.sr_mhz))

    x0 = initial_guess(san_event.az_deg, san_event.el_deg, float(np.median(san_one_way_ranges_km)))
    points_ecef_m = []
    measured_total_paths_m = []
    beijing_local_times_ns = []
    for match in matches:
        measured = np.array(
            [
                float(san_total_paths_m[match["san_idx"]]),
                float(dan_total_paths_m[match["dan_idx"]]),
                float(wen_total_paths_m[match["wen_idx"]]),
            ],
            dtype=np.float64,
        )
        point = solve_position_from_total_paths_m(measured, x0)
        x0 = point
        llh = jcoord.ecef2geodetic(point[0], point[1], point[2])
        if not np.isfinite(llh[0]) or not np.isfinite(llh[1]) or not np.isfinite(llh[2]):
            continue
        if float(llh[0]) > MAX_LAT_DEG:
            continue
        points_ecef_m.append(point)
        measured_total_paths_m.append(measured)
        beijing_local_times_ns.append(match["time_ns"])

    if len(points_ecef_m) < MIN_POINTS:
        return None

    points_ecef_m = np.asarray(points_ecef_m, dtype=np.float64)
    measured_total_paths_m = np.asarray(measured_total_paths_m, dtype=np.float64)
    matched_times_ns = np.asarray(beijing_local_times_ns, dtype=np.int64)
    if input_times_are_utc:
        utc_times_ns = matched_times_ns
        beijing_local_times_ns = utc_times_ns + SOURCE_TIMEZONE_OFFSET_NS
    else:
        beijing_local_times_ns = matched_times_ns
        utc_times_ns = beijing_local_ns_to_utc_ns(beijing_local_times_ns)
    order = np.argsort(utc_times_ns)
    points_ecef_m = points_ecef_m[order]
    measured_total_paths_m = measured_total_paths_m[order]
    beijing_local_times_ns = beijing_local_times_ns[order]
    utc_times_ns = utc_times_ns[order]
    t_rel_s = (utc_times_ns.astype(np.float64) - float(utc_times_ns[0])) / 1e9

    r0_prior_gcrs_m, v0_prior_gcrs_mps, prior_points_gcrs_m = linear_initial_state_from_uncorrected_positions(
        points_ecef_m,
        utc_times_ns,
    )
    # The LFM ambiguity correction is fitted in measurement space.  The
    # uncorrected triangulated path is only a prior guess for the optimizer.
    p0 = np.concatenate([r0_prior_gcrs_m, v0_prior_gcrs_mps])

    def residual(params):
        predicted_total_paths_m = predict_total_paths_m(params, t_rel_s, utc_times_ns)
        return (predicted_total_paths_m - measured_total_paths_m).ravel()

    result = so.least_squares(
        residual,
        p0,
        method="trf",
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4]),
        max_nfev=80,
    )
    predicted_total_paths_m = predict_total_paths_m(result.x, t_rel_s, utc_times_ns)
    predicted_half_path_diagnostic_m = half_path_from_total_path_m(predicted_total_paths_m)
    measured_half_path_diagnostic_m = half_path_from_total_path_m(measured_total_paths_m)
    half_path_diagnostic_residuals_m = predicted_half_path_diagnostic_m - measured_half_path_diagnostic_m
    total_path_residuals_m = predicted_total_paths_m - measured_total_paths_m
    x_itrs, v_itrs = gcrs_state_to_itrs(result.x[:3], result.x[3:], t_rel_s, utc_times_ns)
    llh = np.asarray([jcoord.ecef2geodetic(x[0], x[1], x[2]) for x in x_itrs], dtype=np.float64)

    return {
        "event_id": event_id,
        "time_ns": utc_times_ns,
        "beijing_local_time_ns": beijing_local_times_ns,
        "t_rel_s": t_rel_s,
        "r0_gcrs_m": result.x[:3],
        "v0_gcrs_mps": result.x[3:],
        "r0_prior_gcrs_m": r0_prior_gcrs_m,
        "v0_prior_gcrs_mps": v0_prior_gcrs_mps,
        "prior_points_ecef_m": points_ecef_m,
        "prior_points_gcrs_m": prior_points_gcrs_m,
        "prior_speed_km_s": float(np.linalg.norm(v0_prior_gcrs_mps) / 1e3),
        "speed_km_s": float(np.linalg.norm(result.x[3:]) / 1e3),
        "itrs_fit_m": x_itrs,
        "itrs_fit_v_mps": v_itrs,
        "lat_deg": llh[:, 0],
        "lon_deg": llh[:, 1],
        "alt_km": llh[:, 2] / 1e3,
        "measured_total_paths_m": measured_total_paths_m,
        "predicted_total_paths_m": predicted_total_paths_m,
        "total_path_residuals_m": total_path_residuals_m,
        "measured_half_path_diagnostic_m": measured_half_path_diagnostic_m,
        "predicted_half_path_diagnostic_m": predicted_half_path_diagnostic_m,
        "half_path_diagnostic_residuals_m": half_path_diagnostic_residuals_m,
        "rms_total_path_residual_m": float(np.sqrt(np.mean(total_path_residuals_m**2))),
        "median_abs_total_path_residual_m": float(np.median(np.abs(total_path_residuals_m))),
        "rms_half_path_diagnostic_residual_m": float(np.sqrt(np.mean(half_path_diagnostic_residuals_m**2))),
        "median_abs_half_path_diagnostic_residual_m": float(np.median(np.abs(half_path_diagnostic_residuals_m))),
        "duration_s": float(t_rel_s[-1] - t_rel_s[0]),
        "t0_ns": int(utc_times_ns[0]),
        "t0_beijing_local_ns": int(beijing_local_times_ns[0]),
        "t0_utc": stuffr.unix2datestr(int(utc_times_ns[0]) / 1e9),
        "t0_beijing_local": stuffr.unix2datestr(int(beijing_local_times_ns[0]) / 1e9),
        "n_points": int(len(utc_times_ns)),
        "optimizer_success": bool(result.success),
        "optimizer_cost": float(result.cost),
        "optimizer_nfev": int(result.nfev),
    }


def write_h5(path, fits):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["coordinate_frame"] = "GCRS"
        h.attrs["model"] = "constant velocity, no deceleration; LFM range-Doppler ambiguity included"
        h.attrs["source_time_zone"] = "Beijing local time, UTC+8"
        h.attrs["source_time_correction"] = "UTC time_ns = raw MATLAB local time_ns - 8 hours"
        h.attrs["source_timezone_offset_hours"] = SOURCE_TIMEZONE_OFFSET_HOURS
        h.attrs["lfm_total_path_bias_m"] = "validated satellite convention: +(f0/chirp_rate)*path_length_rate"
        h.attrs["half_path_diagnostic_coordinate_m"] = "diagnostic half-path coordinate only; not used for fitting"
        h.attrs["fit_residual_coordinate"] = "total propagation path length"
        h.attrs["radar_frequency_hz"] = RADAR_FREQUENCY_HZ
        h.attrs["radar_wavelength_m"] = RADAR_WAVELENGTH_M
        h.attrs["bandwidth_hz"] = BANDWIDTH_HZ
        h.attrs["lfm_duration_s"] = LFM_DURATION_S
        h.attrs["chirp_rate_hz_per_s"] = CHIRP_RATE_HZ_PER_S
        h.attrs["danzhou_first_sample_delay_us"] = DAN_CENTER_US
        h.attrs["wenchang_first_sample_delay_us"] = WEN_CENTER_US
        h.attrs["max_lat_deg"] = MAX_LAT_DEG
        h["link_names"] = LINK_NAMES.astype(string_dtype)
        h["link_tx_positions_m"] = LINK_TX_POSITIONS_M
        h["link_rx_positions_m"] = LINK_RX_POSITIONS_M

        h["event_id"] = np.asarray([fit["event_id"] for fit in fits], dtype=string_dtype)
        h["t0_ns"] = np.asarray([fit["t0_ns"] for fit in fits], dtype=np.int64)
        h["t0_beijing_local_ns"] = np.asarray([fit["t0_beijing_local_ns"] for fit in fits], dtype=np.int64)
        h["t0_utc"] = np.asarray([fit["t0_utc"] for fit in fits], dtype=string_dtype)
        h["t0_beijing_local"] = np.asarray([fit["t0_beijing_local"] for fit in fits], dtype=string_dtype)
        h["n_points"] = np.asarray([fit["n_points"] for fit in fits], dtype=np.int32)
        h["duration_s"] = np.asarray([fit["duration_s"] for fit in fits], dtype=np.float64)
        h["r0_gcrs_m"] = np.asarray([fit["r0_gcrs_m"] for fit in fits], dtype=np.float64)
        h["v0_gcrs_mps"] = np.asarray([fit["v0_gcrs_mps"] for fit in fits], dtype=np.float64)
        h["r0_prior_gcrs_m"] = np.asarray([fit["r0_prior_gcrs_m"] for fit in fits], dtype=np.float64)
        h["v0_prior_gcrs_mps"] = np.asarray([fit["v0_prior_gcrs_mps"] for fit in fits], dtype=np.float64)
        h["prior_speed_km_s"] = np.asarray([fit["prior_speed_km_s"] for fit in fits], dtype=np.float64)
        h["speed_km_s"] = np.asarray([fit["speed_km_s"] for fit in fits], dtype=np.float64)
        h["rms_total_path_residual_m"] = np.asarray([fit["rms_total_path_residual_m"] for fit in fits], dtype=np.float64)
        h["median_abs_total_path_residual_m"] = np.asarray(
            [fit["median_abs_total_path_residual_m"] for fit in fits], dtype=np.float64
        )
        h["rms_half_path_diagnostic_residual_m"] = np.asarray(
            [fit["rms_half_path_diagnostic_residual_m"] for fit in fits], dtype=np.float64
        )
        h["median_abs_half_path_diagnostic_residual_m"] = np.asarray(
            [fit["median_abs_half_path_diagnostic_residual_m"] for fit in fits], dtype=np.float64
        )
        h["start_alt_km"] = np.asarray([fit["alt_km"][0] for fit in fits], dtype=np.float64)
        h["end_alt_km"] = np.asarray([fit["alt_km"][-1] for fit in fits], dtype=np.float64)
        h["optimizer_success"] = np.asarray([fit["optimizer_success"] for fit in fits], dtype=bool)
        h["optimizer_nfev"] = np.asarray([fit["optimizer_nfev"] for fit in fits], dtype=np.int32)

        points = h.create_group("points")
        for fit in fits:
            group = points.create_group(fit["event_id"])
            group["time_ns"] = fit["time_ns"]
            group["beijing_local_time_ns"] = fit["beijing_local_time_ns"]
            group["t_rel_s"] = fit["t_rel_s"]
            group["itrs_fit_m"] = fit["itrs_fit_m"]
            group["itrs_fit_v_mps"] = fit["itrs_fit_v_mps"]
            group["prior_points_ecef_m"] = fit["prior_points_ecef_m"]
            group["prior_points_gcrs_m"] = fit["prior_points_gcrs_m"]
            group["lat_deg"] = fit["lat_deg"]
            group["lon_deg"] = fit["lon_deg"]
            group["alt_km"] = fit["alt_km"]
            group["measured_total_paths_m"] = fit["measured_total_paths_m"]
            group["predicted_total_paths_m"] = fit["predicted_total_paths_m"]
            group["total_path_residuals_m"] = fit["total_path_residuals_m"]
            group["measured_half_path_diagnostic_m"] = fit["measured_half_path_diagnostic_m"]
            group["predicted_half_path_diagnostic_m"] = fit["predicted_half_path_diagnostic_m"]
            group["half_path_diagnostic_residuals_m"] = fit["half_path_diagnostic_residuals_m"]


def main():
    triplets = pair_tristatic_events(load_events(SAN_PATTERN), load_events(DAN_PATTERN), load_events(WEN_PATTERN))
    fits = []
    for idx, (san_event, dan_event, wen_event) in enumerate(triplets):
        event_id = f"tri_{idx:04d}_{san_event.t0_ns}"
        fit = solve_triplet(event_id, san_event, dan_event, wen_event)
        if fit is not None:
            fits.append(fit)
        if (idx + 1) % 25 == 0:
            print(f"processed {idx + 1}/{len(triplets)} triplets; fits={len(fits)}", flush=True)

    if not fits:
        raise RuntimeError("No GCRS trajectory fits were produced.")

    os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)
    write_h5(OUTPUT_H5, fits)

    speeds = np.asarray([fit["speed_km_s"] for fit in fits], dtype=np.float64)
    total_rms = np.asarray([fit["rms_total_path_residual_m"] for fit in fits], dtype=np.float64)
    half_path_rms = np.asarray([fit["rms_half_path_diagnostic_residual_m"] for fit in fits], dtype=np.float64)
    n_points = np.asarray([fit["n_points"] for fit in fits], dtype=np.int32)
    print(f"fits: {len(fits)}")
    print(f"points total: {int(np.sum(n_points))}")
    print(f"speed km/s median/range: {np.nanmedian(speeds):.3f} / {np.nanmin(speeds):.3f} to {np.nanmax(speeds):.3f}")
    print(
        "total-path RMS residual m median/range: "
        f"{np.nanmedian(total_rms):.1f} / {np.nanmin(total_rms):.1f} to {np.nanmax(total_rms):.1f}"
    )
    print(
        "half-path diagnostic RMS residual m median/range: "
        f"{np.nanmedian(half_path_rms):.1f} / {np.nanmin(half_path_rms):.1f} to {np.nanmax(half_path_rms):.1f}"
    )
    print(OUTPUT_H5)


if __name__ == "__main__":
    main()
