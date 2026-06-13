import os

import astropy.units as u
import h5py
import jcoord
import numpy as np
import stuffr
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time

from grid_search_delays_beam_axis import (
    DAN_CENTER_US,
    DAN_PATTERN,
    MAX_LAT_DEG,
    SAN_PATTERN,
    WEN_CENTER_US,
    WEN_PATTERN,
    delay_us_to_range_km,
    gate_to_delay_us,
    initial_guess,
    load_events,
    nearest_index,
    pair_tristatic_events,
    range_gates_to_km,
    solve_position,
)


OUTPUT_H5 = os.path.join("results", "gcrs_trajectory_fits_v20260610.h5")
MIN_POINTS = 3
MATCH_TOLERANCE_MS = 7.5


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


def fit_constant_velocity(points_gcrs_m, times_ns):
    t_rel_s = (np.asarray(times_ns, dtype=np.float64) - float(times_ns[0])) / 1e9
    design = np.column_stack([np.ones_like(t_rel_s), t_rel_s])
    coeffs = np.linalg.lstsq(design, points_gcrs_m, rcond=None)[0]
    r0_m = coeffs[0, :]
    v0_mps = coeffs[1, :]
    fit_m = design @ coeffs
    residuals_m = points_gcrs_m - fit_m
    residual_norm_m = np.linalg.norm(residuals_m, axis=1)
    return t_rel_s, r0_m, v0_mps, fit_m, residuals_m, residual_norm_m


def solve_triplet(event_id, san_event, dan_event, wen_event):
    matches = match_pulses_with_time(san_event, dan_event, wen_event)
    if len(matches) < MIN_POINTS:
        return None

    san_ranges_km = range_gates_to_km(san_event.range_gate, san_event.r0_km, san_event.sr_mhz)
    dan_ranges_km = delay_us_to_range_km(DAN_CENTER_US + gate_to_delay_us(dan_event.range_gate, dan_event.sr_mhz))
    wen_ranges_km = delay_us_to_range_km(WEN_CENTER_US + gate_to_delay_us(wen_event.range_gate, wen_event.sr_mhz))

    x0 = initial_guess(san_event.az_deg, san_event.el_deg, float(np.median(san_ranges_km)))
    points_ecef_m = []
    times_ns = []
    lat_deg = []
    lon_deg = []
    alt_km = []

    for match in matches:
        point = solve_position(
            float(san_ranges_km[match["san_idx"]]),
            float(dan_ranges_km[match["dan_idx"]]),
            float(wen_ranges_km[match["wen_idx"]]),
            x0,
        )
        x0 = point
        llh = jcoord.ecef2geodetic(point[0], point[1], point[2])
        if not np.isfinite(llh[0]) or not np.isfinite(llh[1]) or not np.isfinite(llh[2]):
            continue
        if float(llh[0]) > MAX_LAT_DEG:
            continue
        points_ecef_m.append(point)
        times_ns.append(match["time_ns"])
        lat_deg.append(float(llh[0]))
        lon_deg.append(float(llh[1]))
        alt_km.append(float(llh[2] / 1e3))

    if len(points_ecef_m) < MIN_POINTS:
        return None

    points_ecef_m = np.asarray(points_ecef_m, dtype=np.float64)
    times_ns = np.asarray(times_ns, dtype=np.int64)
    order = np.argsort(times_ns)
    points_ecef_m = points_ecef_m[order]
    times_ns = times_ns[order]
    lat_deg = np.asarray(lat_deg, dtype=np.float64)[order]
    lon_deg = np.asarray(lon_deg, dtype=np.float64)[order]
    alt_km = np.asarray(alt_km, dtype=np.float64)[order]

    points_gcrs_m = ecef_to_gcrs(points_ecef_m, times_ns)
    t_rel_s, r0_m, v0_mps, fit_gcrs_m, residuals_m, residual_norm_m = fit_constant_velocity(points_gcrs_m, times_ns)

    return {
        "event_id": event_id,
        "time_ns": times_ns,
        "t_rel_s": t_rel_s,
        "ecef_m": points_ecef_m,
        "gcrs_m": points_gcrs_m,
        "fit_gcrs_m": fit_gcrs_m,
        "residuals_m": residuals_m,
        "residual_norm_m": residual_norm_m,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "alt_km": alt_km,
        "r0_gcrs_m": r0_m,
        "v0_gcrs_mps": v0_mps,
        "speed_km_s": float(np.linalg.norm(v0_mps) / 1e3),
        "duration_s": float(t_rel_s[-1] - t_rel_s[0]),
        "rms_residual_m": float(np.sqrt(np.mean(residual_norm_m**2))),
        "median_residual_m": float(np.median(residual_norm_m)),
        "t0_ns": int(times_ns[0]),
        "t0_utc": stuffr.unix2datestr(int(times_ns[0]) / 1e9),
        "n_points": int(len(times_ns)),
    }


def write_h5(path, fits):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["coordinate_frame"] = "GCRS"
        h.attrs["model"] = "constant velocity, no deceleration; t0 is first retained detection"
        h.attrs["danzhou_first_sample_delay_us"] = DAN_CENTER_US
        h.attrs["wenchang_first_sample_delay_us"] = WEN_CENTER_US
        h.attrs["max_lat_deg"] = MAX_LAT_DEG

        h["event_id"] = np.asarray([fit["event_id"] for fit in fits], dtype=string_dtype)
        h["t0_ns"] = np.asarray([fit["t0_ns"] for fit in fits], dtype=np.int64)
        h["t0_utc"] = np.asarray([fit["t0_utc"] for fit in fits], dtype=string_dtype)
        h["n_points"] = np.asarray([fit["n_points"] for fit in fits], dtype=np.int32)
        h["duration_s"] = np.asarray([fit["duration_s"] for fit in fits], dtype=np.float64)
        h["r0_gcrs_m"] = np.asarray([fit["r0_gcrs_m"] for fit in fits], dtype=np.float64)
        h["v0_gcrs_mps"] = np.asarray([fit["v0_gcrs_mps"] for fit in fits], dtype=np.float64)
        h["speed_km_s"] = np.asarray([fit["speed_km_s"] for fit in fits], dtype=np.float64)
        h["rms_residual_m"] = np.asarray([fit["rms_residual_m"] for fit in fits], dtype=np.float64)
        h["median_residual_m"] = np.asarray([fit["median_residual_m"] for fit in fits], dtype=np.float64)
        h["start_alt_km"] = np.asarray([fit["alt_km"][0] for fit in fits], dtype=np.float64)
        h["end_alt_km"] = np.asarray([fit["alt_km"][-1] for fit in fits], dtype=np.float64)
        h["start_lat_deg"] = np.asarray([fit["lat_deg"][0] for fit in fits], dtype=np.float64)
        h["start_lon_deg"] = np.asarray([fit["lon_deg"][0] for fit in fits], dtype=np.float64)

        points = h.create_group("points")
        for fit in fits:
            group = points.create_group(fit["event_id"])
            group["time_ns"] = fit["time_ns"]
            group["t_rel_s"] = fit["t_rel_s"]
            group["ecef_m"] = fit["ecef_m"]
            group["gcrs_m"] = fit["gcrs_m"]
            group["fit_gcrs_m"] = fit["fit_gcrs_m"]
            group["residuals_m"] = fit["residuals_m"]
            group["residual_norm_m"] = fit["residual_norm_m"]
            group["lat_deg"] = fit["lat_deg"]
            group["lon_deg"] = fit["lon_deg"]
            group["alt_km"] = fit["alt_km"]


def main():
    triplets = pair_tristatic_events(load_events(SAN_PATTERN), load_events(DAN_PATTERN), load_events(WEN_PATTERN))
    fits = []
    for idx, (san_event, dan_event, wen_event) in enumerate(triplets):
        event_id = f"tri_{idx:04d}_{san_event.t0_ns}"
        fit = solve_triplet(event_id, san_event, dan_event, wen_event)
        if fit is not None:
            fits.append(fit)

    if not fits:
        raise RuntimeError("No GCRS trajectory fits were produced.")

    os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)
    write_h5(OUTPUT_H5, fits)

    speeds = np.asarray([fit["speed_km_s"] for fit in fits], dtype=np.float64)
    rms = np.asarray([fit["rms_residual_m"] for fit in fits], dtype=np.float64)
    n_points = np.asarray([fit["n_points"] for fit in fits], dtype=np.int32)
    print(f"fits: {len(fits)}")
    print(f"points total: {int(np.sum(n_points))}")
    print(f"speed km/s median/range: {np.nanmedian(speeds):.3f} / {np.nanmin(speeds):.3f} to {np.nanmax(speeds):.3f}")
    print(f"RMS residual m median/range: {np.nanmedian(rms):.1f} / {np.nanmin(rms):.1f} to {np.nanmax(rms):.1f}")
    print(OUTPUT_H5)


if __name__ == "__main__":
    main()
