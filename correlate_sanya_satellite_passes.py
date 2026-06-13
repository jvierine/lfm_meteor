#!/usr/bin/env python3
"""Correlate Sanya high-SNR range detections with TLE-predicted satellite passes.

This script is intentionally self-contained so it can be copied to the
revontuli server and run next to the Sanya matched-filter products.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
from dataclasses import dataclass

import h5py
import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

try:
    from sanya_opts import SANYA_RANGE_CORRECTION_KM, SANYA_TLE_RANGE_OFFSET_KM
except Exception:
    SANYA_TLE_RANGE_OFFSET_KM = 16.0186
    SANYA_RANGE_CORRECTION_KM = -SANYA_TLE_RANGE_OFFSET_KM

try:
    from mpi4py import MPI
except Exception:
    MPI = None


VERSION = "v20260613c"
C_M_PER_S = 299792458.0
SANYA_LAT_DEG = 18.3492
SANYA_LON_DEG = 109.6222
SANYA_ALT_KM = 0.05
SANYA_TX_AZ_DEG = 15.0
SANYA_TX_EL_DEG = 75.0


@dataclass
class Detection:
    event_id: str
    source_file: str
    time_ns: int
    unix_s: float
    range_km: float
    snr_db: float
    ipp_us: float
    range_gate: int


@dataclass
class TleRecord:
    sat_id: str
    name: str
    line1: str
    line2: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sanya-dir", default="results/head_echoes/sanya")
    p.add_argument("--tle", default="tle/space_track_gp_snapshot_2024-04-22_latest_per_object.tle")
    p.add_argument("--output-dir", default=f"results/satellite_correlation/{VERSION}")
    p.add_argument("--snr-min-db", type=float, default=35.0)
    p.add_argument("--beam-half-angle-deg", type=float, default=10.0)
    p.add_argument("--coarse-step-s", type=float, default=60.0)
    p.add_argument("--fine-step-s", type=float, default=1.0)
    p.add_argument("--coarse-candidate-angle-deg", type=float, default=40.0)
    p.add_argument(
        "--range-tolerance-km",
        type=float,
        default=0.0,
        help="Diagnostic-only when <=0. Positive values reject detections by aliased range residual.",
    )
    p.add_argument("--min-matched-pulses", type=int, default=3)
    p.add_argument("--max-satellites", type=int, default=0)
    p.add_argument("--satellite-start", type=int, default=0)
    p.add_argument("--satellite-stop", type=int, default=0)
    p.add_argument("--pass-pad-s", type=float, default=120.0)
    p.add_argument(
        "--source-timezone-offset-hours",
        type=float,
        default=8.0,
        help=(
            "Subtract this offset from HDF5 times when the file lacks explicit UTC metadata. "
            "The old Sanya server products store MATLAB Beijing-local timestamps in times_ns."
        ),
    )
    return p.parse_args()


def iso_utc(unix_s: float) -> str:
    return dt.datetime.fromtimestamp(float(unix_s), tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def times_from_unix(ts, unix_s: np.ndarray):
    dts = [dt.datetime.fromtimestamp(float(s), tz=dt.timezone.utc) for s in np.asarray(unix_s, dtype=float)]
    return ts.utc(dts)


def azel_unit(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.vstack((np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)))


def angular_sep_deg(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    beam = azel_unit(np.asarray(SANYA_TX_AZ_DEG), np.asarray(SANYA_TX_EL_DEG)).reshape(3, 1)
    los = azel_unit(np.asarray(az_deg), np.asarray(el_deg))
    dot = np.sum(beam * los, axis=0)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def read_tles(path: str) -> list[TleRecord]:
    records: list[TleRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = [line.rstrip("\n") for line in fh if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name = lines[i][2:7].strip()
            records.append(TleRecord(name.lstrip("0") or name, name, lines[i], lines[i + 1]))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name = lines[i].strip()
            cat = lines[i + 1][2:7].strip()
            records.append(TleRecord(cat.lstrip("0") or cat, name, lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return records


def scalar_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "decode"):
        return value.decode("utf-8", "replace")
    return str(value)


def read_detections(sanya_dir: str, snr_min_db: float, source_timezone_offset_hours: float) -> tuple[list[Detection], str, float]:
    detections: list[Detection] = []
    correction_mode = "unknown"
    correction_ns_default = int(round(source_timezone_offset_hours * 3600.0 * 1e9))
    for path in sorted(glob.glob(os.path.join(sanya_dir, "*.h5"))):
        with h5py.File(path, "r") as h:
            snr = h["snr_peak_db"][()]
            mask = np.asarray(snr >= snr_min_db)
            if not np.any(mask):
                continue
            times_ns = h["times_ns"][()].astype(np.int64)
            if h.attrs.get("times_ns_time_scale", "") == "UTC":
                correction_ns = 0
                correction_mode = "hdf5_times_ns_marked_utc"
            else:
                correction_ns = correction_ns_default
                correction_mode = f"hdf5_times_ns_minus_{source_timezone_offset_hours:g}h_due_missing_utc_metadata"
            times_ns = times_ns - correction_ns
            ranges = h["range_km"][()]
            gates = h["range_gate"][()]
            ipp_us = float(h["ipp_us"][()])
            event_id = scalar_string(h["event_id"][()])
            source_file = scalar_string(h["source_file"][()])
            for idx in np.where(mask)[0]:
                t_ns = int(times_ns[idx])
                detections.append(
                    Detection(
                        event_id=event_id,
                        source_file=source_file,
                        time_ns=t_ns,
                        unix_s=t_ns / 1e9,
                        range_km=float(ranges[idx]),
                        snr_db=float(snr[idx]),
                        ipp_us=ipp_us,
                        range_gate=int(gates[idx]),
                    )
                )
    detections.sort(key=lambda d: d.unix_s)
    return detections, correction_mode, source_timezone_offset_hours


def split_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        if start is not None and ((not val) or i == len(mask) - 1):
            end = i if val and i == len(mask) - 1 else i - 1
            runs.append((start, end))
            start = None
    return runs


def observe_satellite(sat: EarthSatellite, observer, ts, unix_s: np.ndarray):
    t = times_from_unix(ts, unix_s)
    return observe_satellite_time(sat, observer, t)


def observe_satellite_time(sat: EarthSatellite, observer, t):
    topocentric = (sat - observer).at(t)
    alt, az, distance = topocentric.altaz()
    sep = angular_sep_deg(az.degrees, alt.degrees)
    return sep, distance.km, az.degrees, alt.degrees


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.nanpercentile(np.asarray(values, dtype=float), q))


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD if MPI is not None else None
    rank = comm.Get_rank() if comm is not None else 0
    size = comm.Get_size() if comm is not None else 1
    os.makedirs(args.output_dir, exist_ok=True)

    detections, time_correction_mode, source_timezone_offset_hours = read_detections(
        args.sanya_dir, args.snr_min_db, args.source_timezone_offset_hours
    )
    if not detections:
        raise SystemExit("No detections survived the SNR threshold.")
    det_times = np.asarray([d.unix_s for d in detections], dtype=float)
    det_ranges = np.asarray([d.range_km for d in detections], dtype=float)
    det_snrs = np.asarray([d.snr_db for d in detections], dtype=float)
    det_events = np.asarray([d.event_id for d in detections], dtype=object)
    det_ipps = np.asarray([d.ipp_us for d in detections], dtype=float)
    alias_km = 0.5 * C_M_PER_S * float(np.nanmedian(det_ipps)) * 1e-6 / 1e3

    tles = read_tles(args.tle)
    n_tles_total = len(tles)
    if args.satellite_start or args.satellite_stop:
        stop = args.satellite_stop if args.satellite_stop else len(tles)
        tles = tles[args.satellite_start : stop]
    if args.max_satellites:
        tles = tles[: args.max_satellites]
    n_tles_active = len(tles)
    tles_for_rank = tles[rank::size]

    ts = load.timescale()
    observer = wgs84.latlon(SANYA_LAT_DEG, SANYA_LON_DEG, elevation_m=SANYA_ALT_KM * 1e3)
    t0 = math.floor(det_times.min() / args.coarse_step_s) * args.coarse_step_s
    t1 = math.ceil(det_times.max() / args.coarse_step_s) * args.coarse_step_s
    coarse_times = np.arange(t0, t1 + 0.5 * args.coarse_step_s, args.coarse_step_s)
    coarse_sf_time = times_from_unix(ts, coarse_times)
    det_sf_time = times_from_unix(ts, det_times)

    pass_rows: list[dict] = []
    match_rows_raw: list[dict] = []
    candidate_windows: list[tuple[TleRecord, EarthSatellite, float, float, float]] = []

    for si, tle in enumerate(tles_for_rank):
        if si % 100 == 0:
            print(f"rank {rank}/{size} satellite {si}/{len(tles_for_rank)}", flush=True)
        sat = EarthSatellite(tle.line1, tle.line2, tle.name, ts)
        try:
            sep, rng, az, el = observe_satellite_time(sat, observer, coarse_sf_time)
        except Exception as exc:
            print(f"skip {tle.sat_id}: {exc}", flush=True)
            continue
        coarse_candidate = np.asarray(sep <= args.coarse_candidate_angle_deg)
        if not np.any(coarse_candidate):
            continue
        for run_start, run_end in split_true_runs(coarse_candidate):
            start_s = max(t0, coarse_times[max(0, run_start - 1)] - args.pass_pad_s)
            stop_s = min(t1, coarse_times[min(len(coarse_times) - 1, run_end + 1)] + args.pass_pad_s)
            fine_times = np.arange(start_s, stop_s + 0.5 * args.fine_step_s, args.fine_step_s)
            fsep, frng, faz, fel = observe_satellite(sat, observer, ts, fine_times)
            fine_inside = np.asarray(fsep <= args.beam_half_angle_deg)
            if not np.any(fine_inside):
                continue
            imin = int(np.nanargmin(fsep))
            pass_start_s = float(fine_times[np.where(fine_inside)[0][0]])
            pass_stop_s = float(fine_times[np.where(fine_inside)[0][-1]])
            pass_row = {
                "sat_id": tle.sat_id,
                "tle_name": tle.name,
                "pass_start_utc": iso_utc(pass_start_s),
                "pass_stop_utc": iso_utc(pass_stop_s),
                "closest_utc": iso_utc(fine_times[imin]),
                "min_beam_angle_deg": float(fsep[imin]),
                "slant_range_km_at_closest": float(frng[imin]),
                "az_deg_at_closest": float(faz[imin]),
                "el_deg_at_closest": float(fel[imin]),
            }
            pass_rows.append(pass_row)
            print(
                "POTENTIAL_PASS "
                f"rank={rank} sat={tle.sat_id} closest={pass_row['closest_utc']} "
                f"min_angle_deg={float(fsep[imin]):.3f} range_km={float(frng[imin]):.1f}",
                flush=True,
            )
            candidate_windows.append((tle, sat, pass_start_s, pass_stop_s, float(fsep[imin])))

    for ci, (tle, sat, start_s, stop_s, min_sep) in enumerate(candidate_windows):
        if ci % 50 == 0:
            print(f"candidate window {ci}/{len(candidate_windows)}", flush=True)
        idx = np.where((det_times >= start_s) & (det_times <= stop_s))[0]
        if len(idx) == 0:
            continue
        sep, rng, az, el = observe_satellite_time(sat, observer, det_sf_time[idx])
        inside = sep <= args.beam_half_angle_deg
        if not np.any(inside):
            continue
        for local_i in np.where(inside)[0]:
            i = int(idx[local_i])
            alias_n = int(np.rint((rng[local_i] - det_ranges[i]) / alias_km))
            pred_alias = float(rng[local_i] - alias_n * alias_km)
            residual = float(det_ranges[i] - pred_alias)
            if args.range_tolerance_km > 0.0 and abs(residual) > args.range_tolerance_km:
                continue
            print(
                "DETECTION_IN_PASS "
                f"rank={rank} sat={tle.sat_id} event={det_events[i]} "
                f"time={iso_utc(det_times[i])} snr_db={float(det_snrs[i]):.1f} "
                f"obs_range_km={float(det_ranges[i]):.3f} alias_n={alias_n} "
                f"range_offset_km={residual:.3f}",
                flush=True,
            )
            match_rows_raw.append(
                {
                    "event_id": det_events[i],
                    "sat_id": tle.sat_id,
                    "tle_name": tle.name,
                    "alias_n": alias_n,
                    "time_utc": iso_utc(det_times[i]),
                    "time_unix_s": float(det_times[i]),
                    "observed_range_km": float(det_ranges[i]),
                    "predicted_slant_range_km": float(rng[local_i]),
                    "predicted_aliased_range_km": pred_alias,
                    "range_offset_km": residual,
                    "beam_angle_deg": float(sep[local_i]),
                    "sat_az_deg": float(az[local_i]),
                    "sat_el_deg": float(el[local_i]),
                    "snr_db": float(det_snrs[i]),
                    "pass_min_beam_angle_deg": min_sep,
                }
            )

    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in match_rows_raw:
        key = (row["event_id"], row["sat_id"], int(row["alias_n"]))
        grouped.setdefault(key, []).append(row)

    match_rows: list[dict] = []
    for (event_id, sat_id, alias_n), rows in grouped.items():
        if len(rows) < args.min_matched_pulses:
            continue
        offsets = [float(r["range_offset_km"]) for r in rows]
        snrs = [float(r["snr_db"]) for r in rows]
        angles = [float(r["beam_angle_deg"]) for r in rows]
        obs = [float(r["observed_range_km"]) for r in rows]
        pred = [float(r["predicted_aliased_range_km"]) for r in rows]
        slant = [float(r["predicted_slant_range_km"]) for r in rows]
        times = [float(r["time_unix_s"]) for r in rows]
        match_rows.append(
            {
                "event_id": event_id,
                "sat_id": sat_id,
                "tle_name": rows[0]["tle_name"],
                "alias_n": alias_n,
                "n_pulses": len(rows),
                "start_utc": iso_utc(min(times)),
                "stop_utc": iso_utc(max(times)),
                "median_snr_db": float(np.median(snrs)),
                "max_snr_db": float(np.max(snrs)),
                "median_beam_angle_deg": float(np.median(angles)),
                "min_beam_angle_deg": float(np.min(angles)),
                "median_observed_range_km": float(np.median(obs)),
                "median_predicted_slant_range_km": float(np.median(slant)),
                "median_predicted_aliased_range_km": float(np.median(pred)),
                "median_range_offset_km": float(np.median(offsets)),
                "mean_range_offset_km": float(np.mean(offsets)),
                "rms_range_offset_km": float(np.sqrt(np.mean(np.square(offsets)))),
                "p10_range_offset_km": percentile(offsets, 10),
                "p90_range_offset_km": percentile(offsets, 90),
            }
        )

    if comm is not None:
        gathered_pass = comm.gather(pass_rows, root=0)
        gathered_raw = comm.gather(match_rows_raw, root=0)
        gathered_grouped = comm.gather(match_rows, root=0)
        if rank != 0:
            return
        pass_rows = [row for rows in gathered_pass for row in rows]
        match_rows_raw = [row for rows in gathered_raw for row in rows]
        match_rows = [row for rows in gathered_grouped for row in rows]

    pass_rows.sort(key=lambda r: (r["closest_utc"], r["sat_id"]))
    match_rows.sort(key=lambda r: (-int(r["n_pulses"]), abs(float(r["median_range_offset_km"]))))
    match_rows_raw.sort(key=lambda r: (r["time_utc"], r["sat_id"]))

    pass_fields = [
        "sat_id",
        "tle_name",
        "pass_start_utc",
        "pass_stop_utc",
        "closest_utc",
        "min_beam_angle_deg",
        "slant_range_km_at_closest",
        "az_deg_at_closest",
        "el_deg_at_closest",
    ]
    raw_fields = [
        "event_id",
        "sat_id",
        "tle_name",
        "alias_n",
        "time_utc",
        "observed_range_km",
        "predicted_slant_range_km",
        "predicted_aliased_range_km",
        "range_offset_km",
        "beam_angle_deg",
        "sat_az_deg",
        "sat_el_deg",
        "snr_db",
        "pass_min_beam_angle_deg",
    ]
    match_fields = [
        "event_id",
        "sat_id",
        "tle_name",
        "alias_n",
        "n_pulses",
        "start_utc",
        "stop_utc",
        "median_snr_db",
        "max_snr_db",
        "median_beam_angle_deg",
        "min_beam_angle_deg",
        "median_observed_range_km",
        "median_predicted_slant_range_km",
        "median_predicted_aliased_range_km",
        "median_range_offset_km",
        "mean_range_offset_km",
        "rms_range_offset_km",
        "p10_range_offset_km",
        "p90_range_offset_km",
    ]
    write_csv(os.path.join(args.output_dir, "sanya_satellite_passes.csv"), pass_rows, pass_fields)
    write_csv(os.path.join(args.output_dir, "sanya_satellite_detection_matches_raw.csv"), match_rows_raw, raw_fields)
    write_csv(os.path.join(args.output_dir, "sanya_satellite_detection_matches_grouped.csv"), match_rows, match_fields)

    summary = {
        "version": VERSION,
        "sanya_dir": args.sanya_dir,
        "tle": args.tle,
        "output_dir": args.output_dir,
        "snr_min_db": args.snr_min_db,
        "beam_half_angle_deg": args.beam_half_angle_deg,
        "range_tolerance_km": args.range_tolerance_km,
        "range_match_mode": "angle_only_range_diagnostic" if args.range_tolerance_km <= 0.0 else "angle_and_range",
        "min_matched_pulses": args.min_matched_pulses,
        "coarse_step_s": args.coarse_step_s,
        "fine_step_s": args.fine_step_s,
        "coarse_candidate_angle_deg": args.coarse_candidate_angle_deg,
        "coarse_guard_rationale": (
            "Default 40 deg guard = 10 deg beam plus allowance for a near-overhead LEO "
            "moving about 0.9 deg/s across half of a 60 s coarse interval, with margin."
        ),
        "satellite_start": args.satellite_start,
        "satellite_stop": args.satellite_stop if args.satellite_stop else n_tles_total,
        "n_tles_total": n_tles_total,
        "n_tles": n_tles_active,
        "mpi_size": size,
        "n_high_snr_detections": len(detections),
        "n_pass_windows": len(pass_rows),
        "n_raw_detection_matches": len(match_rows_raw),
        "n_grouped_detection_matches": len(match_rows),
        "detection_start_utc": iso_utc(det_times.min()),
        "detection_stop_utc": iso_utc(det_times.max()),
        "alias_range_km": alias_km,
        "sanya_lat_deg": SANYA_LAT_DEG,
        "sanya_lon_deg": SANYA_LON_DEG,
        "sanya_alt_km": SANYA_ALT_KM,
        "sanya_tx_az_deg": SANYA_TX_AZ_DEG,
        "sanya_tx_el_deg": SANYA_TX_EL_DEG,
        "sanya_tle_range_offset_km": SANYA_TLE_RANGE_OFFSET_KM,
        "sanya_range_correction_km": SANYA_RANGE_CORRECTION_KM,
        "sanya_range_correction_sign": "corrected_observed_range_km = observed_range_km + sanya_range_correction_km",
        "time_correction_mode": time_correction_mode,
        "source_timezone_offset_hours": source_timezone_offset_hours,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
