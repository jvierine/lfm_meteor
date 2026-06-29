#!/usr/bin/env python3
"""Rank Sanya monostatic events by off-center micro-Doppler peak strength."""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np
import scipy.signal as sig

from plot_sanya_microdoppler_gallery import load_sanya_event
from plot_tristatic_microdoppler_fft import microdoppler_image


SCRIPT_VERSION = "v20260619a"
DEFAULT_EVENT_GLOB = os.path.join("results", "head_echoes", "sanya", "sanya_*.h5")
DEFAULT_OUTPUT = os.path.join("results", f"sanya_microdoppler_secondary_peak_scan_{SCRIPT_VERSION}.h5")


def strongest_secondary(freq_hz: np.ndarray, power_db: np.ndarray, guard_khz: float, width_khz: float) -> tuple[float, float, float]:
    use = (np.abs(freq_hz) >= guard_khz * 1e3) & (np.abs(freq_hz) <= 0.5 * width_khz * 1e3) & np.isfinite(power_db)
    if not np.any(use):
        return np.nan, np.nan, np.nan
    f = freq_hz[use]
    p = power_db[use]
    peaks, props = sig.find_peaks(p, prominence=1.5)
    if peaks.size:
        best_local = int(peaks[np.argmax(p[peaks])])
        prominence = float(props["prominences"][np.argmax(p[peaks])])
    else:
        best_local = int(np.nanargmax(p))
        prominence = np.nan
    return float(f[best_local] / 1e3), float(p[best_local]), prominence


def longest_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def analyze_event(path: str, args: argparse.Namespace) -> dict:
    data = load_sanya_event(path)
    result = microdoppler_image(data, args)
    n_pulses = int(len(data["times_ns"]))
    if result is None:
        return {
            "event_id": data["event_id"],
            "source_h5": path,
            "first_pulse_unix_us": int(data["times_ns"][0] // 1000),
            "n_pulses": n_pulses,
            "n_valid": 0,
            "max_secondary_db": np.nan,
            "max_secondary_khz": np.nan,
            "median_candidate_db": np.nan,
            "candidate_fraction": 0.0,
            "longest_candidate_run": 0,
            "score": -np.inf,
        }

    freq_hz = result["freq_hz"]
    image_db = result["image_db"]
    sec_khz = np.full(image_db.shape[1], np.nan, dtype=np.float64)
    sec_db = np.full(image_db.shape[1], np.nan, dtype=np.float64)
    prom_db = np.full(image_db.shape[1], np.nan, dtype=np.float64)
    for idx in range(image_db.shape[1]):
        if not np.any(np.isfinite(image_db[:, idx])):
            continue
        sec_khz[idx], sec_db[idx], prom_db[idx] = strongest_secondary(freq_hz, image_db[:, idx], args.guard_khz, args.width_khz)

    candidate = np.isfinite(sec_db) & (sec_db >= args.threshold_db)
    n_candidate = int(np.count_nonzero(candidate))
    n_valid = int(result["n_valid"])
    longest = longest_run(candidate)
    score = float(np.nanmax(sec_db)) if np.any(np.isfinite(sec_db)) else -np.inf
    score += 2.0 * min(longest, 5)
    score += 10.0 * (n_candidate / max(n_valid, 1))
    return {
        "event_id": data["event_id"],
        "source_h5": path,
        "first_pulse_unix_us": int(data["times_ns"][0] // 1000),
        "n_pulses": n_pulses,
        "n_valid": n_valid,
        "max_secondary_db": float(np.nanmax(sec_db)) if np.any(np.isfinite(sec_db)) else np.nan,
        "max_secondary_khz": float(sec_khz[int(np.nanargmax(sec_db))]) if np.any(np.isfinite(sec_db)) else np.nan,
        "median_candidate_db": float(np.nanmedian(sec_db[candidate])) if n_candidate else np.nan,
        "candidate_fraction": float(n_candidate / max(n_valid, 1)),
        "longest_candidate_run": int(longest),
        "score": score,
    }


def write_h5(rows: list[dict], args: argparse.Namespace) -> None:
    rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["event_glob"] = args.event_glob
        h.attrs["guard_khz"] = float(args.guard_khz)
        h.attrs["threshold_db"] = float(args.threshold_db)
        h.attrs["width_khz"] = float(args.width_khz)
        h.attrs["rank_note"] = "Candidate ranking only; not a final fragmentation classifier."
        for key in rows[0]:
            values = [r[key] for r in rows]
            if key in {"event_id", "source_h5"}:
                h.create_dataset(key, data=np.asarray(values, dtype=object), dtype=string_dtype)
            elif key in {"first_pulse_unix_us", "n_pulses", "n_valid", "longest_candidate_run"}:
                h[key] = np.asarray(values, dtype=np.int64)
            else:
                h[key] = np.asarray(values, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event-glob", default=DEFAULT_EVENT_GLOB)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--zero-pad-factor", type=int, default=64)
    p.add_argument("--gate-upsample-factor", type=int, default=1)
    p.add_argument("--width-khz", type=float, default=200.0)
    p.add_argument("--db-floor", type=float, default=-45.0)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--snr-min-db", type=float, default=-np.inf)
    p.add_argument("--guard-khz", type=float, default=8.0)
    p.add_argument("--threshold-db", type=float, default=-12.0)
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args()

    paths = sorted(glob.glob(args.event_glob))
    if args.max_events is not None:
        paths = paths[: args.max_events]
    if not paths:
        raise FileNotFoundError(args.event_glob)
    rows = []
    for idx, path in enumerate(paths, start=1):
        rows.append(analyze_event(path, args))
        if idx % 100 == 0:
            print(f"processed {idx}/{len(paths)}", flush=True)
    write_h5(rows, args)
    strong = sum(r["candidate_fraction"] > 0.0 for r in rows)
    persistent = sum(r["longest_candidate_run"] >= 3 for r in rows)
    print(f"n_events={len(rows)}")
    print(f"n_any_candidate={strong}")
    print(f"n_persistent_run_ge3={persistent}")
    print(f"output={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
