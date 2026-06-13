#!/usr/bin/env python3
"""Estimate Danzhou/Wenchang delays by minimizing distance to the Sanya beam."""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool, cpu_count

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np

import sanya_opts as sc
from grid_search_delays_beam_axis import (
    C,
    beam_axis,
    build_trajectories,
    delay_us_to_range_km,
    gate_to_delay_us,
    initial_guess,
    solve_three_spheres,
)


_TRAJECTORIES = None
_AXIS_ORIGIN = None
_AXIS_DIRECTION = None
_MAX_LAT_DEG = None
_MIN_KEPT = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dan-start-us", type=float, default=250.0)
    p.add_argument("--dan-stop-us", type=float, default=450.0)
    p.add_argument("--wen-start-us", type=float, default=250.0)
    p.add_argument("--wen-stop-us", type=float, default=450.0)
    p.add_argument("--step-us", type=float, default=5.0)
    p.add_argument("--max-lat-deg", type=float, default=18.7)
    p.add_argument("--workers", type=int, default=min(8, cpu_count()))
    p.add_argument("--min-kept-fraction", type=float, default=0.9)
    p.add_argument("--output-h5", default=os.path.join("results", "remote_delay_beam_axis_fast.h5"))
    p.add_argument("--output-png", default=os.path.join("results", "remote_delay_beam_axis_fast.png"))
    return p.parse_args()


def init_worker(trajectories, axis_origin, axis_direction, max_lat_deg, min_kept):
    global _TRAJECTORIES
    global _AXIS_ORIGIN
    global _AXIS_DIRECTION
    global _MAX_LAT_DEG
    global _MIN_KEPT
    _TRAJECTORIES = trajectories
    _AXIS_ORIGIN = axis_origin
    _AXIS_DIRECTION = axis_direction
    _MAX_LAT_DEG = max_lat_deg
    _MIN_KEPT = min_kept


def fast_points(traj, dan_delay0_us, wen_delay0_us):
    p_san = np.asarray(sc.p_san, dtype=np.float64)
    p_dan = np.asarray(sc.p_dan, dtype=np.float64)
    p_wen = np.asarray(sc.p_wen, dtype=np.float64)
    dan_ranges_km = delay_us_to_range_km(dan_delay0_us + gate_to_delay_us(traj.dan_gates, traj.dan_sr_mhz))
    wen_ranges_km = delay_us_to_range_km(wen_delay0_us + gate_to_delay_us(traj.wen_gates, traj.wen_sr_mhz))
    x0 = initial_guess(traj.san_az_deg, traj.san_el_deg, float(np.median(traj.san_ranges_km)))
    points = []
    n_invalid = 0
    for san_range_km, dan_range_km, wen_range_km in zip(traj.san_ranges_km, dan_ranges_km, wen_ranges_km):
        san_range_m = float(san_range_km) * 1e3
        dan_target_m = 2.0 * float(dan_range_km) * 1e3 - san_range_m
        wen_target_m = 2.0 * float(wen_range_km) * 1e3 - san_range_m
        xhat = solve_three_spheres(
            p_san,
            p_dan,
            p_wen,
            san_range_m,
            dan_target_m,
            wen_target_m,
            x0,
        )
        if xhat is None:
            n_invalid += 1
            continue
        x0 = xhat
        points.append(xhat)
    if not points:
        return np.empty((0, 3), dtype=np.float64), n_invalid
    return np.asarray(points, dtype=np.float64), n_invalid


def score_pair(dan_delay0_us, wen_delay0_us):
    total_km = 0.0
    n_kept = 0
    n_rejected = 0
    n_invalid = 0
    altitudes = []
    for traj in _TRAJECTORIES:
        points, invalid = fast_points(traj, dan_delay0_us, wen_delay0_us)
        n_invalid += invalid
        if points.size == 0:
            continue
        llh = np.asarray([jcoord.ecef2geodetic(p[0], p[1], p[2]) for p in points], dtype=np.float64)
        alt_km = llh[:, 2] / 1e3
        keep = np.isfinite(llh[:, 0]) & np.isfinite(llh[:, 1]) & np.isfinite(alt_km) & (llh[:, 0] <= _MAX_LAT_DEG)
        n_rejected += int((~keep).sum())
        if not np.any(keep):
            continue
        rel = points[keep] - _AXIS_ORIGIN
        distances_m = np.linalg.norm(np.cross(rel, _AXIS_DIRECTION), axis=1)
        total_km += float(np.sum(distances_m) / 1e3)
        altitudes.extend(alt_km[keep].tolist())
        n_kept += int(keep.sum())
    if n_kept == 0:
        return np.inf, np.nan, np.nan, n_kept, n_rejected, n_invalid
    if n_kept < _MIN_KEPT:
        return np.inf, np.nan, np.nan, n_kept, n_rejected, n_invalid
    penalty_km = 1.0e3 * n_invalid
    altitudes = np.asarray(altitudes, dtype=np.float64)
    return (
        total_km + penalty_km,
        float(np.mean(altitudes)),
        float(np.median(altitudes)),
        n_kept,
        n_rejected,
        n_invalid,
    )


def evaluate(job):
    i, j, dan_delay0_us, wen_delay0_us = job
    score_km, mean_alt_km, median_alt_km, n_kept, n_rejected, n_invalid = score_pair(
        dan_delay0_us,
        wen_delay0_us,
    )
    return i, j, dan_delay0_us, wen_delay0_us, score_km, mean_alt_km, median_alt_km, n_kept, n_rejected, n_invalid


def main() -> None:
    args = parse_args()
    trajectories = build_trajectories()
    axis_origin, axis_direction = beam_axis()
    n_possible = int(sum(len(traj.san_ranges_km) for traj in trajectories))
    min_kept = int(np.ceil(args.min_kept_fraction * n_possible))
    dan_grid = np.arange(args.dan_start_us, args.dan_stop_us + 0.5 * args.step_us, args.step_us)
    wen_grid = np.arange(args.wen_start_us, args.wen_stop_us + 0.5 * args.step_us, args.step_us)
    jobs = [
        (i, j, float(dan), float(wen))
        for i, dan in enumerate(dan_grid)
        for j, wen in enumerate(wen_grid)
    ]

    score_grid = np.full((len(dan_grid), len(wen_grid)), np.nan, dtype=np.float64)
    mean_alt_grid = np.full_like(score_grid, np.nan)
    median_alt_grid = np.full_like(score_grid, np.nan)
    kept_grid = np.zeros_like(score_grid, dtype=np.int32)
    rejected_grid = np.zeros_like(score_grid, dtype=np.int32)
    invalid_grid = np.zeros_like(score_grid, dtype=np.int32)
    best = None

    print(
        f"Evaluating {len(jobs)} delay pairs with {args.workers} workers; "
        f"{len(trajectories)} trajectories; Sanya correction {sc.SANYA_RANGE_CORRECTION_KM:+.4f} km; "
        f"min kept {min_kept}/{n_possible}.",
        flush=True,
    )
    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(trajectories, axis_origin, axis_direction, args.max_lat_deg, min_kept),
    ) as pool:
        for result in pool.imap_unordered(evaluate, jobs):
            i, j, dan, wen, score, mean_alt, median_alt, n_kept, n_rejected, n_invalid = result
            score_grid[i, j] = score
            mean_alt_grid[i, j] = mean_alt
            median_alt_grid[i, j] = median_alt
            kept_grid[i, j] = n_kept
            rejected_grid[i, j] = n_rejected
            invalid_grid[i, j] = n_invalid
            if best is None or score < best["score_km"]:
                best = {
                    "dan_delay_us": dan,
                    "wen_delay_us": wen,
                    "score_km": score,
                    "mean_alt_km": mean_alt,
                    "median_alt_km": median_alt,
                    "n_kept": int(n_kept),
                    "n_rejected": int(n_rejected),
                    "n_invalid": int(n_invalid),
                }
                print(
                    f"New best: D_dan={dan:.3f} us D_wen={wen:.3f} us "
                    f"score={score:.3f} km kept={n_kept} invalid={n_invalid}",
                    flush=True,
                )

    os.makedirs(os.path.dirname(args.output_h5), exist_ok=True)
    with h5py.File(args.output_h5, "w") as h:
        h["dan_delay_grid_us"] = dan_grid
        h["wen_delay_grid_us"] = wen_grid
        h["score_grid_km"] = score_grid
        h["mean_alt_grid_km"] = mean_alt_grid
        h["median_alt_grid_km"] = median_alt_grid
        h["kept_grid"] = kept_grid
        h["rejected_grid"] = rejected_grid
        h["invalid_grid"] = invalid_grid
        h.attrs["sanya_range_correction_km"] = sc.SANYA_RANGE_CORRECTION_KM
        h.attrs["metric"] = "sum of distances to Sanya transmit beam axis with invalid-point penalty"
        h.attrs["max_lat_deg"] = args.max_lat_deg
        h.attrs["min_kept_fraction"] = args.min_kept_fraction
        h.attrs["min_kept_points"] = min_kept
        h.attrs["n_possible_points"] = n_possible
        for key, value in best.items():
            h.attrs[f"best_{key}"] = value

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(wen_grid, dan_grid, score_grid, shading="auto")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Beam-axis score (km)")
    ax.scatter([best["wen_delay_us"]], [best["dan_delay_us"]], c="red", marker="x", s=100)
    ax.set_xlabel("Wenchang first-sample delay (us)")
    ax.set_ylabel("Danzhou first-sample delay (us)")
    ax.set_title("Remote Delay Fit to Sanya Beam Axis")
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=220)
    plt.close(fig)

    print("")
    print(f"Best Danzhou first-sample delay: {best['dan_delay_us']:.3f} us")
    print(f"Best Wenchang first-sample delay: {best['wen_delay_us']:.3f} us")
    print(f"Best beam-axis score: {best['score_km']:.3f} km")
    print(f"Best mean/median altitude: {best['mean_alt_km']:.3f} / {best['median_alt_km']:.3f} km")
    print(f"Best kept/rejected/invalid points: {best['n_kept']}/{best['n_rejected']}/{best['n_invalid']}")
    print(args.output_h5)
    print(args.output_png)


if __name__ == "__main__":
    main()
