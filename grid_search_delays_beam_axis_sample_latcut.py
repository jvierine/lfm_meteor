import os
from multiprocessing import Pool, cpu_count

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np

from grid_search_delays_beam_axis import (
    DAN_CENTER_US,
    WEN_CENTER_US,
    beam_axis,
    build_trajectories,
    solve_trajectory_points,
)


DELTA_US = 4.0
STEP_US = 0.25
MAX_LAT_DEG = 18.7
OUTPUT_H5 = os.path.join("results", "delay_grid_search_beam_axis_sample_latcut.h5")
OUTPUT_PNG = os.path.join("results", "delay_grid_search_beam_axis_sample_latcut.png")
N_WORKERS = min(8, cpu_count())

_TRAJECTORIES = None
_AXIS_ORIGIN = None
_AXIS_DIRECTION = None


def init_worker():
    global _TRAJECTORIES
    global _AXIS_ORIGIN
    global _AXIS_DIRECTION
    _TRAJECTORIES = build_trajectories()
    _AXIS_ORIGIN, _AXIS_DIRECTION = beam_axis()


def score_grid_latcut(dan_delay0_us, wen_delay0_us):
    total_km = 0.0
    all_alt_km = []
    n_kept = 0
    n_rejected = 0
    for traj in _TRAJECTORIES:
        points = solve_trajectory_points(traj, dan_delay0_us, wen_delay0_us)
        llh = np.asarray([jcoord.ecef2geodetic(p[0], p[1], p[2]) for p in points], dtype=np.float64)
        alt_km = llh[:, 2] / 1e3
        keep = np.isfinite(llh[:, 0]) & np.isfinite(llh[:, 1]) & np.isfinite(alt_km) & (llh[:, 0] <= MAX_LAT_DEG)
        n_rejected += int((~keep).sum())
        if not np.any(keep):
            continue
        kept_points = points[keep]
        rel = kept_points - _AXIS_ORIGIN
        distances_m = np.linalg.norm(np.cross(rel, _AXIS_DIRECTION), axis=1)
        total_km += float(np.sum(distances_m) / 1e3)
        all_alt_km.extend(alt_km[keep].tolist())
        n_kept += int(keep.sum())

    all_alt_km = np.asarray(all_alt_km, dtype=np.float64)
    if all_alt_km.size == 0:
        return np.inf, np.nan, np.nan, 0, n_rejected
    return total_km, float(np.mean(all_alt_km)), float(np.median(all_alt_km)), n_kept, n_rejected


def evaluate_delay_pair(job):
    i, j, dan_delay0_us, wen_delay0_us = job
    score_km, mean_alt_km, median_alt_km, n_kept, n_rejected = score_grid_latcut(
        float(dan_delay0_us),
        float(wen_delay0_us),
    )
    return (
        i,
        j,
        float(dan_delay0_us),
        float(wen_delay0_us),
        score_km,
        mean_alt_km,
        median_alt_km,
        n_kept,
        n_rejected,
    )


def main():
    trajectories = build_trajectories()
    dan_grid = np.arange(DAN_CENTER_US - DELTA_US, DAN_CENTER_US + DELTA_US + 0.5 * STEP_US, STEP_US)
    wen_grid = np.arange(WEN_CENTER_US - DELTA_US, WEN_CENTER_US + DELTA_US + 0.5 * STEP_US, STEP_US)

    score_grid_km = np.zeros((len(dan_grid), len(wen_grid)), dtype=np.float64)
    mean_alt_grid_km = np.zeros_like(score_grid_km)
    median_alt_grid_km = np.zeros_like(score_grid_km)
    kept_grid = np.zeros_like(score_grid_km, dtype=np.int32)
    rejected_grid = np.zeros_like(score_grid_km, dtype=np.int32)

    best = None
    completed_by_row = np.zeros(len(dan_grid), dtype=np.int32)
    jobs = [
        (i, j, float(dan_delay0_us), float(wen_delay0_us))
        for i, dan_delay0_us in enumerate(dan_grid)
        for j, wen_delay0_us in enumerate(wen_grid)
    ]

    print(
        f"Evaluating {len(jobs)} delay pairs with {N_WORKERS} workers "
        f"for {len(trajectories)} trajectories, keeping lat <= {MAX_LAT_DEG:.3f} deg.",
        flush=True,
    )
    with Pool(processes=N_WORKERS, initializer=init_worker) as pool:
        for result in pool.imap_unordered(evaluate_delay_pair, jobs):
            i, j, dan_delay0_us, wen_delay0_us, score_km, mean_alt_km, median_alt_km, n_kept, n_rejected = result
            score_grid_km[i, j] = score_km
            mean_alt_grid_km[i, j] = mean_alt_km
            median_alt_grid_km[i, j] = median_alt_km
            kept_grid[i, j] = n_kept
            rejected_grid[i, j] = n_rejected
            completed_by_row[i] += 1
            if best is None or score_km < best["score_km"]:
                best = {
                    "dan_delay0_us": float(dan_delay0_us),
                    "wen_delay0_us": float(wen_delay0_us),
                    "score_km": float(score_km),
                    "mean_alt_km": float(mean_alt_km),
                    "median_alt_km": float(median_alt_km),
                    "n_kept": int(n_kept),
                    "n_rejected": int(n_rejected),
                }
                print(
                    f"New best: D_dan={best['dan_delay0_us']:.3f} us "
                    f"D_wen={best['wen_delay0_us']:.3f} us "
                    f"score={best['score_km']:.3f} km "
                    f"kept={best['n_kept']} rejected={best['n_rejected']}",
                    flush=True,
                )
            if completed_by_row[i] == len(wen_grid):
                print(
                    f"Completed D_dan row {i + 1:02d}/{len(dan_grid)}: {dan_grid[i]:.3f} us",
                    flush=True,
                )

    with h5py.File(OUTPUT_H5, "w") as h:
        h["dan_delay_grid_us"] = dan_grid
        h["wen_delay_grid_us"] = wen_grid
        h["score_grid_km"] = score_grid_km
        h["mean_alt_grid_km"] = mean_alt_grid_km
        h["median_alt_grid_km"] = median_alt_grid_km
        h["kept_grid"] = kept_grid
        h["rejected_grid"] = rejected_grid
        h.attrs["center_dan_delay_us"] = DAN_CENTER_US
        h.attrs["center_wen_delay_us"] = WEN_CENTER_US
        h.attrs["delta_us"] = DELTA_US
        h.attrs["step_us"] = STEP_US
        h.attrs["max_lat_deg"] = MAX_LAT_DEG
        h.attrs["metric"] = "sum of point distances to Sanya transmit beam axis after latitude outlier cut"
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["best_dan_delay_us"] = best["dan_delay0_us"]
        h.attrs["best_wen_delay_us"] = best["wen_delay0_us"]
        h.attrs["best_score_km"] = best["score_km"]
        h.attrs["best_mean_alt_km"] = best["mean_alt_km"]
        h.attrs["best_median_alt_km"] = best["median_alt_km"]
        h.attrs["best_n_kept"] = best["n_kept"]
        h.attrs["best_n_rejected"] = best["n_rejected"]

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(wen_grid, dan_grid, score_grid_km, shading="auto")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(f"Sum of distances to beam axis (km), lat <= {MAX_LAT_DEG:.1f} deg")
    ax.scatter([best["wen_delay0_us"]], [best["dan_delay0_us"]], c="red", marker="x", s=100)
    ax.set_xlabel("Wenchang first-sample delay (us)")
    ax.set_ylabel("Danzhou first-sample delay (us)")
    ax.set_title("Sample-step Beam-Axis Delay Search with Latitude Cut")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220)
    plt.close(fig)

    print("")
    print(f"Trajectories used: {len(trajectories)}")
    print(f"Latitude cut: lat <= {MAX_LAT_DEG:.3f} deg")
    print(f"Best Danzhou first-sample delay: {best['dan_delay0_us']:.3f} us")
    print(f"Best Wenchang first-sample delay: {best['wen_delay0_us']:.3f} us")
    print(f"Best beam-axis score: {best['score_km']:.3f} km")
    print(f"Best mean altitude: {best['mean_alt_km']:.3f} km")
    print(f"Best median altitude: {best['median_alt_km']:.3f} km")
    print(f"Best kept/rejected points: {best['n_kept']}/{best['n_rejected']}")
    print(OUTPUT_H5)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
