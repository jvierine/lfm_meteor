import os
from multiprocessing import Pool, cpu_count

import h5py
import matplotlib.pyplot as plt
import numpy as np

from grid_search_delays_beam_axis import (
    DAN_CENTER_US,
    WEN_CENTER_US,
    beam_axis,
    build_trajectories,
    score_grid,
)


DELTA_US = 20.0
STEP_US = 2.0
OUTPUT_H5 = os.path.join("results", "delay_grid_search_beam_axis_fine.h5")
OUTPUT_PNG = os.path.join("results", "delay_grid_search_beam_axis_fine.png")
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


def evaluate_delay_pair(job):
    i, j, dan_delay0_us, wen_delay0_us = job
    score_km, mean_alt_km, median_alt_km = score_grid(
        _TRAJECTORIES,
        _AXIS_ORIGIN,
        _AXIS_DIRECTION,
        float(dan_delay0_us),
        float(wen_delay0_us),
    )
    return i, j, float(dan_delay0_us), float(wen_delay0_us), score_km, mean_alt_km, median_alt_km


def main():
    trajectories = build_trajectories()
    dan_grid = np.arange(DAN_CENTER_US - DELTA_US, DAN_CENTER_US + DELTA_US + 0.5 * STEP_US, STEP_US)
    wen_grid = np.arange(WEN_CENTER_US - DELTA_US, WEN_CENTER_US + DELTA_US + 0.5 * STEP_US, STEP_US)

    score_grid_km = np.zeros((len(dan_grid), len(wen_grid)), dtype=np.float64)
    mean_alt_grid_km = np.zeros_like(score_grid_km)
    median_alt_grid_km = np.zeros_like(score_grid_km)

    best = None
    completed_by_row = np.zeros(len(dan_grid), dtype=np.int32)
    jobs = [
        (i, j, float(dan_delay0_us), float(wen_delay0_us))
        for i, dan_delay0_us in enumerate(dan_grid)
        for j, wen_delay0_us in enumerate(wen_grid)
    ]
    print(
        f"Evaluating {len(jobs)} delay pairs with {N_WORKERS} workers "
        f"for {len(trajectories)} trajectories.",
        flush=True,
    )
    with Pool(processes=N_WORKERS, initializer=init_worker) as pool:
        for result in pool.imap_unordered(evaluate_delay_pair, jobs):
            i, j, dan_delay0_us, wen_delay0_us, score_km, mean_alt_km, median_alt_km = result
            score_grid_km[i, j] = score_km
            mean_alt_grid_km[i, j] = mean_alt_km
            median_alt_grid_km[i, j] = median_alt_km
            completed_by_row[i] += 1
            if best is None or score_km < best["score_km"]:
                best = {
                    "dan_delay0_us": float(dan_delay0_us),
                    "wen_delay0_us": float(wen_delay0_us),
                    "score_km": float(score_km),
                    "mean_alt_km": float(mean_alt_km),
                    "median_alt_km": float(median_alt_km),
                }
                print(
                    f"New best: D_dan={best['dan_delay0_us']:.3f} us "
                    f"D_wen={best['wen_delay0_us']:.3f} us "
                    f"score={best['score_km']:.3f} km "
                    f"mean_alt={best['mean_alt_km']:.3f} km",
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
        h.attrs["center_dan_delay_us"] = DAN_CENTER_US
        h.attrs["center_wen_delay_us"] = WEN_CENTER_US
        h.attrs["delta_us"] = DELTA_US
        h.attrs["step_us"] = STEP_US
        h.attrs["metric"] = "sum of point distances to Sanya transmit beam axis"
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["best_dan_delay_us"] = best["dan_delay0_us"]
        h.attrs["best_wen_delay_us"] = best["wen_delay0_us"]
        h.attrs["best_score_km"] = best["score_km"]
        h.attrs["best_mean_alt_km"] = best["mean_alt_km"]
        h.attrs["best_median_alt_km"] = best["median_alt_km"]

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(wen_grid, dan_grid, score_grid_km, shading="auto")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Sum of distances to beam axis (km)")
    ax.scatter([best["wen_delay0_us"]], [best["dan_delay0_us"]], c="red", marker="x", s=100)
    ax.set_xlabel("Wenchang first-sample delay (us)")
    ax.set_ylabel("Danzhou first-sample delay (us)")
    ax.set_title("Fine Tri-static Beam-Axis Delay Search")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220)
    plt.close(fig)

    print("")
    print(f"Trajectories used: {len(trajectories)}")
    print(f"Best Danzhou first-sample delay: {best['dan_delay0_us']:.3f} us")
    print(f"Best Wenchang first-sample delay: {best['wen_delay0_us']:.3f} us")
    print(f"Best beam-axis score: {best['score_km']:.3f} km")
    print(f"Best mean altitude: {best['mean_alt_km']:.3f} km")
    print(f"Best median altitude: {best['median_alt_km']:.3f} km")
    print(OUTPUT_H5)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
