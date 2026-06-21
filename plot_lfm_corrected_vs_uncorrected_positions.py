import os

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np

from fit_gcrs_trajectories_lfm_ambiguity import (
    LINK_RX_POSITIONS_M,
    LINK_TX_POSITIONS_M,
    gcrs_state_to_itrs,
    lfm_total_path_bias_m,
    link_total_paths_and_rates_m,
    solve_position_from_total_paths_m,
)


INPUT_H5 = os.path.join("results", "gcrs_trajectory_fits_lfm_ambiguity_v20260613b.h5")
OUTPUT_PNG = os.path.join("results", "lfm_corrected_vs_uncorrected_positions_debug.png")


def ecef_to_llh(points):
    return np.asarray([jcoord.ecef2geodetic(point[0], point[1], point[2]) for point in points], dtype=np.float64)


def total_path_biases_m(positions_itrs_m, velocities_itrs_mps):
    _, path_rates_mps = link_total_paths_and_rates_m(
        positions_itrs_m,
        velocities_itrs_mps,
        LINK_TX_POSITIONS_M,
        LINK_RX_POSITIONS_M,
    )
    return lfm_total_path_bias_m(path_rates_mps)


def solve_corrected_points(measured_total_paths_m, total_path_biases_m, x0):
    corrected_total_paths_m = measured_total_paths_m - total_path_biases_m
    corrected_points = []
    guess = np.asarray(x0, dtype=np.float64)
    for total_paths_m in corrected_total_paths_m:
        point = solve_position_from_total_paths_m(total_paths_m, guess)
        corrected_points.append(point)
        guess = point
    return np.asarray(corrected_points, dtype=np.float64), corrected_total_paths_m


def load_debug_points(path):
    uncorrected_chunks = []
    corrected_chunks = []
    fitted_chunks = []
    bias_chunks = []

    with h5py.File(path, "r") as h:
        event_ids = [event_id.decode("utf-8") if isinstance(event_id, bytes) else str(event_id) for event_id in h["event_id"][:]]
        r0_by_event = dict(zip(event_ids, h["r0_gcrs_m"][:]))
        v0_by_event = dict(zip(event_ids, h["v0_gcrs_mps"][:]))

        for event_id in event_ids:
            group = h["points"][event_id]
            times_ns = group["time_ns"][:]
            t_rel_s = group["t_rel_s"][:]
            measured_total_paths_m = group["measured_total_paths_m"][:]
            uncorrected = group["prior_points_ecef_m"][:]
            fitted = group["itrs_fit_m"][:]
            fit_positions, fit_velocities = gcrs_state_to_itrs(
                r0_by_event[event_id],
                v0_by_event[event_id],
                t_rel_s,
                times_ns,
            )
            biases_m = total_path_biases_m(fit_positions, fit_velocities)
            corrected, _ = solve_corrected_points(measured_total_paths_m, biases_m, uncorrected[0])
            uncorrected_chunks.append(uncorrected)
            corrected_chunks.append(corrected)
            fitted_chunks.append(fitted)
            bias_chunks.append(biases_m)

    return {
        "uncorrected": np.vstack(uncorrected_chunks),
        "corrected": np.vstack(corrected_chunks),
        "fitted": np.vstack(fitted_chunks),
        "biases_m": np.vstack(bias_chunks),
        "n_events": len(event_ids),
    }


def plot_debug(data):
    uncorrected_llh = ecef_to_llh(data["uncorrected"])
    corrected_llh = ecef_to_llh(data["corrected"])
    fitted_llh = ecef_to_llh(data["fitted"])
    shift_m = np.linalg.norm(data["corrected"] - data["uncorrected"], axis=1)
    fit_shift_m = np.linalg.norm(data["fitted"] - data["uncorrected"], axis=1)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    ax_map, ax_lat, ax_lon, ax_hist = axes.ravel()

    ax_map.scatter(
        uncorrected_llh[:, 1],
        uncorrected_llh[:, 0],
        s=7,
        color="#7a7a7a",
        alpha=0.35,
        linewidths=0,
        label="Uncorrected path-delay solve",
    )
    ax_map.scatter(
        corrected_llh[:, 1],
        corrected_llh[:, 0],
        s=7,
        color="#d95f02",
        alpha=0.45,
        linewidths=0,
        label="Doppler-corrected path-delay solve",
    )
    step = max(1, len(shift_m) // 350)
    ax_map.quiver(
        uncorrected_llh[::step, 1],
        uncorrected_llh[::step, 0],
        corrected_llh[::step, 1] - uncorrected_llh[::step, 1],
        corrected_llh[::step, 0] - uncorrected_llh[::step, 0],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        alpha=0.25,
        width=0.002,
    )
    ax_map.set_xlabel("Longitude (deg)")
    ax_map.set_ylabel("Latitude (deg)")
    ax_map.set_title("Horizontal Position")
    ax_map.grid(True, alpha=0.25)
    ax_map.legend(loc="best")

    ax_lat.scatter(uncorrected_llh[:, 0], uncorrected_llh[:, 2] / 1e3, s=6, color="#7a7a7a", alpha=0.30, linewidths=0)
    ax_lat.scatter(corrected_llh[:, 0], corrected_llh[:, 2] / 1e3, s=6, color="#d95f02", alpha=0.40, linewidths=0)
    ax_lat.scatter(fitted_llh[:, 0], fitted_llh[:, 2] / 1e3, s=4, color="#1b9e77", alpha=0.25, linewidths=0, label="GCRS fit samples")
    ax_lat.set_xlabel("Latitude (deg)")
    ax_lat.set_ylabel("Height (km)")
    ax_lat.set_title("Latitude vs Height")
    ax_lat.grid(True, alpha=0.25)
    ax_lat.legend(loc="best")

    ax_lon.scatter(uncorrected_llh[:, 1], uncorrected_llh[:, 2] / 1e3, s=6, color="#7a7a7a", alpha=0.30, linewidths=0)
    ax_lon.scatter(corrected_llh[:, 1], corrected_llh[:, 2] / 1e3, s=6, color="#d95f02", alpha=0.40, linewidths=0)
    ax_lon.scatter(fitted_llh[:, 1], fitted_llh[:, 2] / 1e3, s=4, color="#1b9e77", alpha=0.25, linewidths=0)
    ax_lon.set_xlabel("Longitude (deg)")
    ax_lon.set_ylabel("Height (km)")
    ax_lon.set_title("Longitude vs Height")
    ax_lon.grid(True, alpha=0.25)

    ax_hist.hist(shift_m / 1e3, bins=40, color="#d95f02", alpha=0.72, label="Corrected path-delay solve")
    ax_hist.hist(fit_shift_m / 1e3, bins=40, color="#1b9e77", alpha=0.45, label="GCRS fit samples")
    ax_hist.set_xlabel("Shift from uncorrected position (km)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Position Shift")
    ax_hist.grid(True, alpha=0.25)
    ax_hist.legend(loc="best")

    fig.suptitle(
        "Debug: LFM Doppler-Corrected Path-Delay Positions vs Uncorrected Tri-static Positions\n"
        f"{len(shift_m)} points from {data['n_events']} trajectories; "
        f"median corrected shift={np.nanmedian(shift_m):.1f} m"
    )
    fig.savefig(OUTPUT_PNG, dpi=220)
    plt.close(fig)

    return {
        "n_points": int(len(shift_m)),
        "n_events": int(data["n_events"]),
        "median_shift_m": float(np.nanmedian(shift_m)),
        "p95_shift_m": float(np.nanpercentile(shift_m, 95.0)),
        "max_shift_m": float(np.nanmax(shift_m)),
        "median_fit_shift_m": float(np.nanmedian(fit_shift_m)),
        "median_total_path_biases_m": np.nanmedian(data["biases_m"], axis=0),
    }


def main():
    if not os.path.exists(INPUT_H5):
        raise FileNotFoundError(f"Missing {INPUT_H5}; run fit_gcrs_trajectories_lfm_ambiguity.py first.")
    data = load_debug_points(INPUT_H5)
    stats = plot_debug(data)
    print(f"points: {stats['n_points']}")
    print(f"trajectories: {stats['n_events']}")
    print(f"corrected-vs-uncorrected shift median/p95/max: {stats['median_shift_m']:.2f} / {stats['p95_shift_m']:.2f} / {stats['max_shift_m']:.2f} m")
    print(f"GCRS-fit-vs-uncorrected shift median: {stats['median_fit_shift_m']:.2f} m")
    print(
        "median LFM total-path biases Sanya/Danzhou/Wenchang: "
        f"{stats['median_total_path_biases_m'][0]:.2f} / "
        f"{stats['median_total_path_biases_m'][1]:.2f} / "
        f"{stats['median_total_path_biases_m'][2]:.2f} m"
    )
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
