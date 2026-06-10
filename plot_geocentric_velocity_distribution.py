import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


INPUT_H5 = os.path.join("results", "gcrs_trajectory_fits_lfm_ambiguity_v20260610.h5")
OUTPUT_H5 = os.path.join("results", "geocentric_velocity_distribution_v20260611.h5")
OUTPUT_PNG = os.path.join("results", "geocentric_velocity_distribution.png")
PAPER_OUTPUT_PNG = "/Users/jvi019/src/sanya_tristatic_paper/figures/geocentric_velocity_distribution.png"


def load_speeds(path):
    with h5py.File(path, "r") as h:
        event_id = np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h["event_id"][:]])
        speed_km_s = h["speed_km_s"][:]
        n_points = h["n_points"][:]
        rms_total_path_residual_m = h["rms_total_path_residual_m"][:]
    keep = np.isfinite(speed_km_s)
    return event_id[keep], speed_km_s[keep], n_points[keep], rms_total_path_residual_m[keep]


def write_h5(path, event_id, speed_km_s, n_points, rms_total_path_residual_m, bins, counts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["input_h5"] = INPUT_H5
        h.attrs["speed_coordinate_frame"] = "GCRS geocentric fitted velocity magnitude"
        h.attrs["n_meteors"] = int(speed_km_s.size)
        h.attrs["median_speed_km_s"] = float(np.nanmedian(speed_km_s))
        h.attrs["mean_speed_km_s"] = float(np.nanmean(speed_km_s))
        h.attrs["p05_speed_km_s"] = float(np.nanpercentile(speed_km_s, 5.0))
        h.attrs["p95_speed_km_s"] = float(np.nanpercentile(speed_km_s, 95.0))
        h["event_id"] = event_id.astype(string_dtype)
        h["speed_km_s"] = speed_km_s
        h["n_points"] = n_points
        h["rms_total_path_residual_m"] = rms_total_path_residual_m
        h["histogram_bin_edges_km_s"] = bins
        h["histogram_counts"] = counts


def plot_distribution(speed_km_s):
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    bin_min = np.floor(np.nanmin(speed_km_s) / 2.0) * 2.0
    bin_max = np.ceil(np.nanmax(speed_km_s) / 2.0) * 2.0
    bins = np.arange(bin_min, bin_max + 2.0, 2.0)
    counts, _ = np.histogram(speed_km_s, bins=bins)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.hist(speed_km_s, bins=bins, color="#2f6f73", edgecolor="white", linewidth=0.8)
    median = np.nanmedian(speed_km_s)
    p05, p95 = np.nanpercentile(speed_km_s, [5.0, 95.0])
    ax.axvline(median, color="#c84b31", linewidth=2.0, label=f"Median {median:.1f} km s$^{{-1}}$")
    ax.axvspan(p05, p95, color="#c84b31", alpha=0.12, label=f"5-95%: {p05:.1f}-{p95:.1f} km s$^{{-1}}$")
    ax.set_xlabel("Geocentric speed (km s$^{-1}$)")
    ax.set_ylabel("Meteor count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    os.makedirs(os.path.dirname(PAPER_OUTPUT_PNG), exist_ok=True)
    fig.savefig(PAPER_OUTPUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return bins, counts


def main():
    event_id, speed_km_s, n_points, rms_total_path_residual_m = load_speeds(INPUT_H5)
    bins, counts = plot_distribution(speed_km_s)
    write_h5(OUTPUT_H5, event_id, speed_km_s, n_points, rms_total_path_residual_m, bins, counts)
    print(f"meteors: {speed_km_s.size}")
    print(f"speed km/s median/mean: {np.nanmedian(speed_km_s):.3f} / {np.nanmean(speed_km_s):.3f}")
    print(f"speed km/s 5-95%: {np.nanpercentile(speed_km_s, 5.0):.3f} to {np.nanpercentile(speed_km_s, 95.0):.3f}")
    print(f"speed km/s range: {np.nanmin(speed_km_s):.3f} to {np.nanmax(speed_km_s):.3f}")
    print(OUTPUT_H5)
    print(OUTPUT_PNG)
    print(PAPER_OUTPUT_PNG)


if __name__ == "__main__":
    main()
