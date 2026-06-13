import os
import glob

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np

from grid_search_delays_beam_axis import DAN_CENTER_US, MAX_LAT_DEG, WEN_CENTER_US, build_trajectories, solve_trajectory_points
import sanya_opts as sc


OUTPUT_PNG = os.path.join("results", "meteor_height_histogram.png")
PAPER_OUTPUT_PNG = "/Users/jvi019/src/sanya_tristatic_paper/figures/meteor_height_histogram.png"
INPUT_H5 = os.path.join("results", "all_tristatic_ballistic_snr_weighted_v20260613b.h5")
BIN_SIZE_KM = 1.0
COMMON_VOLUME_ALT_KM = 94.988
MONOSTATIC_SANYA_H5 = os.path.join("results", "sanya_monostatic_ranges_v20260613b.h5")
MONOSTATIC_SANYA_PATTERN = os.path.join("results", "head_echoes", "sanya", "sanya_*.h5")
SANYA_AZ_DEG = 15.0
SANYA_EL_DEG = 75.0


def sanya_slant_ranges_to_heights_km(ranges_km, az_deg=SANYA_AZ_DEG, el_deg=SANYA_EL_DEG):
    altitudes_km = []
    for range_km in np.asarray(ranges_km, dtype=np.float64):
        llh = jcoord.az_el_r2geodetic(
            sc.lat0[0],
            sc.lon0[0],
            sc.alt0[0] * 1e3,
            float(az_deg),
            float(el_deg),
            float(range_km) * 1e3,
        )
        altitudes_km.append(float(llh[2] / 1e3))
    altitudes_km = np.asarray(altitudes_km, dtype=np.float64)
    return altitudes_km[np.isfinite(altitudes_km)]


def collect_heights():
    if os.path.exists(INPUT_H5):
        chunks = []
        with h5py.File(INPUT_H5, "r") as h:
            for event_id in h["event_id"][:]:
                name = event_id.decode("utf-8") if isinstance(event_id, bytes) else str(event_id)
                chunks.append(np.asarray(h["points"][name]["alt_km"][:], dtype=np.float64))
        if not chunks:
            raise RuntimeError(f"No fitted trajectory samples found in {INPUT_H5}")
        altitudes_km = np.concatenate(chunks)
        altitudes_km = altitudes_km[np.isfinite(altitudes_km)]
        return altitudes_km, 0

    altitudes_km = []
    n_rejected = 0
    for trajectory in build_trajectories():
        points = solve_trajectory_points(trajectory, DAN_CENTER_US, WEN_CENTER_US)
        llh = np.asarray([jcoord.ecef2geodetic(point[0], point[1], point[2]) for point in points], dtype=np.float64)
        alt_km = llh[:, 2] / 1e3
        keep = np.isfinite(llh[:, 0]) & np.isfinite(llh[:, 1]) & np.isfinite(alt_km) & (llh[:, 0] <= MAX_LAT_DEG)
        n_rejected += int((~keep).sum())
        altitudes_km.extend(alt_km[keep].tolist())
    return np.asarray(altitudes_km, dtype=np.float64), n_rejected


def collect_sanya_monostatic_heights():
    if os.path.exists(MONOSTATIC_SANYA_H5):
        with h5py.File(MONOSTATIC_SANYA_H5, "r") as h:
            # Treat Sanya monostatic detections as slant ranges and convert
            # them to altitude along the fixed Sanya transmit beam. This keeps
            # the plotted quantity height, not range, even if an old cache has
            # ambiguous naming.
            if "range_km" in h:
                return sanya_slant_ranges_to_heights_km(h["range_km"][()])
            altitudes_km = np.asarray(h["height_km"][()], dtype=np.float64)
            return altitudes_km[np.isfinite(altitudes_km)]

    altitudes_km = []
    for path in sorted(glob.glob(MONOSTATIC_SANYA_PATTERN)):
        with h5py.File(path, "r") as h:
            ranges_km = np.asarray(h["range_km"][()], dtype=np.float64)
            az_deg = float(h["az"][()])
            el_deg = float(h["el"][()])
        altitudes_km.extend(sanya_slant_ranges_to_heights_km(ranges_km, az_deg, el_deg).tolist())
    return np.asarray(altitudes_km, dtype=np.float64)


def main():
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    alt_km, n_rejected = collect_heights()
    mono_alt_km = collect_sanya_monostatic_heights()
    if alt_km.size == 0:
        raise RuntimeError("No retained meteor heights available for histogram.")

    if mono_alt_km.size > 0:
        all_alt_km = np.concatenate([alt_km, mono_alt_km])
    else:
        all_alt_km = alt_km
    bin_start = np.floor(np.nanmin(all_alt_km))
    bin_stop = np.ceil(np.nanmax(all_alt_km))
    bins = np.arange(bin_start, bin_stop + BIN_SIZE_KM, BIN_SIZE_KM)

    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    tri_counts, _, _ = ax.hist(
        alt_km,
        bins=bins,
        orientation="horizontal",
        color="#315f72",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        label=f"Tri-static ({alt_km.size})",
    )

    mono_counts = None
    if mono_alt_km.size > 0:
        ax_mono = ax.twiny()
        mono_counts, _, _ = ax_mono.hist(
            mono_alt_km,
            bins=bins,
            orientation="horizontal",
            histtype="step",
            color="#b34a2e",
            linewidth=1.8,
            label=f"Sanya monostatic ({mono_alt_km.size})",
        )
        ax_mono.set_xlabel("Sanya monostatic count")
        ax_mono.tick_params(axis="x", colors="#b34a2e")
    else:
        ax_mono = None

    ax.axhline(np.nanmedian(alt_km), color="black", linestyle="--", linewidth=1.6, label=f"Median {np.nanmedian(alt_km):.1f} km")
    ax.axhline(np.nanmean(alt_km), color="#0f8b8d", linestyle="-", linewidth=1.6, label=f"Mean {np.nanmean(alt_km):.1f} km")
    ax.axhline(
        COMMON_VOLUME_ALT_KM,
        color="#7b1fa2",
        linestyle=":",
        linewidth=2.0,
        label=f"Beam intersection {COMMON_VOLUME_ALT_KM:.1f} km",
    )
    ymax = max(float(np.nanmax(tri_counts)), 1.0)
    ax.annotate(
        "Beam-axis intersection",
        xy=(0.82 * ymax, COMMON_VOLUME_ALT_KM),
        xytext=(0.28 * ymax, COMMON_VOLUME_ALT_KM + 2.0),
        arrowprops={"arrowstyle": "->", "color": "#7b1fa2", "lw": 1.2},
        color="#7b1fa2",
        ha="left",
    )
    ax.set_xlabel("Tri-static count")
    ax.set_ylabel("Height (km)")
    ax.grid(True, axis="x", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if ax_mono is not None:
        mono_handles, mono_labels = ax_mono.get_legend_handles_labels()
        handles.extend(mono_handles)
        labels.extend(mono_labels)
    ax.legend(handles, labels, loc="upper right")
    fig.tight_layout()

    fig.savefig(OUTPUT_PNG, dpi=220)
    os.makedirs(os.path.dirname(PAPER_OUTPUT_PNG), exist_ok=True)
    fig.savefig(PAPER_OUTPUT_PNG, dpi=220)
    plt.close(fig)

    print(f"points: {alt_km.size}")
    print(f"rejected latitude outliers: {n_rejected}")
    print(f"height range: {np.nanmin(alt_km):.3f} to {np.nanmax(alt_km):.3f} km")
    print(f"height mean/median: {np.nanmean(alt_km):.3f} / {np.nanmedian(alt_km):.3f} km")
    print(f"common-volume beam intersection height: {COMMON_VOLUME_ALT_KM:.3f} km")
    print(f"sanya monostatic points: {mono_alt_km.size}")
    if mono_alt_km.size > 0:
        print(f"sanya monostatic height range: {np.nanmin(mono_alt_km):.3f} to {np.nanmax(mono_alt_km):.3f} km")
    print(f"bins: {bin_start:.0f} to {bin_stop:.0f} km in {BIN_SIZE_KM:.0f} km steps")
    print(INPUT_H5 if os.path.exists(INPUT_H5) else "legacy point solver")
    print(OUTPUT_PNG)
    print(PAPER_OUTPUT_PNG)


if __name__ == "__main__":
    main()
