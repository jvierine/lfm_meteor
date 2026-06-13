#!/usr/bin/env python3
"""Plot Sanya satellite beam-axis offset versus range offset."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        default="results/satellite_correlation/v20260613c_snr15_full",
        help="Directory containing the satellite correlation grouped CSV.",
    )
    p.add_argument(
        "--output",
        default="/Users/jvi019/src/sanya_tristatic_paper/figures/satellite_candidates/sanya_offset_vs_beam_16km.png",
    )
    p.add_argument("--center-offset-km", type=float, default=16.0)
    p.add_argument("--offset-half-width-km", type=float, default=2.0)
    p.add_argument("--min-pulses", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    grouped_path = os.path.join(args.results_dir, "sanya_satellite_detection_matches_grouped.csv")
    grouped = pd.read_csv(grouped_path, dtype={"sat_id": str})

    lo = args.center_offset_km - args.offset_half_width_km
    hi = args.center_offset_km + args.offset_half_width_km
    selected = grouped[
        (grouped["median_range_offset_km"] >= lo)
        & (grouped["median_range_offset_km"] <= hi)
        & (grouped["n_pulses"] >= args.min_pulses)
    ].copy()
    selected = selected.sort_values(["median_beam_angle_deg", "n_pulses"], ascending=[True, False])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    csv_output = os.path.splitext(args.output)[0] + ".csv"
    selected.to_csv(csv_output, index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    sizes = 18.0 + 12.0 * np.log10(selected["n_pulses"].to_numpy(dtype=float))
    sc = ax.scatter(
        selected["median_beam_angle_deg"],
        selected["median_range_offset_km"],
        c=selected["median_snr_db"],
        s=sizes,
        cmap="viridis",
        alpha=0.82,
        linewidths=0.25,
        edgecolors="black",
    )
    ax.axhline(args.center_offset_km, color="black", lw=1.1, ls="--", alpha=0.75)
    ax.axhspan(lo, hi, color="0.5", alpha=0.08, zorder=0)
    ax.set_xlabel("Median beam off-axis angle (deg)")
    ax.set_ylabel("Median diagnostic range offset (km)")
    ax.set_title(f"Sanya satellite associations with {lo:.0f}-{hi:.0f} km range offset")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(lo, hi)
    ax.set_xlim(left=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Median SNR (dB)")

    for _, row in selected.head(12).iterrows():
        if row["median_beam_angle_deg"] <= 2.0 or row["n_pulses"] >= 200:
            ax.annotate(
                str(row["sat_id"]),
                (row["median_beam_angle_deg"], row["median_range_offset_km"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
            )

    fig.savefig(args.output, dpi=200)
    print(f"Selected {len(selected)} grouped associations")
    print(args.output)
    print(csv_output)


if __name__ == "__main__":
    main()
