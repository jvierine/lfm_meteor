#!/usr/bin/env python3
"""Make a contact sheet for Sanya satellite candidate plots."""

from __future__ import annotations

import argparse
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plot-dir",
        default="/Users/jvi019/src/sanya_tristatic_paper/figures/satellite_candidates",
    )
    p.add_argument("--output", default="candidate_group_contact_sheet.png")
    p.add_argument("--cols", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(os.path.join(args.plot_dir, "candidate_group_plots.csv"))
    n = len(manifest)
    rows = (n + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(12, 3.7 * rows), constrained_layout=True)
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, manifest.iterrows()):
        image = mpimg.imread(row["path"])
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(
            f"NORAD {row['sat_id']} alias {int(row['alias_n'])}, "
            f"n={int(row['n_pulses'])}, offset={row['median_range_offset_km']:.2f} km",
            fontsize=9,
        )
    for ax in axes[n:]:
        ax.set_axis_off()
    output = os.path.join(args.plot_dir, args.output)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
