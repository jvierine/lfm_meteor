#!/usr/bin/env python3
"""Plot Sanya satellite candidate detection groups from the TLE correlation run."""

from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        default="results/satellite_correlation/v20260613c_full",
        help="Directory containing the satellite correlation CSV files.",
    )
    p.add_argument(
        "--output-dir",
        default="/Users/jvi019/src/sanya_tristatic_paper/figures/satellite_candidates",
    )
    p.add_argument("--min-pulses", type=int, default=10)
    p.add_argument("--max-groups", type=int, default=12)
    p.add_argument("--max-abs-offset-km", type=float, default=50.0)
    return p.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def seconds_from_start(utc_series: pd.Series) -> np.ndarray:
    t = pd.to_datetime(utc_series, utc=True)
    return (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)


def plot_group(raw: pd.DataFrame, group: pd.Series, out_png: str) -> None:
    rows = raw[
        (raw["event_id"] == group["event_id"])
        & (raw["sat_id"].astype(str) == str(group["sat_id"]))
        & (raw["alias_n"].astype(int) == int(group["alias_n"]))
    ].copy()
    rows = rows.sort_values("time_utc")
    if rows.empty:
        return

    t_rel = seconds_from_start(rows["time_utc"])
    beam = rows["beam_angle_deg"].to_numpy(dtype=float)
    snr = rows["snr_db"].to_numpy(dtype=float)
    obs = rows["observed_range_km"].to_numpy(dtype=float)
    pred = rows["predicted_aliased_range_km"].to_numpy(dtype=float)
    offset = rows["range_offset_km"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)

    sc = axes[0].scatter(beam, snr, c=t_rel, s=24, cmap="viridis", edgecolor="none")
    axes[0].set_xlabel("Beam off-axis angle (deg)")
    axes[0].set_ylabel("Peak SNR (dB)")
    axes[0].grid(True, alpha=0.25)
    cb = fig.colorbar(sc, ax=axes[0])
    cb.set_label("Time since first pulse (s)")

    axes[1].plot(t_rel, obs, "o", ms=3.2, label="Observed Sanya range")
    axes[1].plot(t_rel, pred, "-", lw=1.8, label="TLE predicted aliased range")
    axes[1].plot(t_rel, offset, ".", ms=2.4, alpha=0.45, label="Offset")
    axes[1].set_xlabel("Time since first pulse (s)")
    axes[1].set_ylabel("Range or offset (km)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    title = (
        f"{group['event_id']}  NORAD {group['sat_id']}  alias {int(group['alias_n'])}\n"
        f"n={int(group['n_pulses'])}, median offset={float(group['median_range_offset_km']):.2f} km, "
        f"median beam={float(group['median_beam_angle_deg']):.2f} deg"
    )
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    grouped_path = os.path.join(args.results_dir, "sanya_satellite_detection_matches_grouped.csv")
    raw_path = os.path.join(args.results_dir, "sanya_satellite_detection_matches_raw.csv")
    grouped = pd.read_csv(grouped_path)
    raw = pd.read_csv(raw_path)

    grouped = grouped[grouped["n_pulses"] >= args.min_pulses].copy()
    grouped["abs_median_offset_km"] = grouped["median_range_offset_km"].abs()
    grouped = grouped[grouped["abs_median_offset_km"] <= args.max_abs_offset_km]
    grouped = grouped.sort_values(["abs_median_offset_km", "n_pulses"], ascending=[True, False])
    selected = grouped.head(args.max_groups)

    manifest = []
    for _, group in selected.iterrows():
        stem = safe_name(
            f"sat_candidate_{group['event_id']}_norad{group['sat_id']}_alias{int(group['alias_n'])}"
        )
        out_png = os.path.join(args.output_dir, f"{stem}.png")
        plot_group(raw, group, out_png)
        manifest.append(
            {
                "event_id": group["event_id"],
                "sat_id": str(group["sat_id"]),
                "alias_n": int(group["alias_n"]),
                "n_pulses": int(group["n_pulses"]),
                "median_range_offset_km": float(group["median_range_offset_km"]),
                "median_beam_angle_deg": float(group["median_beam_angle_deg"]),
                "path": out_png,
            }
        )

    manifest_path = os.path.join(args.output_dir, "candidate_group_plots.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"Wrote {len(manifest)} plots")
    print(manifest_path)
    for item in manifest:
        print(
            f"{item['path']}  sat={item['sat_id']} alias={item['alias_n']} "
            f"n={item['n_pulses']} offset={item['median_range_offset_km']:.2f} km"
        )


if __name__ == "__main__":
    main()
