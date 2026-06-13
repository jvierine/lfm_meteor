#!/usr/bin/env python3
"""Plot the closest Sanya beam-angle satellite passes and any coincident echoes."""

from __future__ import annotations

import argparse
import os
import re

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results/satellite_correlation/v20260613c_full")
    p.add_argument(
        "--output-dir",
        default="/Users/jvi019/src/sanya_tristatic_paper/figures/satellite_low_beam_angle",
    )
    p.add_argument("--max-passes", type=int, default=24)
    p.add_argument("--snr-min-db", type=float, default=35.0)
    p.add_argument("--contact-cols", type=int, default=3)
    return p.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def rel_seconds(utc_series: pd.Series, t0) -> np.ndarray:
    t = pd.to_datetime(utc_series, utc=True)
    return (t - t0).dt.total_seconds().to_numpy(dtype=float)


def plot_pass(pass_row: pd.Series, raw: pd.DataFrame, out_png: str, snr_min_db: float) -> dict:
    sat_id = str(pass_row["sat_id"])
    t_start = pd.Timestamp(pass_row["pass_start_utc"])
    t_stop = pd.Timestamp(pass_row["pass_stop_utc"])
    t_mid = pd.Timestamp(pass_row["closest_utc"])

    sat_raw = raw[raw["sat_id"].astype(str) == sat_id].copy()
    if not sat_raw.empty:
        sat_raw["timestamp"] = pd.to_datetime(sat_raw["time_utc"], utc=True)
        rows = sat_raw[(sat_raw["timestamp"] >= t_start) & (sat_raw["timestamp"] <= t_stop)].copy()
    else:
        rows = sat_raw

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    fig.suptitle(
        f"NORAD {sat_id}: closest {pass_row['closest_utc']}, "
        f"min angle {float(pass_row['min_beam_angle_deg']):.3f} deg, "
        f"slant range {float(pass_row['slant_range_km_at_closest']):.0f} km",
        fontsize=10,
    )

    axes[0].axvline(float(pass_row["min_beam_angle_deg"]), color="0.7", lw=1, ls="--")
    axes[0].set_xlabel("Beam off-axis angle (deg)")
    axes[0].set_ylabel("Peak SNR (dB)")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel("Time from closest approach (s)")
    axes[1].set_ylabel("Range (km)")
    axes[1].grid(True, alpha=0.25)

    n_echo = len(rows)
    if n_echo:
        rows = rows.sort_values("timestamp")
        trel = (rows["timestamp"] - t_mid).dt.total_seconds().to_numpy(dtype=float)
        snr = rows["snr_db"].to_numpy(dtype=float)
        beam = rows["beam_angle_deg"].to_numpy(dtype=float)
        obs = rows["observed_range_km"].to_numpy(dtype=float)
        pred = rows["predicted_aliased_range_km"].to_numpy(dtype=float)
        alias = rows["alias_n"].to_numpy(dtype=int)

        sc0 = axes[0].scatter(beam, snr, c=trel, cmap="viridis", s=18, edgecolor="none")
        cb0 = fig.colorbar(sc0, ax=axes[0])
        cb0.set_label("Time from closest approach (s)")

        sc1 = axes[1].scatter(trel, obs, c=snr, cmap="plasma", s=18, edgecolor="none", label="Observed")
        for alias_n in sorted(set(alias)):
            idx = alias == alias_n
            order = np.argsort(trel[idx])
            axes[1].plot(trel[idx][order], pred[idx][order], lw=1.2, label=f"Predicted alias {alias_n}")
        cb1 = fig.colorbar(sc1, ax=axes[1])
        cb1.set_label("Peak SNR (dB)")
        axes[1].legend(loc="best", fontsize=7)
    else:
        message = f"No SNR >= {snr_min_db:.0f} dB Sanya detections during this in-beam interval"
        for ax in axes:
            ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
        axes[0].set_xlim(0, 10)
        axes[0].set_ylim(snr_min_db - 5, snr_min_db + 15)
        axes[1].set_xlim((t_start - t_mid).total_seconds(), (t_stop - t_mid).total_seconds())
        axes[1].set_ylim(60, 210)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return {
        "sat_id": sat_id,
        "closest_utc": pass_row["closest_utc"],
        "min_beam_angle_deg": float(pass_row["min_beam_angle_deg"]),
        "slant_range_km_at_closest": float(pass_row["slant_range_km_at_closest"]),
        "pass_start_utc": pass_row["pass_start_utc"],
        "pass_stop_utc": pass_row["pass_stop_utc"],
        "n_echo_detections": int(n_echo),
        "path": out_png,
    }


def make_contact_sheet(manifest: pd.DataFrame, output: str, cols: int) -> None:
    n = len(manifest)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 2.7 * rows), constrained_layout=True)
    axes = np.ravel(axes)
    for ax, (_, row) in zip(axes, manifest.iterrows()):
        ax.imshow(mpimg.imread(row["path"]))
        ax.set_axis_off()
        ax.set_title(
            f"{row['sat_id']}  angle={row['min_beam_angle_deg']:.3f} deg  echoes={int(row['n_echo_detections'])}",
            fontsize=8,
        )
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    passes = pd.read_csv(os.path.join(args.results_dir, "sanya_satellite_passes.csv"))
    raw = pd.read_csv(os.path.join(args.results_dir, "sanya_satellite_detection_matches_raw.csv"))

    selected = passes.sort_values("min_beam_angle_deg").head(args.max_passes)
    manifest = []
    for _, pass_row in selected.iterrows():
        stem = safe_name(
            f"low_beam_norad{pass_row['sat_id']}_{pass_row['closest_utc']}_"
            f"angle{float(pass_row['min_beam_angle_deg']):.3f}"
        )
        out_png = os.path.join(args.output_dir, f"{stem}.png")
        manifest.append(plot_pass(pass_row, raw, out_png, args.snr_min_db))

    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(args.output_dir, "low_beam_angle_pass_plots.csv")
    manifest_df.to_csv(manifest_path, index=False)
    contact = os.path.join(args.output_dir, "low_beam_angle_contact_sheet.png")
    make_contact_sheet(manifest_df, contact, args.contact_cols)
    print(f"Wrote {len(manifest_df)} low-beam-angle pass plots")
    print(contact)
    print(manifest_path)
    print(manifest_df[["sat_id", "closest_utc", "min_beam_angle_deg", "n_echo_detections", "path"]].to_string(index=False))


if __name__ == "__main__":
    main()
