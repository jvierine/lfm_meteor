#!/usr/bin/env python3
"""Plot Sanya meteor head-echo detections as a corrected range histogram."""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sanya_opts import SANYA_RANGE_CORRECTION_KM, SANYA_TLE_RANGE_OFFSET_KM


DEFAULT_INPUT_GLOB = "results/tristatic_head_echoes/sanya/sanya_*.h5"
DEFAULT_OUTPUT = "results/sanya_head_echo_range_histogram_corrected.png"
DEFAULT_PAPER_OUTPUT = (
    "/Users/jvi019/src/sanya_tristatic_paper/figures/"
    "sanya_head_echo_range_histogram_corrected.png"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--paper-output", default=DEFAULT_PAPER_OUTPUT)
    p.add_argument("--bin-size-km", type=float, default=1.0)
    p.add_argument("--snr-min-db", type=float, default=None)
    return p.parse_args()


def read_ranges(paths: list[str], snr_min_db: float | None) -> tuple[np.ndarray, np.ndarray, int]:
    ranges = []
    snrs = []
    n_files_with_data = 0
    for path in paths:
        with h5py.File(path, "r") as h:
            if "range_km" not in h:
                continue
            range_km = np.asarray(h["range_km"][()], dtype=np.float64)
            keep = np.isfinite(range_km)
            if snr_min_db is not None and "snr_peak_db" in h:
                snr_db = np.asarray(h["snr_peak_db"][()], dtype=np.float64)
                keep &= np.isfinite(snr_db) & (snr_db >= snr_min_db)
            else:
                snr_db = np.full(range_km.shape, np.nan, dtype=np.float64)
            if np.any(keep):
                ranges.append(range_km[keep])
                snrs.append(snr_db[keep])
                n_files_with_data += 1
    if not ranges:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), n_files_with_data
    return np.concatenate(ranges), np.concatenate(snrs), n_files_with_data


def save_histogram_csv(path: str, bins: np.ndarray, counts: np.ndarray) -> None:
    out = pd.DataFrame(
        {
            "range_bin_start_km": bins[:-1],
            "range_bin_stop_km": bins[1:],
            "range_bin_center_km": 0.5 * (bins[:-1] + bins[1:]),
            "count": counts.astype(int),
        }
    )
    out.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No Sanya head-echo files matched {args.input_glob!r}")

    raw_range_km, snr_db, n_files_with_data = read_ranges(paths, args.snr_min_db)
    if raw_range_km.size == 0:
        raise RuntimeError("No finite Sanya range detections found.")

    corrected_range_km = raw_range_km + SANYA_RANGE_CORRECTION_KM
    bin_start = args.bin_size_km * np.floor(np.nanmin(corrected_range_km) / args.bin_size_km)
    bin_stop = args.bin_size_km * np.ceil(np.nanmax(corrected_range_km) / args.bin_size_km)
    bins = np.arange(bin_start, bin_stop + args.bin_size_km, args.bin_size_km)
    counts, _ = np.histogram(corrected_range_km, bins=bins)

    fig, ax = plt.subplots(figsize=(5.2, 6.2), constrained_layout=True)
    ax.hist(
        corrected_range_km,
        bins=bins,
        orientation="horizontal",
        color="#315f72",
        edgecolor="white",
        linewidth=0.55,
        alpha=0.92,
    )
    median_range = float(np.nanmedian(corrected_range_km))
    mean_range = float(np.nanmean(corrected_range_km))
    ax.axhline(median_range, color="black", lw=1.5, ls="--", label=f"Median {median_range:.1f} km")
    ax.axhline(mean_range, color="#b34a2e", lw=1.5, label=f"Mean {mean_range:.1f} km")
    ax.set_xlabel("Head-echo detections")
    ax.set_ylabel("Corrected Sanya range (km)")
    title = "Sanya meteor head-echo range distribution"
    if args.snr_min_db is not None:
        title += f" (SNR >= {args.snr_min_db:g} dB)"
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="best")
    ax.text(
        0.99,
        0.97,
        f"Applied range correction: {SANYA_RANGE_CORRECTION_KM:+.4f} km\n"
        f"TLE offset: {SANYA_TLE_RANGE_OFFSET_KM:+.4f} km",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=220)
    csv_output = os.path.splitext(args.output)[0] + ".csv"
    save_histogram_csv(csv_output, bins, counts)

    if args.paper_output:
        os.makedirs(os.path.dirname(args.paper_output), exist_ok=True)
        fig.savefig(args.paper_output, dpi=220)
    plt.close(fig)

    print(f"input files matched: {len(paths)}")
    print(f"input files with detections: {n_files_with_data}")
    print(f"detections: {corrected_range_km.size}")
    print(f"range correction applied: {SANYA_RANGE_CORRECTION_KM:+.4f} km")
    print(f"corrected range min/max: {np.nanmin(corrected_range_km):.3f} / {np.nanmax(corrected_range_km):.3f} km")
    print(f"corrected range mean/median: {mean_range:.3f} / {median_range:.3f} km")
    if np.any(np.isfinite(snr_db)):
        print(f"SNR mean/median: {np.nanmean(snr_db):.3f} / {np.nanmedian(snr_db):.3f} dB")
    print(args.output)
    print(csv_output)
    if args.paper_output:
        print(args.paper_output)


if __name__ == "__main__":
    main()
