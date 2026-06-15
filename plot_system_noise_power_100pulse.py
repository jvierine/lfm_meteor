#!/usr/bin/env python3
"""Plot the reduced 4 MHz system-noise-power reference product."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-h5",
        default="/Users/jvi019/src/lfm_meteor/results/sanya_4mhz_system_noise_power_100pulse.h5",
        help="Reduced 100-pulse system-noise-power HDF5 product.",
    )
    p.add_argument(
        "--output-dir",
        default="/Users/jvi019/src/sanya_tristatic_paper/memos/figures",
        help="Directory for PDF and PNG outputs.",
    )
    p.add_argument(
        "--basename",
        default="memo20_system_noise_power_100pulse",
        help="Output filename stem.",
    )
    p.add_argument(
        "--median-temperature-k",
        type=float,
        default=130.0,
        help="Temperature assigned to each station median noise power.",
    )
    p.add_argument(
        "--low-rate-median-window",
        type=int,
        default=401,
        help="Odd-numbered rolling window, in 100-pulse bins, for the low-rate median filter.",
    )
    p.add_argument(
        "--low-rate-mad-sigma",
        type=float,
        default=3.0,
        help="Low-rate outlier threshold in robust MAD sigma units.",
    )
    p.add_argument(
        "--wenchang-low-rate-mad-sigma",
        type=float,
        default=0.9,
        help="Station-specific low-rate outlier threshold for Wenchang.",
    )
    p.add_argument(
        "--wenchang-low-rate-median-window",
        type=int,
        default=4001,
        help="Station-specific rolling window, in 100-pulse bins, for Wenchang.",
    )
    p.add_argument(
        "--wenchang-min-floor-ratio",
        type=float,
        default=1.08,
        help="For Wenchang, keep only points within this ratio of the rolling-minimum low-rate floor.",
    )
    return p.parse_args()


def ns_to_matplotlib_dates(ns: np.ndarray) -> np.ndarray:
    dt = ns.astype("datetime64[ns]").astype("datetime64[ms]").astype(object)
    return mdates.date2num(dt)


def rolling_median_mad_outliers(values: np.ndarray, window: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if window < 3:
        raise ValueError("--low-rate-median-window must be at least 3")
    if window % 2 == 0:
        window += 1

    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    outlier = ~finite
    filtered = np.full(vals.shape, np.nan, dtype=np.float64)
    if np.count_nonzero(finite) < window:
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals[finite] - med))
        sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nan
        if np.isfinite(sigma) and sigma > 0:
            outlier |= np.abs(vals - med) > threshold * sigma
        filtered[finite & ~outlier] = vals[finite & ~outlier]
        return outlier, filtered

    pad = window // 2
    padded = np.pad(vals, pad, mode="edge")
    windows = sliding_window_view(padded, window)
    local_median = np.nanmedian(windows, axis=1)
    local_mad = np.nanmedian(np.abs(windows - local_median[:, None]), axis=1)
    local_sigma = 1.4826 * local_mad
    valid_sigma = np.isfinite(local_sigma) & (local_sigma > 0)
    residual = np.abs(vals - local_median)
    outlier |= valid_sigma & (residual > threshold * local_sigma)
    filtered[finite & ~outlier] = vals[finite & ~outlier]
    return outlier, filtered


def rolling_minimum_floor_outliers(values: np.ndarray, window: int, max_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if window < 3:
        raise ValueError("--wenchang-low-rate-median-window must be at least 3")
    if window % 2 == 0:
        window += 1
    if max_ratio <= 1.0:
        raise ValueError("--wenchang-min-floor-ratio must be larger than 1")

    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    outlier = ~finite
    filtered = np.full(vals.shape, np.nan, dtype=np.float64)
    if np.count_nonzero(finite) < window:
        floor = np.nanmin(vals)
        outlier |= vals > floor + np.log(max_ratio)
        filtered[finite & ~outlier] = vals[finite & ~outlier]
        return outlier, filtered

    pad = window // 2
    padded = np.pad(vals, pad, mode="edge")
    windows = sliding_window_view(padded, window)
    local_floor = np.nanmin(windows, axis=1)
    outlier |= vals > local_floor + np.log(max_ratio)
    filtered[finite & ~outlier] = vals[finite & ~outlier]
    return outlier, filtered


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input_h5, "r") as h:
        names = [x.decode() if hasattr(x, "decode") else str(x) for x in h["site_names"][:]]
        station_id = h["bins/station_id"][:]
        time_utc_ns = h["bins/time_utc_mid_ns"][:]
        power = h["bins/noise_power_mean_raw_voltage"][:].astype(np.float64)
        n_rejected = h["bins/n_rejected"][:]
        n_total = h["bins/n_total"][:]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 300,
        }
    )
    colors = {
        "Sanya": "#1f77b4",
        "Danzhou": "#2ca02c",
        "Wenchang": "#d62728",
    }

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, constrained_layout=True)
    legend_handles = []

    for sid, name in enumerate(names):
        m = station_id == sid
        if not np.any(m):
            continue
        median = np.nanmedian(power[m])
        temperature_k = args.median_temperature_k * power[m] / median
        if name == "Wenchang":
            outlier, _ = rolling_minimum_floor_outliers(
                np.log(temperature_k),
                args.wenchang_low_rate_median_window,
                args.wenchang_min_floor_ratio,
            )
        else:
            outlier, _ = rolling_median_mad_outliers(
                np.log(temperature_k),
                args.low_rate_median_window,
                args.low_rate_mad_sigma,
            )
        clean = ~outlier
        x = ns_to_matplotlib_dates(time_utc_ns[m])
        rejected_fraction = n_rejected[m] / np.maximum(n_total[m], 1)
        color = colors.get(name, f"C{sid}")
        axes[0].scatter(x[outlier], temperature_k[outlier], s=2.0, alpha=0.035, linewidths=0, color=color, rasterized=True)
        axes[0].scatter(x[clean], temperature_k[clean], s=3.0, alpha=0.70, linewidths=0, color=color, rasterized=True)
        axes[1].scatter(x[outlier], rejected_fraction[outlier], s=2.0, alpha=0.035, linewidths=0, color=color, rasterized=True)
        axes[1].scatter(x[clean], rejected_fraction[clean], s=3.0, alpha=0.70, linewidths=0, color=color, rasterized=True)
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="None", color=color, markerfacecolor=color, markeredgewidth=0, markersize=6, label=name)
        )

    axes[0].axhline(args.median_temperature_k, color="0.25", lw=0.8, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$T_{\mathrm{sys}}$ (K)")
    axes[0].set_title(f"4 MHz system noise, station medians set to {args.median_temperature_k:g} K")
    axes[0].legend(handles=legend_handles, loc="upper left", ncol=3, frameon=False)

    axes[1].set_ylabel("Fraction of rejected pulses")
    axes[1].set_xlabel("UTC time")
    axes[1].set_ylim(bottom=-0.01)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(formatter)

    for ax in axes:
        ax.grid(True, color="0.88", lw=0.6)
        ax.set_axisbelow(True)

    pdf = outdir / f"{args.basename}.pdf"
    png = outdir / f"{args.basename}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
