#!/usr/bin/env python3
"""Plot tri-static meteor positions with Sanya/Danzhou/Wenchang beam lines."""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np

import sanya_opts as sc
from grid_search_delays_beam_axis import (
    DAN_PATTERN,
    SAN_PATTERN,
    WEN_PATTERN,
    build_trajectories,
    solve_trajectory_points,
)


DEFAULT_OUTPUT = "results/memo_meteor_positions_latlon_height_beams.png"
DEFAULT_PAPER_OUTPUT = (
    "/Users/jvi019/src/sanya_tristatic_paper/figures/"
    "memo_meteor_positions_latlon_height_beams.png"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dan-delay-us", type=float, default=359.0)
    p.add_argument("--wen-delay-us", type=float, default=360.0)
    p.add_argument("--max-lat-deg", type=float, default=18.7)
    p.add_argument("--beam-max-range-km", type=float, default=240.0)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--paper-output", default=DEFAULT_PAPER_OUTPUT)
    return p.parse_args()


def ecef_to_llh(points: np.ndarray) -> np.ndarray:
    return np.asarray([jcoord.ecef2geodetic(p[0], p[1], p[2]) for p in points], dtype=np.float64)


def median_pointing(pattern: str) -> tuple[float, float]:
    az = []
    el = []
    for path in sorted(glob.glob(pattern)):
        with h5py.File(path, "r") as h:
            az.append(float(h["az"][()]))
            el.append(float(h["el"][()]))
    if not az:
        raise FileNotFoundError(f"No event files matched {pattern!r}")
    return float(np.median(az)), float(np.median(el))


def beam_line_llh(site_idx: int, az_deg: float, el_deg: float, max_range_km: float) -> np.ndarray:
    ranges_m = np.linspace(0.0, max_range_km * 1e3, 700)
    lat0 = float(sc.lat0[site_idx])
    lon0 = float(sc.lon0[site_idx])
    alt0_m = float(sc.alt0[site_idx] * 1e3)
    return np.asarray(
        [
            jcoord.az_el_r2geodetic(lat0, lon0, alt0_m, az_deg, el_deg, range_m)
            for range_m in ranges_m
        ],
        dtype=np.float64,
    )


def solve_positions(dan_delay_us: float, wen_delay_us: float) -> tuple[np.ndarray, int]:
    trajectories = build_trajectories()
    chunks = []
    for traj in trajectories:
        points = solve_trajectory_points(traj, dan_delay_us, wen_delay_us)
        chunks.append(ecef_to_llh(points))
    if not chunks:
        raise RuntimeError("No tri-static positions were solved.")
    return np.vstack(chunks), len(trajectories)


def main() -> None:
    args = parse_args()

    llh, n_trajectories = solve_positions(args.dan_delay_us, args.wen_delay_us)
    lat_deg = llh[:, 0]
    lon_deg = llh[:, 1]
    alt_km = llh[:, 2] / 1e3
    finite = np.isfinite(lat_deg) & np.isfinite(lon_deg) & np.isfinite(alt_km)
    keep = finite & (lat_deg <= args.max_lat_deg)
    n_rejected = int(np.count_nonzero(finite & ~keep))
    lat_deg = lat_deg[keep]
    lon_deg = lon_deg[keep]
    alt_km = alt_km[keep]

    pointings = {
        "Sanya tx/rx": (0, *median_pointing(SAN_PATTERN), "black", "-"),
        "Danzhou rx": (1, *median_pointing(DAN_PATTERN), "#b34a2e", "--"),
        "Wenchang rx": (2, *median_pointing(WEN_PATTERN), "#2f7d4b", "--"),
    }

    alt_lo = float(np.nanmin(alt_km) - 7.0)
    alt_hi = float(np.nanmax(alt_km) + 7.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), sharey=True, constrained_layout=True)
    sca = axes[0].scatter(
        lat_deg,
        alt_km,
        c=alt_km,
        s=8,
        alpha=0.46,
        linewidths=0,
        cmap="viridis",
        label="Tri-static meteor positions",
    )
    axes[1].scatter(
        lon_deg,
        alt_km,
        c=alt_km,
        s=8,
        alpha=0.46,
        linewidths=0,
        cmap="viridis",
    )

    for label, (site_idx, az_deg, el_deg, color, linestyle) in pointings.items():
        line = beam_line_llh(site_idx, az_deg, el_deg, args.beam_max_range_km)
        line_alt_km = line[:, 2] / 1e3
        mask = np.isfinite(line_alt_km) & (line_alt_km >= alt_lo) & (line_alt_km <= alt_hi)
        axes[0].plot(
            line[mask, 0],
            line_alt_km[mask],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{label} ({az_deg:.1f}/{el_deg:.1f} deg)",
        )
        axes[1].plot(
            line[mask, 1],
            line_alt_km[mask],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{label} ({az_deg:.1f}/{el_deg:.1f} deg)",
        )

    axes[0].set_xlabel("Latitude (deg)")
    axes[0].set_ylabel("Height (km)")
    axes[1].set_xlabel("Longitude (deg)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_ylim(alt_lo, alt_hi)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    cb = fig.colorbar(sca, ax=axes, pad=0.01)
    cb.set_label("Height (km)")
    fig.suptitle(
        "Tri-static meteor head-echo positions and radar beam lines\n"
        f"Sanya correction {sc.SANYA_RANGE_CORRECTION_KM:+.4f} km; "
        f"Danzhou delay {args.dan_delay_us:.0f} us; Wenchang delay {args.wen_delay_us:.0f} us"
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=220)
    if args.paper_output:
        os.makedirs(os.path.dirname(args.paper_output), exist_ok=True)
        fig.savefig(args.paper_output, dpi=220)
    plt.close(fig)

    print(f"points: {lat_deg.size}")
    print(f"rejected latitude outliers: {n_rejected}")
    print(f"trajectories: {n_trajectories}")
    print(f"height range: {np.nanmin(alt_km):.3f} to {np.nanmax(alt_km):.3f} km")
    print(f"latitude range: {np.nanmin(lat_deg):.6f} to {np.nanmax(lat_deg):.6f} deg")
    print(f"longitude range: {np.nanmin(lon_deg):.6f} to {np.nanmax(lon_deg):.6f} deg")
    for label, (_site_idx, az_deg, el_deg, _color, _linestyle) in pointings.items():
        print(f"{label}: az={az_deg:.6f} deg el={el_deg:.6f} deg")
    print(args.output)
    if args.paper_output:
        print(args.paper_output)


if __name__ == "__main__":
    main()
