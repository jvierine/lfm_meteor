#!/usr/bin/env python3
"""Plot Sanya-only range versus radial velocity and range-fit RMS diagnostics.

The radial velocity is estimated independently for each Sanya detection event
by fitting an unweighted polynomial to corrected Sanya monostatic range versus
UTC time and evaluating the polynomial derivative at each detection.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = "results/sanya_monostatic_ranges_v20260613b.h5"
DEFAULT_OUTPUT = "results/sanya_range_vs_radial_velocity.png"
DEFAULT_RMS_OUTPUT = "results/sanya_range_fit_rms_histogram.png"
DEFAULT_CSV = "results/sanya_range_vs_radial_velocity.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--rms-output", default=DEFAULT_RMS_OUTPUT)
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--poly-degree", type=int, default=2)
    p.add_argument("--min-points", type=int, default=5)
    p.add_argument("--min-snr-db", type=float, default=None)
    p.add_argument(
        "--max-fit-rms-m",
        type=float,
        default=100.0,
        help="Maximum per-event polynomial range-fit RMS to show/write, in meters.",
    )
    return p.parse_args()


def read_string_array(dataset) -> np.ndarray:
    values = dataset[()]
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def fit_range_rate(time_ns: np.ndarray, range_km: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    t_s = (time_ns.astype(np.float64) - float(time_ns[0])) / 1e9
    t0 = float(np.mean(t_s))
    t_fit = t_s - t0
    degree = max(1, min(int(degree), len(t_s) - 1))
    coeff = np.polyfit(t_fit, range_km.astype(np.float64), degree)
    fitted = np.polyval(coeff, t_fit)
    rate_km_s = np.polyval(np.polyder(coeff), t_fit)
    rms_m = float(np.sqrt(np.mean((range_km - fitted) ** 2.0)) * 1e3)
    return rate_km_s, rms_m


def load_points(
    path: str,
    min_points: int,
    poly_degree: int,
    min_snr_db: float | None,
    max_fit_rms_m: float | None,
) -> tuple[list[dict], list[dict], dict]:
    with h5py.File(path, "r") as h:
        event_id = read_string_array(h["event_id"])
        time_ns = h["time_ns"][()]
        range_km = h["range_km"][()]
        height_km = h["height_km"][()]
        snr_db = h["snr_peak_db"][()]
        raw_range_km = h["raw_range_km"][()]
        meta = dict(h.attrs)

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, eid in enumerate(event_id):
        if min_snr_db is not None and snr_db[idx] < min_snr_db:
            continue
        if not np.isfinite(range_km[idx]) or not np.isfinite(height_km[idx]):
            continue
        groups[eid].append(idx)

    rows: list[dict] = []
    event_rows: list[dict] = []
    for eid, indices in sorted(groups.items()):
        if len(indices) < min_points:
            continue
        order = np.argsort(time_ns[indices])
        idx = np.asarray(indices, dtype=int)[order]
        rate_km_s, fit_rms_m = fit_range_rate(time_ns[idx], range_km[idx], poly_degree)
        event_rows.append(
            {
                "event_id": eid,
                "n_event_points": int(len(idx)),
                "range_poly_fit_rms_m": fit_rms_m,
                "survives_rms_filter": bool(max_fit_rms_m is None or fit_rms_m <= max_fit_rms_m),
            }
        )
        if max_fit_rms_m is not None and fit_rms_m > max_fit_rms_m:
            continue
        for ii, rate in zip(idx, rate_km_s):
            rows.append(
                {
                    "event_id": eid,
                    "time_ns": int(time_ns[ii]),
                    "height_km": float(height_km[ii]),
                    "range_km": float(range_km[ii]),
                    "raw_range_km": float(raw_range_km[ii]),
                    "sanya_radial_velocity_km_s": float(rate),
                    "snr_peak_db": float(snr_db[ii]),
                    "n_event_points": int(len(idx)),
                    "range_poly_fit_rms_m": fit_rms_m,
                }
            )
    return rows, event_rows, meta


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(
    path: str,
    rows: list[dict],
    meta: dict,
    poly_degree: int,
    min_points: int,
    max_fit_rms_m: float | None,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    height = np.asarray([row["height_km"] for row in rows], dtype=float)
    range_km = np.asarray([row["range_km"] for row in rows], dtype=float)
    velocity = np.asarray([row["sanya_radial_velocity_km_s"] for row in rows], dtype=float)
    snr = np.asarray([row["snr_peak_db"] for row in rows], dtype=float)
    fit_rms = np.asarray([row["range_poly_fit_rms_m"] for row in rows], dtype=float)
    keep = (
        np.isfinite(height)
        & np.isfinite(range_km)
        & np.isfinite(velocity)
        & np.isfinite(snr)
        & np.isfinite(fit_rms)
        & (range_km > 40.0)
        & (range_km < 220.0)
        & (np.abs(velocity) < 100.0)
    )

    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    sc = ax.scatter(
        velocity[keep],
        range_km[keep],
        c=snr[keep],
        s=10,
        cmap="viridis",
        alpha=0.62,
        linewidths=0,
    )
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.45)
    ax.set_xlabel("Sanya radial velocity $d\\rho/dt$ (km/s)")
    ax.set_ylabel("Corrected Sanya range (km)")
    ax.set_title("Sanya monostatic detections")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Sanya peak SNR (dB)")
    ax.grid(True, alpha=0.22)
    ax.text(
        0.02,
        0.02,
        f"{int(np.count_nonzero(keep))} detections\n"
        f"degree {poly_degree}; >= {min_points} points/event\n"
        f"range-fit RMS <= {max_fit_rms_m:g} m",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.25",
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_rms_histogram(
    path: str,
    event_rows: list[dict],
    max_fit_rms_m: float | None,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    rms = np.asarray([row["range_poly_fit_rms_m"] for row in event_rows], dtype=float)
    npts = np.asarray([row["n_event_points"] for row in event_rows], dtype=int)
    keep = np.isfinite(rms)
    rms = rms[keep]
    npts = npts[keep]
    if rms.size == 0:
        raise RuntimeError("No finite event RMS values found.")

    finite_max = float(np.nanpercentile(rms, 99.0))
    hist_max = max(10.0, finite_max)
    bins = np.linspace(0.0, hist_max, 45)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.hist(rms, bins=bins, color="#496d89", alpha=0.86, edgecolor="white", linewidth=0.7)
    if max_fit_rms_m is not None:
        ax.axvline(
            max_fit_rms_m,
            color="#b34a2e",
            lw=2.0,
            label=f"RMS filter: {max_fit_rms_m:g} m",
        )
    ax.set_xlabel("Per-event polynomial range-fit RMS (m)")
    ax.set_ylabel("Number of Sanya monostatic events")
    ax.set_title("Sanya monostatic range-fit RMS distribution")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(loc="upper right")
    n_survive = int(np.count_nonzero(rms <= max_fit_rms_m)) if max_fit_rms_m is not None else int(rms.size)
    ax.text(
        0.98,
        0.95,
        f"{rms.size} fitted events\n"
        f"{n_survive} pass RMS filter\n"
        f"median RMS {np.nanmedian(rms):.1f} m",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="0.25",
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows, event_rows, meta = load_points(
        args.input,
        args.min_points,
        args.poly_degree,
        args.min_snr_db,
        args.max_fit_rms_m,
    )
    if not rows:
        raise RuntimeError("No Sanya events survived the selection.")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.rms_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    write_csv(args.csv, rows)
    make_plot(args.output, rows, meta, args.poly_degree, args.min_points, args.max_fit_rms_m)
    make_rms_histogram(args.rms_output, event_rows, args.max_fit_rms_m)
    heights = np.asarray([row["height_km"] for row in rows], dtype=float)
    rates = np.asarray([row["sanya_radial_velocity_km_s"] for row in rows], dtype=float)
    print(f"wrote {args.output}")
    print(f"wrote {args.rms_output}")
    print(f"wrote {args.csv}")
    print(f"n={len(rows)} height={np.nanmin(heights):.2f}..{np.nanmax(heights):.2f} km")
    print(f"radial velocity={np.nanmin(rates):.2f}..{np.nanmax(rates):.2f} km/s")
    print(f"fitted events={len(event_rows)}")
    print(f"max polynomial range-fit RMS={args.max_fit_rms_m} m")


if __name__ == "__main__":
    main()
