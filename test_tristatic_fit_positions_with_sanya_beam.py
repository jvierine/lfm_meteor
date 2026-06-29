#!/usr/bin/env python3
"""Test whether fitted tri-static positions are consistent with the Sanya TX beam."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np

import fit_gcrs_trajectories_lfm_ambiguity as gfit
import plot_memo09_antenna_gain_patterns as gain_model
import plot_sanya_beam_position_histogram as beam_hist


DEFAULT_CATALOG_DIR = Path("results/tristatic_calibrated_chirp_v20260624a_first20")
DEFAULT_OUTPUT_BASE = Path("results/sanya_beam_fit_test_v20260624a")
DEFAULT_MIN_EVENTS = 5
DEFAULT_MIN_POSITIONS = 100
DEFAULT_MIN_FRACTION_ABOVE_MINUS_13DB = 0.65
DEFAULT_MAX_MEDIAN_ABS_OFFSET_DEG = 1.0
DEFAULT_MIN_MEDIAN_GAIN_DB = -8.0
DEFAULT_POSITION_FILTER = "sanya_path_keep"


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("zero-length vector")
    return vector / norm


def load_catalog_positions(catalog_dir: Path, position_filter: str = DEFAULT_POSITION_FILTER):
    paths = sorted(glob.glob(str(catalog_dir / "joint_delay_doppler_fft_tri_*.h5")))
    positions = []
    event_ids = []
    event_position_counts = []
    for path in paths:
        with h5py.File(path, "r") as h:
            if "joint_fit/x_itrs_m" not in h:
                continue
            x = np.asarray(h["joint_fit/x_itrs_m"][:], dtype=np.float64)
            if "joint_fit/keep_rows" in h:
                keep = np.asarray(h["joint_fit/keep_rows"][:], dtype=bool)
            else:
                keep = np.ones(len(x), dtype=bool)
            if position_filter == "sanya_path_keep" and "joint_fit/path_keep" in h:
                keep &= np.asarray(h["joint_fit/path_keep"][:, 0], dtype=bool)
            elif position_filter == "any_path_keep" and "joint_fit/path_keep" in h:
                keep &= np.any(np.asarray(h["joint_fit/path_keep"][:], dtype=bool), axis=1)
            elif position_filter != "all_rows":
                raise ValueError(f"unknown position_filter={position_filter!r}")
            finite = keep & np.all(np.isfinite(x), axis=1)
            if not np.any(finite):
                continue
            positions.append(x[finite])
            event_ids.append(decode(h.attrs.get("event_id", Path(path).stem.replace("joint_delay_doppler_fft_", ""))))
            event_position_counts.append(int(np.count_nonzero(finite)))
    if not positions:
        raise RuntimeError(f"No fitted positions found in {catalog_dir}")
    return np.vstack(positions), np.asarray(event_ids, dtype=object), np.asarray(event_position_counts, dtype=np.int64)


def sanya_beam_coordinates(positions_ecef_m):
    tx = np.asarray(gfit.LINK_TX_POSITIONS_M[0], dtype=np.float64)
    lat_deg, lon_deg, _alt_m = jcoord.ecef2geodetic(*tx)
    los_ecef = unit(np.asarray(positions_ecef_m, dtype=np.float64) - tx[None, :])
    los_enu = beam_hist.ecef_to_enu_vectors(los_ecef, lat_deg, lon_deg)

    site = gain_model.SITES[0]
    pointing_enu = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    east_axis, north_axis = beam_hist.local_sky_axes(pointing_enu)
    east_deg, north_deg = beam_hist.angular_offsets_deg(los_enu, pointing_enu, east_axis, north_axis)
    gain_db = beam_hist.sanya_beam_relative_gain_db(east_deg, north_deg)
    return east_deg, north_deg, gain_db


def summarize(east_deg, north_deg, gain_db, n_events, n_positions):
    radial_offset_deg = np.hypot(east_deg, north_deg)
    return {
        "n_events": int(n_events),
        "n_positions": int(n_positions),
        "median_east_offset_deg": float(np.nanmedian(east_deg)),
        "median_north_offset_deg": float(np.nanmedian(north_deg)),
        "median_abs_offset_deg": float(np.nanmedian(radial_offset_deg)),
        "p90_abs_offset_deg": float(np.nanpercentile(radial_offset_deg, 90.0)),
        "median_gain_db": float(np.nanmedian(gain_db)),
        "p10_gain_db": float(np.nanpercentile(gain_db, 10.0)),
        "fraction_above_minus_3db": float(np.nanmean(gain_db >= -3.0)),
        "fraction_above_minus_10db": float(np.nanmean(gain_db >= -10.0)),
        "fraction_above_minus_13db": float(np.nanmean(gain_db >= -13.0)),
        "fraction_above_minus_20db": float(np.nanmean(gain_db >= -20.0)),
    }


def evaluate(summary, args):
    failures = []
    if summary["n_events"] < args.min_events:
        failures.append(f"n_events {summary['n_events']} < {args.min_events}")
    if summary["n_positions"] < args.min_positions:
        failures.append(f"n_positions {summary['n_positions']} < {args.min_positions}")
    if summary["fraction_above_minus_13db"] < args.min_fraction_above_minus_13db:
        failures.append(
            "fraction_above_minus_13db "
            f"{summary['fraction_above_minus_13db']:.3f} < {args.min_fraction_above_minus_13db:.3f}"
        )
    if summary["median_abs_offset_deg"] > args.max_median_abs_offset_deg:
        failures.append(
            f"median_abs_offset_deg {summary['median_abs_offset_deg']:.3f} > {args.max_median_abs_offset_deg:.3f}"
        )
    if summary["median_gain_db"] < args.min_median_gain_db:
        failures.append(f"median_gain_db {summary['median_gain_db']:.3f} < {args.min_median_gain_db:.3f}")
    return failures


def write_h5(output_base, catalog_dir, event_ids, event_counts, east_deg, north_deg, gain_db, summary, failures, args):
    os.makedirs(output_base.parent, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(output_base.with_suffix(".h5"), "w") as h:
        h.attrs["script"] = Path(__file__).name
        h.attrs["catalog_dir"] = str(catalog_dir)
        h.attrs["position_filter"] = str(args.position_filter)
        h.attrs["test_passed"] = len(failures) == 0
        h.attrs["failure_reasons"] = "; ".join(failures)
        h.attrs["min_events"] = int(args.min_events)
        h.attrs["min_positions"] = int(args.min_positions)
        h.attrs["min_fraction_above_minus_13db"] = float(args.min_fraction_above_minus_13db)
        h.attrs["max_median_abs_offset_deg"] = float(args.max_median_abs_offset_deg)
        h.attrs["min_median_gain_db"] = float(args.min_median_gain_db)
        for key, value in summary.items():
            h.attrs[key] = value
        h.create_dataset("event_id", data=np.asarray(event_ids, dtype=object), dtype=string_dtype)
        h["event_position_count"] = event_counts
        h["east_offset_deg"] = np.asarray(east_deg, dtype=np.float64)
        h["north_offset_deg"] = np.asarray(north_deg, dtype=np.float64)
        h["relative_gain_db"] = np.asarray(gain_db, dtype=np.float64)


def plot_result(output_base, east_deg, north_deg, gain_db, summary, failures):
    fig, ax = plt.subplots(figsize=(5.8, 5.1), constrained_layout=True)
    sc = ax.scatter(east_deg, north_deg, c=gain_db, s=9, cmap="viridis", vmin=-25, vmax=0, alpha=0.75, edgecolors="none")
    grid = np.linspace(-2.6, 2.6, 260)
    east_grid, north_grid = np.meshgrid(grid, grid)
    gain_grid = beam_hist.sanya_beam_relative_gain_db(east_grid, north_grid)
    ax.contour(east_grid, north_grid, gain_grid, levels=[-20.0, -13.0, -10.0, -3.0], colors="0.15", linewidths=0.8)
    ax.axhline(0.0, color="0.3", lw=0.7, alpha=0.6)
    ax.axvline(0.0, color="0.3", lw=0.7, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("Sanya beam east offset (deg)")
    ax.set_ylabel("Sanya beam north offset (deg)")
    ax.set_title("Tri-static fitted positions vs. Sanya TX beam")
    status = "PASS" if not failures else "FAIL"
    ax.text(
        0.02,
        0.02,
        (
            f"{status}\n"
            f"N={summary['n_positions']} positions, {summary['n_events']} events\n"
            f"median gain={summary['median_gain_db']:.1f} dB\n"
            f"frac > -13 dB={summary['fraction_above_minus_13db']:.2f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.035)
    cb.set_label("Sanya relative gain (dB)")
    os.makedirs(output_base.parent, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=240)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--min-events", type=int, default=DEFAULT_MIN_EVENTS)
    parser.add_argument("--min-positions", type=int, default=DEFAULT_MIN_POSITIONS)
    parser.add_argument("--min-fraction-above-minus-13db", type=float, default=DEFAULT_MIN_FRACTION_ABOVE_MINUS_13DB)
    parser.add_argument("--max-median-abs-offset-deg", type=float, default=DEFAULT_MAX_MEDIAN_ABS_OFFSET_DEG)
    parser.add_argument("--min-median-gain-db", type=float, default=DEFAULT_MIN_MEDIAN_GAIN_DB)
    parser.add_argument(
        "--position-filter",
        choices=("sanya_path_keep", "any_path_keep", "all_rows"),
        default=DEFAULT_POSITION_FILTER,
        help="Which fitted trajectory samples to include in the beam-consistency statistic.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    positions, event_ids, event_counts = load_catalog_positions(args.catalog_dir, position_filter=args.position_filter)
    east_deg, north_deg, gain_db = sanya_beam_coordinates(positions)
    finite = np.isfinite(east_deg) & np.isfinite(north_deg) & np.isfinite(gain_db)
    east_deg = east_deg[finite]
    north_deg = north_deg[finite]
    gain_db = gain_db[finite]
    summary = summarize(east_deg, north_deg, gain_db, len(event_ids), len(gain_db))
    failures = evaluate(summary, args)
    write_h5(args.output_base, args.catalog_dir, event_ids, event_counts, east_deg, north_deg, gain_db, summary, failures, args)
    plot_result(args.output_base, east_deg, north_deg, gain_db, summary, failures)

    print(f"catalog_dir={args.catalog_dir}")
    print(f"position_filter={args.position_filter}")
    for key, value in summary.items():
        print(f"{key}={value}")
    print(f"output_h5={args.output_base.with_suffix('.h5')}")
    print(f"output_png={args.output_base.with_suffix('.png')}")
    if failures:
        print("test_passed=False")
        for failure in failures:
            print(f"failure={failure}")
        raise SystemExit(1)
    print("test_passed=True")


if __name__ == "__main__":
    main()
