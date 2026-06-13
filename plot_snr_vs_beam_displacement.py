import glob
import os
import argparse
from dataclasses import dataclass

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

import sanya_opts as sc
from grid_search_delays_beam_axis import (
    DAN_PATTERN,
    MAX_LAT_DEG,
    SAN_PATTERN,
    WEN_PATTERN,
    beam_axis,
    delay_us_to_range_km,
    gate_to_delay_us,
    initial_guess,
    nearest_index,
    overlap_ns,
    range_gates_to_km,
    solve_position,
)


OUTPUT_PNG = os.path.join("results", "snr_vs_beam_displacement.png")
PAPER_OUTPUT_PNG = "/Users/jvi019/src/sanya_tristatic_paper/figures/memo_snr_beam_displacement.png"
SANYA_BEAM_WIDTH_DEG = 0.9
DEFAULT_DAN_DELAY_US = sc.DANZHOU_FIRST_SAMPLE_DELAY_US
DEFAULT_WEN_DELAY_US = sc.WENCHANG_FIRST_SAMPLE_DELAY_US


@dataclass
class Event:
    path: str
    site: str
    times_ns: np.ndarray
    range_gate: np.ndarray
    snr_peak_db: np.ndarray
    r0_km: float
    sr_mhz: float
    az_deg: float
    el_deg: float
    t0_ns: int
    t1_ns: int


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return value


def load_event(path):
    with h5py.File(path, "r") as h:
        times_ns = h["times_ns"][()].astype(np.int64)
        echoes = h["echoes"][()]
        if "range_gate" in h:
            range_gate = h["range_gate"][()].astype(np.int32)
        else:
            range_gate = np.argmax(np.abs(echoes), axis=1).astype(np.int32)
        if "snr_peak_db" in h:
            snr_peak_db = h["snr_peak_db"][()].astype(np.float64)
        else:
            power_db = 10.0 * np.log10(np.maximum(np.abs(echoes) ** 2.0, 1e-12))
            snr_peak_db = np.max(power_db, axis=1)
        return Event(
            path=path,
            site=str(decode_scalar(h["site"][()])).lower(),
            times_ns=times_ns,
            range_gate=range_gate,
            snr_peak_db=snr_peak_db,
            r0_km=float(h["r0"][()]),
            sr_mhz=float(h["sr_mhz"][()]) if "sr_mhz" in h else 4.0,
            az_deg=float(h["az"][()]),
            el_deg=float(h["el"][()]),
            t0_ns=int(times_ns.min()),
            t1_ns=int(times_ns.max()),
        )


def load_events(pattern):
    return [load_event(path) for path in sorted(glob.glob(pattern))]


def best_overlap(event, candidates):
    best = None
    best_overlap_ns = 0
    for candidate in candidates:
        shared = overlap_ns(event, candidate)
        if shared > best_overlap_ns:
            best_overlap_ns = shared
            best = candidate
    return best


def pair_tristatic_events(san_events, dan_events, wen_events):
    triplets = []
    for san_event in san_events:
        dan_event = best_overlap(san_event, dan_events)
        wen_event = best_overlap(san_event, wen_events)
        if dan_event is None or wen_event is None:
            continue
        if overlap_ns(san_event, dan_event) == 0 or overlap_ns(san_event, wen_event) == 0:
            continue
        triplets.append((san_event, dan_event, wen_event))
    return triplets


def match_pulses(san_event, dan_event, wen_event, tolerance_ms=7.5):
    tolerance_ns = int(tolerance_ms * 1e6)
    matches = []
    for san_idx, san_t in enumerate(san_event.times_ns):
        dan_idx = nearest_index(dan_event.times_ns, san_t)
        wen_idx = nearest_index(wen_event.times_ns, san_t)
        if dan_idx is None or wen_idx is None:
            continue
        dan_t = int(dan_event.times_ns[dan_idx])
        wen_t = int(wen_event.times_ns[wen_idx])
        if abs(dan_t - int(san_t)) > tolerance_ns:
            continue
        if abs(wen_t - int(san_t)) > tolerance_ns:
            continue
        matches.append((san_idx, dan_idx, wen_idx))
    return matches


def enu_basis(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float64)
    north = np.array(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        dtype=np.float64,
    )
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=np.float64)
    return east, north, up


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dan-delay-us", type=float, default=DEFAULT_DAN_DELAY_US)
    p.add_argument("--wen-delay-us", type=float, default=DEFAULT_WEN_DELAY_US)
    p.add_argument("--output", default=OUTPUT_PNG)
    p.add_argument("--paper-output", default=PAPER_OUTPUT_PNG)
    return p.parse_args()


def collect_points(dan_delay_us, wen_delay_us):
    axis_origin, axis_direction = beam_axis()
    east, north, _ = enu_basis(sc.lat0[0], sc.lon0[0])
    rows = []
    triplets = pair_tristatic_events(load_events(SAN_PATTERN), load_events(DAN_PATTERN), load_events(WEN_PATTERN))
    for san_event, dan_event, wen_event in triplets:
        matches = match_pulses(san_event, dan_event, wen_event)
        if len(matches) < 3:
            continue
        san_ranges = (
            range_gates_to_km(san_event.range_gate, san_event.r0_km, san_event.sr_mhz)
            + sc.SANYA_RANGE_CORRECTION_KM
        )
        dan_ranges = delay_us_to_range_km(dan_delay_us + gate_to_delay_us(dan_event.range_gate, dan_event.sr_mhz))
        wen_ranges = delay_us_to_range_km(wen_delay_us + gate_to_delay_us(wen_event.range_gate, wen_event.sr_mhz))
        x0 = initial_guess(san_event.az_deg, san_event.el_deg, float(np.median(san_ranges)))
        for san_idx, dan_idx, wen_idx in matches:
            point = solve_position(
                float(san_ranges[san_idx]),
                float(dan_ranges[dan_idx]),
                float(wen_ranges[wen_idx]),
                x0,
            )
            x0 = point
            llh = jcoord.ecef2geodetic(point[0], point[1], point[2])
            lat_deg = float(llh[0])
            lon_deg = float(llh[1])
            alt_km = float(llh[2] / 1e3)
            if not np.isfinite(lat_deg) or not np.isfinite(lon_deg) or not np.isfinite(alt_km):
                continue
            if lat_deg > MAX_LAT_DEG:
                continue
            along_m = np.dot(point - axis_origin, axis_direction)
            closest = axis_origin + along_m * axis_direction
            displacement = point - closest
            east_m = np.dot(displacement, east)
            north_m = np.dot(displacement, north)
            rows.append(
                {
                    "east_deg": float(np.rad2deg(np.arctan2(east_m, along_m))),
                    "north_deg": float(np.rad2deg(np.arctan2(north_m, along_m))),
                    "sanya_snr_db": float(san_event.snr_peak_db[san_idx]),
                    "danzhou_snr_db": float(dan_event.snr_peak_db[dan_idx]),
                    "wenchang_snr_db": float(wen_event.snr_peak_db[wen_idx]),
                }
            )
    return rows


def main():
    args = parse_args()
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    rows = collect_points(args.dan_delay_us, args.wen_delay_us)
    if not rows:
        raise RuntimeError("No matched tri-static points available for SNR displacement plot.")

    east_deg = np.asarray([row["east_deg"] for row in rows], dtype=np.float64)
    north_deg = np.asarray([row["north_deg"] for row in rows], dtype=np.float64)
    snr_by_site = {
        "Sanya": np.asarray([row["sanya_snr_db"] for row in rows], dtype=np.float64),
        "Danzhou": np.asarray([row["danzhou_snr_db"] for row in rows], dtype=np.float64),
        "Wenchang": np.asarray([row["wenchang_snr_db"] for row in rows], dtype=np.float64),
    }
    all_snr = np.concatenate(list(snr_by_site.values()))
    vmin = float(np.nanpercentile(all_snr, 1.0))
    vmax = float(np.nanpercentile(all_snr, 99.0))
    xy_lim = float(np.ceil(max(np.nanmax(np.abs(east_deg)), np.nanmax(np.abs(north_deg))) * 2.0) / 2.0)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.15), sharex=True, sharey=True, constrained_layout=True)
    scatter = None
    beam_circle_radius_deg = 0.5 * SANYA_BEAM_WIDTH_DEG
    for ax, (site, snr_db) in zip(axes, snr_by_site.items()):
        scatter = ax.scatter(
            east_deg,
            north_deg,
            c=snr_db,
            s=9,
            alpha=0.7,
            linewidths=0,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        beam_circle = plt.Circle(
            (0.0, 0.0),
            beam_circle_radius_deg,
            fill=False,
            color="white",
            linewidth=1.8,
            alpha=0.95,
        )
        ax.add_patch(beam_circle)
        ax.axvline(0.0, color="white", linewidth=1.0, alpha=0.8)
        ax.axhline(0.0, color="white", linewidth=1.0, alpha=0.8)
        ax.set_title(site)
        ax.set_xlabel("East-west angular displacement (deg)")
        ax.set_xlim(-xy_lim, xy_lim)
        ax.set_ylim(-xy_lim, xy_lim)
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.grid(True, alpha=0.22)
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("North-south angular displacement (deg)")
    axes[0].text(
        0.03,
        0.95,
        f"{SANYA_BEAM_WIDTH_DEG:.1f} deg beam width",
        color="white",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 2},
    )
    cb = fig.colorbar(scatter, ax=axes, shrink=0.92, pad=0.02)
    cb.set_label("Matched-filter peak SNR (dB)")

    fig.savefig(args.output, dpi=220)
    if args.paper_output:
        os.makedirs(os.path.dirname(args.paper_output), exist_ok=True)
        fig.savefig(args.paper_output, dpi=220)
    plt.close(fig)

    print(f"points: {len(rows)}")
    print(f"east angular displacement range: {np.nanmin(east_deg):.4f} to {np.nanmax(east_deg):.4f} deg")
    print(f"north angular displacement range: {np.nanmin(north_deg):.4f} to {np.nanmax(north_deg):.4f} deg")
    print(f"Sanya beam-width circle: {SANYA_BEAM_WIDTH_DEG:.3f} deg diameter")
    print(f"Sanya range correction: {sc.SANYA_RANGE_CORRECTION_KM:+.4f} km")
    print(f"Danzhou/Wenchang delays: {args.dan_delay_us:.3f} / {args.wen_delay_us:.3f} us")
    for site, snr_db in snr_by_site.items():
        print(f"{site} SNR median/range: {np.nanmedian(snr_db):.2f} dB / {np.nanmin(snr_db):.2f} to {np.nanmax(snr_db):.2f} dB")
    print(args.output)
    if args.paper_output:
        print(args.paper_output)


if __name__ == "__main__":
    main()
