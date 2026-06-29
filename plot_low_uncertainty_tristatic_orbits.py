"""Plot low-uncertainty Sanya tri-static heliocentric orbit ensemble."""

from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

import estimate_orbit_element_uncertainty as orbit_unc


INPUT_H5 = "results/orbit_element_uncertainty_snr_scatter_20samp.h5"
OUTPUT_PDF = "/Users/jvi019/src/sanya_tristatic_paper/figures/orbit_xy_low_uncertainty_tristatic_events.pdf"


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def load_selected(args):
    rows = []
    with h5py.File(args.input, "r") as h:
        event_ids = [decode(x) for x in h["event_id"][:]]
        e_std = h["e_std"][:]
        inc_std = h["inc_deg_std"][:]
        reached = h["above_atmosphere_reached_fraction"][:]
        snr = h["median_sanya_snr_db"][:]
        n_points = h["n_points"][:]
        q_std = np.full(len(event_ids), np.nan, dtype=np.float64)
        inc_mean = np.full(len(event_ids), np.nan, dtype=np.float64)
        for idx, event_id in enumerate(event_ids):
            kepler = h["kepler_samples"][event_id][:]
            a_au = kepler[0, :] / orbit_unc.AU_M
            e = kepler[1, :]
            q = a_au * (1.0 - e)
            q_std[idx] = np.nanstd(q, ddof=1)
            inc_mean[idx] = np.nanmean(kepler[2, :])
        keep = (
            np.isfinite(q_std)
            & np.isfinite(e_std)
            & np.isfinite(inc_std)
            & (reached >= args.min_reached_fraction)
            & (q_std <= args.max_q_std_au)
            & (e_std <= args.max_e_std)
            & (inc_std <= args.max_inc_std_deg)
        )
        for idx in np.flatnonzero(keep):
            rows.append(
                {
                    "event_id": event_ids[idx],
                    "kepler": h["kepler_samples"][event_ids[idx]][:, 0],
                    "median_sanya_snr_db": float(snr[idx]),
                    "inclination_deg": float(inc_mean[idx]),
                    "n_points": int(n_points[idx]),
                    "q_std": float(q_std[idx]),
                    "e_std": float(e_std[idx]),
                    "inc_std": float(inc_std[idx]),
                }
            )
    return rows


def plot_orbits(rows, args):
    if not rows:
        raise RuntimeError("No events passed the low-uncertainty selection.")

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 21,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 8.8), constrained_layout=True)
    draw_planet_orbit_rings(ax)
    inc = np.asarray([row["inclination_deg"] for row in rows], dtype=np.float64)
    norm = plt.Normalize(vmin=0.0, vmax=180.0)
    cmap = plt.get_cmap("turbo")

    for row in rows:
        xy = orbit_unc.ellipse_xy_from_kepler(row["kepler"])
        if xy is None:
            continue
        color = cmap(norm(row["inclination_deg"]))
        ax.plot(xy[0], xy[1], color=color, alpha=0.72, lw=1.25)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.015)
    cb.set_label("Meteor orbit inclination (deg)")

    criteria = (
        f"{len(rows)} events; "
        rf"$\sigma_e \leq {args.max_e_std:g}$, "
        rf"$\sigma_q \leq {args.max_q_std_au:g}$ AU, "
        rf"$\sigma_i \leq {args.max_inc_std_deg:g}^\circ$"
    )
    ax.text(
        0.04,
        0.05,
        criteria,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 4},
    )
    ax.text(
        0.98,
        0.02,
        "200 km above-atmosphere covariance samples",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color="0.45",
    )
    ax.set_xlim(-args.lim_au, args.lim_au)
    ax.set_ylim(-args.lim_au, args.lim_au)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Heliocentric ecliptic x (AU)")
    ax.set_ylabel("Heliocentric ecliptic y (AU)")
    ax.grid(color="0.90", lw=0.8)
    ax.legend(loc="upper center", ncol=3, framealpha=0.9)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(os.path.splitext(args.output)[0] + ".png", dpi=220)
    plt.close(fig)


def draw_planet_orbit_rings(ax) -> None:
    planets = [
        ("Mercury", 0.387, "#999999"),
        ("Venus", 0.723, "#b79a74"),
        ("Earth", 1.000, "#5b8fc9"),
        ("Mars", 1.524, "#c7705f"),
        ("Jupiter", 5.204, "#9b8565"),
    ]
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    for name, radius, color in planets:
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), color=color, lw=1.15, alpha=0.85, label=name)
    ax.scatter([0.0], [0.0], s=140, color="#ffd21f", edgecolor="#333333", linewidth=1.0, zorder=4, label="Sun")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=INPUT_H5)
    p.add_argument("--output", default=OUTPUT_PDF)
    p.add_argument("--max-e-std", type=float, default=0.1)
    p.add_argument("--max-q-std-au", type=float, default=0.1)
    p.add_argument("--max-inc-std-deg", type=float, default=5.0)
    p.add_argument("--min-reached-fraction", type=float, default=1.0)
    p.add_argument("--lim-au", type=float, default=6.0)
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_selected(args)
    plot_orbits(rows, args)
    print(f"selected {len(rows)} events")
    print(f"wrote {args.output}")
    print(f"wrote {os.path.splitext(args.output)[0] + '.png'}")


if __name__ == "__main__":
    main()
