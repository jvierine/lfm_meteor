import argparse
import os
import shutil

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import fit_gcrs_trajectories_lfm_ambiguity as gfit
import plot_memo09_antenna_gain_patterns as gain_model


INPUT_H5 = "results/all_tristatic_ballistic_snr_weighted_v20260613b.h5"
OUTPUT_BASE = "results/sanya_beam_position_histogram"
ARTICLE_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/figures"
GRID_LIMIT_DEG = 2.6
GRID_N = 501
HIST_BINS = 120


def decode_strings(values):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("zero-length vector")
    return vector / norm


def local_sky_axes(pointing_enu):
    east = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    north = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    pointing = unit(pointing_enu)
    east_axis = east - float(east @ pointing) * pointing
    north_axis = north - float(north @ pointing) * pointing
    east_axis = unit(east_axis)
    north_axis = north_axis - float(north_axis @ east_axis) * east_axis
    north_axis = north_axis - float(north_axis @ pointing) * pointing
    north_axis = unit(north_axis)
    return east_axis, north_axis


def ecef_to_enu_vectors(vectors_ecef, lat_deg, lon_deg):
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float64)
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], dtype=np.float64)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=np.float64)
    vectors = np.asarray(vectors_ecef, dtype=np.float64)
    return np.stack([vectors @ east, vectors @ north, vectors @ up], axis=-1)


def angular_offsets_deg(directions_enu, pointing_enu, east_axis, north_axis):
    directions = unit(directions_enu)
    pointing = unit(pointing_enu)
    denom = np.maximum(directions @ pointing, 1e-12)
    east_deg = np.rad2deg(np.arctan2(directions @ east_axis, denom))
    north_deg = np.rad2deg(np.arctan2(directions @ north_axis, denom))
    return east_deg, north_deg


def directions_from_east_north_offsets(pointing_enu, east_axis, north_axis, east_deg, north_deg):
    directions = (
        pointing_enu
        + np.tan(np.deg2rad(east_deg))[..., None] * east_axis
        + np.tan(np.deg2rad(north_deg))[..., None] * north_axis
    )
    return unit(directions)


def collect_positions(input_h5):
    with h5py.File(input_h5, "r") as h:
        event_ids = decode_strings(h["event_id"][:])
        points = []
        for event_id in event_ids:
            group = h["points"][event_id]
            x_itrs_m = np.asarray(group["x_itrs_m"][:], dtype=np.float64)
            finite = np.all(np.isfinite(x_itrs_m), axis=1)
            if np.any(finite):
                points.append(x_itrs_m[finite])
    if not points:
        raise RuntimeError(f"No finite x_itrs_m positions found in {input_h5}")
    return np.vstack(points), len(event_ids)


def sanya_beam_relative_gain_db(east_grid_deg, north_grid_deg):
    site = gain_model.SITES[0]
    pointing = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    east_axis, north_axis = local_sky_axes(pointing)
    directions = directions_from_east_north_offsets(pointing, east_axis, north_axis, east_grid_deg, north_grid_deg)
    _normal, tilt_axis, panel_cross_axis = gain_model.panel_axes(site)
    power = gain_model.aperture_power(
        directions,
        pointing,
        tilt_axis,
        panel_cross_axis,
        site.dim_tilt_plane_m,
        site.dim_cross_tilt_m,
    )
    return 10.0 * np.log10(np.maximum(power, 1e-10))


def make_plot(input_h5, output_base, copy_to_article=False):
    positions_ecef_m, n_events = collect_positions(input_h5)
    sanya_lat_deg, sanya_lon_deg, _sanya_alt_m = jcoord.ecef2geodetic(*gfit.LINK_TX_POSITIONS_M[0])
    los_ecef = unit(positions_ecef_m - gfit.LINK_TX_POSITIONS_M[0][None, :])
    los_enu = ecef_to_enu_vectors(los_ecef, sanya_lat_deg, sanya_lon_deg)

    site = gain_model.SITES[0]
    pointing = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    east_axis, north_axis = local_sky_axes(pointing)
    east_deg, north_deg = angular_offsets_deg(los_enu, pointing, east_axis, north_axis)
    finite = np.isfinite(east_deg) & np.isfinite(north_deg)
    east_deg = east_deg[finite]
    north_deg = north_deg[finite]

    grid = np.linspace(-GRID_LIMIT_DEG, GRID_LIMIT_DEG, GRID_N)
    east_grid, north_grid = np.meshgrid(grid, grid)
    gain_db = sanya_beam_relative_gain_db(east_grid, north_grid)

    hist_range = [[-GRID_LIMIT_DEG, GRID_LIMIT_DEG], [-GRID_LIMIT_DEG, GRID_LIMIT_DEG]]
    in_window = (
        (east_deg >= -GRID_LIMIT_DEG)
        & (east_deg <= GRID_LIMIT_DEG)
        & (north_deg >= -GRID_LIMIT_DEG)
        & (north_deg <= GRID_LIMIT_DEG)
    )
    counts, east_edges, north_edges = np.histogram2d(east_deg, north_deg, bins=HIST_BINS, range=hist_range)
    counts = counts.T
    max_count = float(np.nanmax(counts)) if np.any(counts > 0.0) else 1.0
    counts = np.ma.masked_less_equal(counts, 0.0)

    with plt.rc_context(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    ):
        fig, ax = plt.subplots(figsize=(6.2, 5.25), constrained_layout=True)
        hist = ax.pcolormesh(
            east_edges,
            north_edges,
            counts,
            cmap="magma",
            norm=LogNorm(vmin=1.0, vmax=max(max_count, 1.0)),
            shading="auto",
            alpha=0.82,
            rasterized=True,
        )
        contours = ax.contour(
            east_grid,
            north_grid,
            gain_db,
            levels=[-30.0, -20.0, -13.3, -10.0, -3.0],
            colors=["0.55", "0.35", "0.20", "0.10", "0.0"],
            linewidths=[0.7, 0.8, 0.9, 1.0, 1.25],
            linestyles=[":", "--", "-.", "-", "-"],
        )
        ax.axhline(0.0, color="0.2", lw=0.7, alpha=0.6)
        ax.axvline(0.0, color="0.2", lw=0.7, alpha=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-GRID_LIMIT_DEG, GRID_LIMIT_DEG)
        ax.set_ylim(-GRID_LIMIT_DEG, GRID_LIMIT_DEG)
        ax.set_xlabel("East (deg)")
        ax.set_ylabel("North (deg)")
        ax.set_title("Tri-static head echo positions")
        ax.text(
            0.02,
            0.02,
            f"{int(np.count_nonzero(in_window)):,} of {len(east_deg):,} pulse positions shown\n{n_events} tri-static events",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="0.05",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
        )
        legend_handles = [
            Line2D([0], [0], color="0.0", lw=1.25, ls="-", label="-3 dB"),
            Line2D([0], [0], color="0.20", lw=0.9, ls="-.", label="-13 dB"),
            Line2D([0], [0], color="0.35", lw=0.8, ls="--", label="-20 dB"),
            Line2D([0], [0], color="0.55", lw=0.7, ls=":", label="-30 dB"),
        ]
        ax.legend(handles=legend_handles, title="Sanya gain", loc="upper right", frameon=True, framealpha=0.86)
        cb = fig.colorbar(hist, ax=ax, fraction=0.046, pad=0.035)
        count_ticks = [tick for tick in (1, 2, 4, 6, 8, 10) if tick <= max_count]
        if count_ticks:
            cb.set_ticks(count_ticks)
            cb.set_ticklabels([str(tick) for tick in count_ticks])
        cb.set_label("Pulse count per angular bin")

        png = f"{output_base}.png"
        pdf = f"{output_base}.pdf"
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"positions={len(east_deg)} events={n_events}")
    print(f"east_offset_deg_median={np.nanmedian(east_deg):.4f} north_offset_deg_median={np.nanmedian(north_deg):.4f}")

    if copy_to_article:
        os.makedirs(ARTICLE_FIGURE_DIR, exist_ok=True)
        for path in (png, pdf):
            dest = os.path.join(ARTICLE_FIGURE_DIR, os.path.basename(path))
            shutil.copy2(path, dest)
            print(f"copied {dest}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a 2D histogram of derived tri-static meteor positions over the Sanya beam pattern."
    )
    parser.add_argument("--input", default=INPUT_H5)
    parser.add_argument("--output-base", default=OUTPUT_BASE)
    parser.add_argument("--copy-to-article", action="store_true")
    args = parser.parse_args()
    make_plot(args.input, args.output_base, copy_to_article=args.copy_to_article)


if __name__ == "__main__":
    main()
