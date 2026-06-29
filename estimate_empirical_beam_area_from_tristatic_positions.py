#!/usr/bin/env python3
"""Estimate Sanya effective collecting area from fitted tri-static positions."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde

import fit_gcrs_trajectories_lfm_ambiguity as gfit
import plot_memo09_antenna_gain_patterns as gain_model
import plot_sanya_beam_position_histogram as beam_hist


DEFAULT_CATALOG_DIR = Path("results/tristatic")
DEFAULT_OUTPUT_BASE = Path("results/empirical_tristatic_beam_area_v20260624a")
DEFAULT_PAPER_FIGURE_DIR = Path("/Users/jvi019/src/sanya_tristatic_paper/figures")
DEFAULT_PAPER_TABLE_DIR = Path("/Users/jvi019/src/sanya_tristatic_paper/tables")


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


def local_enu_to_ecef_matrix(lat_deg, lon_deg):
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float64)
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], dtype=np.float64)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=np.float64)
    return np.column_stack([east, north, up])


def beam_axes_ecef():
    tx = np.asarray(gfit.LINK_TX_POSITIONS_M[0], dtype=np.float64)
    lat_deg, lon_deg, _alt_m = jcoord.ecef2geodetic(*tx)
    site = gain_model.SITES[0]
    pointing_enu = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    east_axis_enu, north_axis_enu = beam_hist.local_sky_axes(pointing_enu)
    enu_to_ecef = local_enu_to_ecef_matrix(lat_deg, lon_deg)
    pointing_ecef = unit(enu_to_ecef @ pointing_enu)
    east_axis_ecef = unit(enu_to_ecef @ east_axis_enu)
    north_axis_ecef = unit(enu_to_ecef @ north_axis_enu)
    return tx, pointing_ecef, east_axis_ecef, north_axis_ecef, float(site.pointing_el_deg)


def load_positions(catalog_dir: Path):
    paths = sorted(glob.glob(str(catalog_dir / "joint_delay_doppler_fft_tri_*.h5")))
    pulse_positions = []
    event_centroids = []
    event_ids = []
    for path in paths:
        with h5py.File(path, "r") as h:
            if "joint_fit/x_itrs_m" not in h:
                continue
            x = np.asarray(h["joint_fit/x_itrs_m"][:], dtype=np.float64)
            keep = np.asarray(h["joint_fit/keep_rows"][:], dtype=bool) if "joint_fit/keep_rows" in h else np.ones(len(x), dtype=bool)
            finite = keep & np.all(np.isfinite(x), axis=1)
            if not np.any(finite):
                continue
            x = x[finite]
            pulse_positions.append(x)
            event_centroids.append(np.nanmean(x, axis=0))
            event_ids.append(str(h.attrs.get("event_id", Path(path).stem.replace("joint_delay_doppler_fft_", ""))))
    if not pulse_positions:
        raise RuntimeError(f"No fitted positions found in {catalog_dir}")
    return np.vstack(pulse_positions), np.vstack(event_centroids), np.asarray(event_ids, dtype=object)


def project_offsets_km(positions_ecef_m):
    tx, pointing, east_axis, north_axis, elevation_deg = beam_axes_ecef()
    rel = np.asarray(positions_ecef_m, dtype=np.float64) - tx[None, :]
    along = rel @ pointing
    offset = rel - along[:, None] * pointing[None, :]
    east_km = (offset @ east_axis) / 1e3
    north_km = (offset @ north_axis) / 1e3
    return east_km, north_km, elevation_deg


def kde_effective_area_km2(east_km, north_km, grid_n=240, pad_fraction=0.25):
    points = np.vstack([east_km, north_km])
    kde = gaussian_kde(points)
    x_min, x_max = np.nanmin(east_km), np.nanmax(east_km)
    y_min, y_max = np.nanmin(north_km), np.nanmax(north_km)
    x_pad = max((x_max - x_min) * pad_fraction, 0.25)
    y_pad = max((y_max - y_min) * pad_fraction, 0.25)
    x = np.linspace(x_min - x_pad, x_max + x_pad, grid_n)
    y = np.linspace(y_min - y_pad, y_max + y_pad, grid_n)
    xx, yy = np.meshgrid(x, y)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    peak = float(np.nanmax(density))
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    area_perp = float(np.nansum(density / peak) * dx * dy)
    return area_perp, x, y, density


def hull_area_km2(east_km, north_km, percentile):
    r = np.hypot(east_km - np.nanmedian(east_km), north_km - np.nanmedian(north_km))
    keep = r <= np.nanpercentile(r, percentile)
    pts = np.column_stack([east_km[keep], north_km[keep]])
    if len(pts) < 3:
        return np.nan
    return float(ConvexHull(pts).volume)


def summarize_area(east_km, north_km, elevation_deg, grid_n):
    area_perp, x, y, density = kde_effective_area_km2(east_km, north_km, grid_n=grid_n)
    horizontal_factor = 1.0 / np.sin(np.deg2rad(elevation_deg))
    return {
        "kde_area_perpendicular_km2": area_perp,
        "kde_area_horizontal_km2": float(area_perp * horizontal_factor),
        "hull90_perpendicular_km2": hull_area_km2(east_km, north_km, 90.0),
        "hull95_perpendicular_km2": hull_area_km2(east_km, north_km, 95.0),
        "hull98_perpendicular_km2": hull_area_km2(east_km, north_km, 98.0),
        "horizontal_projection_factor": float(horizontal_factor),
        "median_east_km": float(np.nanmedian(east_km)),
        "median_north_km": float(np.nanmedian(north_km)),
        "x_grid_km": x,
        "y_grid_km": y,
        "density": density,
    }


def write_h5(path, pulse_xy, centroid_xy, pulse_summary, centroid_summary, event_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["script"] = Path(__file__).name
        h.attrs["method"] = "Gaussian KDE of beam-normal fitted-position distribution normalized to unit peak; horizontal area = perpendicular area / sin(elevation)"
        h.attrs["default_area_source"] = "pulse_position_kde_horizontal_km2"
        h.attrs["pulse_position_kde_horizontal_km2"] = pulse_summary["kde_area_horizontal_km2"]
        h.attrs["event_centroid_kde_horizontal_km2"] = centroid_summary["kde_area_horizontal_km2"]
        h.create_dataset("event_id", data=np.asarray(event_ids, dtype=object), dtype=string_dtype)
        h["pulse_east_km"] = pulse_xy[0]
        h["pulse_north_km"] = pulse_xy[1]
        h["event_centroid_east_km"] = centroid_xy[0]
        h["event_centroid_north_km"] = centroid_xy[1]
        for name, summary in (("pulse_positions", pulse_summary), ("event_centroids", centroid_summary)):
            g = h.create_group(name)
            for key, value in summary.items():
                if isinstance(value, np.ndarray):
                    g[key] = value
                else:
                    g.attrs[key] = value


def write_macros(path, pulse_summary, centroid_summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "% Generated by /Users/jvi019/src/lfm_meteor/estimate_empirical_beam_area_from_tristatic_positions.py",
            rf"\newcommand{{\empiricalBeamAreaPulseKmSq}}{{{pulse_summary['kde_area_horizontal_km2']:.2f}}}",
            rf"\newcommand{{\empiricalBeamAreaCentroidKmSq}}{{{centroid_summary['kde_area_horizontal_km2']:.2f}}}",
            rf"\newcommand{{\empiricalBeamAreaPulsePerpKmSq}}{{{pulse_summary['kde_area_perpendicular_km2']:.2f}}}",
            rf"\newcommand{{\empiricalBeamAreaCentroidPerpKmSq}}{{{centroid_summary['kde_area_perpendicular_km2']:.2f}}}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def plot_area(output_base, pulse_xy, centroid_xy, pulse_summary, centroid_summary):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True, sharex=True, sharey=True)
    for ax, xy, summary, title in (
        (axes[0], pulse_xy, pulse_summary, "Pulse positions"),
        (axes[1], centroid_xy, centroid_summary, "Event centroids"),
    ):
        x = summary["x_grid_km"]
        y = summary["y_grid_km"]
        density = summary["density"]
        response = density / np.nanmax(density)
        mesh = ax.pcolormesh(x, y, response, shading="auto", cmap="magma", rasterized=True)
        ax.contour(x, y, response, levels=[0.1, 0.25, 0.5], colors="white", linewidths=[0.8, 0.9, 1.0])
        ax.scatter(xy[0], xy[1], s=4 if title.startswith("Pulse") else 10, c="cyan", alpha=0.35, edgecolors="none")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}\nA_eff={summary['kde_area_horizontal_km2']:.2f} km$^2$")
        ax.set_xlabel("Beam-normal east offset (km)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Beam-normal north offset (km)")
    cb = fig.colorbar(mesh, ax=axes, pad=0.02, fraction=0.045)
    cb.set_label("Relative empirical response")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--paper-figure-dir", type=Path, default=DEFAULT_PAPER_FIGURE_DIR)
    parser.add_argument("--paper-macro-output", type=Path, default=DEFAULT_PAPER_TABLE_DIR / "empirical_beam_area_macros.tex")
    parser.add_argument("--grid-n", type=int, default=260)
    parser.add_argument("--copy-to-paper", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pulse_pos, centroid_pos, event_ids = load_positions(args.catalog_dir)
    pulse_e, pulse_n, elevation_deg = project_offsets_km(pulse_pos)
    cent_e, cent_n, _ = project_offsets_km(centroid_pos)
    pulse_summary = summarize_area(pulse_e, pulse_n, elevation_deg, args.grid_n)
    centroid_summary = summarize_area(cent_e, cent_n, elevation_deg, args.grid_n)
    output_h5 = args.output_base.with_suffix(".h5")
    write_h5(output_h5, (pulse_e, pulse_n), (cent_e, cent_n), pulse_summary, centroid_summary, event_ids)
    write_macros(args.paper_macro_output, pulse_summary, centroid_summary)
    png, pdf = plot_area(args.output_base, (pulse_e, pulse_n), (cent_e, cent_n), pulse_summary, centroid_summary)
    copied = []
    if args.copy_to_paper:
        args.paper_figure_dir.mkdir(parents=True, exist_ok=True)
        for src in (png, pdf):
            dst = args.paper_figure_dir / src.name
            dst.write_bytes(src.read_bytes())
            copied.append(dst)

    print(f"n_pulse_positions={len(pulse_e)}")
    print(f"n_event_centroids={len(cent_e)}")
    print(f"pulse_area_horizontal_km2={pulse_summary['kde_area_horizontal_km2']:.6g}")
    print(f"centroid_area_horizontal_km2={centroid_summary['kde_area_horizontal_km2']:.6g}")
    print(f"pulse_area_perpendicular_km2={pulse_summary['kde_area_perpendicular_km2']:.6g}")
    print(f"centroid_area_perpendicular_km2={centroid_summary['kde_area_perpendicular_km2']:.6g}")
    print(f"output_h5={output_h5}")
    print(f"plot_png={png}")
    print(f"plot_pdf={pdf}")
    print(f"macro_output={args.paper_macro_output}")
    for path in copied:
        print(f"paper_copy={path}")


if __name__ == "__main__":
    main()
