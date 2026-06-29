#!/usr/bin/env python3
"""Plot Sanya Sun-centered radiants folded into the upper-right quadrant."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import plot_sun_centered_ecliptic_radiants as radiant_plot


DEFAULT_INPUT_H5 = Path("results/sun_centered_ecliptic_radiants.h5")
DEFAULT_FLUX_H5 = Path("results/sky_averaged_flux_empirical_centroid_area_v20260624c.h5")
DEFAULT_OUTPUT_BASE = Path("results/folded_upper_right_radiants_v20260624a")
DEFAULT_PAPER_FIGURE_DIR = Path("/Users/jvi019/src/sanya_tristatic_paper/figures")
APEX_MERIDIAN_DEG = 270.0
LON_BINS = np.linspace(0.0, 180.0, 19)
LAT_BINS = np.linspace(0.0, 90.0, 10)


def decode_strings(values):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def load_radiants(path: Path):
    with h5py.File(path, "r") as h:
        return {
            "event_id": decode_strings(h["event_id"][:]),
            "lambda_minus_sun_deg": np.asarray(h["lambda_minus_sun_deg"][:], dtype=float),
            "beta_ecliptic_deg": np.asarray(h["beta_ecliptic_deg"][:], dtype=float),
            "speed_km_s": np.asarray(h["speed_km_s"][:], dtype=float),
            "t0_ns": np.asarray(h["t0_ns"][:], dtype=np.int64),
        }


def load_flux_product(path: Path, event_id):
    if not path.exists():
        return (
            np.full(len(event_id), np.nan, dtype=float),
            np.full(len(event_id), np.nan, dtype=float),
        )
    with h5py.File(path, "r") as h:
        flux_event_id = decode_strings(h["event_id"][:])
        flux_weight = np.asarray(h["flux_weight_per_km2_day"][:], dtype=float)
        exposure_sum_h = np.asarray(h["family_effective_exposure_sum_h"][:], dtype=float)
    index = {eid: i for i, eid in enumerate(flux_event_id)}
    weights = np.full(len(event_id), np.nan, dtype=float)
    exposure = np.full(len(event_id), np.nan, dtype=float)
    for i, eid in enumerate(event_id):
        j = index.get(eid)
        if j is not None:
            weights[i] = flux_weight[j]
            exposure[i] = exposure_sum_h[j]
    return weights, exposure


def fold_coordinates(lambda_minus_sun_deg, beta_deg):
    plot_lon_deg = radiant_plot.centered_plot_longitude_deg(lambda_minus_sun_deg)
    folded_lon_deg = np.abs(plot_lon_deg)
    folded_beta_deg = np.abs(beta_deg)
    return folded_lon_deg, folded_beta_deg


def write_h5(path: Path, data, folded_lon_deg, folded_beta_deg, flux_weight, exposure_sum_h):
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["script"] = Path(__file__).name
        h.attrs["input_h5"] = str(DEFAULT_INPUT_H5)
        h.attrs["folding"] = "all radiants mirrored into plot longitude >= 0 and ecliptic latitude >= 0"
        h.attrs["apex_meridian_lambda_minus_sun_deg"] = APEX_MERIDIAN_DEG
        h.create_dataset("event_id", data=np.asarray(data["event_id"], dtype=object), dtype=string_dtype)
        h["lambda_minus_sun_deg"] = data["lambda_minus_sun_deg"]
        h["beta_ecliptic_deg"] = data["beta_ecliptic_deg"]
        h["speed_km_s"] = data["speed_km_s"]
        h["t0_ns"] = data["t0_ns"]
        h["folded_plot_longitude_deg"] = folded_lon_deg
        h["folded_beta_ecliptic_deg"] = folded_beta_deg
        h["flux_weight_per_km2_day"] = flux_weight
        h["symmetry_family_effective_exposure_sum_h"] = exposure_sum_h


def plot(output_base: Path, folded_lon_deg, folded_beta_deg, speed_km_s):
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12.5,
            "axes.titlesize": 13.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    sc = ax.scatter(
        folded_lon_deg,
        folded_beta_deg,
        c=speed_km_s,
        s=26,
        cmap="turbo",
        alpha=0.78,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(0.0, 90.0)
    ax.set_xlabel(r"Folded longitude from apex meridian (deg)")
    ax.set_ylabel(r"Folded ecliptic latitude, $|\beta|$ (deg)")
    ax.set_title("Sun-centered radiants folded by apex/ecliptic symmetry")
    ax.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"Fitted geocentric velocity, $v_g$ (km s$^{-1}$)")
    ax.text(
        0.02,
        0.98,
        f"n = {len(folded_lon_deg)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.85, "pad": 2.5},
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_flux_map(output_base: Path, folded_lon_deg, folded_beta_deg, flux_weight):
    finite = np.isfinite(folded_lon_deg) & np.isfinite(folded_beta_deg)
    counts, lon_edges, lat_edges = np.histogram2d(
        folded_lon_deg[finite],
        folded_beta_deg[finite],
        bins=(LON_BINS, LAT_BINS),
    )
    finite_flux = finite & np.isfinite(flux_weight) & (flux_weight > 0.0)
    flux, _, _ = np.histogram2d(
        folded_lon_deg[finite_flux],
        folded_beta_deg[finite_flux],
        bins=(LON_BINS, LAT_BINS),
        weights=flux_weight[finite_flux],
    )
    masked_flux = np.ma.masked_where(counts <= 0.0, flux).T

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12.5,
            "axes.titlesize": 13.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        masked_flux,
        cmap="viridis",
        shading="auto",
        edgecolors="white",
        linewidth=0.4,
    )
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if counts[i, j] > 0.0:
                ax.text(
                    0.5 * (lon_edges[i] + lon_edges[i + 1]),
                    0.5 * (lat_edges[j] + lat_edges[j + 1]),
                    f"{int(counts[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if flux[i, j] > 0.55 * np.nanmax(flux) else "0.15",
                )
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(0.0, 90.0)
    ax.set_xlabel(r"Folded longitude from apex meridian (deg)")
    ax.set_ylabel(r"Folded ecliptic latitude, $|\beta|$ (deg)")
    ax.set_title("Folded radiant count-based flux estimate")
    ax.grid(False)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(r"Flux contribution (km$^{-2}$ day$^{-1}$ per bin)")
    ax.text(
        0.02,
        0.98,
        rf"$\sum\Phi$ = {np.nansum(flux):.0f} km$^{{-2}}$ day$^{{-1}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.85, "pad": 2.5},
    )
    png = output_base.with_name(output_base.name + "_count_flux.png")
    pdf = output_base.with_name(output_base.name + "_count_flux.pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_exposure_map(output_base: Path, folded_lon_deg, folded_beta_deg, exposure_sum_h):
    finite = np.isfinite(folded_lon_deg) & np.isfinite(folded_beta_deg)
    counts, lon_edges, lat_edges = np.histogram2d(
        folded_lon_deg[finite],
        folded_beta_deg[finite],
        bins=(LON_BINS, LAT_BINS),
    )
    finite_exposure = finite & np.isfinite(exposure_sum_h) & (exposure_sum_h > 0.0)
    exposure_sum, _, _ = np.histogram2d(
        folded_lon_deg[finite_exposure],
        folded_beta_deg[finite_exposure],
        bins=(LON_BINS, LAT_BINS),
        weights=exposure_sum_h[finite_exposure],
    )
    exposure_count, _, _ = np.histogram2d(
        folded_lon_deg[finite_exposure],
        folded_beta_deg[finite_exposure],
        bins=(LON_BINS, LAT_BINS),
    )
    mean_exposure = np.divide(
        exposure_sum,
        exposure_count,
        out=np.full_like(exposure_sum, np.nan, dtype=float),
        where=exposure_count > 0.0,
    )
    masked_exposure = np.ma.masked_where(counts <= 0.0, mean_exposure).T

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        masked_exposure,
        cmap="cividis",
        shading="auto",
        edgecolors="white",
        linewidth=0.4,
    )
    vmax = float(np.nanmax(mean_exposure)) if np.any(np.isfinite(mean_exposure)) else 1.0
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            if counts[i, j] > 0.0:
                ax.text(
                    0.5 * (lon_edges[i] + lon_edges[i + 1]),
                    0.5 * (lat_edges[j] + lat_edges[j + 1]),
                    f"{int(counts[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if mean_exposure[i, j] < 0.45 * vmax else "0.15",
                )
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(0.0, 90.0)
    ax.set_xlabel(r"Folded longitude from apex meridian (deg)")
    ax.set_ylabel(r"Folded ecliptic latitude, $|\beta|$ (deg)")
    ax.set_title("Zenithal-equivalent observing hours")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Mean symmetry-family effective exposure (h)")
    ax.text(
        0.02,
        0.98,
        f"median = {np.nanmedian(exposure_sum_h):.2f} h",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.85, "pad": 2.5},
    )
    png = output_base.with_name(output_base.name + "_effective_hours.png")
    pdf = output_base.with_name(output_base.name + "_effective_hours.pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT_H5)
    parser.add_argument("--flux-h5", type=Path, default=DEFAULT_FLUX_H5)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--copy-to-paper", action="store_true")
    args = parser.parse_args()

    data = load_radiants(args.input_h5)
    folded_lon_deg, folded_beta_deg = fold_coordinates(data["lambda_minus_sun_deg"], data["beta_ecliptic_deg"])
    flux_weight, exposure_sum_h = load_flux_product(args.flux_h5, data["event_id"])
    write_h5(args.output_base.with_suffix(".h5"), data, folded_lon_deg, folded_beta_deg, flux_weight, exposure_sum_h)
    png, pdf = plot(args.output_base, folded_lon_deg, folded_beta_deg, data["speed_km_s"])
    flux_png, flux_pdf = plot_flux_map(args.output_base, folded_lon_deg, folded_beta_deg, flux_weight)
    exposure_png, exposure_pdf = plot_exposure_map(args.output_base, folded_lon_deg, folded_beta_deg, exposure_sum_h)

    print(f"n={len(folded_lon_deg)}")
    print(f"output_h5={args.output_base.with_suffix('.h5')}")
    print(f"output_png={png}")
    print(f"output_pdf={pdf}")
    print(f"flux_png={flux_png}")
    print(f"flux_pdf={flux_pdf}")
    print(f"exposure_png={exposure_png}")
    print(f"exposure_pdf={exposure_pdf}")
    if args.copy_to_paper:
        args.output_base.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        for src in (png, pdf, flux_png, flux_pdf, exposure_png, exposure_pdf):
            dst = DEFAULT_PAPER_FIGURE_DIR / src.name
            dst.write_bytes(src.read_bytes())
            print(f"paper_copy={dst}")


if __name__ == "__main__":
    main()
