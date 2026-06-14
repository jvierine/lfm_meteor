"""Plot physical rectangular-aperture gain patterns for the Sanya system."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


PAPER_FIGURE_PDF = "/Users/jvi019/src/sanya_tristatic_paper/figures/memo09_antenna_gain_patterns.pdf"
PAPER_FIGURE_PNG = "/Users/jvi019/src/sanya_tristatic_paper/figures/memo09_antenna_gain_patterns.png"
SUMMARY_CSV = "/Users/jvi019/src/sanya_tristatic_paper/figures/memo09_antenna_gain_pattern_summary.csv"

FREQUENCY_HZ = 450.0e6
LIGHT_SPEED_M_S = 299_792_458.0
WAVELENGTH_M = LIGHT_SPEED_M_S / FREQUENCY_HZ
HALF_POWER_DB = -10.0 * math.log10(2.0)


@dataclass(frozen=True)
class SitePattern:
    name: str
    pointing_az_deg: float
    pointing_el_deg: float
    boresight_az_deg: float
    boresight_el_deg: float
    dim_tilt_plane_m: float
    dim_cross_tilt_m: float


SITES = (
    SitePattern("Sanya", 14.996337890625, 74.9981689453125, 0.0, 90.0, 40.0, 40.0),
    SitePattern("Danzhou", 151.2652587890625, 37.3260498046875, 158.3, 70.0, 32.0, 24.0),
    SitePattern("Wenchang", 225.7855224609375, 29.2950439453125, 221.9, 70.0, 32.0, 24.0),
)


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("zero-length vector")
    return v / norm


def azel_to_enu(az_deg: float, el_deg: float) -> np.ndarray:
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array(
        [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)],
        dtype=np.float64,
    )


def angular_separation_deg(u1: np.ndarray, u2: np.ndarray) -> float:
    dot = float(np.clip(np.dot(unit(u1), unit(u2)), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))


def panel_axes(site: SitePattern) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = unit(azel_to_enu(site.boresight_az_deg, site.boresight_el_deg))
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, up))) > 0.999999:
        tilt_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        tilt_axis = unit(up - np.dot(up, normal) * normal)
    cross_axis = unit(np.cross(normal, tilt_axis))
    return normal, tilt_axis, cross_axis


def offset_basis(pointing: np.ndarray, tilt_axis: np.ndarray, cross_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scan = tilt_axis - np.dot(tilt_axis, pointing) * pointing
    if np.linalg.norm(scan) < 1e-12:
        scan = cross_axis - np.dot(cross_axis, pointing) * pointing
    scan = unit(scan)
    cross = unit(np.cross(pointing, scan))
    return scan, cross


def directions_from_offsets(
    pointing: np.ndarray,
    scan_axis: np.ndarray,
    cross_axis: np.ndarray,
    scan_offset_deg: np.ndarray,
    cross_offset_deg: np.ndarray,
) -> np.ndarray:
    sx = np.tan(np.deg2rad(scan_offset_deg))
    sy = np.tan(np.deg2rad(cross_offset_deg))
    direction = pointing[..., None] if pointing.ndim else pointing
    direction = pointing + sx[..., None] * scan_axis + sy[..., None] * cross_axis
    return direction / np.linalg.norm(direction, axis=-1, keepdims=True)


def aperture_power(
    direction: np.ndarray,
    pointing: np.ndarray,
    tilt_axis: np.ndarray,
    cross_axis: np.ndarray,
    dim_tilt_plane_m: float,
    dim_cross_tilt_m: float,
) -> np.ndarray:
    u = dim_tilt_plane_m / WAVELENGTH_M * (np.sum(direction * tilt_axis, axis=-1) - np.dot(pointing, tilt_axis))
    v = dim_cross_tilt_m / WAVELENGTH_M * (np.sum(direction * cross_axis, axis=-1) - np.dot(pointing, cross_axis))
    return np.sinc(u) ** 2 * np.sinc(v) ** 2


def hpbw_along_axis(
    pointing: np.ndarray,
    scan_axis: np.ndarray,
    cross_axis: np.ndarray,
    tilt_axis: np.ndarray,
    panel_cross_axis: np.ndarray,
    site: SitePattern,
    axis_name: str,
) -> float:
    offsets = np.linspace(0.0, 5.0, 5001)
    if axis_name == "scan":
        scan_offsets = offsets
        cross_offsets = np.zeros_like(offsets)
    else:
        scan_offsets = np.zeros_like(offsets)
        cross_offsets = offsets
    direction = directions_from_offsets(pointing, scan_axis, cross_axis, scan_offsets, cross_offsets)
    rel_db = 10.0 * np.log10(
        np.maximum(
            aperture_power(direction, pointing, tilt_axis, panel_cross_axis, site.dim_tilt_plane_m, site.dim_cross_tilt_m),
            1e-12,
        )
    )
    below = np.flatnonzero(rel_db <= HALF_POWER_DB)
    if len(below) == 0:
        return float("nan")
    idx = int(below[0])
    if idx == 0:
        one_sided = offsets[0]
    else:
        x0, x1 = offsets[idx - 1], offsets[idx]
        y0, y1 = rel_db[idx - 1], rel_db[idx]
        one_sided = x0 + (HALF_POWER_DB - y0) * (x1 - x0) / (y1 - y0)
    return float(2.0 * one_sided)


def ideal_normal_gain_dbi(site: SitePattern) -> float:
    aperture_area_m2 = site.dim_tilt_plane_m * site.dim_cross_tilt_m
    gain_linear = 4.0 * math.pi * aperture_area_m2 / WAVELENGTH_M**2
    return 10.0 * math.log10(gain_linear)


def site_summary(site: SitePattern) -> dict[str, float | str]:
    pointing = unit(azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    normal, tilt_axis, cross_axis = panel_axes(site)
    scan_axis, plot_cross_axis = offset_basis(pointing, tilt_axis, cross_axis)
    scan_angle = angular_separation_deg(normal, pointing)
    cos_scan = max(float(np.dot(normal, pointing)), 1e-6)
    projected_area_loss_db = -10.0 * math.log10(cos_scan)
    normal_gain = ideal_normal_gain_dbi(site)
    peak_gain = normal_gain - projected_area_loss_db
    return {
        "site": site.name,
        "pointing_az_deg": site.pointing_az_deg,
        "pointing_el_deg": site.pointing_el_deg,
        "boresight_az_deg": site.boresight_az_deg,
        "boresight_el_deg": site.boresight_el_deg,
        "scan_angle_deg": scan_angle,
        "cos_scan": cos_scan,
        "dim_tilt_plane_m": site.dim_tilt_plane_m,
        "dim_cross_tilt_m": site.dim_cross_tilt_m,
        "aperture_area_m2": site.dim_tilt_plane_m * site.dim_cross_tilt_m,
        "normal_gain_dbi": normal_gain,
        "projected_area_loss_db": projected_area_loss_db,
        "steered_peak_gain_dbi": peak_gain,
        "scan_hpbw_deg": hpbw_along_axis(
            pointing, scan_axis, plot_cross_axis, tilt_axis, cross_axis, site, "scan"
        ),
        "cross_hpbw_deg": hpbw_along_axis(
            pointing, scan_axis, plot_cross_axis, tilt_axis, cross_axis, site, "cross"
        ),
    }


def gain_pattern_dbi(
    site: SitePattern,
    scan_offset_deg: np.ndarray,
    cross_offset_deg: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | str]]:
    pointing = unit(azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    normal, tilt_axis, panel_cross_axis = panel_axes(site)
    scan_axis, plot_cross_axis = offset_basis(pointing, tilt_axis, panel_cross_axis)
    direction = directions_from_offsets(pointing, scan_axis, plot_cross_axis, scan_offset_deg, cross_offset_deg)
    summary = site_summary(site)
    power = aperture_power(direction, pointing, tilt_axis, panel_cross_axis, site.dim_tilt_plane_m, site.dim_cross_tilt_m)
    relative_db = 10.0 * np.log10(np.maximum(power, 1e-8))
    return float(summary["steered_peak_gain_dbi"]) + relative_db, summary


def write_summary(rows: list[dict[str, float | str]]) -> None:
    fieldnames = list(rows[0].keys())
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    summaries = [site_summary(site) for site in SITES]
    write_summary(summaries)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    extent_deg = 3.0
    grid = np.linspace(-extent_deg, extent_deg, 501)
    x, y = np.meshgrid(grid, grid)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), sharex=True, sharey=True, constrained_layout=True)
    mappable = None
    levels = [-20.0, -10.0, -3.0]
    for ax, site, summary in zip(axes, SITES, summaries):
        gain, _ = gain_pattern_dbi(site, x, y)
        peak_gain = float(summary["steered_peak_gain_dbi"])
        mappable = ax.pcolormesh(
            x,
            y,
            gain,
            shading="auto",
            cmap="viridis",
            vmin=20.0,
            vmax=46.0,
            rasterized=True,
        )
        rel = gain - peak_gain
        contours = ax.contour(x, y, rel, levels=levels, colors="white", linewidths=[0.7, 0.8, 1.2])
        ax.clabel(contours, fmt=lambda v: f"{v:.0f} dB", fontsize=8)
        ax.plot(0.0, 0.0, marker="+", color="black", ms=9, mew=1.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{summary['site']}\n"
            f"boresight={float(summary['normal_gain_dbi']):.1f} dBi, steered={peak_gain:.2f} dBi"
        )
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.grid(color="white", alpha=0.18, lw=0.5)
    for ax in axes:
        ax.set_xlabel("Tilt-plane offset from pointing (deg)")
    axes[0].set_ylabel("Cross-tilt offset from pointing (deg)")
    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axes, shrink=0.92, pad=0.02)
        cb.set_label("Gain (dBi)")

    for path in (PAPER_FIGURE_PDF, PAPER_FIGURE_PNG):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(PAPER_FIGURE_PDF)
    fig.savefig(PAPER_FIGURE_PNG, dpi=220)

    for row in summaries:
        print(
            f"{row['site']}: scan={float(row['scan_angle_deg']):.3f} deg, "
            f"ideal-normal={float(row['normal_gain_dbi']):.3f} dBi, "
            f"projected-peak={float(row['steered_peak_gain_dbi']):.3f} dBi, "
            f"HPBW={float(row['scan_hpbw_deg']):.3f}/"
            f"{float(row['cross_hpbw_deg']):.3f} deg"
        )
    print(f"frequency_hz={FREQUENCY_HZ:.1f}")
    print(f"wavelength_m={WAVELENGTH_M:.6f}")
    print(f"figure_pdf={PAPER_FIGURE_PDF}")
    print(f"figure_png={PAPER_FIGURE_PNG}")
    print(f"summary_csv={SUMMARY_CSV}")


if __name__ == "__main__":
    main()
