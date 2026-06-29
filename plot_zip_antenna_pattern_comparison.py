"""Compare supplied SYISR MATLAB array-factor code with the Memo 9 model.

This script ports the beam-pattern calculation from the MATLAB code in

  /Users/jvi019/Dropbox/Work/Documents/2026/cas_visit/Antenna Pattern.zip

and compares it against the rectangular-aperture approximation used by
``plot_memo09_antenna_gain_patterns.py``.  Sanya is evaluated with the
phase-II 104 x 80 active-element grid noted in the supplied MATLAB comments;
the remote receivers retain the supplied 64 x 64 geometry.  The comparison is
made in angular offsets from each station's fixed pointing.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import plot_memo09_antenna_gain_patterns as memo09
import sanya_opts as sc


PAPER_FIGURE_PDF = (
    "/Users/jvi019/src/sanya_tristatic_paper/figures/"
    "memo09_zip_antenna_pattern_comparison.pdf"
)
PAPER_FIGURE_PNG = (
    "/Users/jvi019/src/sanya_tristatic_paper/figures/"
    "memo09_zip_antenna_pattern_comparison.png"
)

FREQUENCY_HZ = sc.RADAR_FREQUENCY_HZ
LIGHT_SPEED_M_S = sc.C
WAVELENGTH_M = sc.RADAR_WAVELENGTH_M

# Values translated from ArrayPattern.m and get_SYISR_Gain_Beamwidth.m.  The
# Sanya transmitter is modeled with the phase-II aperture noted in the MATLAB
# comments, while the remote receive arrays remain at the supplied 64 x 64
# geometry.
DX_EAST_WEST_M = 0.50
DY_NORTH_SOUTH_M = 0.38


@dataclass(frozen=True)
class ArrayGeometry:
    n_east_west: int
    m_north_south: int


ARRAY_GEOMETRY = {
    "Sanya": ArrayGeometry(n_east_west=80, m_north_south=104),
    "Danzhou": ArrayGeometry(n_east_west=64, m_north_south=64),
    "Wenchang": ArrayGeometry(n_east_west=64, m_north_south=64),
}


@dataclass(frozen=True)
class PatternSummary:
    site: str
    matlab_scan_hpbw_deg: float
    matlab_cross_hpbw_deg: float
    memo09_scan_hpbw_deg: float
    memo09_cross_hpbw_deg: float


def enu_to_azel(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    east = direction[..., 0]
    north = direction[..., 1]
    up = direction[..., 2]
    az = np.mod(np.arctan2(east, north), 2.0 * np.pi)
    el = np.arcsin(np.clip(up, -1.0, 1.0))
    return az, el


def centered_offsets_to_directions(
    site: memo09.SitePattern,
    scan_offset_deg: np.ndarray,
    cross_offset_deg: np.ndarray,
) -> np.ndarray:
    pointing = memo09.unit(memo09.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    _, tilt_axis, panel_cross_axis = memo09.panel_axes(site)
    scan_axis, plot_cross_axis = memo09.offset_basis(pointing, tilt_axis, panel_cross_axis)
    return memo09.directions_from_offsets(
        pointing,
        scan_axis,
        plot_cross_axis,
        scan_offset_deg,
        cross_offset_deg,
    )


def ratio_sin(num_factor: float, denom_factor: float, psi: np.ndarray, scale: float) -> np.ndarray:
    numerator = np.sin(num_factor * psi)
    denominator = np.sin(denom_factor * psi)
    out = scale * np.abs(numerator / np.where(np.abs(denominator) > 1e-12, denominator, 1.0))
    limit = scale * abs(num_factor / denom_factor)
    return np.where(np.abs(denominator) > 1e-12, out, limit)


def matlab_array_factor_relative_db(
    site: memo09.SitePattern,
    scan_offset_deg: np.ndarray,
    cross_offset_deg: np.ndarray,
) -> np.ndarray:
    """Port of the ideal active-element array factor in ArrayPattern.m."""
    geometry = ARRAY_GEOMETRY[site.name]
    direction = centered_offsets_to_directions(site, scan_offset_deg, cross_offset_deg)
    phi, theta = enu_to_azel(direction)

    drx = 2.0 * np.pi * DX_EAST_WEST_M / WAVELENGTH_M
    dry = 2.0 * np.pi * DY_NORTH_SOUTH_M / WAVELENGTH_M

    phi_point = np.deg2rad(site.pointing_az_deg)
    theta_point = np.deg2rad(site.pointing_el_deg)

    # The supplied MATLAB code calls these phi_zhuan and theta_zhuan.  For the
    # remote receivers, use the known mechanical boresight tilt toward Sanya.
    phi_rot = np.deg2rad(site.boresight_az_deg)
    theta_rot = np.deg2rad(90.0 - site.boresight_el_deg)

    delta_x = -drx * np.cos(theta_point) * np.sin(phi_point - phi_rot)
    delta_y = -dry * (
        np.cos(theta_point) * np.cos(theta_rot) * np.cos(phi_point - phi_rot)
        + np.sin(theta_point) * np.sin(theta_rot)
    )

    psi_x = drx * np.cos(theta) * np.sin(phi - phi_rot) + delta_x
    psi_y = dry * (
        np.cos(theta) * np.cos(theta_rot) * np.cos(phi - phi_rot)
        + np.sin(theta) * np.sin(theta_rot)
    ) + delta_y

    e_x = ratio_sin(geometry.n_east_west / 2.0, 0.5, psi_x, 1.0 / geometry.n_east_west)
    e_y = ratio_sin(geometry.m_north_south / 2.0, 1.0, psi_y, 2.0 / geometry.m_north_south)
    stagger_pair = 0.5 * np.abs(1.0 + np.exp(1j * (psi_y + 0.5 * psi_x)))
    field_like = np.maximum(stagger_pair * e_x * e_y, 1e-8)

    rel_db = 10.0 * np.log10(field_like)
    return rel_db - np.nanmax(rel_db)


def memo09_relative_db(
    site: memo09.SitePattern,
    scan_offset_deg: np.ndarray,
    cross_offset_deg: np.ndarray,
) -> np.ndarray:
    gain, summary = memo09.gain_pattern_dbi(site, scan_offset_deg, cross_offset_deg)
    return gain - float(summary["steered_peak_gain_dbi"])


def hpbw_from_map(offset_grid_deg: np.ndarray, rel_db_line: np.ndarray) -> float:
    center_index = len(offset_grid_deg) // 2
    right_offsets = offset_grid_deg[center_index:]
    right_db = rel_db_line[center_index:]
    below = np.flatnonzero(right_db <= memo09.HALF_POWER_DB)
    if len(below) == 0:
        return float("nan")
    idx = int(below[0])
    if idx == 0:
        one_sided = right_offsets[0]
    else:
        x0, x1 = right_offsets[idx - 1], right_offsets[idx]
        y0, y1 = right_db[idx - 1], right_db[idx]
        one_sided = x0 + (memo09.HALF_POWER_DB - y0) * (x1 - x0) / (y1 - y0)
    return float(2.0 * one_sided)


def summarize(site: memo09.SitePattern, grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> PatternSummary:
    translated = matlab_array_factor_relative_db(site, x, y)
    rectangular = memo09_relative_db(site, x, y)
    center = len(grid) // 2
    return PatternSummary(
        site=site.name,
        matlab_scan_hpbw_deg=hpbw_from_map(grid, translated[center, :]),
        matlab_cross_hpbw_deg=hpbw_from_map(grid, translated[:, center]),
        memo09_scan_hpbw_deg=hpbw_from_map(grid, rectangular[center, :]),
        memo09_cross_hpbw_deg=hpbw_from_map(grid, rectangular[:, center]),
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    extent_deg = 5.0
    grid = np.linspace(-extent_deg, extent_deg, 701)
    x, y = np.meshgrid(grid, grid)

    summaries = [summarize(site, grid, x, y) for site in memo09.SITES]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.6, 7.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    levels = [-20.0, -10.0, -3.0]
    mappable = None

    for col, (site, summary) in enumerate(zip(memo09.SITES, summaries)):
        patterns = [
            (
                matlab_array_factor_relative_db(site, x, y),
                f"Supplied-code array factor ({ARRAY_GEOMETRY[site.name].m_north_south} x {ARRAY_GEOMETRY[site.name].n_east_west})",
                summary.matlab_scan_hpbw_deg,
                summary.matlab_cross_hpbw_deg,
            ),
            (
                memo09_relative_db(site, x, y),
                "Memo 9 rectangular aperture",
                summary.memo09_scan_hpbw_deg,
                summary.memo09_cross_hpbw_deg,
            ),
        ]
        for row, (rel_db, row_title, scan_hpbw, cross_hpbw) in enumerate(patterns):
            ax = axes[row, col]
            mappable = ax.pcolormesh(
                x,
                y,
                rel_db,
                shading="auto",
                cmap="viridis",
                vmin=-30.0,
                vmax=0.0,
                rasterized=True,
            )
            contours = ax.contour(x, y, rel_db, levels=levels, colors="white", linewidths=[0.7, 0.8, 1.1])
            ax.clabel(contours, fmt=lambda v: f"{v:.0f} dB", fontsize=7)
            ax.plot(0.0, 0.0, marker="+", color="black", ms=8, mew=1.5)
            ax.set_aspect("equal", adjustable="box")
            if row == 0:
                ax.set_title(site.name)
            ax.text(
                0.03,
                0.97,
                f"{row_title}\nHPBW {scan_hpbw:.2f} x {cross_hpbw:.2f} deg",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2.5},
            )
            ax.grid(color="white", alpha=0.18, lw=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Tilt-plane offset from pointing (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cross-tilt offset from pointing (deg)")

    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axes, shrink=0.94, pad=0.015)
        cb.set_label("Relative gain (dB)")

    for path in (PAPER_FIGURE_PDF, PAPER_FIGURE_PNG):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(PAPER_FIGURE_PDF)
    fig.savefig(PAPER_FIGURE_PNG, dpi=220)

    for summary in summaries:
        print(
            f"{summary.site}: MATLAB-port HPBW="
            f"{summary.matlab_scan_hpbw_deg:.3f}/{summary.matlab_cross_hpbw_deg:.3f} deg; "
            f"Memo09 HPBW={summary.memo09_scan_hpbw_deg:.3f}/"
            f"{summary.memo09_cross_hpbw_deg:.3f} deg"
        )
    print(f"figure_pdf={PAPER_FIGURE_PDF}")
    print(f"figure_png={PAPER_FIGURE_PNG}")


if __name__ == "__main__":
    main()
