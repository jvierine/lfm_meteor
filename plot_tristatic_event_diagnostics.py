#!/usr/bin/env python3
"""Generate article-quality diagnostics for fitted tri-static meteor events."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

try:
    import jcoord
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "for_yihui"))
    import jcoord

import sanya_opts as sc


DEFAULT_INPUT = "results/all_tristatic_ballistic_snr_weighted_v20260613b.h5"
DEFAULT_OUTPUT_DIR = "results/tristatic_event_diagnostics_v20260613b"
SITE_NAMES = ("Sanya", "Danzhou", "Wenchang")
SITE_COLORS = ("#1f4e79", "#b85c38", "#2f7d4b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_INPUT, help="Fitted tri-static HDF5 file.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for event figures.")
    p.add_argument("--event-id", action="append", help="Only plot the given event id; may be repeated.")
    p.add_argument("--max-events", type=int, default=None, help="Limit the number of events for test runs.")
    p.add_argument("--samples", type=int, default=72, help="Number of 95 percent trajectory samples to draw.")
    p.add_argument("--dpi", type=int, default=240, help="PNG resolution.")
    p.add_argument("--png", action=argparse.BooleanOptionalAction, default=True, help="Write per-event PNG files.")
    p.add_argument("--pdf", action=argparse.BooleanOptionalAction, default=True, help="Write a combined multi-page PDF.")
    return p.parse_args()


def enu_basis(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return east, north, up


ORIGIN_ECEF_M = np.asarray(jcoord.geodetic2ecef(float(sc.lat0[0]), float(sc.lon0[0]), float(sc.alt0[0]) * 1e3))
EAST, NORTH, UP = enu_basis(float(sc.lat0[0]), float(sc.lon0[0]))
ENU_MATRIX = np.vstack([EAST, NORTH, UP])


def ecef_to_enu_m(points_ecef_m: np.ndarray) -> np.ndarray:
    points = np.asarray(points_ecef_m, dtype=np.float64)
    return (points - ORIGIN_ECEF_M) @ ENU_MATRIX.T


def sigma_from_snr_db(snr_db: np.ndarray, sigma_floor_m: float, sigma_0_m: float) -> np.ndarray:
    snr_amp = 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 20.0)
    return np.sqrt(sigma_floor_m**2 + (sigma_0_m / np.maximum(snr_amp, 1e-6)) ** 2)


def stable_rng(event_id: str) -> np.random.Generator:
    digest = hashlib.sha256(event_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
    return np.random.default_rng(seed)


def component_std(values: np.ndarray, n_points: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == (3,):
        return np.tile(arr.reshape(1, 3), (n_points, 1))
    if arr.shape == (n_points, 3):
        return arr
    if arr.shape == (n_points,):
        return np.tile(arr.reshape(n_points, 1), (1, 3))
    if arr.size == 1:
        return np.full((n_points, 3), float(arr.reshape(-1)[0]))
    return np.full((n_points, 3), np.nan)


def project_component_std_to_enu(std_xyz_m: np.ndarray) -> np.ndarray:
    var_xyz = np.asarray(std_xyz_m, dtype=np.float64) ** 2
    return np.sqrt(var_xyz @ (ENU_MATRIX.T**2))


def symmetric_limits(*arrays: np.ndarray, pad_frac: float = 0.11) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if math.isclose(lo, hi):
        span = max(abs(lo), 1.0) * 0.1
    else:
        span = hi - lo
    return lo - pad_frac * span, hi + pad_frac * span


def format_pm(value: float, sigma: float, unit: str, scale: float = 1.0, digits: int = 3) -> str:
    val = value / scale
    sig = sigma / scale
    if not np.isfinite(sig):
        return f"{val:.{digits}f} {unit}"
    return f"{val:.{digits}f} +/- {sig:.{digits}f} {unit}"


def draw_geometry_panel(
    ax: plt.Axes,
    x_km: np.ndarray,
    y_km: np.ndarray,
    xerr_km: np.ndarray,
    yerr_km: np.ndarray,
    sample_x_km: list[np.ndarray],
    sample_y_km: list[np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    for sx, sy in zip(sample_x_km, sample_y_km):
        ax.plot(sx, sy, color="#7aa6c2", linewidth=0.8, alpha=0.10, zorder=1)
    ax.plot(x_km, y_km, color="#0f172a", linewidth=2.2, label="Best-fit trajectory", zorder=4)
    ax.errorbar(
        x_km,
        y_km,
        xerr=xerr_km,
        yerr=yerr_km,
        fmt="o",
        ms=4.2,
        mfc="#f8fafc",
        mec="#0f172a",
        mew=0.9,
        ecolor="#475569",
        elinewidth=0.85,
        capsize=2.2,
        alpha=0.92,
        label="Fitted positions, 95% bars",
        zorder=5,
    )
    ax.scatter(x_km[0], y_km[0], s=58, color="#d9480f", edgecolor="white", linewidth=0.9, zorder=6, label="Start")
    ax.scatter(x_km[-1], y_km[-1], s=58, color="#2f7d4b", edgecolor="white", linewidth=0.9, zorder=6, label="End")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)


def event_summary_text(
    event_id: str,
    g: h5py.Group,
    enu_km: np.ndarray,
    enu_std_m: np.ndarray,
    vel_enu_kms: np.ndarray,
    vel_enu_std_mps: np.ndarray,
) -> str:
    params = np.asarray(g["params"])
    param_std = np.asarray(g["parameter_std"])
    b_drag = 10.0 ** float(params[6])
    b_sigma = math.log(10.0) * b_drag * float(param_std[6])
    speed = np.asarray(g["speed_km_s"])
    alt = np.asarray(g["alt_km"])
    residuals = np.asarray(g["residuals_m"])
    sigma = np.asarray(g["sigma_m"])
    fit_count = min(residuals.shape[0], sigma.shape[0])
    residuals = residuals[:fit_count]
    sigma = sigma[:fit_count]
    normalized = residuals / np.maximum(sigma, 1e-9)
    rms = float(np.sqrt(np.mean(residuals**2)))
    nrms = float(np.sqrt(np.mean(normalized**2)))

    lines = [
        "Estimated parameters (1 sigma)",
        f"event: {event_id}",
        f"points: {len(speed):d}",
        f"RMS residual: {rms:.2f} m",
        f"normalized RMS: {nrms:.2f}",
        "",
        "Initial ENU state at Sanya origin",
        f"E0: {format_pm(enu_km[0, 0], 1e-3 * enu_std_m[0, 0], 'km')}",
        f"N0: {format_pm(enu_km[0, 1], 1e-3 * enu_std_m[0, 1], 'km')}",
        f"U0: {format_pm(enu_km[0, 2], 1e-3 * enu_std_m[0, 2], 'km')}",
        f"vE0: {format_pm(vel_enu_kms[0, 0], 1e-3 * vel_enu_std_mps[0, 0], 'km/s')}",
        f"vN0: {format_pm(vel_enu_kms[0, 1], 1e-3 * vel_enu_std_mps[0, 1], 'km/s')}",
        f"vU0: {format_pm(vel_enu_kms[0, 2], 1e-3 * vel_enu_std_mps[0, 2], 'km/s')}",
        "",
        f"speed start/end: {speed[0]:.3f} / {speed[-1]:.3f} km/s",
        f"height start/end: {alt[0]:.3f} / {alt[-1]:.3f} km",
        f"C_D A/m: {b_drag:.3g} +/- {b_sigma:.3g} m^2/kg",
        f"log10(C_D A/m): {params[6]:.3f} +/- {param_std[6]:.3f}",
        "",
        "Thin blue curves: 95% uncertainty samples",
        "Source: plot_tristatic_event_diagnostics.py",
    ]
    return "\n".join(lines)


def plot_event(
    h5: h5py.File,
    event_id: str,
    output_dir: Path,
    n_samples: int,
    dpi: int,
    write_png: bool,
    pdf: PdfPages | None,
) -> dict[str, str]:
    g = h5["points"][event_id]
    x_itrs_m = np.asarray(g["x_itrs_m"], dtype=np.float64)
    v_itrs_mps = np.asarray(g["v_itrs_mps"], dtype=np.float64)
    t_rel_s = np.asarray(g["t_rel_s"], dtype=np.float64)
    residuals_m = np.asarray(g["residuals_m"], dtype=np.float64)
    snr_db = np.asarray(g["snr_db"], dtype=np.float64)
    sigma_m = np.asarray(g["sigma_m"], dtype=np.float64)
    fit_count = min(residuals_m.shape[0], snr_db.shape[0], sigma_m.shape[0], t_rel_s.shape[0])
    residuals_m = residuals_m[:fit_count]
    snr_db = snr_db[:fit_count]
    sigma_m = sigma_m[:fit_count]
    t_rel_s = t_rel_s[:fit_count]

    n_points = x_itrs_m.shape[0]
    enu_km = ecef_to_enu_m(x_itrs_m) / 1e3
    vel_enu_kms = (v_itrs_mps @ ENU_MATRIX.T) / 1e3

    pos_std_xyz_m = component_std(np.asarray(g["position_std_m"]), n_points)
    vel_std_xyz_mps = component_std(np.asarray(g["velocity_std_mps"]), n_points)
    pos_std_enu_m = project_component_std_to_enu(pos_std_xyz_m)
    vel_std_enu_mps = project_component_std_to_enu(vel_std_xyz_mps)
    pos_95_enu_km = 1.96 * pos_std_enu_m / 1e3

    rng = stable_rng(event_id)
    sample_enu_km: list[np.ndarray] = []
    for _ in range(max(0, n_samples)):
        draw_xyz = rng.normal(size=x_itrs_m.shape) * (1.96 * pos_std_xyz_m)
        sample_enu_km.append(ecef_to_enu_m(x_itrs_m + draw_xyz) / 1e3)

    plt.rcParams.update(
        {
            "font.size": 12.5,
            "axes.titlesize": 14.5,
            "axes.labelsize": 13.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 9.8,
            "figure.titlesize": 16.5,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
        }
    )
    fig = plt.figure(figsize=(15.9, 10.6), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1.15, 1.15, 0.95], height_ratios=[1.0, 0.86, 0.92])

    ax_en = fig.add_subplot(gs[0:2, 0])
    ax_eu = fig.add_subplot(gs[0:2, 1])
    ax_snr = fig.add_subplot(gs[2, 0])
    ax_time = fig.add_subplot(gs[2, 1])
    ax_text = fig.add_subplot(gs[:, 2])

    draw_geometry_panel(
        ax_en,
        enu_km[:, 0],
        enu_km[:, 1],
        pos_95_enu_km[:, 0],
        pos_95_enu_km[:, 1],
        [s[:, 0] for s in sample_enu_km],
        [s[:, 1] for s in sample_enu_km],
        "East from Sanya (km)",
        "North from Sanya (km)",
        "East-west trajectory in map view",
    )
    draw_geometry_panel(
        ax_eu,
        enu_km[:, 0],
        enu_km[:, 2],
        pos_95_enu_km[:, 0],
        pos_95_enu_km[:, 2],
        [s[:, 0] for s in sample_enu_km],
        [s[:, 2] for s in sample_enu_km],
        "East from Sanya (km)",
        "Up from Sanya (km)",
        "East-west trajectory in height",
    )

    ax_en.set_xlim(*symmetric_limits(enu_km[:, 0], *[s[:, 0] for s in sample_enu_km]))
    ax_en.set_ylim(*symmetric_limits(enu_km[:, 1], *[s[:, 1] for s in sample_enu_km]))
    ax_eu.set_xlim(*symmetric_limits(enu_km[:, 0], *[s[:, 0] for s in sample_enu_km]))
    ax_eu.set_ylim(*symmetric_limits(enu_km[:, 2], *[s[:, 2] for s in sample_enu_km]))
    ax_en.legend(loc="best", frameon=True, framealpha=0.92)

    for site_idx, (site, color) in enumerate(zip(SITE_NAMES, SITE_COLORS)):
        ax_snr.scatter(
            snr_db[:, site_idx],
            np.abs(residuals_m[:, site_idx]),
            s=36,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
            label=site,
        )
    sigma_floor_m = float(h5.attrs["sigma_floor_m"])
    sigma_0_m = float(h5.attrs["sigma_0_m"])
    snr_min = float(np.nanmin(snr_db)) - 1.0
    snr_max = float(np.nanmax(snr_db)) + 1.0
    snr_grid = np.linspace(snr_min, snr_max, 250)
    sigma_curve = sigma_from_snr_db(snr_grid, sigma_floor_m, sigma_0_m)
    ax_snr.plot(snr_grid, sigma_curve, color="#111827", linewidth=2.2, label="1 sigma SNR model")
    ax_snr.plot(snr_grid, 1.96 * sigma_curve, color="#111827", linewidth=1.4, linestyle="--", alpha=0.75, label="95% model")
    ax_snr.set_xlabel("SNR (dB)")
    ax_snr.set_ylabel("|path residual| (m)")
    ax_snr.set_title("Residual magnitude vs. SNR error model")
    ax_snr.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.8)
    ax_snr.legend(loc="best", frameon=True, framealpha=0.92, ncol=2)
    y_hi = float(np.nanmax([np.nanmax(np.abs(residuals_m)), np.nanmax(1.96 * sigma_curve)]))
    ax_snr.set_ylim(0.0, max(5.0, 1.12 * y_hi))

    res_rms_t = np.sqrt(np.mean(residuals_m**2, axis=1))
    sigma_rms_t = np.sqrt(np.mean(sigma_m**2, axis=1))
    ax_time.plot(t_rel_s * 1e3, res_rms_t, color="#0f172a", linewidth=2.2, label="RMS residual magnitude")
    ax_time.plot(t_rel_s * 1e3, sigma_rms_t, color="#d9480f", linewidth=2.0, label="RMS 1 sigma model")
    ax_time.fill_between(t_rel_s * 1e3, 0.0, 1.96 * sigma_rms_t, color="#f59f00", alpha=0.16, label="95% model band")
    ax_time.set_xlabel("Time from first fitted sample (ms)")
    ax_time.set_ylabel("Path residual magnitude (m)")
    ax_time.set_title("Temporal residual diagnostic")
    ax_time.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.8)
    ax_time.legend(loc="best", frameon=True, framealpha=0.92)
    ax_time.set_ylim(0.0, max(5.0, 1.12 * float(np.nanmax([np.nanmax(res_rms_t), np.nanmax(1.96 * sigma_rms_t)]))))

    ax_text.axis("off")
    ax_text.set_facecolor("#f8fafc")
    summary = event_summary_text(event_id, g, enu_km, pos_std_enu_m, vel_enu_kms, vel_std_enu_mps)
    ax_text.text(
        0.04,
        0.985,
        summary,
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        family="DejaVu Sans Mono",
        fontsize=10.8,
        linespacing=1.34,
        bbox=dict(boxstyle="round,pad=0.65", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{event_id}_diagnostic.png"
    if write_png:
        fig.savefig(png_path, dpi=dpi, facecolor="white")
    if pdf is not None:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)

    return {
        "event_id": event_id,
        "png": str(png_path if write_png else ""),
        "n_points": str(n_points),
        "rms_residual_m": f"{float(np.sqrt(np.mean(residuals_m**2))):.6f}",
        "normalized_rms": f"{float(np.sqrt(np.mean((residuals_m / np.maximum(sigma_m, 1e-9)) ** 2))):.6f}",
    }


def iter_event_ids(h5: h5py.File, requested: list[str] | None, max_events: int | None) -> list[str]:
    ids = sorted(h5["points"].keys())
    if requested:
        wanted = set(requested)
        missing = sorted(wanted.difference(ids))
        if missing:
            raise KeyError(f"Requested event ids not found: {', '.join(missing)}")
        ids = [event_id for event_id in ids if event_id in wanted]
    if max_events is not None:
        ids = ids[: max(0, max_events)]
    return ids


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.csv"
    pdf_path = output_dir / "all_tristatic_event_diagnostics.pdf"

    with h5py.File(input_path, "r") as h5:
        event_ids = iter_event_ids(h5, args.event_id, args.max_events)
        if not event_ids:
            raise RuntimeError("No event ids selected.")

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_ctx = PdfPages(pdf_path) if args.pdf else None
        rows = []
        try:
            for idx, event_id in enumerate(event_ids, start=1):
                print(f"[{idx:03d}/{len(event_ids):03d}] plotting {event_id}", flush=True)
                rows.append(plot_event(h5, event_id, output_dir, args.samples, args.dpi, args.png, pdf_ctx))
        finally:
            if pdf_ctx is not None:
                pdf_ctx.close()

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["event_id", "png", "n_points", "rms_residual_m", "normalized_rms"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} event diagnostics")
    if args.png:
        print(f"png directory: {output_dir}")
    if args.pdf:
        print(f"combined pdf: {pdf_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
