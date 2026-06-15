#!/usr/bin/env python3
"""Fit receiver temperature from raw noise-floor power and pygdsm sky noise."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import plot_pygdsm_station_sky_noise as sky_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--system-noise-h5",
        default="/Users/jvi019/src/lfm_meteor/results/sanya_4mhz_system_noise_power_100pulse.h5",
        help="Reduced low-rate 100-pulse system-noise HDF5 product.",
    )
    p.add_argument("--frequency-mhz", type=float, default=450.0)
    p.add_argument("--cadence-min", type=float, default=2.5)
    p.add_argument("--beam-radius-deg", type=float, default=5.0)
    p.add_argument("--beam-grid-step-deg", type=float, default=0.1)
    p.add_argument("--remote-effective-aperture-scale", type=float, default=1.0)
    p.add_argument("--floor-quantile", type=float, default=0.5)
    p.add_argument("--danzhou-floor-quantile", type=float, default=0.10)
    p.add_argument("--wenchang-floor-quantile", type=float, default=0.10)
    p.add_argument("--residual-mad-sigma", type=float, default=4.0)
    p.add_argument("--fit-iterations", type=int, default=4)
    p.add_argument(
        "--use-all-wenchang-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all finite Wenchang samples in the fit so the galactic-plane pass constrains the scale.",
    )
    p.add_argument(
        "--use-all-danzhou-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all finite Danzhou local-floor samples in the fit.",
    )
    p.add_argument(
        "--output-dir",
        default="/Users/jvi019/src/sanya_tristatic_paper/memos/figures",
    )
    p.add_argument(
        "--basename",
        default="memo20_system_noise_floor_gdsm_trec_fit_450mhz",
    )
    return p.parse_args()


def load_low_rate(path: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h:
        names = [x.decode() if hasattr(x, "decode") else str(x) for x in h["site_names"][:]]
        station_id = h["bins/station_id"][:]
        time_ns = h["bins/time_utc_mid_ns"][:]
        power = h["bins/noise_power_mean_raw_voltage"][:].astype(np.float64)
    return names, station_id, time_ns, power


def bin_quantile(
    time_ns: np.ndarray,
    values: np.ndarray,
    bin_edges_ns: np.ndarray,
    quantile: float,
    good: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("floor quantile must be between 0 and 1")
    centers = (bin_edges_ns[:-1] + np.diff(bin_edges_ns) // 2).astype(np.int64)
    binned = np.full(len(centers), np.nan, dtype=np.float64)
    finite = np.isfinite(time_ns) & np.isfinite(values)
    if good is not None:
        finite &= good
    idx = np.searchsorted(bin_edges_ns, time_ns[finite], side="right") - 1
    valid = (idx >= 0) & (idx < len(centers))
    idx = idx[valid]
    vals = values[finite][valid]
    for bi in np.unique(idx):
        binned[bi] = np.nanquantile(vals[idx == bi], quantile)
    ok = np.isfinite(binned)
    return centers[ok], binned[ok]


def ns_to_mpl(ns: np.ndarray) -> np.ndarray:
    dt = ns.astype("datetime64[ns]").astype("datetime64[ms]").astype(object)
    return mdates.date2num(dt)


def fit_raw_power_to_sky_temperature(
    power: np.ndarray,
    t_sky: np.ndarray,
    residual_mad_sigma: float,
    iterations: int,
    clip_residuals: bool,
) -> tuple[float, float, np.ndarray, float, float]:
    good = np.isfinite(power) & np.isfinite(t_sky)
    if np.count_nonzero(good) < 3:
        raise RuntimeError("Not enough finite samples for T_rec fit")
    if iterations < 1:
        raise ValueError("--fit-iterations must be at least 1")
    if residual_mad_sigma <= 0.0:
        raise ValueError("--residual-mad-sigma must be positive")

    fit_mask = good.copy()
    slope = intercept = np.nan
    robust_sigma_power = np.nan
    n_iter = iterations if clip_residuals else 1
    for _ in range(n_iter):
        slope, intercept = np.polyfit(t_sky[fit_mask], power[fit_mask], deg=1)
        residual = power - (slope * t_sky + intercept)
        med = np.nanmedian(residual[fit_mask])
        mad = np.nanmedian(np.abs(residual[fit_mask] - med))
        robust_sigma_power = 1.4826 * mad
        if not clip_residuals:
            break
        if not np.isfinite(robust_sigma_power) or robust_sigma_power <= 0.0:
            break
        next_mask = good & (np.abs(residual - med) <= residual_mad_sigma * robust_sigma_power)
        if np.array_equal(next_mask, fit_mask) or np.count_nonzero(next_mask) < 3:
            break
        fit_mask = next_mask

    if not np.isfinite(slope) or slope <= 0.0:
        raise RuntimeError(f"Invalid fitted calibration slope: {slope}")
    t_rec = intercept / slope
    band_half_width_k = 1.96 * robust_sigma_power / slope
    return float(slope), float(t_rec), fit_mask, float(robust_sigma_power / slope), float(band_half_width_k)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    names, station_id, time_ns, power = load_low_rate(args.system_noise_h5)
    start_ns = int(np.nanmin(time_ns))
    stop_ns = int(np.nanmax(time_ns))
    step_ns = int(round(args.cadence_min * 60.0 * 1e9))
    bin_edges_ns = np.arange(start_ns, stop_ns + step_ns, step_ns, dtype=np.int64)
    if bin_edges_ns[-1] < stop_ns:
        bin_edges_ns = np.append(bin_edges_ns, stop_ns)
    else:
        bin_edges_ns[-1] = stop_ns

    times = sky_model.make_times(start_ns, stop_ns, args.cadence_min)
    sky_x_ns = np.asarray(times.unix * 1e9, dtype=np.float64)
    sky_x_mpl = sky_model.time_to_mpl(times)
    gsm = sky_model.pygdsm.GlobalSkyModel(freq_unit="MHz", include_cmb=False)

    colors = {
        "Sanya": "#1f77b4",
        "Danzhou": "#2ca02c",
        "Wenchang": "#d62728",
    }

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True)
    fit_rows = []

    for ax, name in zip(axes, names):
        sid = names.index(name)
        m = station_id == sid
        if name == "Wenchang":
            floor_quantile = args.wenchang_floor_quantile
        elif name == "Danzhou":
            floor_quantile = args.danzhou_floor_quantile
        else:
            floor_quantile = args.floor_quantile
        floor_good = None
        floor_ns, floor_power_binned = bin_quantile(time_ns[m], power[m], bin_edges_ns, floor_quantile, floor_good)
        t_sky, _n_samples = sky_model.station_sky_temperature(
            gsm,
            name,
            times,
            args.frequency_mhz,
            False,
            args.beam_radius_deg,
            args.beam_grid_step_deg,
            args.remote_effective_aperture_scale,
        )
        t_sky_at_floor = np.interp(floor_ns.astype(np.float64), sky_x_ns, t_sky)
        clip_residuals = not (
            (name == "Wenchang" and args.use_all_wenchang_samples)
            or (name == "Danzhou" and args.use_all_danzhou_samples)
        )
        calibration_power_per_k, t_rec, fit_mask, t_rec_std, band_half_width_k = fit_raw_power_to_sky_temperature(
            floor_power_binned,
            t_sky_at_floor,
            args.residual_mad_sigma,
            args.fit_iterations,
            clip_residuals,
        )
        floor_tsys = floor_power_binned / calibration_power_per_k
        fitted_tsys = t_sky + t_rec
        fit_rows.append(
            (
                name,
                t_rec,
                t_rec_std,
                float(np.nanmedian(floor_tsys)),
                calibration_power_per_k,
                floor_quantile,
                int(np.count_nonzero(fit_mask)),
                int(fit_mask.size),
            )
        )

        color = colors.get(name, f"C{sid}")
        ax.plot(sky_x_mpl, t_sky, color="0.45", lw=1.3, ls="--", label=r"pygdsm $T_{\mathrm{sky}}$")
        ax.fill_between(
            sky_x_mpl,
            fitted_tsys - band_half_width_k,
            fitted_tsys + band_half_width_k,
            color="0.15",
            alpha=0.24,
            linewidth=0,
            label="95% fit band",
            zorder=1,
        )
        ax.plot(sky_x_mpl, fitted_tsys - band_half_width_k, color="0.15", lw=0.9, alpha=0.75, ls=":")
        ax.plot(sky_x_mpl, fitted_tsys + band_half_width_k, color="0.15", lw=0.9, alpha=0.75, ls=":")
        ax.plot(sky_x_mpl, fitted_tsys, color="0.02", lw=2.4, alpha=0.95, label=r"$T_{\mathrm{sky}}+T_{\mathrm{rec}}$", zorder=3)
        rejected = ~fit_mask
        ax.scatter(ns_to_mpl(floor_ns[rejected]), floor_tsys[rejected], s=8, color=color, alpha=0.12, linewidths=0)
        ax.scatter(
            ns_to_mpl(floor_ns[fit_mask]),
            floor_tsys[fit_mask],
            s=18,
            color=color,
            alpha=0.90,
            linewidths=0.25,
            edgecolors="white",
            label="measured floor",
            zorder=4,
        )
        ax.set_ylabel(r"$T$ (K)")
        ax.set_title(name)
        ax.grid(True, color="0.88", lw=0.6)
        ax.set_axisbelow(True)
        ax.text(
            0.015,
            0.92,
            rf"$\hat T_{{\mathrm{{rec}}}}={t_rec:.1f}$ K" + "\n" + rf"median $T_{{\mathrm{{sys}}}}={np.nanmedian(floor_tsys):.1f}$ K",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.75", "boxstyle": "round,pad=0.25", "alpha": 0.85},
        )

    axes[0].legend(loc="lower left", ncol=3, frameon=False)
    axes[-1].set_xlabel("UTC time")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    pdf = outdir / f"{args.basename}.pdf"
    png = outdir / f"{args.basename}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)

    for name, t_rec, t_rec_std, med_tsys, calibration_power_per_k, floor_quantile, n_fit, n_total in fit_rows:
        print(
            f"{name}: T_rec={t_rec:.3f} K, robust_sigma={t_rec_std:.3f} K, "
            f"median_Tsys={med_tsys:.3f} K, power_per_K={calibration_power_per_k:.6e}, "
            f"floor_quantile={floor_quantile:.3f}, fit_samples={n_fit}/{n_total}"
        )
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
