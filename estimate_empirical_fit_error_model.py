#!/usr/bin/env python3
"""Estimate empirical delay and Doppler residual scales from joint fits."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import fit_event_joint_delay_doppler_fft as joint_fit
import fit_gcrs_trajectories_lfm_ambiguity as gfit


DEFAULT_CATALOG_DIR = Path("results/tristatic_pulse200_local_20260629")
DEFAULT_OUTPUT_BASE = Path("results/empirical_joint_fit_error_model_v20260629a")
SITE_LABELS = ("Sanya", "Danzhou", "Wenchang")
STUDENT_T_NU = 4.0


def truthy(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def model_sigma(x, snr_db, beam_offset_deg):
    floor, snr_coeff, beam_coeff = np.exp(np.asarray(x, dtype=np.float64))
    snr_linear_amplitude = 10.0 ** (-np.asarray(snr_db, dtype=np.float64) / 20.0)
    return np.sqrt(
        floor**2.0
        + (snr_coeff * snr_linear_amplitude) ** 2.0
        + (beam_coeff * np.asarray(beam_offset_deg, dtype=np.float64)) ** 2.0
    )


def robust_nll(x, residual, snr_db, beam_offset_deg, nu=STUDENT_T_NU):
    sigma = model_sigma(x, snr_db, beam_offset_deg)
    z2 = (np.asarray(residual, dtype=np.float64) / sigma) ** 2.0
    return float(np.sum(np.log(sigma) + 0.5 * (nu + 1.0) * np.log1p(z2 / nu)))


def fit_scale_model(residual, snr_db, beam_offset_deg):
    residual = np.asarray(residual, dtype=np.float64)
    snr_db = np.asarray(snr_db, dtype=np.float64)
    beam_offset_deg = np.asarray(beam_offset_deg, dtype=np.float64)
    good = np.isfinite(residual) & np.isfinite(snr_db) & np.isfinite(beam_offset_deg)
    residual = residual[good]
    snr_db = snr_db[good]
    beam_offset_deg = beam_offset_deg[good]
    if residual.size < 20:
        raise RuntimeError("not enough residual samples")

    robust_scale = np.nanmedian(np.abs(residual)) / 0.67448975
    robust_scale = max(float(robust_scale), 1e-6)
    snr_guess = robust_scale * 10.0 ** (np.nanmedian(snr_db) / 20.0)
    beam_guess = robust_scale / max(float(np.nanmedian(beam_offset_deg)), 0.2)
    x0 = np.log([0.35 * robust_scale, snr_guess, 0.35 * beam_guess])
    result = minimize(
        robust_nll,
        x0,
        args=(residual, snr_db, beam_offset_deg),
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-6},
    )
    x = result.x if result.success else x0
    coeff = np.exp(x)
    sigma = model_sigma(x, snr_db, beam_offset_deg)
    standardized = residual / sigma
    return {
        "coefficients": coeff,
        "success": bool(result.success),
        "message": str(result.message),
        "n": int(residual.size),
        "median_abs_residual": float(np.nanmedian(np.abs(residual))),
        "rms_residual": float(np.sqrt(np.nanmean(residual**2.0))),
        "standardized_median_abs": float(np.nanmedian(np.abs(standardized))),
        "standardized_rms": float(np.sqrt(np.nanmean(standardized**2.0))),
        "residual": residual,
        "snr_db": snr_db,
        "beam_offset_deg": beam_offset_deg,
        "sigma": sigma,
    }


def collect_samples(catalog_dir: Path):
    delay_residual = []
    delay_snr = []
    delay_beam_offset = []
    delay_station = []
    doppler_residual_mps = []
    doppler_snr = []
    doppler_beam_offset = []
    doppler_station = []
    event_id = []
    event_bad = []

    for path in sorted(glob.glob(str(catalog_dir / "joint_delay_doppler_fft_tri_*.h5"))):
        with h5py.File(path, "r") as h:
            this_event_id = str(h.attrs.get("event_id", Path(path).stem))
            j = h["joint_fit"]
            if truthy(j.attrs.get("bad_fit_detected", False)):
                event_bad.append(this_event_id)
                continue
            snr_db = np.asarray(h["fft_observations"]["fft_snr_db"][:], dtype=np.float64)
            beam_east_deg, beam_north_deg = joint_fit.sanya_beam_offsets_deg(np.asarray(j["x_itrs_m"][:], dtype=np.float64))
            beam_offset_deg = np.sqrt(beam_east_deg**2.0 + beam_north_deg**2.0)
            beam_matrix = np.broadcast_to(beam_offset_deg[:, None], snr_db.shape)
            station_matrix = np.broadcast_to(np.arange(3, dtype=np.int16)[None, :], snr_db.shape)

            path_keep = np.asarray(j["path_keep"][:], dtype=bool)
            path_resid_m = np.asarray(j["path_residuals_m"][:], dtype=np.float64)
            good_delay = path_keep & np.isfinite(path_resid_m) & np.isfinite(snr_db) & np.isfinite(beam_matrix)
            delay_residual.append(path_resid_m[good_delay])
            delay_snr.append(snr_db[good_delay])
            delay_beam_offset.append(beam_matrix[good_delay])
            delay_station.append(station_matrix[good_delay])

            fft_keep = np.asarray(j["fft_keep"][:], dtype=bool)
            path_rate_resid_mps = np.asarray(j["path_rate_residuals_mps"][:], dtype=np.float64)
            good_doppler = fft_keep & np.isfinite(path_rate_resid_mps) & np.isfinite(snr_db) & np.isfinite(beam_matrix)
            doppler_residual_mps.append(path_rate_resid_mps[good_doppler])
            doppler_snr.append(snr_db[good_doppler])
            doppler_beam_offset.append(beam_matrix[good_doppler])
            doppler_station.append(station_matrix[good_doppler])
            event_id.append(this_event_id)

    return {
        "event_id": np.asarray(event_id, dtype=object),
        "event_bad": np.asarray(event_bad, dtype=object),
        "delay_residual_m": np.concatenate(delay_residual),
        "delay_snr_db": np.concatenate(delay_snr),
        "delay_beam_offset_deg": np.concatenate(delay_beam_offset),
        "delay_station": np.concatenate(delay_station),
        "doppler_residual_mps": np.concatenate(doppler_residual_mps),
        "doppler_snr_db": np.concatenate(doppler_snr),
        "doppler_beam_offset_deg": np.concatenate(doppler_beam_offset),
        "doppler_station": np.concatenate(doppler_station),
    }


def binned_abs_stats(residual, snr_db, beam_offset_deg):
    snr_edges = np.asarray([15.0, 20.0, 25.0, 30.0, 40.0, 60.0], dtype=np.float64)
    beam_edges = np.asarray([0.0, 0.4, 0.8, 1.2, 2.0, 3.0], dtype=np.float64)
    shape = (len(snr_edges) - 1, len(beam_edges) - 1)
    n = np.zeros(shape, dtype=np.int64)
    median_abs = np.full(shape, np.nan, dtype=np.float64)
    rms = np.full(shape, np.nan, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    snr_db = np.asarray(snr_db, dtype=np.float64)
    beam_offset_deg = np.asarray(beam_offset_deg, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            mask = (
                (snr_db >= snr_edges[i])
                & (snr_db < snr_edges[i + 1])
                & (beam_offset_deg >= beam_edges[j])
                & (beam_offset_deg < beam_edges[j + 1])
            )
            n[i, j] = int(np.count_nonzero(mask))
            if n[i, j] > 0:
                values = residual[mask]
                median_abs[i, j] = float(np.nanmedian(np.abs(values)))
                rms[i, j] = float(np.sqrt(np.nanmean(values**2.0)))
    return snr_edges, beam_edges, n, median_abs, rms


def write_h5(output_base: Path, catalog_dir: Path, samples: dict, delay_fit: dict, doppler_fit: dict):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(output_base) + ".h5", "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["catalog_dir"] = str(catalog_dir)
        h.attrs["model"] = "sigma = sqrt(floor^2 + (snr_coeff*10^(-SNR_dB/20))^2 + (beam_coeff*theta_deg)^2)"
        h.attrs["likelihood"] = f"Student-t negative log likelihood, nu={STUDENT_T_NU:g}"
        h.attrs["doppler_units"] = "equivalent total-path-rate residual, m/s"
        h.attrs["beat_hz_conversion"] = "sigma_beat_hz = sigma_path_rate_mps / radar_wavelength_m"
        h.attrs["radar_wavelength_m"] = float(gfit.RADAR_WAVELENGTH_M)
        h["event_id"] = samples["event_id"].astype(string_dtype)
        h["event_bad"] = samples["event_bad"].astype(string_dtype)
        for prefix, fit in (("delay", delay_fit), ("doppler", doppler_fit)):
            g = h.create_group(prefix)
            g.attrs["n"] = fit["n"]
            g.attrs["success"] = fit["success"]
            g.attrs["message"] = fit["message"]
            g.attrs["floor"] = fit["coefficients"][0]
            g.attrs["snr_coeff"] = fit["coefficients"][1]
            g.attrs["beam_coeff_per_deg"] = fit["coefficients"][2]
            g.attrs["median_abs_residual"] = fit["median_abs_residual"]
            g.attrs["rms_residual"] = fit["rms_residual"]
            g.attrs["standardized_median_abs"] = fit["standardized_median_abs"]
            g.attrs["standardized_rms"] = fit["standardized_rms"]
            for key in ("residual", "snr_db", "beam_offset_deg", "sigma"):
                g[key] = fit[key]
            snr_edges, beam_edges, n, median_abs, rms = binned_abs_stats(
                fit["residual"],
                fit["snr_db"],
                fit["beam_offset_deg"],
            )
            g["bin_snr_edges_db"] = snr_edges
            g["bin_beam_offset_edges_deg"] = beam_edges
            g["bin_n"] = n
            g["bin_median_abs_residual"] = median_abs
            g["bin_rms_residual"] = rms


def plot_model(output_base: Path, delay_fit: dict, doppler_fit: dict):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    snr_grid = np.linspace(15.0, 55.0, 250)
    beam_examples = [0.0, 0.5, 1.0, 1.5]
    with plt.rc_context(
        {
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "savefig.dpi": 300,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.45), constrained_layout=True)
        for ax, fit, ylabel in (
            (axes[0], delay_fit, "Delay residual scale (m)"),
            (axes[1], doppler_fit, "Doppler path-rate residual scale (m/s)"),
        ):
            x = np.log(fit["coefficients"])
            sample = np.linspace(0, len(fit["residual"]) - 1, min(4500, len(fit["residual"]))).astype(int)
            ax.scatter(
                fit["snr_db"][sample],
                np.abs(fit["residual"][sample]),
                c=fit["beam_offset_deg"][sample],
                s=5,
                cmap="viridis",
                alpha=0.16,
                edgecolors="none",
            )
            for theta in beam_examples:
                sigma = model_sigma(x, snr_grid, np.full_like(snr_grid, theta))
                ax.plot(snr_grid, sigma, lw=1.6, label=rf"$\theta={theta:.1f}^\circ$")
            ax.set_yscale("log")
            ax.set_xlabel("Matched-filter SNR (dB)")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="upper right", frameon=True, framealpha=0.86)
        fig.savefig(str(output_base) + ".png", bbox_inches="tight")
        fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
        plt.close(fig)


def print_summary(delay_fit: dict, doppler_fit: dict):
    d = delay_fit["coefficients"]
    v = doppler_fit["coefficients"]
    print("Empirical robust scatter model:")
    print("  sigma = sqrt(floor^2 + (snr_coeff * 10^(-SNR_dB/20))^2 + (beam_coeff * theta_deg)^2)")
    print(f"delay_m: floor={d[0]:.3g}, snr_coeff={d[1]:.3g} m, beam_coeff={d[2]:.3g} m/deg, n={delay_fit['n']}")
    print(
        f"doppler_mps: floor={v[0]:.3g}, snr_coeff={v[1]:.3g} m/s, "
        f"beam_coeff={v[2]:.3g} m/s/deg, n={doppler_fit['n']}"
    )
    print(
        f"doppler_hz coefficients: floor={v[0] / gfit.RADAR_WAVELENGTH_M:.3g}, "
        f"snr_coeff={v[1] / gfit.RADAR_WAVELENGTH_M:.3g} Hz, "
        f"beam_coeff={v[2] / gfit.RADAR_WAVELENGTH_M:.3g} Hz/deg"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    args = parser.parse_args()

    samples = collect_samples(args.catalog_dir)
    delay_fit = fit_scale_model(
        samples["delay_residual_m"],
        samples["delay_snr_db"],
        samples["delay_beam_offset_deg"],
    )
    doppler_fit = fit_scale_model(
        samples["doppler_residual_mps"],
        samples["doppler_snr_db"],
        samples["doppler_beam_offset_deg"],
    )
    write_h5(args.output_base, args.catalog_dir, samples, delay_fit, doppler_fit)
    plot_model(args.output_base, delay_fit, doppler_fit)
    print_summary(delay_fit, doppler_fit)
    print(str(args.output_base) + ".h5")
    print(str(args.output_base) + ".png")
    print(str(args.output_base) + ".pdf")


if __name__ == "__main__":
    main()
