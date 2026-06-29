#!/usr/bin/env python3
"""Estimate sky-averaged Sanya meteor flux from Sun-centered radiants.

The estimator follows the radiant-visibility convention used by
plot_sun_centered_ecliptic_radiants.py.  Each observed radiant is assigned to a
four-member symmetry family: reflection about the ecliptic and reflection about
the apex meridian.  The event contribution is divided by the sum of the
zenith-corrected effective observing time of the family.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import plot_sun_centered_ecliptic_radiants as radiants


DEFAULT_RADIANT_H5 = Path("results/sun_centered_ecliptic_radiants.h5")
DEFAULT_MASS_H5 = Path("results/joint_fft_mass_distribution_v20260618a.h5")
DEFAULT_OUTPUT_BASE = Path("results/sky_averaged_flux_v20260624a")
DEFAULT_PAPER_MEMO_FIGURE_DIR = Path("/Users/jvi019/src/sanya_tristatic_paper/memos/figures")
DEFAULT_PAPER_TABLE_DIR = Path("/Users/jvi019/src/sanya_tristatic_paper/tables")

APEX_MERIDIAN_DEG = 270.0
DEFAULT_BEAM_FULL_WIDTH_DEG = 0.9
DEFAULT_COMMON_VOLUME_ALTITUDE_KM = 95.0
DEFAULT_POINTING_ELEVATION_DEG = 75.0
EARTH_RADIUS_KM = 6371.0


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def beam_horizontal_area_km2(
    full_width_deg: float,
    common_volume_altitude_km: float = DEFAULT_COMMON_VOLUME_ALTITUDE_KM,
    pointing_elevation_deg: float = DEFAULT_POINTING_ELEVATION_DEG,
) -> float:
    """Approximate Sanya 3-dB beam footprint area on a horizontal meteor layer."""
    elevation_rad = np.deg2rad(pointing_elevation_deg)
    slant_range_km = common_volume_altitude_km / np.sin(elevation_rad)
    half_width_rad = np.deg2rad(0.5 * full_width_deg)
    radius_perpendicular_km = slant_range_km * np.tan(half_width_rad)
    area_perpendicular_km2 = np.pi * radius_perpendicular_km**2
    return float(area_perpendicular_km2 / np.sin(elevation_rad))


def load_radiants(path: Path) -> dict[str, np.ndarray | str]:
    with h5py.File(path, "r") as h:
        return {
            "event_id": decode_strings(h["event_id"][:]),
            "lambda_minus_sun_deg": np.asarray(h["lambda_minus_sun_deg"][:], dtype=np.float64),
            "beta_ecliptic_deg": np.asarray(h["beta_ecliptic_deg"][:], dtype=np.float64),
            "speed_km_s": np.asarray(h["speed_km_s"][:], dtype=np.float64),
            "t0_ns": np.asarray(h["t0_ns"][:], dtype=np.int64),
            "fixed_ecliptic_equinox_utc": str(h.attrs["fixed_ecliptic_equinox_utc"]),
        }


def load_mass_rows(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h:
        return {
            "event_id": decode_strings(h["event_id"][:]),
            "selected": np.asarray(h["selected"][:], dtype=bool),
            "initial_mass_kg": np.asarray(h["initial_mass_kg"][:], dtype=np.float64),
            "mass_95_lo_kg": np.asarray(h["mass_95_lo_kg"][:], dtype=np.float64),
            "mass_95_hi_kg": np.asarray(h["mass_95_hi_kg"][:], dtype=np.float64),
            "initial_radius_m": np.asarray(h["initial_radius_m"][:], dtype=np.float64),
        }


def mirror_family(lambda_minus_sun_deg: np.ndarray, beta_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lambda_minus_sun_deg, dtype=np.float64)
    beta = np.asarray(beta_deg, dtype=np.float64)
    lon_apex_mirror = radiants.wrap360(2.0 * APEX_MERIDIAN_DEG - lon)
    family_lon = np.vstack([lon, lon_apex_mirror, lon, lon_apex_mirror]).T
    family_beta = np.vstack([beta, beta, -beta, -beta]).T
    return family_lon, family_beta


def exposure_for_family(
    family_lon_deg: np.ndarray,
    family_beta_deg: np.ndarray,
    t0_ns: np.ndarray,
    fixed_equinox_iso: str,
) -> tuple[np.ndarray, np.ndarray]:
    _t0, _t1, sample_times = radiants.observation_times(t0_ns)
    # The grid function already contains the exact transformation and zenithal
    # correction used in the paper figure.  We compute a fine grid once and use
    # nearest-neighbour interpolation here; the memo records this approximation.
    plot_lon_mesh, beta_mesh, _visible_h, effective_h = radiants.radiant_visibility_grid(
        sample_times,
        fixed_equinox_iso,
        n_lon=721,
        n_lat=361,
    )
    sun_centered_mesh = radiants.plot_longitude_to_sun_centered_deg(plot_lon_mesh[0, :])
    beta_grid = beta_mesh[:, 0]

    family_exposure_h = np.zeros_like(family_lon_deg, dtype=np.float64)
    for j in range(family_lon_deg.shape[1]):
        plot_lon = radiants.centered_plot_longitude_deg(family_lon_deg[:, j])
        lon_idx = np.clip(np.rint((plot_lon - plot_lon_mesh[0, 0]) / (plot_lon_mesh[0, 1] - plot_lon_mesh[0, 0])).astype(int), 0, plot_lon_mesh.shape[1] - 1)
        beta_idx = np.clip(np.rint((family_beta_deg[:, j] - beta_grid[0]) / (beta_grid[1] - beta_grid[0])).astype(int), 0, beta_grid.size - 1)
        family_exposure_h[:, j] = effective_h[beta_idx, lon_idx]

    # Keep the grid longitude vector as a diagnostic; it is useful when checking
    # whether mirrored radiants land in the intended Sun-centered convention.
    _ = sun_centered_mesh
    return family_exposure_h, effective_h


def event_weights_per_km2_day(family_exposure_h: np.ndarray, area_km2: float) -> np.ndarray:
    family_exposure_sum_h = np.sum(family_exposure_h, axis=1)
    valid = family_exposure_sum_h > 0.0
    weights = np.full(family_exposure_sum_h.shape, np.nan, dtype=np.float64)
    weights[valid] = 24.0 * family_exposure_h.shape[1] / (area_km2 * family_exposure_sum_h[valid])
    return weights


def align_mass_to_radiants(radiant_event_id: np.ndarray, mass_rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mass_index = {event_id: i for i, event_id in enumerate(mass_rows["event_id"])}
    n = len(radiant_event_id)
    selected = np.zeros(n, dtype=bool)
    mass = np.full(n, np.nan, dtype=np.float64)
    mass_lo = np.full(n, np.nan, dtype=np.float64)
    mass_hi = np.full(n, np.nan, dtype=np.float64)
    radius_m = np.full(n, np.nan, dtype=np.float64)
    for i, event_id in enumerate(radiant_event_id):
        j = mass_index.get(event_id)
        if j is None:
            continue
        selected[i] = bool(mass_rows["selected"][j])
        mass[i] = float(mass_rows["initial_mass_kg"][j])
        mass_lo[i] = float(mass_rows["mass_95_lo_kg"][j])
        mass_hi[i] = float(mass_rows["mass_95_hi_kg"][j])
        radius_m[i] = float(mass_rows["initial_radius_m"][j])
    selected &= np.isfinite(mass) & (mass > 0.0)
    return {
        "mass_selected": selected,
        "initial_mass_kg": mass,
        "mass_95_lo_kg": mass_lo,
        "mass_95_hi_kg": mass_hi,
        "initial_radius_m": radius_m,
    }


def summarize(weights: np.ndarray, mass_info: dict[str, np.ndarray]) -> dict[str, float]:
    finite = np.isfinite(weights) & (weights > 0.0)
    selected = finite & mass_info["mass_selected"]
    earth_area_km2 = 4.0 * np.pi * EARTH_RADIUS_KM**2
    all_flux = float(np.nansum(weights[finite]))
    all_flux_sigma = float(np.sqrt(np.nansum(weights[finite] ** 2)))
    mass_subset_number_flux = float(np.nansum(weights[selected]))
    mass_subset_number_flux_sigma = float(np.sqrt(np.nansum(weights[selected] ** 2)))
    mass_flux_kg_km2_day = float(np.nansum(weights[selected] * mass_info["initial_mass_kg"][selected]))
    mass_flux_lo_kg_km2_day = float(np.nansum(weights[selected] * mass_info["mass_95_lo_kg"][selected]))
    mass_flux_hi_kg_km2_day = float(np.nansum(weights[selected] * mass_info["mass_95_hi_kg"][selected]))
    weighted_mean_mass = mass_flux_kg_km2_day / mass_subset_number_flux if mass_subset_number_flux > 0.0 else np.nan
    extrapolated_mass_flux = all_flux * weighted_mean_mass if np.isfinite(weighted_mean_mass) else np.nan
    return {
        "n_radiants": int(np.count_nonzero(finite)),
        "n_mass_selected": int(np.count_nonzero(selected)),
        "sky_averaged_flux_per_km2_day": all_flux,
        "sky_averaged_flux_poisson_sigma_per_km2_day": all_flux_sigma,
        "mass_subset_number_flux_per_km2_day": mass_subset_number_flux,
        "mass_subset_number_flux_poisson_sigma_per_km2_day": mass_subset_number_flux_sigma,
        "mass_subset_mass_flux_kg_km2_day": mass_flux_kg_km2_day,
        "mass_subset_mass_flux_95lo_kg_km2_day": mass_flux_lo_kg_km2_day,
        "mass_subset_mass_flux_95hi_kg_km2_day": mass_flux_hi_kg_km2_day,
        "mass_subset_global_kg_day": mass_flux_kg_km2_day * earth_area_km2,
        "mass_subset_global_95lo_kg_day": mass_flux_lo_kg_km2_day * earth_area_km2,
        "mass_subset_global_95hi_kg_day": mass_flux_hi_kg_km2_day * earth_area_km2,
        "mass_subset_weighted_mean_mass_kg": weighted_mean_mass,
        "extrapolated_all_tristatic_global_kg_day": extrapolated_mass_flux * earth_area_km2,
        "earth_area_km2": earth_area_km2,
    }


def write_h5(
    output_h5: Path,
    radiant_data: dict[str, np.ndarray | str],
    family_lon: np.ndarray,
    family_beta: np.ndarray,
    family_exposure_h: np.ndarray,
    weights: np.ndarray,
    mass_info: dict[str, np.ndarray],
    summary: dict[str, float],
    args: argparse.Namespace,
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(output_h5, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["radiant_h5"] = str(args.radiant_h5)
        h.attrs["mass_h5"] = str(args.mass_h5)
        h.attrs["symmetry"] = "mirrors about apex meridian lambda-lambda_sun=270 deg and ecliptic latitude beta=0"
        h.attrs["zenithal_correction"] = "same sin(elevation)^1.47 effective-hour convention as plot_sun_centered_ecliptic_radiants.py"
        h.attrs["beam_full_width_deg"] = float(args.beam_full_width_deg)
        h.attrs["collecting_area_km2"] = float(args.collecting_area_km2)
        h.attrs["common_volume_altitude_km"] = float(args.common_volume_altitude_km)
        h.attrs["pointing_elevation_deg"] = float(args.pointing_elevation_deg)
        for key, value in summary.items():
            h.attrs[key] = value
        h.create_dataset("event_id", data=np.asarray(radiant_data["event_id"], dtype=object), dtype=string_dtype)
        for key in ("lambda_minus_sun_deg", "beta_ecliptic_deg", "speed_km_s", "t0_ns"):
            h[key] = np.asarray(radiant_data[key])
        h["family_lambda_minus_sun_deg"] = family_lon
        h["family_beta_ecliptic_deg"] = family_beta
        h["family_effective_exposure_h"] = family_exposure_h
        h["family_effective_exposure_sum_h"] = np.sum(family_exposure_h, axis=1)
        h["flux_weight_per_km2_day"] = weights
        for key, value in mass_info.items():
            h[key] = value


def plot_summary(output_base: Path, weights: np.ndarray, family_exposure_h: np.ndarray, mass_info: dict[str, np.ndarray], summary: dict[str, float]) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    finite = np.isfinite(weights) & (weights > 0.0)
    selected = finite & mass_info["mass_selected"]
    exposure_sum = np.sum(family_exposure_h, axis=1)

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
    axes[0].hist(exposure_sum[finite], bins=24, color="#4e79a7", alpha=0.72, edgecolor="white")
    axes[0].set_xlabel("Symmetry-family effective exposure (h)")
    axes[0].set_ylabel("Events")
    axes[0].set_title("Radiant exposure denominator")
    axes[1].hist(weights[finite], bins=24, color="#59a14f", alpha=0.72, edgecolor="white")
    axes[1].set_xlabel(r"Event flux weight (km$^{-2}$ day$^{-1}$)")
    axes[1].set_ylabel("Events")
    axes[1].set_title("Per-event flux contribution")
    fig.suptitle("Symmetry-corrected radiant exposure")
    p = output_base.with_name(output_base.name + "_exposure_weights.png")
    fig.savefig(p, dpi=300)
    paths.append(p)
    plt.close(fig)

    beam_widths = np.linspace(0.5, 1.4, 100)
    areas = np.asarray([beam_horizontal_area_km2(width) for width in beam_widths])
    flux_at_width = summary["sky_averaged_flux_per_km2_day"] * summary["collecting_area_km2"] / areas if "collecting_area_km2" in summary else np.nan
    mass_subset_global = summary["mass_subset_global_kg_day"] * summary["collecting_area_km2"] / areas if "collecting_area_km2" in summary else np.nan
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(beam_widths, flux_at_width, color="#1f77b4", lw=2.0, label="All fitted radiants")
    ax.set_xlabel("Assumed Sanya full beam width (deg)")
    ax.set_ylabel(r"Flux (km$^{-2}$ day$^{-1}$)")
    ax.axvline(DEFAULT_BEAM_FULL_WIDTH_DEG, color="0.2", lw=1.0, ls="--")
    ax2 = ax.twinx()
    ax2.plot(beam_widths, mass_subset_global, color="#b07aa1", lw=2.0, label="Dynamic-mass subset")
    ax2.set_ylabel(r"Mass subset global influx (kg day$^{-1}$)")
    handles = [line for line in ax.get_lines() + ax2.get_lines() if not line.get_label().startswith("_")]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper right", frameon=True)
    ax.set_title("Collecting-area sensitivity")
    p = output_base.with_name(output_base.name + "_area_sensitivity.png")
    fig.savefig(p, dpi=300)
    paths.append(p)
    plt.close(fig)

    if np.any(selected):
        log_mass = np.log10(mass_info["initial_mass_kg"][selected])
        mass_weights = weights[selected] * mass_info["initial_mass_kg"][selected] * summary["earth_area_km2"]
        order = np.argsort(log_mass)
        cumulative = np.cumsum(mass_weights[order])
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
        axes[0].hist(log_mass, bins=18, weights=weights[selected], color="#f28e2b", alpha=0.72, edgecolor="white")
        axes[0].set_xlabel(r"$\log_{10}(m_0/\mathrm{kg})$")
        axes[0].set_ylabel(r"Number flux (km$^{-2}$ day$^{-1}$)")
        axes[0].set_title("Mass-estimate subset")
        axes[1].plot(log_mass[order], cumulative, color="#9c755f", lw=2.0)
        axes[1].set_xlabel(r"$\log_{10}(m_0/\mathrm{kg})$")
        axes[1].set_ylabel(r"Cumulative global mass (kg day$^{-1}$)")
        axes[1].set_title("Dynamic-mass contribution")
        p = output_base.with_name(output_base.name + "_mass_contribution.png")
        fig.savefig(p, dpi=300)
        paths.append(p)
        plt.close(fig)
    return paths


def write_macros(path: Path, summary: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "% Generated by /Users/jvi019/src/lfm_meteor/estimate_sky_averaged_flux.py",
            rf"\newcommand{{\skyFluxRadiantN}}{{{int(summary['n_radiants'])}}}",
            rf"\newcommand{{\skyFluxMassN}}{{{int(summary['n_mass_selected'])}}}",
            rf"\newcommand{{\skyFluxPerKmSqDay}}{{{summary['sky_averaged_flux_per_km2_day']:.2f}}}",
            rf"\newcommand{{\skyFluxSigmaPerKmSqDay}}{{{summary['sky_averaged_flux_poisson_sigma_per_km2_day']:.2f}}}",
            rf"\newcommand{{\skyFluxMassSubsetPerKmSqDay}}{{{summary['mass_subset_number_flux_per_km2_day']:.2f}}}",
            rf"\newcommand{{\skyFluxMassSubsetGlobalKgDay}}{{{summary['mass_subset_global_kg_day']:.2e}}}",
            rf"\newcommand{{\skyFluxMassSubsetGlobalLoKgDay}}{{{summary['mass_subset_global_95lo_kg_day']:.2e}}}",
            rf"\newcommand{{\skyFluxMassSubsetGlobalHiKgDay}}{{{summary['mass_subset_global_95hi_kg_day']:.2e}}}",
            rf"\newcommand{{\skyFluxExtrapolatedGlobalKgDay}}{{{summary['extrapolated_all_tristatic_global_kg_day']:.2e}}}",
            rf"\newcommand{{\skyFluxWeightedMeanMassKg}}{{{summary['mass_subset_weighted_mean_mass_kg']:.2e}}}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def copy_plots(paths: list[Path], target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in paths:
        dst = target_dir / src.name
        dst.write_bytes(src.read_bytes())
        copied.append(dst)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiant-h5", type=Path, default=DEFAULT_RADIANT_H5)
    parser.add_argument("--mass-h5", type=Path, default=DEFAULT_MASS_H5)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--paper-memo-figure-dir", type=Path, default=DEFAULT_PAPER_MEMO_FIGURE_DIR)
    parser.add_argument("--paper-macro-output", type=Path, default=DEFAULT_PAPER_TABLE_DIR / "sky_averaged_flux_macros.tex")
    parser.add_argument("--beam-full-width-deg", type=float, default=DEFAULT_BEAM_FULL_WIDTH_DEG)
    parser.add_argument("--common-volume-altitude-km", type=float, default=DEFAULT_COMMON_VOLUME_ALTITUDE_KM)
    parser.add_argument("--pointing-elevation-deg", type=float, default=DEFAULT_POINTING_ELEVATION_DEG)
    parser.add_argument("--collecting-area-km2", type=float, default=np.nan)
    parser.add_argument("--copy-to-paper", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.collecting_area_km2):
        args.collecting_area_km2 = beam_horizontal_area_km2(
            args.beam_full_width_deg,
            args.common_volume_altitude_km,
            args.pointing_elevation_deg,
        )

    radiant_data = load_radiants(args.radiant_h5)
    mass_rows = load_mass_rows(args.mass_h5)
    family_lon, family_beta = mirror_family(
        np.asarray(radiant_data["lambda_minus_sun_deg"]),
        np.asarray(radiant_data["beta_ecliptic_deg"]),
    )
    family_exposure_h, _effective_grid = exposure_for_family(
        family_lon,
        family_beta,
        np.asarray(radiant_data["t0_ns"]),
        str(radiant_data["fixed_ecliptic_equinox_utc"]),
    )
    weights = event_weights_per_km2_day(family_exposure_h, args.collecting_area_km2)
    mass_info = align_mass_to_radiants(np.asarray(radiant_data["event_id"]), mass_rows)
    summary = summarize(weights, mass_info)
    summary["collecting_area_km2"] = float(args.collecting_area_km2)

    output_h5 = args.output_base.with_suffix(".h5")
    write_h5(output_h5, radiant_data, family_lon, family_beta, family_exposure_h, weights, mass_info, summary, args)
    paths = plot_summary(args.output_base, weights, family_exposure_h, mass_info, summary)
    write_macros(args.paper_macro_output, summary)
    copied = copy_plots(paths, args.paper_memo_figure_dir) if args.copy_to_paper else []

    print(f"n_radiants={summary['n_radiants']}")
    print(f"n_mass_selected={summary['n_mass_selected']}")
    print(f"collecting_area_km2={args.collecting_area_km2:.6f}")
    print(f"sky_flux_per_km2_day={summary['sky_averaged_flux_per_km2_day']:.6g}")
    print(f"sky_flux_sigma_per_km2_day={summary['sky_averaged_flux_poisson_sigma_per_km2_day']:.6g}")
    print(f"mass_subset_global_kg_day={summary['mass_subset_global_kg_day']:.6g}")
    print(f"extrapolated_all_tristatic_global_kg_day={summary['extrapolated_all_tristatic_global_kg_day']:.6g}")
    print(f"output_h5={output_h5}")
    print(f"macro_output={args.paper_macro_output}")
    for path in paths:
        print(f"plot={path}")
    for path in copied:
        print(f"paper_plot={path}")


if __name__ == "__main__":
    main()
