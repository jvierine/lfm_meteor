import argparse
import os
import shutil
import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np

import fit_all_ceplecha_snr_weighted as cepl


DEFAULT_CATALOG_DIR = "results/tristatic"
DEFAULT_WHIPPLE_DIR = "results/tristatic_whipple_jacchia_bootstrap_orbit100_20260701"
DEFAULT_OUTPUT_BASE = "results/joint_fft_mass_distribution_v20260618a"
PAPER_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/figures"
DEFAULT_MAX_SYNTHETIC_VELOCITY_RMS_MPS = 1000.0
DEFAULT_MAX_SYNTHETIC_VELOCITY_MAX_MPS = 1000.0
DEFAULT_MAX_SYNTHETIC_PATH_RATE_RMS_MPS = 1000.0
DEFAULT_MAX_SYNTHETIC_PATH_RMS_M = 2.0


def decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def mass_from_radius(radius_m):
    return (4.0 / 3.0) * np.pi * cepl.METEOROID_DENSITY_KG_M3 * np.asarray(radius_m, dtype=np.float64) ** 3.0


def log10_mass_to_diameter_um(log10_mass_kg):
    mass_kg = 10.0 ** np.asarray(log10_mass_kg, dtype=np.float64)
    radius_m = (3.0 * mass_kg / (4.0 * np.pi * cepl.METEOROID_DENSITY_KG_M3)) ** (1.0 / 3.0)
    return 2.0e6 * radius_m


def diameter_um_to_log10_mass(diameter_um):
    diameter_um = np.asarray(diameter_um, dtype=np.float64)
    radius_m = 0.5e-6 * np.maximum(diameter_um, 1.0e-9)
    mass_kg = mass_from_radius(radius_m)
    return np.log10(mass_kg)


def radius_mass_95_bounds(radius_m, log10_radius_std):
    if not np.isfinite(radius_m) or radius_m <= 0.0 or not np.isfinite(log10_radius_std):
        return np.nan, np.nan, np.nan, np.nan
    sigma_radius_m = np.log(10.0) * float(radius_m) * float(log10_radius_std)
    if not np.isfinite(sigma_radius_m) or sigma_radius_m <= 0.0:
        return np.nan, np.nan, np.nan, np.nan
    radius_95_lo_m = float(radius_m) - 1.96 * sigma_radius_m
    radius_95_hi_m = float(radius_m) + 1.96 * sigma_radius_m
    radius_95_lo_m = float(np.clip(radius_95_lo_m, cepl.MIN_RADIUS_M, cepl.MAX_RADIUS_M))
    radius_95_hi_m = float(np.clip(radius_95_hi_m, cepl.MIN_RADIUS_M, cepl.MAX_RADIUS_M))
    if radius_95_lo_m >= float(radius_m) or radius_95_hi_m <= float(radius_m):
        return np.nan, np.nan, np.nan, np.nan
    return (
        radius_95_lo_m,
        radius_95_hi_m,
        float(mass_from_radius(radius_95_lo_m)),
        float(mass_from_radius(radius_95_hi_m)),
    )


def finite_attr(group, name):
    value = float(group.attrs.get(name, np.nan))
    return value if np.isfinite(value) else np.nan


def load_joint_rows(catalog_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(catalog_dir, "joint_delay_doppler_fft_tri_*.h5"))):
        with h5py.File(path, "r") as h:
            j = h["joint_fit"]
            radius_m = float(j.attrs["initial_radius_m"])
            log10_radius_std = float(j.attrs.get("log10_radius_std", np.nan))
            radius_95_lo_m, radius_95_hi_m, mass_95_lo_kg, mass_95_hi_kg = radius_mass_95_bounds(
                radius_m,
                log10_radius_std,
            )
            bootstrap_radius_95_lo_m = finite_attr(j, "bootstrap_radius0_lo95_m")
            bootstrap_radius_95_hi_m = finite_attr(j, "bootstrap_radius0_hi95_m")
            bootstrap_mass_95_lo_kg = finite_attr(j, "bootstrap_mass0_lo95_kg")
            bootstrap_mass_95_hi_kg = finite_attr(j, "bootstrap_mass0_hi95_kg")
            if (
                np.isfinite(bootstrap_mass_95_lo_kg)
                and np.isfinite(bootstrap_mass_95_hi_kg)
                and bootstrap_mass_95_lo_kg > 0.0
                and bootstrap_mass_95_hi_kg > bootstrap_mass_95_lo_kg
            ):
                mass_95_lo_kg = bootstrap_mass_95_lo_kg
                mass_95_hi_kg = bootstrap_mass_95_hi_kg
                radius_95_lo_m = bootstrap_radius_95_lo_m
                radius_95_hi_m = bootstrap_radius_95_hi_m
                uncertainty_source = 1
            else:
                uncertainty_source = 0
            rows.append(
                {
                    "event_id": decode(h.attrs["event_id"]),
                    "initial_radius_m": radius_m,
                    "initial_mass_kg": float(j.attrs["initial_mass_kg"]),
                    "radius_95_lo_m": radius_95_lo_m,
                    "radius_95_hi_m": radius_95_hi_m,
                    "mass_95_lo_kg": mass_95_lo_kg,
                    "mass_95_hi_kg": mass_95_hi_kg,
                    "log10_radius_std": log10_radius_std,
                    "uncertainty_source": uncertainty_source,
                    "n_points": int(j.attrs["n_points"]),
                    "n_fft_observations": int(j.attrs["n_fft_observations"]),
                    "rms_fft_residual_hz": float(j.attrs["rms_fft_residual_hz"]),
                    "rms_total_path_residual_m": float(j.attrs["rms_total_path_residual_m"]),
                    "initial_speed_km_s": float(j["speed_km_s"][0]),
                }
            )
    return rows


def load_shrinking_radius_to_whipple_rows(catalog_dir, whipple_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(catalog_dir, "shrinking_radius_to_whipple_tri_*.h5"))):
        with h5py.File(path, "r") as h:
            event_id = decode(h.attrs["event_id"])
            whipple_path = os.path.join(whipple_dir, f"joint_delay_doppler_fft_{event_id}.h5")
            if not os.path.exists(whipple_path):
                print(f"warning: missing Whipple-Jacchia file for {event_id}: {whipple_path}")
                continue
            with h5py.File(whipple_path, "r") as hw:
                j = hw["joint_fit"]
                speed = float(j["speed_km_s"][0])
                n_points = int(j.attrs["n_points"])
                n_fft_observations = int(j.attrs["n_fft_observations"])
                rms_fft_residual_hz = float(j.attrs["rms_fft_residual_hz"])
                rms_total_path_residual_m = float(j.attrs["rms_total_path_residual_m"])
                try:
                    shrinking_v = np.asarray(h["v_gcrs_mps"][()], dtype=np.float64)
                    whipple_v = np.asarray(j["v_gcrs_mps"][()], dtype=np.float64)
                    if shrinking_v.shape == whipple_v.shape:
                        synthetic_velocity_max_mps = float(np.nanmax(np.linalg.norm(shrinking_v - whipple_v, axis=1)))
                    else:
                        synthetic_velocity_max_mps = finite_attr(h, "synthetic_velocity_max_mps")
                except Exception:
                    synthetic_velocity_max_mps = finite_attr(h, "synthetic_velocity_max_mps")
            rows.append(
                {
                    "event_id": event_id,
                    "initial_radius_m": finite_attr(h, "initial_radius_m"),
                    "initial_mass_kg": finite_attr(h, "initial_mass_kg"),
                    "radius_95_lo_m": finite_attr(h, "bootstrap_initial_radius_lo95_m"),
                    "radius_95_hi_m": finite_attr(h, "bootstrap_initial_radius_hi95_m"),
                    "mass_95_lo_kg": finite_attr(h, "bootstrap_initial_mass_lo95_kg"),
                    "mass_95_hi_kg": finite_attr(h, "bootstrap_initial_mass_hi95_kg"),
                    "bootstrap_initial_radius_median_m": finite_attr(h, "bootstrap_initial_radius_median_m"),
                    "bootstrap_initial_mass_median_kg": finite_attr(h, "bootstrap_initial_mass_median_kg"),
                    "log10_radius_std": np.nan,
                    "uncertainty_source": 1,
                    "optimizer_success": float(bool(h.attrs.get("optimizer_success", False))),
                    "bootstrap_samples_successful": int(h.attrs.get("bootstrap_samples_successful", 0)),
                    "synthetic_velocity_rms_mps": finite_attr(h, "synthetic_velocity_rms_mps"),
                    "synthetic_velocity_max_mps": synthetic_velocity_max_mps,
                    "synthetic_path_rms_m": finite_attr(h, "synthetic_path_rms_m"),
                    "synthetic_path_rate_rms_mps": finite_attr(h, "synthetic_path_rate_rms_mps"),
                    "n_points": n_points,
                    "n_fft_observations": n_fft_observations,
                    "rms_fft_residual_hz": rms_fft_residual_hz,
                    "rms_total_path_residual_m": rms_total_path_residual_m,
                    "initial_speed_km_s": speed,
                }
            )
    return rows


def load_rows(catalog_dir, whipple_dir):
    rows = load_shrinking_radius_to_whipple_rows(catalog_dir, whipple_dir)
    if rows:
        return rows
    return load_joint_rows(catalog_dir)


def selected_mask(rows, args):
    radius = np.asarray([r["initial_radius_m"] for r in rows], dtype=np.float64)
    log10_std = np.asarray([r["log10_radius_std"] for r in rows], dtype=np.float64)
    uncertainty_source = np.asarray([r["uncertainty_source"] for r in rows], dtype=np.float64)
    n_fft = np.asarray([r["n_fft_observations"] for r in rows], dtype=np.float64)
    beat = np.asarray([r["rms_fft_residual_hz"] for r in rows], dtype=np.float64)
    path = np.asarray([r["rms_total_path_residual_m"] for r in rows], dtype=np.float64)
    mass_lo = np.asarray([r["mass_95_lo_kg"] for r in rows], dtype=np.float64)
    mass_hi = np.asarray([r["mass_95_hi_kg"] for r in rows], dtype=np.float64)
    mass = np.asarray([r["initial_mass_kg"] for r in rows], dtype=np.float64)
    radius_lo = np.asarray([r["radius_95_lo_m"] for r in rows], dtype=np.float64)
    radius_hi = np.asarray([r["radius_95_hi_m"] for r in rows], dtype=np.float64)
    optimizer_success = np.asarray([r.get("optimizer_success", 1.0) for r in rows], dtype=np.float64)
    bootstrap_samples_successful = np.asarray(
        [r.get("bootstrap_samples_successful", np.inf) for r in rows],
        dtype=np.float64,
    )
    synthetic_velocity_rms = np.asarray([r.get("synthetic_velocity_rms_mps", np.nan) for r in rows], dtype=np.float64)
    synthetic_velocity_max = np.asarray([r.get("synthetic_velocity_max_mps", np.nan) for r in rows], dtype=np.float64)
    synthetic_path_rms = np.asarray([r.get("synthetic_path_rms_m", np.nan) for r in rows], dtype=np.float64)
    synthetic_path_rate_rms = np.asarray(
        [r.get("synthetic_path_rate_rms_mps", np.nan) for r in rows],
        dtype=np.float64,
    )
    mass_95_width_dex = np.log10(mass_hi) - np.log10(mass_lo)
    mask = (
        np.isfinite(radius)
        & np.isfinite(mass_lo)
        & np.isfinite(mass_hi)
        & np.isfinite(mass)
        & np.isfinite(mass_95_width_dex)
        & (mass_lo > 0.0)
        & (mass_hi > 0.0)
        & (mass_lo < mass)
        & (mass_hi > mass)
        & (radius > 1.01 * cepl.MIN_RADIUS_M)
        & (radius < 0.99 * cepl.MAX_RADIUS_M)
        & (optimizer_success > 0.0)
        & (bootstrap_samples_successful >= args.min_bootstrap_samples)
        & (
            ~np.isfinite(synthetic_velocity_rms)
            | (synthetic_velocity_rms <= float(args.max_synthetic_velocity_rms_mps))
        )
        & (
            ~np.isfinite(synthetic_velocity_max)
            | (synthetic_velocity_max <= float(args.max_synthetic_velocity_max_mps))
        )
        & (
            ~np.isfinite(synthetic_path_rms)
            | (synthetic_path_rms <= float(args.max_synthetic_path_rms_m))
        )
        & (
            ~np.isfinite(synthetic_path_rate_rms)
            | (synthetic_path_rate_rms <= float(args.max_synthetic_path_rate_rms_mps))
        )
        & (mass_95_width_dex <= args.max_mass_95_width_dex)
    )
    if np.isfinite(args.max_log10_radius_std):
        mask &= (uncertainty_source == 1.0) | (np.isfinite(log10_std) & (log10_std <= args.max_log10_radius_std))
    if args.min_fft_observations > 0:
        mask &= n_fft >= args.min_fft_observations
    if np.isfinite(args.max_fft_rms_hz):
        mask &= beat <= args.max_fft_rms_hz
    if np.isfinite(args.max_path_rms_m):
        mask &= path <= args.max_path_rms_m
    return mask


def write_h5(output_base, rows, mask, args):
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_base + ".h5", "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["catalog_dir"] = args.catalog_dir
        h.attrs["whipple_dir"] = args.whipple_dir
        h.attrs["min_fft_observations"] = int(args.min_fft_observations)
        h.attrs["min_bootstrap_samples"] = int(args.min_bootstrap_samples)
        h.attrs["max_fft_rms_hz"] = float(args.max_fft_rms_hz)
        h.attrs["max_path_rms_m"] = float(args.max_path_rms_m)
        h.attrs["max_log10_radius_std"] = float(args.max_log10_radius_std)
        h.attrs["max_mass_95_width_dex"] = float(args.max_mass_95_width_dex)
        h.attrs["max_synthetic_velocity_rms_mps"] = float(args.max_synthetic_velocity_rms_mps)
        h.attrs["max_synthetic_velocity_max_mps"] = float(args.max_synthetic_velocity_max_mps)
        h.attrs["max_synthetic_path_rms_m"] = float(args.max_synthetic_path_rms_m)
        h.attrs["max_synthetic_path_rate_rms_mps"] = float(args.max_synthetic_path_rate_rms_mps)
        h.attrs["n_catalog_events"] = int(len(rows))
        h.attrs["n_selected"] = int(np.count_nonzero(mask))
        h.create_dataset("event_id", data=np.asarray([r["event_id"] for r in rows], dtype=object), dtype=string_dtype)
        h["selected"] = np.asarray(mask, dtype=bool)
        for key in rows[0].keys():
            if key == "event_id":
                continue
            h[key] = np.asarray([r[key] for r in rows], dtype=np.float64)


def plot(output_base, rows, mask, args):
    masses_all = np.asarray([r["initial_mass_kg"] for r in rows], dtype=np.float64)
    masses = masses_all[mask]
    speeds = np.asarray([r["initial_speed_km_s"] for r in rows], dtype=np.float64)[mask]
    mass_lo = np.asarray([r["mass_95_lo_kg"] for r in rows], dtype=np.float64)[mask]
    mass_hi = np.asarray([r["mass_95_hi_kg"] for r in rows], dtype=np.float64)[mask]
    log_mass = np.log10(masses)
    log_mass_lo = np.log10(mass_lo)
    log_mass_hi = np.log10(mass_hi)
    log_mass_xerr = np.vstack([log_mass - log_mass_lo, log_mass_hi - log_mass])
    log_mass_xerr = np.maximum(log_mass_xerr, 0.0)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    log_mass_min = float(np.nanmin(log_mass))
    log_mass_max = float(np.nanmax(log_mass))
    log_mass_pad = max(0.05, 0.02 * (log_mass_max - log_mass_min))
    bins = np.linspace(log_mass_min - log_mass_pad, log_mass_max + log_mass_pad, 25)
    n_mass, _bins, mass_patches = ax.hist(
        log_mass,
        bins=bins,
        histtype="stepfilled",
        alpha=0.58,
        color="#2f7f6f",
        edgecolor="#17463d",
        linewidth=1.0,
        label=rf"Mass estimates, $n={len(masses)}$",
    )
    ax.axvline(np.nanmedian(log_mass), color="#17463d", lw=1.8, ls="--")
    ax.set_xlabel(r"$\log_{10}(m_0)$, initial mass in kg")
    ax.set_ylabel("Number of fitted trajectories")
    ax.set_title(r"Initial-mass distribution", pad=42)
    ax.set_ylim(0.0, float(np.nanmax(n_mass)) / 0.45)

    ax_diameter = ax.secondary_xaxis(
        "top",
        functions=(log10_mass_to_diameter_um, diameter_um_to_log10_mass),
    )
    diameter_min, diameter_max = log10_mass_to_diameter_um(ax.get_xlim())
    candidate_ticks = np.asarray([5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0])
    diameter_ticks = candidate_ticks[
        (candidate_ticks >= 0.9 * min(diameter_min, diameter_max))
        & (candidate_ticks <= 1.1 * max(diameter_min, diameter_max))
    ]
    if diameter_ticks.size >= 2:
        ax_diameter.set_xticks(diameter_ticks)
    ax_diameter.set_xlabel(r"Equivalent diameter ($\mu$m)")

    ax_points = ax.twinx()
    order = np.argsort(log_mass)
    mass_err = ax_points.errorbar(
        log_mass[order],
        speeds[order],
        xerr=log_mass_xerr[:, order],
        fmt="o",
        ms=3.7,
        mfc="white",
        mec="#17463d",
        ecolor="#17463d",
        elinewidth=0.65,
        capsize=1.2,
        alpha=0.6,
        linestyle="none",
        label=r"Individual $m_0$ estimates",
    )
    speed_min = float(np.nanmin(speeds))
    speed_max = float(np.nanmax(speeds))
    speed_span = max(speed_max - speed_min, 1.0)
    display_low_fraction = 0.62
    display_high_fraction = 0.96
    axis_span = speed_span / (display_high_fraction - display_low_fraction)
    axis_low = speed_min - display_low_fraction * axis_span
    axis_high = axis_low + axis_span
    ax_points.set_ylim(axis_low, axis_high)
    speed_tick_start = 10.0 * np.ceil(speed_min / 10.0)
    speed_tick_stop = 10.0 * np.floor(speed_max / 10.0)
    if speed_tick_stop >= speed_tick_start:
        ax_points.set_yticks(np.arange(speed_tick_start, speed_tick_stop + 0.1, 10.0))
    ax_points.set_ylabel("Initial speed (km s$^{-1}$)")
    ax_points.tick_params(axis="y", colors="0.35")
    ax_points.yaxis.label.set_color("0.35")

    ax.legend(
        [mass_patches[0], mass_err],
        [rf"Mass histogram, $n={len(masses)}$", r"Individual $m_0$ estimates"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.82",
        fontsize=9.2,
    )
    fig.savefig(output_base + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_base + ".pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def copy_to_paper(output_base):
    os.makedirs(PAPER_FIGURE_DIR, exist_ok=True)
    copied = []
    figure_bases = ("meteor_mass_distribution", "joint_fft_mass_distribution")
    for ext in ("png", "pdf"):
        src = f"{output_base}.{ext}"
        for figure_base in figure_bases:
            dst = os.path.join(PAPER_FIGURE_DIR, f"{figure_base}.{ext}")
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def main():
    parser = argparse.ArgumentParser(description="Plot joint delay--FFT initial-mass distribution with 95% error bars.")
    parser.add_argument("--catalog-dir", default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--whipple-dir", default=DEFAULT_WHIPPLE_DIR)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--min-fft-observations", type=int, default=0)
    parser.add_argument("--min-bootstrap-samples", type=int, default=20)
    parser.add_argument("--max-fft-rms-hz", type=float, default=np.inf)
    parser.add_argument("--max-path-rms-m", type=float, default=np.inf)
    parser.add_argument("--max-log10-radius-std", type=float, default=0.5)
    parser.add_argument("--max-mass-95-width-dex", type=float, default=1.0)
    parser.add_argument("--max-synthetic-velocity-rms-mps", type=float, default=DEFAULT_MAX_SYNTHETIC_VELOCITY_RMS_MPS)
    parser.add_argument("--max-synthetic-velocity-max-mps", type=float, default=DEFAULT_MAX_SYNTHETIC_VELOCITY_MAX_MPS)
    parser.add_argument("--max-synthetic-path-rms-m", type=float, default=DEFAULT_MAX_SYNTHETIC_PATH_RMS_M)
    parser.add_argument("--max-synthetic-path-rate-rms-mps", type=float, default=DEFAULT_MAX_SYNTHETIC_PATH_RATE_RMS_MPS)
    parser.add_argument("--copy-to-paper", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.catalog_dir, args.whipple_dir)
    if not rows:
        raise RuntimeError(f"No joint event HDF5 files found in {args.catalog_dir}")
    mask = selected_mask(rows, args)
    write_h5(args.output_base, rows, mask, args)
    plot(args.output_base, rows, mask, args)
    copied = copy_to_paper(args.output_base) if args.copy_to_paper else []

    print(f"n_catalog_events={len(rows)}")
    print(f"n_selected={int(np.count_nonzero(mask))}")
    print(f"output_h5={args.output_base}.h5")
    print(f"output_png={args.output_base}.png")
    print(f"output_pdf={args.output_base}.pdf")
    for path in copied:
        print(f"paper_copy={path}")


if __name__ == "__main__":
    main()
