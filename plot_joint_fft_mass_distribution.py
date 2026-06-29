import argparse
import os
import shutil
import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np

import fit_all_ceplecha_snr_weighted as cepl


DEFAULT_CATALOG_DIR = "results/tristatic"
DEFAULT_OUTPUT_BASE = "results/joint_fft_mass_distribution_v20260618a"
PAPER_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/figures"


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


def load_rows(catalog_dir):
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
                    "n_points": int(j.attrs["n_points"]),
                    "n_fft_observations": int(j.attrs["n_fft_observations"]),
                    "rms_fft_residual_hz": float(j.attrs["rms_fft_residual_hz"]),
                    "rms_total_path_residual_m": float(j.attrs["rms_total_path_residual_m"]),
                    "initial_speed_km_s": float(j["speed_km_s"][0]),
                }
            )
    return rows


def selected_mask(rows, args):
    radius = np.asarray([r["initial_radius_m"] for r in rows], dtype=np.float64)
    log10_std = np.asarray([r["log10_radius_std"] for r in rows], dtype=np.float64)
    n_fft = np.asarray([r["n_fft_observations"] for r in rows], dtype=np.float64)
    beat = np.asarray([r["rms_fft_residual_hz"] for r in rows], dtype=np.float64)
    path = np.asarray([r["rms_total_path_residual_m"] for r in rows], dtype=np.float64)
    mass_lo = np.asarray([r["mass_95_lo_kg"] for r in rows], dtype=np.float64)
    mass_hi = np.asarray([r["mass_95_hi_kg"] for r in rows], dtype=np.float64)
    mass = np.asarray([r["initial_mass_kg"] for r in rows], dtype=np.float64)
    radius_lo = np.asarray([r["radius_95_lo_m"] for r in rows], dtype=np.float64)
    radius_hi = np.asarray([r["radius_95_hi_m"] for r in rows], dtype=np.float64)
    return (
        np.isfinite(radius)
        & np.isfinite(radius_lo)
        & np.isfinite(radius_hi)
        & np.isfinite(log10_std)
        & np.isfinite(mass_lo)
        & np.isfinite(mass_hi)
        & np.isfinite(mass)
        & (mass_lo > 0.0)
        & (mass_hi > 0.0)
        & (mass_lo < mass)
        & (mass_hi > mass)
        & (radius > 1.01 * cepl.MIN_RADIUS_M)
        & (radius < 0.99 * cepl.MAX_RADIUS_M)
        & (radius_lo > 1.01 * cepl.MIN_RADIUS_M)
        & (radius_hi < 0.99 * cepl.MAX_RADIUS_M)
        & (log10_std <= args.max_log10_radius_std)
        & (n_fft >= args.min_fft_observations)
        & (beat <= args.max_fft_rms_hz)
        & (path <= args.max_path_rms_m)
    )


def write_h5(output_base, rows, mask, args):
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_base + ".h5", "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["catalog_dir"] = args.catalog_dir
        h.attrs["min_fft_observations"] = int(args.min_fft_observations)
        h.attrs["max_fft_rms_hz"] = float(args.max_fft_rms_hz)
        h.attrs["max_path_rms_m"] = float(args.max_path_rms_m)
        h.attrs["max_log10_radius_std"] = float(args.max_log10_radius_std)
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
    bins = np.linspace(np.nanpercentile(log_mass, 1.0), np.nanpercentile(log_mass, 99.0), 25)
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
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--min-fft-observations", type=int, default=20)
    parser.add_argument("--max-fft-rms-hz", type=float, default=1500.0)
    parser.add_argument("--max-path-rms-m", type=float, default=150.0)
    parser.add_argument("--max-log10-radius-std", type=float, default=0.5)
    parser.add_argument("--copy-to-paper", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.catalog_dir)
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
