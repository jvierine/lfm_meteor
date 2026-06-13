import os

import matplotlib.pyplot as plt
import numpy as np

import test_rank02_range_interpolation as interp


SCRIPT_VERSION = "v20260611b"
UPSAMPLE_FACTOR = 4
OUTPUT_BASE = os.path.join("results", f"rank02_gcrs_xyz_fit_residual_{SCRIPT_VERSION}")
AXIS_LABELS = ("GCRS x", "GCRS y", "GCRS z")


def build_fits():
    fit = interp.load_reference_fit()
    site_data = {site: interp.load_site(site, fit) for site in interp.SITE_ORDER}
    coarse_gates = interp.precompute_coarse_gates(site_data)
    refined = {}
    for site in interp.SITE_ORDER:
        fine_gate, fine_range_km, _power_db = interp.refine_site_ranges(
            site_data[site],
            UPSAMPLE_FACTOR,
            coarse_gates[site],
        )
        if interp.is_root():
            refined[f"{site}_gate"] = fine_gate
            refined[site] = fine_range_km

    if not interp.is_root():
        return None

    measured, times_ns, _source_indices = interp.matched_measurements(site_data, refined)
    const_fit = interp.fit_trajectory(
        measured,
        times_ns,
        site_data["sanya"]["az_deg"],
        site_data["sanya"]["el_deg"],
        float(np.median(refined["sanya"])),
        acceleration=False,
    )
    accel_fit = interp.fit_trajectory(
        measured,
        times_ns,
        site_data["sanya"]["az_deg"],
        site_data["sanya"]["el_deg"],
        float(np.median(refined["sanya"])),
        acceleration=True,
    )
    return const_fit, accel_fit


def km_centered(points_m):
    center = np.nanmean(points_m, axis=0)
    return (points_m - center[None, :]) / 1e3, center


def plot_gcrs_fit(const_fit, accel_fit):
    os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)
    t_s = const_fit["t_rel_s"]
    measured_km, center_m = km_centered(const_fit["points_gcrs_m"])
    const_model_km = (const_fit["model_points_gcrs_m"] - center_m[None, :]) / 1e3
    accel_model_km = (accel_fit["model_points_gcrs_m"] - center_m[None, :]) / 1e3
    const_path_res_m = const_fit["residuals_m"]
    accel_path_res_m = accel_fit["residuals_m"]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 9.2), sharex=True, constrained_layout=True)
    for comp in range(3):
        ax = axes[comp]
        ax.plot(t_s, measured_km[:, comp], "k.", ms=4, label="uncorrected triangulated seed")
        ax.plot(t_s, const_model_km[:, comp], color="#1f77b4", lw=1.6, label="constant velocity")
        ax.plot(t_s, accel_model_km[:, comp], color="#d62728", lw=1.6, label="constant acceleration")
        ax.set_ylabel(f"{AXIS_LABELS[comp]} - mean (km)")
        ax.grid(True, alpha=0.28)
        if comp == 0:
            ax.legend(loc="best")

    ax = axes[3]
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for link_idx, (label, color) in enumerate(zip(("Sanya", "Danzhou", "Wenchang"), colors)):
        ax.plot(t_s, const_path_res_m[:, link_idx], ".", ms=3.5, color=color, alpha=0.35, label=f"{label}, const vel")
        ax.plot(t_s, accel_path_res_m[:, link_idx], "-", lw=1.2, color=color, label=f"{label}, accel")
    ax.axhline(0.0, color="0.35", lw=0.7)
    ax.set_ylabel("Total-path residual (m)")
    ax.set_xlabel("Time since first matched pulse (s)")
    ax.set_title(
        f"Residuals: const vel RMS={const_fit['rms_total_path_residual_m']:.2f} m; "
        f"accel RMS={accel_fit['rms_total_path_residual_m']:.2f} m; "
        f"$a_\\parallel$={accel_fit['along_track_accel_mps2'] / 1e3:.2f} km s$^{{-2}}$"
    )
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower center", ncol=3)

    fig.suptitle(
        f"Rank02 GCRS trajectory and total-path residuals, {UPSAMPLE_FACTOR}x Doppler-corrected interpolation",
        fontsize=12,
    )
    fig.savefig(f"{OUTPUT_BASE}.png", dpi=220)
    fig.savefig(f"{OUTPUT_BASE}.pdf")
    plt.close(fig)


def main():
    fits = build_fits()
    if not interp.is_root():
        return
    const_fit, accel_fit = fits
    plot_gcrs_fit(const_fit, accel_fit)
    print(
        f"constant velocity RMS={const_fit['rms_total_path_residual_m']:.2f} m; "
        f"constant acceleration RMS={accel_fit['rms_total_path_residual_m']:.2f} m; "
        f"along-track acceleration={accel_fit['along_track_accel_mps2'] / 1e3:.3f} km/s^2"
    )
    print(f"wrote {OUTPUT_BASE}.png")
    print(f"wrote {OUTPUT_BASE}.pdf")


if __name__ == "__main__":
    main()
