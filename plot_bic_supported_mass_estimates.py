"""Plot mass estimates only for events with strong shrinking-radius evidence.

The model comparison is

    Delta BIC = BIC_constant_velocity - BIC_shrinking_radius

so only Delta BIC >= STRONG_DELTA_BIC events are interpreted as supporting the
extra shrinking-radius mass parameter. Weak positive values are reported as
model-comparison diagnostics, not measured masses.
"""

import argparse
import csv
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

import review_mass_support_gui as mass_gui
import review_tristatic_fits_gui as fit_gui


DEFAULT_INPUT_H5 = "results/all_tristatic_ceplecha_snr_weighted_v20260616d.h5"
DEFAULT_REVIEW_H5 = "results/tristatic_fit_review.h5"
DEFAULT_OUTPUT_BASE = "results/bic_supported_mass_estimates"
STRONG_DELTA_BIC = 6.0


def compare_event(h, event_id, review_h5):
    group = h["points"][event_id]
    if "all_time_ns" in group:
        time_ns = np.asarray(group["all_time_ns"][:], dtype=np.int64)
        measured = np.asarray(group["all_measured_total_paths_m"][:], dtype=np.float64)
        sigma = np.asarray(group["all_sigma_m"][:], dtype=np.float64) if "all_sigma_m" in group else np.ones_like(measured)
        x_itrs = np.asarray(group["all_x_itrs_m"][:], dtype=np.float64)
        keep_stored = np.asarray(group["all_keep_rows"][:], dtype=bool)
    else:
        time_ns = np.asarray(group["time_ns"][:], dtype=np.int64)
        measured = np.asarray(group["measured_total_paths_m"][:], dtype=np.float64)
        sigma = np.asarray(group["sigma_m"][:], dtype=np.float64) if "sigma_m" in group else np.ones_like(measured)
        x_itrs = np.asarray(group["x_itrs_m"][:], dtype=np.float64)
        keep_stored = np.ones(len(time_ns), dtype=bool)

    manual_reject = mass_gui.manual_mask_from_review(review_h5, event_id, len(time_ns))
    keep = keep_stored & ~manual_reject
    if np.count_nonzero(keep) < fit_gui.MIN_CONSTANT_VELOCITY_POINTS:
        return None

    x_gcrs = fit_gui.ecef_to_gcrs(x_itrs, time_ns)
    t_rel_s = (time_ns.astype(np.float64) - float(time_ns[0])) / 1e9
    p0_linear = np.concatenate([x_gcrs[0], np.polyfit(t_rel_s, x_gcrs, 1)[0]])

    cv_fit = fit_gui.fit_linear_paths(
        measured,
        time_ns,
        sigma,
        keep,
        p0_linear,
        initial_points_itrs_m=x_itrs,
        estimate_lower_bound=False,
    )
    stored = mass_gui.stored_ceplecha_fit(group)
    if np.array_equal(keep, stored["keep_rows"]):
        shrink_fit = stored
    else:
        shrink_fit = fit_gui.fit_ceplecha_paths(
            measured,
            time_ns,
            sigma,
            keep,
            x_itrs,
            p0_fallback=np.asarray(group["params"][:], dtype=np.float64),
        )

    cv_bic, cv_chi2, n_obs = mass_gui.bic_from_normalized_residuals(cv_fit["normalized_residuals"], 6)
    shrink_bic, shrink_chi2, _ = mass_gui.bic_from_normalized_residuals(shrink_fit["normalized_residuals"], 7)
    delta_bic = cv_bic - shrink_bic
    radius_m = float(shrink_fit["all_radius_m"][0])
    mass_kg = float(shrink_fit["all_mass_kg"][0])
    log10_radius_std = float(shrink_fit.get("log10_radius_std", np.nan))
    sigma_radius_m, sigma_mass_kg = fit_gui.radius_mass_uncertainty(radius_m, mass_kg, log10_radius_std)
    speed_km_s = float(np.linalg.norm(shrink_fit["all_v_gcrs_mps"][0]) / 1e3)
    return {
        "event_id": event_id,
        "time_ns": int(time_ns[0]),
        "n_pulses": int(np.count_nonzero(keep)),
        "n_obs": int(n_obs),
        "delta_bic": float(delta_bic),
        "cv_bic": float(cv_bic),
        "shrinking_bic": float(shrink_bic),
        "delta_chi2": float(cv_chi2 - shrink_chi2),
        "initial_radius_um": radius_m * 1e6,
        "initial_radius_sigma_um": sigma_radius_m * 1e6 if np.isfinite(sigma_radius_m) else np.nan,
        "initial_mass_kg": mass_kg,
        "initial_mass_sigma_kg": sigma_mass_kg,
        "initial_speed_km_s": speed_km_s,
    }


def write_csv(path, rows):
    fieldnames = [
        "event_id",
        "time_ns",
        "n_pulses",
        "n_obs",
        "delta_bic",
        "cv_bic",
        "shrinking_bic",
        "delta_chi2",
        "initial_radius_um",
        "initial_radius_sigma_um",
        "initial_mass_kg",
        "initial_mass_sigma_kg",
        "initial_speed_km_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_rows(rows, output_base):
    supported = [row for row in rows if row["delta_bic"] >= STRONG_DELTA_BIC and np.isfinite(row["initial_mass_kg"])]
    all_masses = np.asarray([row["initial_mass_kg"] for row in rows if np.isfinite(row["initial_mass_kg"])])
    masses = np.asarray([row["initial_mass_kg"] for row in supported])
    mass_sigma = np.asarray([row["initial_mass_sigma_kg"] for row in supported])
    speeds = np.asarray([row["initial_speed_km_s"] for row in supported])
    delta_bic = np.asarray([row["delta_bic"] for row in supported])
    radius_um = np.asarray([row["initial_radius_um"] for row in supported])

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)
    bins = np.logspace(
        np.floor(np.log10(np.nanmin(all_masses))) if len(all_masses) else -14,
        np.ceil(np.log10(np.nanmax(all_masses))) if len(all_masses) else -6,
        18,
    )
    axes[0].hist(all_masses, bins=bins, histtype="step", color="0.55", lw=1.3, label="candidate fits")
    axes[0].hist(masses, bins=bins, color="#1b7837", alpha=0.70, label=rf"$\Delta$BIC $\ge {STRONG_DELTA_BIC:.0f}$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Initial mass (kg)")
    axes[0].set_ylabel("Events")
    axes[0].legend(frameon=False)
    axes[0].grid(True, color="0.88", which="both")

    if len(masses):
        xerr = np.where(np.isfinite(mass_sigma), mass_sigma, 0.0)
        sc = axes[1].scatter(masses, speeds, c=delta_bic, s=38, cmap="viridis", edgecolors="0.2", linewidths=0.3)
        axes[1].errorbar(masses, speeds, xerr=xerr, fmt="none", ecolor="0.65", lw=0.8, alpha=0.8)
        cbar = fig.colorbar(sc, ax=axes[1], pad=0.02)
        cbar.set_label(r"$\Delta$BIC")
        for mass, speed, radius in zip(masses, speeds, radius_um):
            if radius > 100.0:
                axes[1].annotate(f"{radius:.0f} um", (mass, speed), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Initial mass (kg)")
    axes[1].set_ylabel("Initial speed (km s$^{-1}$)")
    axes[1].grid(True, color="0.88", which="both")
    fig.suptitle(f"Shrinking-radius mass estimates with strong BIC evidence (n={len(supported)})")

    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    png = f"{output_base}.png"
    pdf = f"{output_base}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf, len(supported)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", default=DEFAULT_INPUT_H5)
    parser.add_argument("--review-h5", default=DEFAULT_REVIEW_H5)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--all-events", action="store_true", help="Compare all events instead of mass-candidate events only.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    with h5py.File(args.input_h5, "r") as h:
        event_ids = fit_gui.decode_strings(h["event_id"][:])
        if args.all_events:
            mask = np.ones(len(event_ids), dtype=bool)
        else:
            mask = mass_gui.mass_candidate_mask(h)
        selected_event_ids = event_ids[mask]
        for i, event_id in enumerate(selected_event_ids, start=1):
            row = compare_event(h, event_id, args.review_h5)
            if row is not None:
                rows.append(row)
            print(f"{i:04d}/{len(selected_event_ids):04d} {event_id}", flush=True)
    csv_path = f"{args.output_base}.csv"
    write_csv(csv_path, rows)
    png, pdf, n_supported = plot_rows(rows, args.output_base)
    print(f"strong Delta BIC events: {n_supported}/{len(rows)}")
    print(f"wrote {csv_path}")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
