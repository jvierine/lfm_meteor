import csv
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so

import fit_all_ballistic_snr_weighted as base_fit
import fit_all_ceplecha_snr_weighted as ceplecha_fit

CEPLECHA_H5_GLOB = os.path.join("results", "all_tristatic_ceplecha_snr_weighted_*.h5")
SOURCE_H5 = os.path.join("results", "all_tristatic_ceplecha_snr_weighted_v20260616d.h5")
REVIEW_H5 = os.path.join("results", "tristatic_fit_review.h5")
LEAVE_ONE_OUT_H5 = os.path.join("results", "meteor_mass_leave_one_out_uncertainty.h5")
MODEL_SELECTION_CSV = os.path.join("results", "tristatic_bayesian_model_selection.csv")
PAPER_FIGURE = "/Users/jvi019/src/sanya_tristatic_paper/figures/meteor_mass_distribution.png"
OUTPUT_CSV = os.path.join("results", "meteor_mass_distribution_frac_unc_lt_50pct_current.csv")
MASS_SUPPORTED_DELTA_BIC = 6.0

def attr_string(group, key, default=""):
    value = group.attrs.get(key, default)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def load_constrained_initial_mass():
    if os.path.exists(MODEL_SELECTION_CSV):
        return load_bayesian_initial_mass()
    if os.path.exists(REVIEW_H5):
        return load_reviewed_initial_mass()
    return load_catalog_initial_mass()


def source_log10_radius_std_by_event():
    if not os.path.exists(SOURCE_H5):
        return {}
    with h5py.File(SOURCE_H5, "r") as h:
        event_id = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in h["event_id"][:]
        ]
        if "log10_radius_std" not in h:
            return {eid: np.nan for eid in event_id}
        values = np.asarray(h["log10_radius_std"][:], dtype=float)
    return dict(zip(event_id, values))


def load_bayesian_initial_mass():
    log10_radius_std_lookup = source_log10_radius_std_by_event()
    rows = []
    with open(MODEL_SELECTION_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            delta_bic = float(row["delta_bic_cv_minus_shrinking"])
            if delta_bic < MASS_SUPPORTED_DELTA_BIC:
                continue
            event_id = row["event_id"]
            initial_radius_m = float(row["initial_radius_um"]) * 1e-6
            initial_mass_kg = float(row["initial_mass_kg"])
            log10_radius_std = float(log10_radius_std_lookup.get(event_id, np.nan))
            rows.append((
                event_id,
                initial_radius_m,
                initial_mass_kg,
                np.nan,
                np.nan,
                log10_radius_std,
                0.0,
                0.0,
                float(row["constant_velocity_speed0_km_s"]),
                np.nan,
                0,
            ))
    return filter_mass_rows(rows)


def load_bayesian_mass_event_ids():
    if not os.path.exists(MODEL_SELECTION_CSV):
        return None
    event_ids = set()
    with open(MODEL_SELECTION_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            try:
                delta_bic = float(row["delta_bic_cv_minus_shrinking"])
            except (KeyError, TypeError, ValueError):
                continue
            if delta_bic >= MASS_SUPPORTED_DELTA_BIC:
                event_ids.add(row["event_id"])
    return event_ids


def source_event_arrays(event_id):
    with h5py.File(SOURCE_H5, "r") as h:
        group = h["points"][event_id]
        return {
            "time_ns": np.asarray(group["all_time_ns"][:], dtype=np.int64),
            "measured_total_paths_m": np.asarray(group["all_measured_total_paths_m"][:], dtype=np.float64),
            "sigma_m": np.asarray(group["all_sigma_m"][:], dtype=np.float64),
            "x_itrs_initial_m": np.asarray(group["all_x_itrs_m"][:], dtype=np.float64),
        }


def leave_one_out_initial_mass(
    measured_total_paths_m,
    times_ns,
    rho_of_alt_m,
    p0,
    sigma_m,
    keep_rows,
):
    measured = np.asarray(measured_total_paths_m, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.int64)
    sigma = np.asarray(sigma_m, dtype=np.float64)
    keep = np.asarray(keep_rows, dtype=bool)
    epoch_time_ns = int(times[0])
    times_fit = times[keep]
    measured_fit = measured[keep]
    sigma_fit = sigma[keep]
    t_rel_s = (times_fit.astype(np.float64) - float(epoch_time_ns)) / 1e9

    def residual(params):
        predicted, *_rest = ceplecha_fit.predict_paths(params, t_rel_s, times_fit, rho_of_alt_m)
        return ((predicted - measured_fit) / sigma_fit).ravel()

    result = so.least_squares(
        residual,
        np.asarray(p0, dtype=np.float64),
        bounds=(
            np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(ceplecha_fit.MIN_RADIUS_M)]),
            np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(ceplecha_fit.MAX_RADIUS_M)]),
        ),
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0]),
        loss=ceplecha_fit.ROBUST_LOSS,
        f_scale=ceplecha_fit.ROBUST_F_SCALE,
        max_nfev=70,
    )
    radius_m = 10.0 ** float(result.x[6])
    return (4.0 / 3.0) * np.pi * ceplecha_fit.METEOROID_DENSITY_KG_M3 * radius_m**3


def cached_leave_one_out_log_mass_std(event_id, review_group):
    os.makedirs(os.path.dirname(LEAVE_ONE_OUT_H5), exist_ok=True)
    with h5py.File(LEAVE_ONE_OUT_H5, "a") as h:
        if event_id in h:
            group = h[event_id]
            return float(group.attrs.get("log10_mass_std", np.nan)), int(group.attrs.get("n_success", 0))

    params = np.asarray(review_group["params"][:], dtype=np.float64)
    manual_reject = np.asarray(review_group["manual_reject_mask"][:], dtype=bool)
    base_keep = ~manual_reject
    if np.count_nonzero(base_keep) <= base_fit.MIN_POINTS:
        return np.nan, 0
    arrays = source_event_arrays(event_id)
    rho_of_alt_m, _meta = base_fit.density_interpolator(
        arrays["time_ns"],
        arrays["x_itrs_initial_m"],
    )

    log_masses = []
    omitted_indices = []
    for omit in np.flatnonzero(base_keep):
        keep = base_keep.copy()
        keep[omit] = False
        if np.count_nonzero(keep) < base_fit.MIN_POINTS:
            continue
        try:
            mass = leave_one_out_initial_mass(
                arrays["measured_total_paths_m"],
                arrays["time_ns"],
                rho_of_alt_m,
                params,
                sigma_m=arrays["sigma_m"],
                keep_rows=keep,
            )
        except Exception:
            continue
        if np.isfinite(mass) and mass > 0.0:
            log_masses.append(np.log10(mass))
            omitted_indices.append(int(omit))

    log_masses = np.asarray(log_masses, dtype=np.float64)
    log10_mass_std = float(np.nanstd(log_masses, ddof=1)) if len(log_masses) >= 2 else np.nan
    with h5py.File(LEAVE_ONE_OUT_H5, "a") as h:
        if event_id in h:
            del h[event_id]
        group = h.create_group(event_id)
        group.attrs["log10_mass_std"] = log10_mass_std
        group.attrs["n_success"] = int(len(log_masses))
        group["log10_initial_mass_kg"] = log_masses
        group["omitted_index"] = np.asarray(omitted_indices, dtype=np.int32)
    return log10_mass_std, int(len(log_masses))


def load_reviewed_initial_mass():
    rows = []
    supported_mass_event_ids = load_bayesian_mass_event_ids()
    with h5py.File(REVIEW_H5, "r") as h:
        reviews = h.get("reviews")
        if reviews is None:
            raise RuntimeError(f"No reviews group in {REVIEW_H5}")
        for event_id, group in reviews.items():
            if int(group.attrs.get("quality", 0)) < 0:
                continue
            if attr_string(group, "model") != "shrinking radius":
                continue
            if supported_mass_event_ids is not None and str(event_id) not in supported_mass_event_ids:
                continue
            required = ("all_radius_m", "all_mass_kg", "all_v_gcrs_mps", "all_residuals_m")
            if any(key not in group for key in required):
                continue
            initial_radius_m = float(np.asarray(group["all_radius_m"][:], dtype=float)[0])
            initial_mass_kg = float(np.asarray(group["all_mass_kg"][:], dtype=float)[0])
            final_radius_m = float(np.asarray(group["all_radius_m"][:], dtype=float)[-1])
            final_mass_kg = float(np.asarray(group["all_mass_kg"][:], dtype=float)[-1])
            log10_radius_std = float(group.attrs.get("log10_radius_std", np.nan))
            residuals_m = np.asarray(group["all_residuals_m"][:], dtype=float)
            rms = float(group.attrs.get("rms_total_path_residual_m", np.sqrt(np.nanmean(residuals_m**2.0))))
            weighted_rms = float(group.attrs.get("weighted_rms", np.nan))
            start_speed_km_s = float(np.linalg.norm(np.asarray(group["all_v_gcrs_mps"][0], dtype=float)) / 1e3)
            loo_log10_mass_std = np.nan
            loo_n_success = 0
            rows.append((
                str(event_id),
                initial_radius_m,
                initial_mass_kg,
                final_radius_m,
                final_mass_kg,
                log10_radius_std,
                rms,
                weighted_rms,
                start_speed_km_s,
                loo_log10_mass_std,
                loo_n_success,
            ))
    return filter_mass_rows(rows)


def load_catalog_initial_mass():
    ceplecha_h5 = sorted(glob.glob(CEPLECHA_H5_GLOB))[-1]
    with h5py.File(ceplecha_h5, "r") as h:
        event_id = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in h["event_id"][:]
        ]
        rows = []
        for idx, event_id_i in enumerate(event_id):
            rows.append((
                event_id_i,
                float(h["initial_radius_m"][idx]),
                float(h["initial_mass_kg"][idx]),
                float(h["final_radius_m"][idx]),
                float(h["final_mass_kg"][idx]),
                float(h["log10_radius_std"][idx]),
                float(h["rms_total_path_residual_m"][idx]),
                float(h["weighted_rms"][idx]),
                float(h["start_speed_km_s"][idx]),
                np.nan,
                0,
            ))
    return filter_mass_rows(rows)


def filter_mass_rows(rows):
    if not rows:
        return tuple(np.asarray([], dtype=float) for _ in range(9))
    table = np.asarray(rows, dtype=object)
    event_id = table[:, 0].astype(str)
    initial_radius_m = table[:, 1].astype(float)
    initial_mass_kg = table[:, 2].astype(float)
    final_radius_m = table[:, 3].astype(float)
    final_mass_kg = table[:, 4].astype(float)
    log10_radius_std = table[:, 5].astype(float)
    rms = table[:, 6].astype(float)
    weighted_rms = table[:, 7].astype(float)
    start_speed_km_s = table[:, 8].astype(float)
    loo_log10_mass_std = table[:, 9].astype(float)
    loo_n_success = table[:, 10].astype(int)

    frac_radius_unc = np.log(10.0) * log10_radius_std
    formal_log10_mass_std = 3.0 * frac_radius_unc / np.log(10.0)
    conservative_log10_mass_std = np.fmax(formal_log10_mass_std, loo_log10_mass_std)
    conservative_log10_mass_std = np.where(
        np.isfinite(conservative_log10_mass_std),
        conservative_log10_mass_std,
        formal_log10_mass_std,
    )
    keep = (
        np.isfinite(initial_mass_kg)
        & (initial_mass_kg > 0.0)
        & np.isfinite(initial_radius_m)
        & (initial_radius_m > 0.0)
        & np.isfinite(log10_radius_std)
        & (log10_radius_std > 0.0)
        & np.isfinite(frac_radius_unc)
        & np.isfinite(start_speed_km_s)
        & np.isfinite(rms)
        & np.isfinite(weighted_rms)
    )
    return (
        event_id[keep],
        initial_radius_m[keep],
        initial_mass_kg[keep],
        final_radius_m[keep],
        final_mass_kg[keep],
        log10_radius_std[keep],
        frac_radius_unc[keep],
        rms[keep],
        start_speed_km_s[keep],
        conservative_log10_mass_std[keep],
        loo_log10_mass_std[keep],
        loo_n_success[keep],
    )


def write_csv(path, ceplecha_rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_id",
            "radius_frac_uncertainty",
            "initial_radius_um",
            "initial_diameter_um",
            "initial_mass_kg",
            "initial_mass_g",
            "final_radius_um",
            "final_mass_kg",
            "log10_radius_std",
            "rms_total_path_residual_m",
            "start_speed_km_s",
            "formal_log10_mass_std",
            "leave_one_out_log10_mass_std",
            "conservative_log10_mass_std",
            "leave_one_out_n_success",
        ])
        for eid, r0, m0, r1, m1, s, fu, rms_i, v0, cons_s, loo_s, loo_n in ceplecha_rows:
            formal_s = 3.0 * fu / np.log(10.0)
            writer.writerow([eid, fu, r0 * 1e6, 2.0 * r0 * 1e6, m0, m0 * 1e3, r1 * 1e6, m1, s, rms_i, v0, formal_s, loo_s, cons_s, loo_n])


def plot_mass_distribution(initial_mass_kg, conservative_log10_mass_std, start_speed_km_s):
    log_mass_kg = np.log10(initial_mass_kg)
    sigma_log_mass = np.asarray(conservative_log10_mass_std, dtype=float)

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    bins = np.linspace(np.nanpercentile(log_mass_kg, 1.0), np.nanpercentile(log_mass_kg, 99.0), 25)
    n_mass, _bins, mass_patches = ax.hist(
        log_mass_kg,
        bins=bins,
        histtype="stepfilled",
        alpha=0.58,
        color="#2f7f6f",
        edgecolor="#17463d",
        linewidth=1.0,
        label=rf"Mass estimates, $n={len(initial_mass_kg)}$",
    )
    ax.axvline(np.median(log_mass_kg), color="#17463d", lw=1.8, ls="--")
    ax.set_xlabel(r"$\log_{10}(m_0)$, initial mass in kg")
    ax.set_ylabel("Number of fitted trajectories")
    ax.set_title(r"Initial-mass distribution")
    ax.set_ylim(0.0, float(np.nanmax(n_mass)) / 0.45)

    ax_points = ax.twinx()
    mass_order = np.argsort(log_mass_kg)
    mass_err = ax_points.errorbar(
        log_mass_kg[mass_order],
        np.asarray(start_speed_km_s, dtype=float)[mass_order],
        xerr=sigma_log_mass[mass_order],
        fmt="o",
        ms=3.6,
        mfc="white",
        mec="#17463d",
        ecolor="#17463d",
        elinewidth=0.65,
        capsize=1.2,
        alpha=0.6,
        linestyle="none",
        label=r"Individual $m_0$ estimates",
    )
    speed = np.asarray(start_speed_km_s, dtype=float)
    speed_min = float(np.nanmin(speed))
    speed_max = float(np.nanmax(speed))
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
        [rf"Mass histogram, $n={len(initial_mass_kg)}$", r"Individual $m_0$ estimates"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.82",
        fontsize=9.2,
    )
    fig.savefig(PAPER_FIGURE, dpi=220)


def summarize(name, values):
    q = np.percentile(values, [5, 25, 50, 75, 95])
    print(f"{name}: p5={q[0]:.6g}, p25={q[1]:.6g}, median={q[2]:.6g}, p75={q[3]:.6g}, p95={q[4]:.6g}")


def main():
    rows = load_constrained_initial_mass()
    _event_id, radius_m, mass_kg = rows[:3]
    write_csv(OUTPUT_CSV, zip(*rows))
    plot_mass_distribution(mass_kg, rows[9], rows[8])
    print(f"mass_estimate_n={len(mass_kg)}")
    summarize("initial_radius_um", radius_m * 1e6)
    summarize("initial_diameter_um", 2.0 * radius_m * 1e6)
    summarize("initial_mass_kg", mass_kg)
    summarize("initial_mass_g", mass_kg * 1e3)
    print(f"figure={PAPER_FIGURE}")
    print(f"csv={os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()
