"""Plot Bayesian shrinking-radius candidate events as standard event plots."""

import argparse
import csv
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so

import bayesian_model_selection_tristatic as bms
import fit_all_ballistic_snr_weighted as base_fit
import fit_all_ceplecha_snr_weighted as ceplecha_fit
import fit_gcrs_trajectories_lfm_ambiguity as gfit
import plot_article_event_fit as event_plot
import review_tristatic_fits_gui as fit_review


SOURCE_H5 = "results/all_tristatic_ceplecha_snr_weighted_v20260616d.h5"
REVIEW_H5 = "results/tristatic_fit_review.h5"
MODEL_SELECTION_CSV = "results/tristatic_bayesian_model_selection.csv"
OUTPUT_BASE = "results/bayesian_model_selection_example"
ARTICLE_FIGURE = "/Users/jvi019/src/sanya_tristatic_paper/figures/bayesian_model_selection_example.png"
RADIUS_UPPER_FLOOR_TEST_FRACTIONS = (0.05, 0.10, 0.30, 0.50, 1.00)
RADIUS_UPPER_FLOOR_DELTA_CHI2 = 1.0


BEAM_CENTER_ECEF_M = np.asarray(
    event_plot.jcoord.geodetic2ecef(
        event_plot.COMMON_VOLUME_LAT_DEG,
        event_plot.COMMON_VOLUME_LON_DEG,
        event_plot.COMMON_VOLUME_ALT_KM * 1e3,
    ),
    dtype=np.float64,
)


def best_event_id():
    with open(MODEL_SELECTION_CSV, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["status"] == "ok"]
    supported = [row for row in rows if float(row["delta_bic_cv_minus_shrinking"]) >= 6.0]
    if not supported:
        supported = [row for row in rows if float(row["delta_bic_cv_minus_shrinking"]) > 0.0]
    best = min(
        supported,
        key=lambda row: float(row["chi2_shrinking_radius"]) / float(row["n_observations"]),
    )
    return best["event_id"], best


def rows_from_selection_csv(path):
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f) if row["status"] == "ok"]


def read_event(event_id):
    with h5py.File(SOURCE_H5, "r") as h:
        group = h["points"][event_id]
        event = {
            "time_ns": np.asarray(group["all_time_ns"][:], dtype=np.int64),
            "measured": np.asarray(group["all_measured_total_paths_m"][:], dtype=np.float64),
            "sigma": np.asarray(group["all_sigma_m"][:], dtype=np.float64),
            "x_itrs_initial": np.asarray(group["all_x_itrs_m"][:], dtype=np.float64),
            "p0_shrinking": np.asarray(group["params"][:], dtype=np.float64),
        }
        if "all_keep_rows" in group:
            event["keep_rows"] = np.asarray(group["all_keep_rows"][:], dtype=bool)
    with h5py.File(REVIEW_H5, "r") as h:
        event["manual_reject_mask"] = np.asarray(h["reviews"][event_id]["manual_reject_mask"][:], dtype=bool)
    if "keep_rows" not in event:
        event["keep_rows"] = ~event["manual_reject_mask"]
    elif len(event["manual_reject_mask"]) == len(event["keep_rows"]):
        event["keep_rows"] = event["keep_rows"] & ~event["manual_reject_mask"]
    event["keep_obs"] = np.repeat(event["keep_rows"][:, None], event["measured"].shape[1], axis=1)
    return event


def fit_constant_velocity(event):
    measured = event["measured"]
    times = event["time_ns"]
    sigma = event["sigma"]
    keep_obs = event["keep_obs"]
    keep_rows = np.any(keep_obs, axis=1)
    epoch_ns = int(times[0])
    x_gcrs_seed = bms.ecef_to_gcrs(event["x_itrs_initial"], times)
    t_rel_all_s = (times.astype(np.float64) - float(epoch_ns)) / 1e9
    p0 = np.concatenate([x_gcrs_seed[0], np.polyfit(t_rel_all_s, x_gcrs_seed, 1)[0]])
    times_fit = times[keep_rows]
    measured_fit = measured[keep_rows]
    sigma_fit = sigma[keep_rows]
    keep_obs_fit = keep_obs[keep_rows]

    def predict(params, query_times_ns):
        t_rel_s = (np.asarray(query_times_ns, dtype=np.float64) - float(epoch_ns)) / 1e9
        x_gcrs = params[:3][None, :] + t_rel_s[:, None] * params[3:6][None, :]
        v_gcrs = np.repeat(params[3:6][None, :], len(t_rel_s), axis=0)
        x_itrs, v_itrs = base_fit.gcrs_state_samples_to_itrs(x_gcrs, v_gcrs, query_times_ns)
        paths_m, path_rates_mps = gfit.link_total_paths_and_rates_m(
            x_itrs,
            v_itrs,
            gfit.LINK_TX_POSITIONS_M,
            gfit.LINK_RX_POSITIONS_M,
        )
        return paths_m + gfit.lfm_total_path_bias_m(path_rates_mps), x_gcrs, v_gcrs, x_itrs, v_itrs

    def residual(params):
        predicted, *_rest = predict(params, times_fit)
        return ((predicted - measured_fit) / sigma_fit)[keep_obs_fit]

    result = so.least_squares(
        residual,
        p0,
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4]),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=220,
    )
    predicted, x_gcrs, v_gcrs, x_itrs, v_itrs = predict(result.x, times)
    return {
        "predicted": predicted,
        "chi2": float(np.sum(residual(result.x) ** 2.0)),
        "x_gcrs": x_gcrs,
        "v_gcrs": v_gcrs,
        "x_itrs": x_itrs,
        "v_itrs": v_itrs,
    }


def fit_shrinking_radius(event):
    measured = event["measured"]
    times = event["time_ns"]
    sigma = event["sigma"]
    keep_obs = event["keep_obs"]
    keep_rows = np.any(keep_obs, axis=1)
    rho_of_alt_m, _meta = base_fit.density_interpolator(times, event["x_itrs_initial"])
    times_fit = times[keep_rows]
    measured_fit = measured[keep_rows]
    sigma_fit = sigma[keep_rows]
    keep_obs_fit = keep_obs[keep_rows]
    t_rel_fit_s = (times_fit.astype(np.float64) - float(times[0])) / 1e9

    def residual(params):
        predicted, *_rest = ceplecha_fit.predict_paths(params, t_rel_fit_s, times_fit, rho_of_alt_m)
        return ((predicted - measured_fit) / sigma_fit)[keep_obs_fit]

    result = so.least_squares(
        residual,
        event["p0_shrinking"],
        bounds=(
            np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(ceplecha_fit.MIN_RADIUS_M)]),
            np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(ceplecha_fit.MAX_RADIUS_M)]),
        ),
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0]),
        loss=ceplecha_fit.ROBUST_LOSS,
        f_scale=ceplecha_fit.ROBUST_F_SCALE,
        max_nfev=220,
    )
    covariance = ceplecha_fit.ceplecha_covariance_summary(
        result,
        int(np.count_nonzero(keep_obs_fit)),
        residual_func=residual,
    )
    t_rel_all_s = (times.astype(np.float64) - float(times[0])) / 1e9
    predicted, x_gcrs, v_gcrs, x_itrs, v_itrs, _radius, _mass, _success, _message = ceplecha_fit.predict_paths(
        result.x,
        t_rel_all_s,
        times,
        rho_of_alt_m,
    )
    radius_m = 10.0 ** float(result.x[6])
    mass_kg = (4.0 / 3.0) * np.pi * ceplecha_fit.METEOROID_DENSITY_KG_M3 * radius_m**3
    log10_radius_std = float(covariance.get("log10_radius_std", np.nan))
    sigma_radius_m, sigma_mass_kg = fit_review.radius_mass_uncertainty(radius_m, mass_kg, log10_radius_std)
    upper_floor_fraction = radius_upper_uncertainty_floor_fraction(result.x, residual)
    radius_upper_m, mass_upper_kg, upper_fraction = radius_mass_upper_limits(
        radius_m,
        mass_kg,
        sigma_radius_m,
        upper_floor_fraction,
    )
    speed0_km_s, sigma_speed0_km_s = speed_uncertainty_km_s(result.x, covariance["parameter_covariance"])
    return {
        "predicted": predicted,
        "chi2": float(np.sum(residual(result.x) ** 2.0)),
        "radius_m": float(radius_m),
        "mass_kg": float(mass_kg),
        "sigma_radius_m": float(sigma_radius_m),
        "sigma_mass_kg": float(sigma_mass_kg),
        "radius_upper_m": float(radius_upper_m),
        "mass_upper_kg": float(mass_upper_kg),
        "radius_upper_fraction": float(upper_fraction),
        "radius_upper_floor_fraction": float(upper_floor_fraction),
        "speed0_km_s": float(speed0_km_s),
        "sigma_speed0_km_s": float(sigma_speed0_km_s),
        "x_gcrs": x_gcrs,
        "v_gcrs": v_gcrs,
        "x_itrs": x_itrs,
        "v_itrs": v_itrs,
    }


def refine_observation_keep(event, sigma_threshold=4.0):
    first_pass = fit_shrinking_radius(event)
    normalized = np.abs((first_pass["predicted"] - event["measured"]) / event["sigma"])
    refined = event["keep_obs"] & np.isfinite(normalized) & (normalized <= sigma_threshold)
    if np.count_nonzero(refined) >= 9:
        event["keep_obs"] = refined
        event["keep_rows"] = np.any(refined, axis=1)
    return event


def altitude_km(points_itrs):
    return np.asarray([event_plot.jcoord.ecef2geodetic(*point)[2] / 1e3 for point in points_itrs], dtype=np.float64)


def projected_components(model, measured_gcrs, along_axis, cross_axis, origin):
    fit_along_m = (model["x_gcrs"] - origin) @ along_axis
    fit_cross_m = (model["x_gcrs"] - origin) @ cross_axis
    meas_along_m = (measured_gcrs - origin) @ along_axis
    meas_cross_m = (measured_gcrs - origin) @ cross_axis
    return {
        "along_residual_m": meas_along_m - fit_along_m,
        "cross_residual_m": meas_cross_m - fit_cross_m,
        "alt_km": altitude_km(model["x_itrs"]),
    }


def rms_m(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(values**2.0)))


def sci_pm_math(value, sigma, unit=""):
    if not np.isfinite(value):
        return rf"\mathrm{{nan}}{unit}"
    if not np.isfinite(sigma):
        return rf"{value:.2g}{unit}"
    if value == 0.0:
        return rf"(0.0 \pm {sigma:.1g}){unit}"
    exponent = int(np.floor(np.log10(abs(value))))
    scale = 10.0**exponent
    mantissa = value / scale
    sigma_mantissa = sigma / scale
    return rf"({mantissa:.2f} \pm {sigma_mantissa:.2f})\times 10^{{{exponent}}}{unit}"


def sci_math(value, unit=""):
    if not np.isfinite(value):
        return rf"\mathrm{{nan}}{unit}"
    if value == 0.0:
        return rf"0{unit}"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.2f}\times 10^{{{exponent}}}{unit}"


def speed_uncertainty_km_s(params, covariance):
    velocity_mps = np.asarray(params[3:6], dtype=np.float64)
    speed_mps = float(np.linalg.norm(velocity_mps))
    if not np.isfinite(speed_mps) or speed_mps <= 0.0:
        return np.nan, np.nan
    cov_v = np.asarray(covariance[3:6, 3:6], dtype=np.float64)
    unit_v = velocity_mps / speed_mps
    sigma_mps = np.sqrt(max(float(unit_v @ cov_v @ unit_v), 0.0))
    return speed_mps / 1e3, sigma_mps / 1e3


def radius_upper_uncertainty_floor_fraction(best_params, residual_func):
    r0 = residual_func(best_params)
    chi2_0 = float(np.sum(r0**2.0))
    floor = 0.0
    for fraction in RADIUS_UPPER_FLOOR_TEST_FRACTIONS:
        perturbed = np.asarray(best_params, dtype=np.float64).copy()
        perturbed[6] += np.log10(1.0 + fraction)
        r = residual_func(perturbed)
        delta_chi2 = float(np.sum(r**2.0) - chi2_0)
        if delta_chi2 < RADIUS_UPPER_FLOOR_DELTA_CHI2:
            floor = max(floor, float(fraction))
    return floor


def radius_mass_upper_limits(radius_m, mass_kg, sigma_radius_m, upper_floor_fraction):
    covariance_fraction = np.nan
    if np.isfinite(radius_m) and radius_m > 0.0 and np.isfinite(sigma_radius_m):
        covariance_fraction = max(0.0, float(sigma_radius_m) / float(radius_m))
    upper_fraction = np.nanmax([covariance_fraction, float(upper_floor_fraction)])
    if not np.isfinite(upper_fraction):
        return np.nan, np.nan, np.nan
    radius_upper_m = float(radius_m) * (1.0 + upper_fraction)
    mass_upper_kg = float(mass_kg) * (radius_upper_m / float(radius_m)) ** 3.0
    return radius_upper_m, mass_upper_kg, float(upper_fraction)


def padded_limits(values, min_half_span):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -float(min_half_span), float(min_half_span)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    center = 0.5 * (low + high)
    half_span = 0.5 * (high - low)
    half_span = max(float(min_half_span), 1.12 * half_span)
    return center - half_span, center + half_span


def plot_event(event_id, best, output_base=OUTPUT_BASE, article_figure=None):
    event = refine_observation_keep(read_event(event_id))
    keep_obs = event["keep_obs"]
    keep_rows = event["keep_rows"]
    rejected_obs = ~keep_obs
    t_rel_s = (event["time_ns"].astype(np.float64) - float(event["time_ns"][0])) / 1e9
    constant_velocity = fit_constant_velocity(event)
    shrinking_radius = fit_shrinking_radius(event)

    sr_path_residual_m = shrinking_radius["predicted"] - event["measured"]
    cv_path_residual_m = constant_velocity["predicted"] - event["measured"]
    sr_normalized_residual = sr_path_residual_m / event["sigma"]
    cv_normalized_residual = cv_path_residual_m / event["sigma"]
    sr_norm_rms_by_pulse = np.full(event["time_ns"].shape, np.nan, dtype=np.float64)
    cv_norm_rms_by_pulse = np.full(event["time_ns"].shape, np.nan, dtype=np.float64)
    for pulse_idx in range(len(t_rel_s)):
        obs = keep_obs[pulse_idx]
        if np.any(obs):
            sr_norm_rms_by_pulse[pulse_idx] = np.sqrt(np.nanmean(sr_normalized_residual[pulse_idx, obs] ** 2.0))
            cv_norm_rms_by_pulse[pulse_idx] = np.sqrt(np.nanmean(cv_normalized_residual[pulse_idx, obs] ** 2.0))
    path_origin_m = event["measured"][0]
    residual_rms = {
        "sr_path_m": rms_m(sr_path_residual_m[keep_obs]),
        "cv_path_m": rms_m(cv_path_residual_m[keep_obs]),
        "sr_normalized": rms_m(sr_normalized_residual[keep_obs]),
        "cv_normalized": rms_m(cv_normalized_residual[keep_obs]),
    }

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.1), constrained_layout=True)
    sr_color = "#1b7837"
    cv_color = "#984ea3"
    link_colors = ("#1f77b4", "#d95f02", "#7570b3")

    ax = axes[0, 0]
    for link_idx, site_label in enumerate(event_plot.SITE_LABELS):
        obs = keep_obs[:, link_idx]
        ax.scatter(
            t_rel_s[obs],
            event["measured"][obs, link_idx] - path_origin_m[link_idx],
            marker="o",
            s=15,
            color=link_colors[link_idx],
            alpha=0.80,
            label=site_label,
        )
        ax.plot(
            t_rel_s,
            shrinking_radius["predicted"][:, link_idx] - path_origin_m[link_idx],
            color=link_colors[link_idx],
            lw=1.25,
            alpha=0.95,
        )
        ax.plot(
            t_rel_s,
            constant_velocity["predicted"][:, link_idx] - path_origin_m[link_idx],
            color=link_colors[link_idx],
            lw=1.0,
            ls="--",
            alpha=0.65,
        )
    if np.any(rejected_obs):
        rejected_t = np.repeat(t_rel_s[:, None], rejected_obs.shape[1], axis=1)[rejected_obs]
        rejected_delay = (event["measured"] - path_origin_m[None, :])[rejected_obs]
        ax.scatter(
            rejected_t,
            rejected_delay,
            marker="x",
            s=17,
            color="0.55",
            alpha=0.45,
            linewidths=0.8,
            label="rejected",
        )
    ax.set_title("Path")
    ax.set_ylabel("Relative path (m)")
    ax.grid(True, color="0.88", lw=0.7)
    ax.legend(loc="best", frameon=False, fontsize=7.4)

    ax = axes[0, 1]
    ax.axhline(0.0, color="0.25", lw=0.8)
    for link_idx, site_label in enumerate(event_plot.SITE_LABELS):
        obs = keep_obs[:, link_idx]
        ax.errorbar(
            t_rel_s[obs],
            sr_path_residual_m[obs, link_idx],
            yerr=event["sigma"][obs, link_idx],
            fmt="o",
            ms=2.8,
            lw=0.6,
            capsize=1.0,
            color=link_colors[link_idx],
            ecolor="0.65",
            alpha=0.90,
            label=site_label,
        )
    ax.set_title("Shrinking radius")
    ax.set_ylabel("Residual (m)")
    ax.set_ylim(*padded_limits(sr_path_residual_m[keep_obs], min_half_span=25.0))
    ax.grid(True, color="0.88", lw=0.7)

    ax = axes[1, 0]
    ax.axhline(0.0, color="0.25", lw=0.8)
    for link_idx, site_label in enumerate(event_plot.SITE_LABELS):
        obs = keep_obs[:, link_idx]
        ax.errorbar(
            t_rel_s[obs],
            cv_path_residual_m[obs, link_idx],
            yerr=event["sigma"][obs, link_idx],
            fmt="o",
            ms=2.8,
            lw=0.6,
            capsize=1.0,
            color=link_colors[link_idx],
            ecolor="0.65",
            alpha=0.90,
            label=site_label,
        )
    ax.set_title("Constant velocity")
    ax.set_xlabel("Time since first pulse (s)")
    ax.set_ylabel("Residual (m)")
    ax.set_ylim(*padded_limits(cv_path_residual_m[keep_obs], min_half_span=25.0))
    ax.grid(True, color="0.88", lw=0.7)

    ax = axes[1, 1]
    ax.scatter(t_rel_s[keep_rows], cv_norm_rms_by_pulse[keep_rows], marker="x", s=18, color=cv_color, alpha=0.75, label="CV")
    ax.scatter(t_rel_s[keep_rows], sr_norm_rms_by_pulse[keep_rows], marker="o", s=16, color=sr_color, alpha=0.90, label="SR")
    ax.set_ylim(bottom=0.0)
    n_observations = int(np.count_nonzero(keep_obs))
    delta_bic = bms.bic(constant_velocity["chi2"], n_observations, bms.K_CONSTANT_VELOCITY) - bms.bic(
        shrinking_radius["chi2"],
        n_observations,
        bms.K_SHRINKING_RADIUS,
    )
    ax.text(
        0.03,
        0.05,
        (
            f"$\\Delta$BIC={delta_bic:.1f}\n"
            rf"$|\boldsymbol{{v}}_0|={shrinking_radius['speed0_km_s']:.2f}"
            rf"\pm {shrinking_radius['sigma_speed0_km_s']:.2f}"
            rf"\,\mathrm{{km\,s^{{-1}}}}$"
            "\n"
            rf"$r_0={sci_math(shrinking_radius['radius_m'] * 1e6, unit=r'\,\mu\mathrm{{m}}')}$"
            rf", $r_0<{sci_math(shrinking_radius['radius_upper_m'] * 1e6, unit=r'\,\mu\mathrm{{m}}')}$"
            "\n"
            rf"$m_0={sci_math(shrinking_radius['mass_kg'], unit=r'\,\mathrm{{kg}}')}$"
            rf", $m_0<{sci_math(shrinking_radius['mass_upper_kg'], unit=r'\,\mathrm{{kg}}')}$"
        ),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
    )
    ax.set_title("Fit RMS")
    ax.set_xlabel("Time since first pulse (s)")
    ax.set_ylabel("Normalized RMS")
    ax.grid(True, color="0.88", lw=0.7)
    ax.legend(loc="best", frameon=False)

    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    png = f"{output_base}_{event_id}.png"
    pdf = f"{output_base}_{event_id}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    if article_figure:
        os.makedirs(os.path.dirname(article_figure), exist_ok=True)
        fig.savefig(article_figure, dpi=300)
    plt.close(fig)
    print(f"event_id={event_id}")
    print(f"delta_bic={delta_bic:.6g}")
    print(f"radius_um={shrinking_radius['radius_m'] * 1e6:.6g}")
    print(f"sigma_radius_um={shrinking_radius['sigma_radius_m'] * 1e6:.6g}")
    print(f"radius_upper_um={shrinking_radius['radius_upper_m'] * 1e6:.6g}")
    print(f"radius_upper_fraction={shrinking_radius['radius_upper_fraction']:.6g}")
    print(f"radius_upper_floor_fraction={shrinking_radius['radius_upper_floor_fraction']:.6g}")
    print(f"mass_kg={shrinking_radius['mass_kg']:.6g}")
    print(f"sigma_mass_kg={shrinking_radius['sigma_mass_kg']:.6g}")
    print(f"mass_upper_kg={shrinking_radius['mass_upper_kg']:.6g}")
    print(f"speed0_km_s={shrinking_radius['speed0_km_s']:.6g}")
    print(f"sigma_speed0_km_s={shrinking_radius['sigma_speed0_km_s']:.6g}")
    print(
        "rms_residual_m="
        f"shrinking_radius_path:{residual_rms['sr_path_m']:.3f},"
        f"constant_velocity_path:{residual_rms['cv_path_m']:.3f},"
        f"shrinking_radius_normalized:{residual_rms['sr_normalized']:.3f},"
        f"constant_velocity_normalized:{residual_rms['cv_normalized']:.3f}"
    )
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    if article_figure:
        print(f"wrote {article_figure}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", default=MODEL_SELECTION_CSV)
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--all-positive", action="store_true")
    parser.add_argument("--output-base", default=OUTPUT_BASE)
    parser.add_argument("--copy-to-article", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = rows_from_selection_csv(args.selection_csv)
    by_id = {row["event_id"]: row for row in rows}

    if args.all_positive:
        targets = [row for row in rows if float(row["delta_bic_cv_minus_shrinking"]) > 0.0]
        targets.sort(key=lambda row: (-float(row["delta_bic_cv_minus_shrinking"]), row["event_id"]))
        print(f"batch_events={len(targets)}")
        for row in targets:
            plot_event(row["event_id"], row, output_base=args.output_base, article_figure=None)
        return

    if args.event_id is not None:
        if args.event_id not in by_id:
            raise SystemExit(f"event_id not found in selection csv: {args.event_id}")
        plot_event(
            args.event_id,
            by_id[args.event_id],
            output_base=args.output_base,
            article_figure=ARTICLE_FIGURE if args.copy_to_article else None,
        )
        return

    event_id, best = best_event_id()
    plot_event(
        event_id,
        best,
        output_base=args.output_base,
        article_figure=ARTICLE_FIGURE,
    )


if __name__ == "__main__":
    main()
