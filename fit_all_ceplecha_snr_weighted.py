"""Fit tri-static Sanya events with the shared Ceplecha drag/ablation model.

This script mirrors ``fit_all_ballistic_snr_weighted.py`` for data loading,
matched-filter refinement, SNR weighting, sigma clipping, and Sanya
tx-target-rx path prediction.  The propagation step is intentionally delegated
to the reusable ``meteor_trajectory_models`` package.
"""

import argparse
import json
import os

import h5py
import jcoord
import numpy as np
import scipy.optimize as so

import fit_all_ballistic_snr_weighted as base
import fit_gcrs_trajectories_lfm_ambiguity as gfit
import sanya_opts as sc
from grid_search_delays_beam_axis import DAN_PATTERN, SAN_PATTERN, WEN_PATTERN, load_events, pair_tristatic_events
from meteor_trajectory_models import integrate_ceplecha


SCRIPT_VERSION = "v20260616d"
OUTPUT_H5 = os.path.join("results", f"all_tristatic_ceplecha_snr_weighted_{SCRIPT_VERSION}.h5")
OUTPUT_JSON = os.path.join("results", f"all_tristatic_ceplecha_snr_weighted_{SCRIPT_VERSION}.json")

MIN_RADIUS_M = 1e-7
MAX_RADIUS_M = 1e-2
METEOROID_DENSITY_KG_M3 = 3000.0
ABLATION_SIGMA_KG_J = 1e-8
SPHERICAL_EARTH_RADIUS_M = 6371.0e3
CEPLECHA_SAMPLE_DT_S = 5e-4
PER_PULSE_NORM_CLIP_RMS = 2.5
PER_PULSE_ABS_CLIP_RMS_M = 75.0
ROBUST_LOSS = "soft_l1"
ROBUST_F_SCALE = 1.0
MAX_CLIP_ITERATIONS = 3
COVARIANCE_POSITION_STEP_M = 30.0
COVARIANCE_VELOCITY_FRACTIONAL_STEP = 0.01
COVARIANCE_MIN_VELOCITY_STEP_MPS = 1.0
COVARIANCE_LOG10_RADIUS_STEP = np.log10(1.1)
RADIUS_START_GRID_UM = np.array([3.0, 7.0, 15.0, 30.0, 80.0, 300.0], dtype=np.float64)
LINE_FIT_SPANS = ((0.0, 1.0), (0.0, 0.8), (0.1, 0.9), (0.2, 1.0))
N_OPTIMIZED_STARTS = 3
MAX_EDGE_REJECT_POINTS = 3
MAX_REJECT_RUN_POINTS = 3
MIN_RETAINED_FRACTION = 0.65
MANUAL_REVIEW_MASK_SOURCE = None


def is_root():
    return base.RANK == 0


def log(message):
    if is_root():
        print(message, flush=True)


def radius_from_drag_guess(log10_b_drag, density_kg_m3=METEOROID_DENSITY_KG_M3):
    """Map the fixed-drag coefficient to the equivalent Ceplecha radius."""

    b_drag = 10.0 ** float(log10_b_drag)
    radius = 3.0 / (4.0 * density_kg_m3 * max(b_drag, 1e-30))
    return float(np.clip(radius, MIN_RADIUS_M, MAX_RADIUS_M))


def initial_ceplecha_guess(points_ecef_m, times_ns):
    p0_ballistic = base.initial_ballistic_guess(points_ecef_m, times_ns, log10_b=1.0)
    radius0_m = radius_from_drag_guess(p0_ballistic[6])
    return np.concatenate([p0_ballistic[:6], [np.log10(radius0_m)]])


def line_state_guess(points_ecef_m, times_ns, span=(0.0, 1.0), log10_radius_m=np.log10(20e-6)):
    points_gcrs_m = gfit.ecef_to_gcrs(points_ecef_m, times_ns)
    t_rel_s = (np.asarray(times_ns, dtype=np.float64) - float(times_ns[0])) / 1e9
    n = len(t_rel_s)
    start = int(np.floor(float(span[0]) * n))
    stop = int(np.ceil(float(span[1]) * n))
    start = max(0, min(start, n - 2))
    stop = max(start + 2, min(stop, n))
    design = np.column_stack([np.ones(stop - start), t_rel_s[start:stop]])
    coeffs = np.linalg.lstsq(design, points_gcrs_m[start:stop], rcond=None)[0]
    return np.concatenate([coeffs[0], coeffs[1], [float(log10_radius_m)]])


def reference_state_guess(reference_fit, epoch_time_ns, log10_radius_m=np.log10(20e-6)):
    if reference_fit is None:
        return None
    dt_s = (float(epoch_time_ns) - float(reference_fit["t0_ns"])) / 1e9
    r0 = np.asarray(reference_fit["r0_gcrs_m"], dtype=np.float64) + dt_s * np.asarray(reference_fit["v0_gcrs_mps"], dtype=np.float64)
    v0 = np.asarray(reference_fit["v0_gcrs_mps"], dtype=np.float64)
    return np.concatenate([r0, v0, [float(log10_radius_m)]])


def unique_initial_guesses(points_ecef_m, times_ns, reference_fit=None):
    base_guesses = []
    for span in LINE_FIT_SPANS:
        try:
            base_guesses.append(line_state_guess(points_ecef_m, times_ns, span=span))
        except Exception:
            continue
    ref_guess = reference_state_guess(reference_fit, int(times_ns[0]))
    if ref_guess is not None:
        base_guesses.append(ref_guess)
    try:
        base_guesses.append(initial_ceplecha_guess(points_ecef_m, times_ns))
    except Exception:
        pass

    guesses = []
    seen = set()
    for base_guess in base_guesses:
        if base_guess is None or not np.all(np.isfinite(base_guess[:6])):
            continue
        for radius_um in RADIUS_START_GRID_UM:
            guess = np.asarray(base_guess, dtype=np.float64).copy()
            guess[6] = np.log10(radius_um * 1e-6)
            key = tuple(np.round(np.concatenate([guess[:3] / 100.0, guess[3:6] / 10.0, [guess[6]]]), 3))
            if key in seen:
                continue
            seen.add(key)
            guesses.append(guess)
    return guesses


def normalized_residual_score(fit):
    normalized = np.asarray(fit["normalized_residuals"], dtype=np.float64)
    per_pulse = np.sqrt(np.nanmean(normalized**2.0, axis=1))
    rms = float(np.sqrt(np.nanmean(normalized**2.0)))
    p90 = float(np.nanpercentile(per_pulse, 90.0))
    edge = float(max(per_pulse[0], per_pulse[-1])) if len(per_pulse) else np.inf
    return rms + 0.15 * p90 + 0.05 * edge


def initial_guess_score(measured, times_ns, rho_of_alt_m, p0, sigma_m=None, keep_rows=None, epoch_time_ns=None):
    measured = np.asarray(measured, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.int64)
    if keep_rows is None:
        keep_rows = np.ones(len(times), dtype=bool)
    if epoch_time_ns is None:
        epoch_time_ns = int(times[0])
    measured_fit = measured[keep_rows]
    times_fit = times[keep_rows]
    t_rel_s = (times_fit.astype(np.float64) - float(epoch_time_ns)) / 1e9
    if sigma_m is None:
        sigma = np.ones_like(measured_fit)
    else:
        sigma = np.asarray(sigma_m, dtype=np.float64)[keep_rows]
    try:
        pred, *_rest = predict_paths(p0, t_rel_s, times_fit, rho_of_alt_m)
    except Exception:
        return np.inf
    normalized = (pred - measured_fit) / sigma
    per_pulse = np.sqrt(np.nanmean(normalized**2.0, axis=1))
    rms = float(np.sqrt(np.nanmean(normalized**2.0)))
    p90 = float(np.nanpercentile(per_pulse, 90.0))
    edge = float(max(per_pulse[0], per_pulse[-1])) if len(per_pulse) else np.inf
    return rms + 0.15 * p90 + 0.05 * edge


def fit_ceplecha_multistart(measured, times_ns, rho_of_alt_m, guesses, sigma_m=None, keep_rows=None, epoch_time_ns=None):
    screened = []
    for p0 in guesses:
        score = initial_guess_score(
            measured,
            times_ns,
            rho_of_alt_m,
            p0,
            sigma_m=sigma_m,
            keep_rows=keep_rows,
            epoch_time_ns=epoch_time_ns,
        )
        if np.isfinite(score):
            screened.append((score, p0))
    screened.sort(key=lambda item: item[0])
    selected = [p0 for _score, p0 in screened[:N_OPTIMIZED_STARTS]]
    if not selected:
        selected = list(guesses[:N_OPTIMIZED_STARTS])

    best = None
    failures = 0
    for p0 in selected:
        try:
            fit = fit_ceplecha(
                measured,
                times_ns,
                rho_of_alt_m,
                p0,
                sigma_m=sigma_m,
                keep_rows=keep_rows,
                epoch_time_ns=epoch_time_ns,
            )
        except Exception:
            failures += 1
            continue
        score = normalized_residual_score(fit)
        fit["multistart_score"] = float(score)
        if best is None or score < best["multistart_score"]:
            best = fit
    if best is None:
        raise RuntimeError(f"All Ceplecha multistart attempts failed; failures={failures}")
    best["multistart_n_guesses"] = int(len(guesses))
    best["multistart_n_screened"] = int(len(screened))
    best["multistart_n_optimized"] = int(len(selected))
    best["multistart_n_failures"] = int(failures)
    return best


def clipping_is_isolated(keep_rows):
    keep = np.asarray(keep_rows, dtype=bool)
    n = len(keep)
    if n == 0 or np.sum(keep) < base.MIN_POINTS:
        return False
    if np.mean(keep) < MIN_RETAINED_FRACTION:
        return False
    kept = np.flatnonzero(keep)
    if kept[0] > MAX_EDGE_REJECT_POINTS or (n - 1 - kept[-1]) > MAX_EDGE_REJECT_POINTS:
        return False
    reject = ~keep
    if not np.any(reject):
        return True
    changes = np.diff(reject.astype(np.int8))
    starts = list(np.flatnonzero(changes == 1) + 1)
    ends = list(np.flatnonzero(changes == -1))
    if reject[0]:
        starts = [0] + starts
    if reject[-1]:
        ends = ends + [n - 1]
    longest_reject_run = max((end - start + 1 for start, end in zip(starts, ends)), default=0)
    return longest_reject_run <= MAX_REJECT_RUN_POINTS


def load_manual_review_masks(path):
    if not path:
        return {}
    masks = {}
    with h5py.File(path, "r") as h:
        if "reviews" not in h:
            return masks
        for event_id, group in h["reviews"].items():
            if "manual_reject_mask" in group:
                masks[event_id] = np.asarray(group["manual_reject_mask"][:], dtype=bool)
    return masks


def review_keep_rows(event_id, n_points, manual_masks):
    if not manual_masks or event_id not in manual_masks:
        return np.ones(n_points, dtype=bool), np.zeros(n_points, dtype=bool), False
    mask = np.asarray(manual_masks[event_id], dtype=bool)
    if len(mask) != n_points:
        log(f"manual review mask length mismatch for {event_id}: mask={len(mask)} points={n_points}; ignoring")
        return np.ones(n_points, dtype=bool), np.zeros(n_points, dtype=bool), False
    keep_rows = ~mask
    if np.count_nonzero(keep_rows) < base.MIN_POINTS:
        log(f"manual review mask for {event_id} leaves too few points; ignoring")
        return np.ones(n_points, dtype=bool), np.zeros(n_points, dtype=bool), False
    return keep_rows, mask, True


def propagate_ceplecha(params, t_rel_s, rho_of_alt_m):
    t_rel = np.asarray(t_rel_s, dtype=np.float64)
    radius0_m = 10.0 ** float(params[6])
    t1 = float(np.max(t_rel)) if len(t_rel) else 0.0
    t1 = max(t1, CEPLECHA_SAMPLE_DT_S)

    result = integrate_ceplecha(
        params[:3],
        params[3:6],
        radius0_m,
        rho_of_alt_m,
        meteoroid_density_kg_m3=METEOROID_DENSITY_KG_M3,
        ablation_sigma_kg_j=ABLATION_SIGMA_KG_J,
        t_span_s=(0.0, t1),
        sample_dt_s=min(CEPLECHA_SAMPLE_DT_S, max(t1 / 5.0, 1e-6)),
        height_function=lambda r: float(np.linalg.norm(r) - SPHERICAL_EARTH_RADIUS_M),
    )
    if result.time_s.size < 2:
        raise RuntimeError(f"Ceplecha integration returned too few samples: {result.message}")

    x_gcrs = np.column_stack([np.interp(t_rel, result.time_s, result.position_m[:, dim]) for dim in range(3)])
    v_gcrs = np.column_stack([np.interp(t_rel, result.time_s, result.velocity_mps[:, dim]) for dim in range(3)])
    radius_m = np.interp(t_rel, result.time_s, result.radius_m)
    mass_kg = np.interp(t_rel, result.time_s, result.mass_kg)
    return x_gcrs, v_gcrs, radius_m, mass_kg, result.success, result.message


def predict_paths(params, t_rel_s, times_ns, rho_of_alt_m):
    x_gcrs, v_gcrs, radius_m, mass_kg, success, message = propagate_ceplecha(params, t_rel_s, rho_of_alt_m)
    x_itrs, v_itrs = base.gcrs_state_samples_to_itrs(x_gcrs, v_gcrs, times_ns)
    total_paths_m, path_rates_mps = gfit.link_total_paths_and_rates_m(
        x_itrs,
        v_itrs,
        gfit.LINK_TX_POSITIONS_M,
        gfit.LINK_RX_POSITIONS_M,
    )
    return (
        total_paths_m + gfit.lfm_total_path_bias_m(path_rates_mps),
        x_gcrs,
        v_gcrs,
        x_itrs,
        v_itrs,
        radius_m,
        mass_kg,
        success,
        message,
    )


def ceplecha_covariance_summary(result, n_residuals, residual_func=None):
    velocity_step = np.maximum(
        COVARIANCE_MIN_VELOCITY_STEP_MPS,
        COVARIANCE_VELOCITY_FRACTIONAL_STEP * np.abs(np.asarray(result.x[3:6], dtype=np.float64)),
    )
    jac_abs_step = np.array(
        [
            COVARIANCE_POSITION_STEP_M,
            COVARIANCE_POSITION_STEP_M,
            COVARIANCE_POSITION_STEP_M,
            velocity_step[0],
            velocity_step[1],
            velocity_step[2],
            COVARIANCE_LOG10_RADIUS_STEP,
        ],
        dtype=np.float64,
    )
    cov = base.linearized_covariance_summary(
        result,
        n_residuals,
        residual_func=residual_func,
        jac_abs_step=jac_abs_step,
        jac_bounds=(
            np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(MIN_RADIUS_M)]),
            np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(MAX_RADIUS_M)]),
        ),
    )
    cov["log10_radius_std"] = float(cov["parameter_std"][6]) if len(cov["parameter_std"]) > 6 else np.nan
    return cov


def fit_ceplecha(
    measured_total_paths_m,
    times_ns,
    rho_of_alt_m,
    p0,
    sigma_m=None,
    keep_rows=None,
    epoch_time_ns=None,
    robust_f_scale=ROBUST_F_SCALE,
    loss=ROBUST_LOSS,
):
    measured = np.asarray(measured_total_paths_m, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.int64)
    if keep_rows is None:
        keep_rows = np.ones(len(times), dtype=bool)
    if epoch_time_ns is None:
        epoch_time_ns = int(times[0])
    measured_fit = measured[keep_rows]
    times_fit = times[keep_rows]
    t_rel_s = (times_fit.astype(np.float64) - float(epoch_time_ns)) / 1e9
    if sigma_m is None:
        sigma = np.ones_like(measured_fit)
        f_scale = 50.0
    else:
        sigma = np.asarray(sigma_m, dtype=np.float64)[keep_rows]
        f_scale = robust_f_scale

    def residual(x):
        pred, *_rest = predict_paths(x, t_rel_s, times_fit, rho_of_alt_m)
        return ((pred - measured_fit) / sigma).ravel()

    result = so.least_squares(
        residual,
        p0,
        bounds=(
            np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(MIN_RADIUS_M)]),
            np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(MAX_RADIUS_M)]),
        ),
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0]),
        loss=loss,
        f_scale=f_scale,
        max_nfev=260,
    )
    pred, x_gcrs, v_gcrs, x_itrs, v_itrs, radius_m, mass_kg, cepl_success, cepl_message = predict_paths(
        result.x,
        t_rel_s,
        times_fit,
        rho_of_alt_m,
    )
    raw_resid = pred - measured_fit
    normalized = raw_resid / sigma
    llh = np.asarray([jcoord.ecef2geodetic(x[0], x[1], x[2]) for x in x_itrs], dtype=np.float64)
    covariance = ceplecha_covariance_summary(result, len(residual(result.x)), residual_func=residual)
    return {
        "params": result.x,
        "parameter_covariance": covariance["parameter_covariance"],
        "parameter_std": covariance["parameter_std"],
        "position_std_m": covariance["position_std_m"],
        "velocity_std_mps": covariance["velocity_std_mps"],
        "log10_radius_std": covariance["log10_radius_std"],
        "covariance_available": covariance["covariance_available"],
        "covariance_degrees_of_freedom": covariance["degrees_of_freedom"],
        "covariance_residual_variance": covariance["residual_variance"],
        "keep_rows": keep_rows,
        "time_ns": times_fit,
        "fit_epoch_time_ns": int(epoch_time_ns),
        "t_rel_s": t_rel_s,
        "measured_total_paths_m": measured_fit,
        "predicted_total_paths_m": pred,
        "residuals_m": raw_resid,
        "normalized_residuals": normalized,
        "x_gcrs_m": x_gcrs,
        "v_gcrs_mps": v_gcrs,
        "x_itrs_m": x_itrs,
        "v_itrs_mps": v_itrs,
        "lat_deg": llh[:, 0],
        "lon_deg": llh[:, 1],
        "alt_km": llh[:, 2] / 1e3,
        "speed_km_s": np.linalg.norm(v_gcrs, axis=1) / 1e3,
        "itrs_speed_km_s": np.linalg.norm(v_itrs, axis=1) / 1e3,
        "radius_m": radius_m,
        "mass_kg": mass_kg,
        "initial_radius_m": float(radius_m[0]),
        "final_radius_m": float(radius_m[-1]),
        "initial_mass_kg": float(mass_kg[0]),
        "final_mass_kg": float(mass_kg[-1]),
        "rms_total_path_residual_m": float(np.sqrt(np.mean(raw_resid**2.0))),
        "median_abs_total_path_residual_m": float(np.median(np.abs(raw_resid))),
        "weighted_rms": float(np.sqrt(np.mean(normalized**2.0))),
        "max_abs_total_path_residual_m": float(np.nanmax(np.abs(raw_resid))),
        "max_abs_link_mean_residual_m": float(np.nanmax(np.abs(np.nanmean(raw_resid, axis=0)))),
        "max_abs_link_median_residual_m": float(np.nanmax(np.abs(np.nanmedian(raw_resid, axis=0)))),
        "max_per_pulse_normalized_residual": float(
            np.nanmax(np.sqrt(np.mean(normalized**2.0, axis=1)))
        ),
        "n_points": int(len(times_fit)),
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "optimizer_cost": float(result.cost),
        "optimizer_loss": loss,
        "ceplecha_success": bool(cepl_success),
        "ceplecha_message": str(cepl_message),
    }


def process_triplet(idx, triplet, ref_fit, sigma_model=None, manual_masks=None):
    san_event, dan_event, wen_event = triplet
    raw_event_id = f"tri_{idx:04d}_{san_event.t0_ns}"
    fit0 = base.match_reference_fit(san_event, ref_fit)
    if fit0 is None:
        return {"event_id": raw_event_id, "status": "missing_reference_fit"}
    event_id = fit0["event_id"]
    try:
        site_data = {
            "sanya": base.load_site_h5(san_event.path, fit0, "sanya"),
            "danzhou": base.load_site_h5(dan_event.path, fit0, "danzhou"),
            "wenchang": base.load_site_h5(wen_event.path, fit0, "wenchang"),
        }
        refined = {}
        for site in ("sanya", "danzhou", "wenchang"):
            gate, range_km, _power_db = base.refine_site(site_data[site])
            refined[f"{site}_gate"] = gate
            refined[f"{site}_range_km"] = range_km
        refined["sanya_range_km"] = refined["sanya_range_km"] + sc.SANYA_RANGE_CORRECTION_KM
        measured, times_ns, beijing_local_times_ns, snr_db, source_indices = base.matched_measurements_from_sites(
            san_event,
            dan_event,
            wen_event,
            site_data,
            refined,
        )
        if len(times_ns) < base.MIN_POINTS:
            return {"event_id": event_id, "status": "too_few_points", "n_points": int(len(times_ns))}
        order = np.argsort(times_ns)
        measured = measured[order]
        times_ns = times_ns[order]
        beijing_local_times_ns = beijing_local_times_ns[order]
        snr_db = snr_db[order]
        source_indices = source_indices[order]
        points, keep_geo = base.triangulate_points(measured, san_event.az_deg, san_event.el_deg)
        measured = measured[keep_geo]
        times_ns = times_ns[keep_geo]
        beijing_local_times_ns = beijing_local_times_ns[keep_geo]
        snr_db = snr_db[keep_geo]
        source_indices = source_indices[keep_geo]
        if len(times_ns) < base.MIN_POINTS:
            return {"event_id": event_id, "status": "too_few_geo_points", "n_points": int(len(times_ns))}
        manual_keep_rows, manual_reject_mask, manual_review_applied = review_keep_rows(event_id, len(times_ns), manual_masks)
        rho_of_alt_m, msis_meta = base.density_interpolator(times_ns, points)
        initial_guesses = unique_initial_guesses(points, times_ns, reference_fit=fit0)
        if sigma_model is None:
            fit = fit_ceplecha_multistart(
                measured,
                times_ns,
                rho_of_alt_m,
                initial_guesses,
                sigma_m=None,
                keep_rows=manual_keep_rows,
                epoch_time_ns=int(times_ns[0]),
            )
            sigma_used = None
        else:
            sigma_used = base.sigma_from_snr_db(snr_db, sigma_model["sigma_floor_m"], sigma_model["sigma_0_m"])
            keep_rows = manual_keep_rows.copy()
            epoch_time_ns = int(times_ns[0])
            fit = fit_ceplecha_multistart(
                measured,
                times_ns,
                rho_of_alt_m,
                initial_guesses,
                sigma_m=sigma_used,
                keep_rows=keep_rows,
                epoch_time_ns=epoch_time_ns,
            )
            clipping_disallowed = False
            for _clip_iter in range(MAX_CLIP_ITERATIONS):
                per_pulse_norm = np.sqrt(np.mean(fit["normalized_residuals"] ** 2.0, axis=1))
                per_pulse_abs_m = np.sqrt(np.mean(fit["residuals_m"] ** 2.0, axis=1))
                candidate_keep = keep_rows.copy()
                kept_indices = np.flatnonzero(fit["keep_rows"])
                candidate_keep[kept_indices] = (
                    (per_pulse_norm < PER_PULSE_NORM_CLIP_RMS)
                    & (per_pulse_abs_m < PER_PULSE_ABS_CLIP_RMS_M)
                )
                if np.sum(candidate_keep) < base.MIN_POINTS or np.array_equal(candidate_keep, keep_rows):
                    break
                if not clipping_is_isolated(candidate_keep):
                    clipping_disallowed = True
                    break
                keep_rows = candidate_keep
                fit = fit_ceplecha(
                    measured,
                    times_ns,
                    rho_of_alt_m,
                    fit["params"],
                    sigma_m=sigma_used,
                    keep_rows=keep_rows,
                    epoch_time_ns=epoch_time_ns,
                )
            fit["initial_n_points"] = int(len(times_ns))
            fit["n_clipped_points"] = int(len(times_ns) - np.sum(fit["keep_rows"]))
            fit["clip_fraction"] = float(fit["n_clipped_points"] / len(times_ns))
            fit["clipping_disallowed"] = bool(clipping_disallowed)
            fit["clipping_is_isolated"] = bool(clipping_is_isolated(fit["keep_rows"]))
        fit["manual_review_applied"] = bool(manual_review_applied)
        fit["manual_reject_mask"] = np.asarray(manual_reject_mask, dtype=bool)
        fit["n_manual_reject_points"] = int(np.count_nonzero(manual_reject_mask))
        all_t_rel_s = (times_ns.astype(np.float64) - float(fit["fit_epoch_time_ns"])) / 1e9
        (
            all_predicted_total_paths_m,
            all_x_gcrs_m,
            all_v_gcrs_mps,
            all_x_itrs_m,
            all_v_itrs_mps,
            all_radius_m,
            all_mass_kg,
            _all_cepl_success,
            _all_cepl_message,
        ) = predict_paths(fit["params"], all_t_rel_s, times_ns, rho_of_alt_m)
        all_llh = np.asarray([jcoord.ecef2geodetic(x[0], x[1], x[2]) for x in all_x_itrs_m], dtype=np.float64)
        all_residuals_m = all_predicted_total_paths_m - measured
        return {
            "event_id": event_id,
            "status": "ok",
            "msis": msis_meta,
            "beijing_local_time_ns": beijing_local_times_ns,
            "snr_db": snr_db,
            "source_indices": source_indices,
            "sigma_m": sigma_used,
            "all_time_ns": times_ns,
            "all_beijing_local_time_ns": beijing_local_times_ns,
            "all_measured_total_paths_m": measured,
            "all_snr_db": snr_db,
            "all_source_indices": source_indices,
            "all_t_rel_s": all_t_rel_s,
            "all_predicted_total_paths_m": all_predicted_total_paths_m,
            "all_residuals_m": all_residuals_m,
            "all_x_gcrs_m": all_x_gcrs_m,
            "all_v_gcrs_mps": all_v_gcrs_mps,
            "all_x_itrs_m": all_x_itrs_m,
            "all_v_itrs_mps": all_v_itrs_mps,
            "all_lat_deg": all_llh[:, 0],
            "all_lon_deg": all_llh[:, 1],
            "all_alt_km": all_llh[:, 2] / 1e3,
            "all_speed_km_s": np.linalg.norm(all_v_gcrs_mps, axis=1) / 1e3,
            "all_itrs_speed_km_s": np.linalg.norm(all_v_itrs_mps, axis=1) / 1e3,
            "all_radius_m": all_radius_m,
            "all_mass_kg": all_mass_kg,
            **fit,
        }
    except Exception as exc:
        return {"event_id": event_id, "status": "error", "error": repr(exc)}


def local_process(triplets, ref_fit, sigma_model=None, manual_masks=None):
    outputs = []
    for idx in range(base.RANK, len(triplets), base.SIZE):
        out = process_triplet(idx, triplets[idx], ref_fit, sigma_model=sigma_model, manual_masks=manual_masks)
        outputs.append(out)
        if out["status"] == "ok":
            print(
                f"[rank {base.RANK}] ok {out['event_id']} n={out['n_points']} "
                f"rms={out['rms_total_path_residual_m']:.1f} m r0={out['initial_radius_m'] * 1e6:.1f} um",
                flush=True,
            )
        elif len(outputs) % 5 == 0:
            print(f"[rank {base.RANK}] processed {len(outputs)} local events; last_status={out['status']}", flush=True)
    return outputs


def gather_outputs(local_outputs):
    return base.gather_outputs(local_outputs)


def write_results(path, outputs, sigma_model):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    ok = [o for o in outputs if o["status"] == "ok"]
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["trajectory_model_package"] = "meteor_trajectory_models"
        h.attrs["dynamics_model"] = (
            "Ceplecha-style spherical meteoroid drag and ablation from meteor_trajectory_models; "
            "GCRS state samples are transformed to ITRS for Sanya tri-static tx-target-rx path prediction."
        )
        h.attrs["meteoroid_density_kg_m3"] = METEOROID_DENSITY_KG_M3
        h.attrs["ablation_sigma_kg_j"] = ABLATION_SIGMA_KG_J
        h.attrs["ceplecha_height_model"] = "spherical height norm(r_gcrs)-6371 km"
        h.attrs["ceplecha_sample_dt_s"] = CEPLECHA_SAMPLE_DT_S
        h.attrs["per_pulse_norm_clip_rms"] = PER_PULSE_NORM_CLIP_RMS
        h.attrs["per_pulse_abs_clip_rms_m"] = PER_PULSE_ABS_CLIP_RMS_M
        h.attrs["robust_loss"] = ROBUST_LOSS
        h.attrs["robust_f_scale"] = ROBUST_F_SCALE
        h.attrs["max_clip_iterations"] = MAX_CLIP_ITERATIONS
        h.attrs["radius_start_grid_um"] = RADIUS_START_GRID_UM
        h.attrs["n_optimized_starts"] = N_OPTIMIZED_STARTS
        h.attrs["max_edge_reject_points"] = MAX_EDGE_REJECT_POINTS
        h.attrs["max_reject_run_points"] = MAX_REJECT_RUN_POINTS
        h.attrs["min_retained_fraction"] = MIN_RETAINED_FRACTION
        if MANUAL_REVIEW_MASK_SOURCE is not None:
            h.attrs["manual_review_h5"] = os.path.abspath(MANUAL_REVIEW_MASK_SOURCE)
        h.attrs["upsample_factor"] = base.UPSAMPLE_FACTOR
        h.attrs["sanya_first_sample_delay_us"] = sc.SANYA_FIRST_SAMPLE_DELAY_US
        h.attrs["sanya_range_correction_km"] = sc.SANYA_RANGE_CORRECTION_KM
        h.attrs["danzhou_first_sample_delay_us"] = sc.DANZHOU_FIRST_SAMPLE_DELAY_US
        h.attrs["wenchang_first_sample_delay_us"] = sc.WENCHANG_FIRST_SAMPLE_DELAY_US
        h.attrs["time_scale"] = "UTC"
        h.attrs["source_timezone_policy"] = (
            "Use event UTC metadata when present; otherwise subtract UTC+8 from raw MATLAB local timestamps."
        )
        h.attrs["fit_state_frame"] = "GCRS"
        h.attrs["path_prediction_frame"] = "ITRS station geometry after transforming GCRS state samples"
        h.attrs["sigma_floor_m"] = sigma_model["sigma_floor_m"]
        h.attrs["sigma_0_m"] = sigma_model["sigma_0_m"]
        h["event_id"] = np.asarray([o["event_id"] for o in ok], dtype=string_dtype)
        h["t0_ns"] = np.asarray([o["time_ns"][0] for o in ok], dtype=np.int64)
        h["r0_gcrs_m"] = np.asarray([o["params"][:3] for o in ok], dtype=np.float64)
        h["v0_gcrs_mps"] = np.asarray([o["params"][3:6] for o in ok], dtype=np.float64)
        h["speed_km_s"] = np.asarray([np.linalg.norm(o["params"][3:6]) / 1e3 for o in ok], dtype=np.float64)
        h["n_points"] = np.asarray([o["n_points"] for o in ok], dtype=np.int32)
        h["rms_total_path_residual_m"] = np.asarray([o["rms_total_path_residual_m"] for o in ok], dtype=np.float64)
        h["median_abs_total_path_residual_m"] = np.asarray([o["median_abs_total_path_residual_m"] for o in ok], dtype=np.float64)
        h["max_abs_total_path_residual_m"] = np.asarray([o["max_abs_total_path_residual_m"] for o in ok], dtype=np.float64)
        h["max_abs_link_mean_residual_m"] = np.asarray([o["max_abs_link_mean_residual_m"] for o in ok], dtype=np.float64)
        h["max_abs_link_median_residual_m"] = np.asarray([o["max_abs_link_median_residual_m"] for o in ok], dtype=np.float64)
        h["max_per_pulse_normalized_residual"] = np.asarray([o["max_per_pulse_normalized_residual"] for o in ok], dtype=np.float64)
        h["weighted_rms"] = np.asarray([o["weighted_rms"] for o in ok], dtype=np.float64)
        h["initial_n_points"] = np.asarray([o.get("initial_n_points", o["n_points"]) for o in ok], dtype=np.int32)
        h["n_clipped_points"] = np.asarray([o.get("n_clipped_points", 0) for o in ok], dtype=np.int32)
        h["clip_fraction"] = np.asarray([o.get("clip_fraction", 0.0) for o in ok], dtype=np.float64)
        h["clipping_disallowed"] = np.asarray([o.get("clipping_disallowed", False) for o in ok], dtype=bool)
        h["clipping_is_isolated"] = np.asarray([o.get("clipping_is_isolated", True) for o in ok], dtype=bool)
        h["multistart_score"] = np.asarray([o.get("multistart_score", np.nan) for o in ok], dtype=np.float64)
        h["multistart_n_guesses"] = np.asarray([o.get("multistart_n_guesses", 0) for o in ok], dtype=np.int32)
        h["multistart_n_screened"] = np.asarray([o.get("multistart_n_screened", 0) for o in ok], dtype=np.int32)
        h["multistart_n_optimized"] = np.asarray([o.get("multistart_n_optimized", 0) for o in ok], dtype=np.int32)
        h["multistart_n_failures"] = np.asarray([o.get("multistart_n_failures", 0) for o in ok], dtype=np.int32)
        h["manual_review_applied"] = np.asarray([o.get("manual_review_applied", False) for o in ok], dtype=bool)
        h["n_manual_reject_points"] = np.asarray([o.get("n_manual_reject_points", 0) for o in ok], dtype=np.int32)
        h["initial_radius_m"] = np.asarray([o["initial_radius_m"] for o in ok], dtype=np.float64)
        h["final_radius_m"] = np.asarray([o["final_radius_m"] for o in ok], dtype=np.float64)
        h["initial_mass_kg"] = np.asarray([o["initial_mass_kg"] for o in ok], dtype=np.float64)
        h["final_mass_kg"] = np.asarray([o["final_mass_kg"] for o in ok], dtype=np.float64)
        h["start_speed_km_s"] = np.asarray([o["speed_km_s"][0] for o in ok], dtype=np.float64)
        h["end_speed_km_s"] = np.asarray([o["speed_km_s"][-1] for o in ok], dtype=np.float64)
        h["start_alt_km"] = np.asarray([o["alt_km"][0] for o in ok], dtype=np.float64)
        h["end_alt_km"] = np.asarray([o["alt_km"][-1] for o in ok], dtype=np.float64)
        h["parameter_std"] = np.asarray([o["parameter_std"] for o in ok], dtype=np.float64)
        h["position_std_m"] = np.asarray([o["position_std_m"] for o in ok], dtype=np.float64)
        h["velocity_std_mps"] = np.asarray([o["velocity_std_mps"] for o in ok], dtype=np.float64)
        h["log10_radius_std"] = np.asarray([o["log10_radius_std"] for o in ok], dtype=np.float64)
        h["covariance_degrees_of_freedom"] = np.asarray([o["covariance_degrees_of_freedom"] for o in ok], dtype=np.int32)
        h["covariance_residual_variance"] = np.asarray([o["covariance_residual_variance"] for o in ok], dtype=np.float64)
        h["covariance_available"] = np.asarray([o["covariance_available"] for o in ok], dtype=bool)

        points = h.create_group("points")
        for o in ok:
            g = points.create_group(o["event_id"])
            for key in [
                "time_ns",
                "beijing_local_time_ns",
                "t_rel_s",
                "measured_total_paths_m",
                "predicted_total_paths_m",
                "residuals_m",
                "normalized_residuals",
                "x_gcrs_m",
                "v_gcrs_mps",
                "x_itrs_m",
                "v_itrs_mps",
                "lat_deg",
                "lon_deg",
                "alt_km",
                "speed_km_s",
                "itrs_speed_km_s",
                "radius_m",
                "mass_kg",
                "snr_db",
                "source_indices",
                "params",
                "parameter_std",
                "parameter_covariance",
                "position_std_m",
                "velocity_std_mps",
            ]:
                g[key] = o[key]
            g.attrs["log10_radius_std"] = o["log10_radius_std"]
            g.attrs["initial_radius_m"] = o["initial_radius_m"]
            g.attrs["final_radius_m"] = o["final_radius_m"]
            g.attrs["initial_mass_kg"] = o["initial_mass_kg"]
            g.attrs["final_mass_kg"] = o["final_mass_kg"]
            g.attrs["ceplecha_success"] = o["ceplecha_success"]
            g.attrs["ceplecha_message"] = o["ceplecha_message"]
            g.attrs["covariance_available"] = o["covariance_available"]
            g.attrs["covariance_degrees_of_freedom"] = o["covariance_degrees_of_freedom"]
            g.attrs["covariance_residual_variance"] = o["covariance_residual_variance"]
            g.attrs["max_abs_total_path_residual_m"] = o["max_abs_total_path_residual_m"]
            g.attrs["max_abs_link_mean_residual_m"] = o["max_abs_link_mean_residual_m"]
            g.attrs["max_abs_link_median_residual_m"] = o["max_abs_link_median_residual_m"]
            g.attrs["max_per_pulse_normalized_residual"] = o["max_per_pulse_normalized_residual"]
            g.attrs["initial_n_points"] = o.get("initial_n_points", o["n_points"])
            g.attrs["n_clipped_points"] = o.get("n_clipped_points", 0)
            g.attrs["clip_fraction"] = o.get("clip_fraction", 0.0)
            g.attrs["clipping_disallowed"] = o.get("clipping_disallowed", False)
            g.attrs["clipping_is_isolated"] = o.get("clipping_is_isolated", True)
            g.attrs["multistart_score"] = o.get("multistart_score", np.nan)
            g.attrs["multistart_n_guesses"] = o.get("multistart_n_guesses", 0)
            g.attrs["multistart_n_screened"] = o.get("multistart_n_screened", 0)
            g.attrs["multistart_n_optimized"] = o.get("multistart_n_optimized", 0)
            g.attrs["multistart_n_failures"] = o.get("multistart_n_failures", 0)
            g.attrs["manual_review_applied"] = o.get("manual_review_applied", False)
            g.attrs["n_manual_reject_points"] = o.get("n_manual_reject_points", 0)
            g.attrs["fit_epoch_time_ns"] = o["fit_epoch_time_ns"]
            g["all_time_ns"] = o["all_time_ns"]
            g["all_beijing_local_time_ns"] = o["all_beijing_local_time_ns"]
            g["all_measured_total_paths_m"] = o["all_measured_total_paths_m"]
            g["all_snr_db"] = o["all_snr_db"]
            g["all_source_indices"] = o["all_source_indices"]
            g["all_keep_rows"] = o["keep_rows"]
            g["manual_reject_mask"] = o.get("manual_reject_mask", np.zeros(len(o["all_time_ns"]), dtype=bool))
            g["all_t_rel_s"] = o["all_t_rel_s"]
            g["all_predicted_total_paths_m"] = o["all_predicted_total_paths_m"]
            g["all_residuals_m"] = o["all_residuals_m"]
            g["all_x_gcrs_m"] = o["all_x_gcrs_m"]
            g["all_v_gcrs_mps"] = o["all_v_gcrs_mps"]
            g["all_x_itrs_m"] = o["all_x_itrs_m"]
            g["all_v_itrs_mps"] = o["all_v_itrs_mps"]
            g["all_lat_deg"] = o["all_lat_deg"]
            g["all_lon_deg"] = o["all_lon_deg"]
            g["all_alt_km"] = o["all_alt_km"]
            g["all_speed_km_s"] = o["all_speed_km_s"]
            g["all_itrs_speed_km_s"] = o["all_itrs_speed_km_s"]
            g["all_radius_m"] = o["all_radius_m"]
            g["all_mass_kg"] = o["all_mass_kg"]
            if o["sigma_m"] is not None:
                g["sigma_m"] = o["sigma_m"]
                g["all_sigma_m"] = o["sigma_m"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-events", type=int, default=None, help="Process only the first N paired tri-static events.")
    parser.add_argument("--output-h5", default=OUTPUT_H5, help="Output HDF5 path.")
    parser.add_argument("--output-json", default=OUTPUT_JSON, help="Output JSON summary path.")
    parser.add_argument(
        "--sigma-floor-m",
        type=float,
        default=None,
        help="Use this empirical path sigma floor and skip the unweighted first pass.",
    )
    parser.add_argument(
        "--sigma-0-m",
        type=float,
        default=None,
        help="Use this empirical path sigma amplitude and skip the unweighted first pass.",
    )
    parser.add_argument(
        "--review-h5",
        default=None,
        help="Optional manual review HDF5 from review_tristatic_fits_gui.py; manual reject masks are applied before fitting.",
    )
    return parser.parse_args()


def main():
    global MANUAL_REVIEW_MASK_SOURCE
    args = parse_args()
    MANUAL_REVIEW_MASK_SOURCE = args.review_h5
    if (args.sigma_floor_m is None) != (args.sigma_0_m is None):
        raise SystemExit("--sigma-floor-m and --sigma-0-m must be supplied together")
    if is_root():
        log(f"loading triplets and reference fits; MPI ranks={base.SIZE}")
    ref_fit = base.load_reference_fits()
    if is_root() and args.review_h5:
        manual_masks = load_manual_review_masks(args.review_h5)
        log(f"loaded manual review masks for {len(manual_masks)} events from {args.review_h5}")
    else:
        manual_masks = None
    if base.COMM is not None:
        manual_masks = base.COMM.bcast(manual_masks, root=0)
    triplets = pair_tristatic_events(load_events(SAN_PATTERN), load_events(DAN_PATTERN), load_events(WEN_PATTERN))
    if args.max_events is not None:
        triplets = triplets[: args.max_events]
    if is_root():
        log(f"triplets={len(triplets)}")

    if args.sigma_floor_m is not None:
        sigma_model = {
            "sigma_floor_m": float(args.sigma_floor_m),
            "sigma_0_m": float(args.sigma_0_m),
            "optimizer_success": True,
            "n_samples": 0,
            "source": "command_line",
        }
        if is_root():
            log(
                "using command-line sigma_path(SNR) = sqrt("
                f"{sigma_model['sigma_floor_m']:.2f}^2 + "
                f"({sigma_model['sigma_0_m']:.2f}/10^(SNR_dB/20))^2) m"
            )
    else:
        if is_root():
            log("first pass unweighted robust Ceplecha fits")
        first_local = local_process(triplets, ref_fit, sigma_model=None, manual_masks=manual_masks)
        first_outputs = gather_outputs(first_local)

        if is_root():
            ok_first = [o for o in first_outputs if o["status"] == "ok"]
            residuals = np.concatenate([o["residuals_m"].ravel() for o in ok_first])
            snr = np.concatenate([o["snr_db"].ravel() for o in ok_first])
            sigma_model = base.fit_sigma_model(residuals, snr)
            log(
                "global sigma_path(SNR) = sqrt("
                f"{sigma_model['sigma_floor_m']:.2f}^2 + "
                f"({sigma_model['sigma_0_m']:.2f}/10^(SNR_dB/20))^2) m; "
                f"samples={sigma_model['n_samples']}"
            )
        else:
            sigma_model = None
        if base.COMM is not None:
            sigma_model = base.COMM.bcast(sigma_model, root=0)

    if is_root():
        log("second pass weighted robust Ceplecha fits with sigma clipping")
    second_local = local_process(triplets, ref_fit, sigma_model=sigma_model, manual_masks=manual_masks)
    second_outputs = gather_outputs(second_local)

    if is_root():
        ok = [o for o in second_outputs if o["status"] == "ok"]
        status_counts = {}
        for o in second_outputs:
            status_counts[o["status"]] = status_counts.get(o["status"], 0) + 1
        write_results(args.output_h5, second_outputs, sigma_model)
        summary = {
            "script": os.path.basename(__file__),
            "script_version": SCRIPT_VERSION,
            "trajectory_model_package": "meteor_trajectory_models",
            "n_triplets": len(triplets),
            "status_counts": status_counts,
            "sigma_model": sigma_model,
            "n_ok": len(ok),
            "meteoroid_density_kg_m3": METEOROID_DENSITY_KG_M3,
            "ablation_sigma_kg_j": ABLATION_SIGMA_KG_J,
            "per_pulse_norm_clip_rms": PER_PULSE_NORM_CLIP_RMS,
            "per_pulse_abs_clip_rms_m": PER_PULSE_ABS_CLIP_RMS_M,
            "robust_loss": ROBUST_LOSS,
            "robust_f_scale": ROBUST_F_SCALE,
            "max_clip_iterations": MAX_CLIP_ITERATIONS,
            "radius_start_grid_um": RADIUS_START_GRID_UM.tolist(),
            "n_optimized_starts": N_OPTIMIZED_STARTS,
            "max_edge_reject_points": MAX_EDGE_REJECT_POINTS,
            "max_reject_run_points": MAX_REJECT_RUN_POINTS,
            "min_retained_fraction": MIN_RETAINED_FRACTION,
            "rms_total_path_residual_m_median": float(np.nanmedian([o["rms_total_path_residual_m"] for o in ok])) if ok else np.nan,
            "rms_total_path_residual_m_range": [
                float(np.nanmin([o["rms_total_path_residual_m"] for o in ok])) if ok else np.nan,
                float(np.nanmax([o["rms_total_path_residual_m"] for o in ok])) if ok else np.nan,
            ],
            "initial_radius_m_median": float(np.nanmedian([o["initial_radius_m"] for o in ok])) if ok else np.nan,
            "final_radius_m_median": float(np.nanmedian([o["final_radius_m"] for o in ok])) if ok else np.nan,
            "initial_mass_kg_median": float(np.nanmedian([o["initial_mass_kg"] for o in ok])) if ok else np.nan,
            "final_mass_kg_median": float(np.nanmedian([o["final_mass_kg"] for o in ok])) if ok else np.nan,
            "position_std_m_median_xyz": np.nanmedian(np.asarray([o["position_std_m"] for o in ok], dtype=np.float64), axis=0).tolist()
            if ok
            else [np.nan, np.nan, np.nan],
            "velocity_std_mps_median_xyz": np.nanmedian(np.asarray([o["velocity_std_mps"] for o in ok], dtype=np.float64), axis=0).tolist()
            if ok
            else [np.nan, np.nan, np.nan],
            "log10_radius_std_median": float(np.nanmedian([o["log10_radius_std"] for o in ok])) if ok else np.nan,
            "covariance_available_count": int(np.count_nonzero([o["covariance_available"] for o in ok])) if ok else 0,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log(f"status_counts={status_counts}")
        log(f"median weighted Ceplecha RMS={summary['rms_total_path_residual_m_median']:.2f} m")
        log(f"median initial radius={summary['initial_radius_m_median'] * 1e6:.2f} um")
        log(f"median final radius={summary['final_radius_m_median'] * 1e6:.2f} um")
        log(f"median position std xyz={summary['position_std_m_median_xyz']} m")
        log(f"median velocity std xyz={summary['velocity_std_mps_median_xyz']} m/s")
        log(f"wrote {args.output_h5}")
        log(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
