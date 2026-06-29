"""Bayesian model selection for reviewed Sanya tri-static trajectories.

The comparison uses the current manual review masks and compares a
constant-velocity six-parameter trajectory with a seven-parameter
shrinking-radius ablation trajectory.  The reported evidence statistic is the
Bayesian information criterion (BIC), which is the usual large-sample Laplace
approximation to the marginal likelihood for Gaussian residuals.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os

import astropy.units as u
import h5py
import jcoord
import numpy as np
import scipy.optimize as so
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time

import fit_all_ballistic_snr_weighted as base_fit
import fit_all_ceplecha_snr_weighted as ceplecha_fit
import fit_gcrs_trajectories_lfm_ambiguity as gfit


DEFAULT_INPUT_H5 = "results/all_tristatic_ceplecha_snr_weighted_v20260616d.h5"
DEFAULT_REVIEW_H5 = "results/tristatic_fit_review.h5"
DEFAULT_OUTPUT_CSV = "results/tristatic_bayesian_model_selection.csv"
DEFAULT_OUTPUT_JSON = "results/tristatic_bayesian_model_selection_summary.json"
DEFAULT_CACHE_H5 = "results/tristatic_bayesian_model_selection_cache.h5"
MIN_CONSTANT_VELOCITY_POINTS = 3
K_CONSTANT_VELOCITY = 6
K_SHRINKING_RADIUS = 7


def decode_strings(values):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def ecef_to_gcrs(points_ecef_m, times_ns):
    obstime = Time(np.asarray(times_ns, dtype=np.float64) / 1e9, format="unix", scale="utc")
    itrs = ITRS(
        CartesianRepresentation(
            points_ecef_m[:, 0] * u.m,
            points_ecef_m[:, 1] * u.m,
            points_ecef_m[:, 2] * u.m,
        ),
        obstime=obstime,
    )
    return itrs.transform_to(GCRS(obstime=obstime)).cartesian.xyz.to_value(u.m).T


def read_review_masks(path):
    masks = {}
    qualities = {}
    with h5py.File(path, "r") as h:
        for event_id, group in h["reviews"].items():
            masks[str(event_id)] = np.asarray(group["manual_reject_mask"][:], dtype=bool)
            qualities[str(event_id)] = int(group.attrs.get("quality", 0))
    return masks, qualities


def fit_constant_velocity(measured_total_paths_m, times_ns, sigma_m, keep_rows, initial_points_itrs_m):
    measured = np.asarray(measured_total_paths_m, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.int64)
    sigma = np.asarray(sigma_m, dtype=np.float64)
    keep = np.asarray(keep_rows, dtype=bool)
    epoch_time_ns = int(times[0])
    x_gcrs0 = ecef_to_gcrs(initial_points_itrs_m, times)
    t_rel_all_s = (times.astype(np.float64) - float(epoch_time_ns)) / 1e9
    p0 = np.concatenate([x_gcrs0[0], np.polyfit(t_rel_all_s, x_gcrs0, 1)[0]])
    times_fit = times[keep]
    measured_fit = measured[keep]
    sigma_fit = sigma[keep]

    def predict(params, query_times_ns):
        t_rel_s = (np.asarray(query_times_ns, dtype=np.float64) - float(epoch_time_ns)) / 1e9
        x_gcrs = params[:3][None, :] + t_rel_s[:, None] * params[3:6][None, :]
        v_gcrs = np.repeat(params[3:6][None, :], len(t_rel_s), axis=0)
        x_itrs, v_itrs = base_fit.gcrs_state_samples_to_itrs(x_gcrs, v_gcrs, query_times_ns)
        total_paths_m, path_rates_mps = gfit.link_total_paths_and_rates_m(
            x_itrs,
            v_itrs,
            gfit.LINK_TX_POSITIONS_M,
            gfit.LINK_RX_POSITIONS_M,
        )
        return total_paths_m + gfit.lfm_total_path_bias_m(path_rates_mps), x_gcrs, v_gcrs

    def residual(params):
        predicted, *_rest = predict(params, times_fit)
        return ((predicted - measured_fit) / sigma_fit).ravel()

    result = so.least_squares(
        residual,
        p0,
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4]),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=160,
    )
    r = residual(result.x)
    _pred, _x_gcrs, v_gcrs = predict(result.x, times)
    return {
        "chi2": float(np.sum(r**2.0)),
        "success": bool(result.success),
        "speed0_km_s": float(np.linalg.norm(v_gcrs[0]) / 1e3),
    }


def fit_shrinking_radius(measured_total_paths_m, times_ns, sigma_m, keep_rows, initial_points_itrs_m, p0):
    measured = np.asarray(measured_total_paths_m, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.int64)
    sigma = np.asarray(sigma_m, dtype=np.float64)
    keep = np.asarray(keep_rows, dtype=bool)
    rho_of_alt_m, _meta = base_fit.density_interpolator(times, np.asarray(initial_points_itrs_m, dtype=np.float64))
    if p0 is None or len(p0) != 7 or not np.all(np.isfinite(p0)):
        p0 = ceplecha_fit.initial_ceplecha_guess(initial_points_itrs_m, times)
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
        p0,
        bounds=(
            np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(ceplecha_fit.MIN_RADIUS_M)]),
            np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(ceplecha_fit.MAX_RADIUS_M)]),
        ),
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0]),
        loss=ceplecha_fit.ROBUST_LOSS,
        f_scale=ceplecha_fit.ROBUST_F_SCALE,
        max_nfev=160,
    )
    r = residual(result.x)
    radius_m = 10.0 ** float(result.x[6])
    mass_kg = (4.0 / 3.0) * np.pi * ceplecha_fit.METEOROID_DENSITY_KG_M3 * radius_m**3
    return {
        "chi2": float(np.sum(r**2.0)),
        "success": bool(result.success),
        "initial_radius_m": float(radius_m),
        "initial_mass_kg": float(mass_kg),
    }


def bic(chi2, n_observations, n_parameters):
    return float(chi2 + n_parameters * np.log(n_observations))


def compute_event(event_id, group, mask):
    if "all_time_ns" not in group:
        raise RuntimeError("Expected all_* event group")
    time_ns = np.asarray(group["all_time_ns"][:], dtype=np.int64)
    measured = np.asarray(group["all_measured_total_paths_m"][:], dtype=np.float64)
    sigma = np.asarray(group["all_sigma_m"][:], dtype=np.float64)
    initial_points = np.asarray(group["all_x_itrs_m"][:], dtype=np.float64)
    p0 = np.asarray(group["params"][:], dtype=np.float64) if "params" in group else None
    keep = ~np.asarray(mask, dtype=bool)
    n_points = int(np.count_nonzero(keep))
    if n_points < MIN_CONSTANT_VELOCITY_POINTS:
        raise RuntimeError("Too few retained points")
    n_obs = int(n_points * measured.shape[1])
    cv = fit_constant_velocity(measured, time_ns, sigma, keep, initial_points)
    sr = fit_shrinking_radius(measured, time_ns, sigma, keep, initial_points, p0)
    bic_cv = bic(cv["chi2"], n_obs, K_CONSTANT_VELOCITY)
    bic_sr = bic(sr["chi2"], n_obs, K_SHRINKING_RADIUS)
    delta_bic_cv_minus_sr = bic_cv - bic_sr
    return {
        "event_id": event_id,
        "n_points": n_points,
        "n_observations": n_obs,
        "chi2_constant_velocity": cv["chi2"],
        "chi2_shrinking_radius": sr["chi2"],
        "bic_constant_velocity": bic_cv,
        "bic_shrinking_radius": bic_sr,
        "delta_bic_cv_minus_shrinking": float(delta_bic_cv_minus_sr),
        "preferred_model": "shrinking_radius" if delta_bic_cv_minus_sr > 0.0 else "constant_velocity",
        "strong_preferred_model": "shrinking_radius" if delta_bic_cv_minus_sr >= 6.0 else "constant_velocity",
        "initial_radius_um": sr["initial_radius_m"] * 1e6,
        "initial_mass_kg": sr["initial_mass_kg"],
        "constant_velocity_speed0_km_s": cv["speed0_km_s"],
        "status": "ok",
    }


def compute_event_from_paths(input_h5, review_h5, event_id):
    masks, qualities = read_review_masks(review_h5)
    with h5py.File(input_h5, "r") as h:
        try:
            row = compute_event(event_id, h["points"][event_id], masks[event_id])
        except Exception as exc:
            return {"event_id": event_id, "status": f"failed: {exc}", "review_quality": int(qualities.get(event_id, 0))}
    row["review_quality"] = int(qualities.get(event_id, 0))
    return row


def cache_load(path, event_id):
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as h:
        if event_id not in h:
            return None
        group = h[event_id]
        return {key: group.attrs[key] for key in group.attrs}


def cache_save(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "a") as h:
        event_id = str(row["event_id"])
        if event_id in h:
            del h[event_id]
        group = h.create_group(event_id)
        for key, value in row.items():
            if key == "event_id":
                continue
            group.attrs[key] = value


def write_outputs(rows, output_csv, output_json):
    ok = [r for r in rows if r.get("status") == "ok"]
    with open(output_csv, "w", newline="") as f:
        fieldnames = [
            "event_id",
            "status",
            "n_points",
            "n_observations",
            "chi2_constant_velocity",
            "chi2_shrinking_radius",
            "bic_constant_velocity",
            "bic_shrinking_radius",
            "delta_bic_cv_minus_shrinking",
            "preferred_model",
            "strong_preferred_model",
            "initial_radius_um",
            "initial_mass_kg",
            "constant_velocity_speed0_km_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    delta = np.asarray([float(r["delta_bic_cv_minus_shrinking"]) for r in ok], dtype=np.float64)
    summary = {
        "n_events": len(rows),
        "n_ok": len(ok),
        "n_prefer_shrinking_radius_delta_bic_gt_0": int(np.count_nonzero(delta > 0.0)),
        "n_strong_shrinking_radius_delta_bic_ge_6": int(np.count_nonzero(delta >= 6.0)),
        "n_prefer_constant_velocity_delta_bic_le_0": int(np.count_nonzero(delta <= 0.0)),
        "delta_bic_cv_minus_shrinking_median": float(np.nanmedian(delta)) if len(delta) else np.nan,
        "delta_bic_cv_minus_shrinking_p25": float(np.nanpercentile(delta, 25)) if len(delta) else np.nan,
        "delta_bic_cv_minus_shrinking_p75": float(np.nanpercentile(delta, 75)) if len(delta) else np.nan,
        "definition": "positive delta_bic_cv_minus_shrinking means shrinking-radius model has smaller BIC",
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", default=DEFAULT_INPUT_H5)
    parser.add_argument("--review-h5", default=DEFAULT_REVIEW_H5)
    parser.add_argument("--cache-h5", default=DEFAULT_CACHE_H5)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    masks, qualities = read_review_masks(args.review_h5)
    rows = []
    with h5py.File(args.input_h5, "r") as h:
        event_ids = [event_id for event_id in decode_strings(h["event_id"][:]) if event_id in masks]
        if args.limit is not None:
            event_ids = event_ids[: args.limit]

    cached_rows = {}
    pending_event_ids = []
    for event_id in event_ids:
        cached = cache_load(args.cache_h5, event_id)
        if cached is None:
            pending_event_ids.append(event_id)
        else:
            cached_rows[event_id] = {"event_id": event_id, **cached}

    computed_rows = {}
    if args.workers <= 1:
        for idx, event_id in enumerate(pending_event_ids, start=1):
            with h5py.File(args.input_h5, "r") as h:
                try:
                    row = compute_event(event_id, h["points"][event_id], masks[event_id])
                except Exception as exc:
                    row = {"event_id": event_id, "status": f"failed: {exc}"}
            row["review_quality"] = int(qualities.get(event_id, 0))
            computed_rows[event_id] = row
            rows.append(row)
            if row.get("status") == "ok":
                cache_save(args.cache_h5, row)
            if idx % 10 == 0:
                print(f"processed {idx}/{len(pending_event_ids)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(compute_event_from_paths, args.input_h5, args.review_h5, event_id): event_id
                for event_id in pending_event_ids
            }
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                event_id = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"event_id": event_id, "status": f"failed: {exc}", "review_quality": int(qualities.get(event_id, 0))}
                computed_rows[event_id] = row
                if row.get("status") == "ok":
                    cache_save(args.cache_h5, row)
                if idx % 10 == 0 or idx == len(pending_event_ids):
                    print(f"processed {idx}/{len(pending_event_ids)}", flush=True)

    rows = [cached_rows.get(event_id) or computed_rows[event_id] for event_id in event_ids]

    summary = write_outputs(rows, args.output_csv, args.output_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
