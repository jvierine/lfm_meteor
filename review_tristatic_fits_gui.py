"""Interactive review GUI for tri-static meteor trajectory fits.

The GUI reads a fitted tri-static HDF5 file, displays one event at a time,
lets the reviewer mark the fit quality, and allows pulse-level outlier masks
to be edited by clicking plotted measurements.  Refit requests run in a
background queue so browsing and point selection stay responsive.
"""

import argparse
import concurrent.futures
import os
from datetime import datetime, timezone

import astropy.units as u
import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from matplotlib.widgets import Button, RadioButtons

import fit_all_ceplecha_snr_weighted as ceplecha_fit
import fit_all_ballistic_snr_weighted as base_fit
import fit_gcrs_trajectories_lfm_ambiguity as gfit
import plot_article_event_fit as event_plot


DEFAULT_INPUT_H5 = "results/all_tristatic_ceplecha_snr_weighted_v20260616d.h5"
DEFAULT_REVIEW_H5 = "results/tristatic_fit_review.h5"
QUALITY_UNKNOWN = 0
QUALITY_GOOD = 1
QUALITY_BAD = -1
QUALITY_LABELS = {
    QUALITY_UNKNOWN: "unknown",
    QUALITY_GOOD: "good",
    QUALITY_BAD: "bad",
}
MODEL_LINEAR = "constant velocity"
MODEL_CEPLECHA = "shrinking radius"
MODEL_LABELS = (MODEL_CEPLECHA, MODEL_LINEAR)
SITE_LABELS = ("Sanya", "Danzhou", "Wenchang")
MIN_CONSTANT_VELOCITY_POINTS = 3
MAX_DEFAULT_RADIUS_FRACTIONAL_UNCERTAINTY = 0.50
RADIUS_LOWER_BOUND_DELTA_CHI2 = 3.841458820694124
RADIUS_LOWER_BOUND_GRID_UM = np.logspace(-1, 4, 18)
GUI_AUTO_OUTLIER_NORM_THRESHOLD = 4.0
GUI_AUTO_OUTLIER_ABS_THRESHOLD_M = 75.0
GUI_AUTO_OUTLIER_MAX_FRACTION = 0.35


def reserve_review_keybindings():
    """Prevent Matplotlib defaults from eating review shortcuts."""

    reserved = {"b", "r", "s", "v"}
    for key in list(plt.rcParams):
        if key.startswith("keymap."):
            plt.rcParams[key] = [value for value in plt.rcParams[key] if value not in reserved]


def decode_strings(values):
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def unix_ns_to_utc_label(time_ns):
    return np.datetime_as_string(np.datetime64(int(time_ns), "ns"), unit="ms").replace("T", " ") + " UTC"


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


def horizontal_offsets_km(points_ecef_m):
    beam_center_ecef_m = np.asarray(
        jcoord.geodetic2ecef(
            event_plot.COMMON_VOLUME_LAT_DEG,
            event_plot.COMMON_VOLUME_LON_DEG,
            event_plot.COMMON_VOLUME_ALT_KM * 1e3,
        ),
        dtype=np.float64,
    )
    return event_plot.horizontal_offsets_km(
        points_ecef_m,
        beam_center_ecef_m,
        event_plot.COMMON_VOLUME_LAT_DEG,
        event_plot.COMMON_VOLUME_LON_DEG,
    )


def norm_uncertainty(vector, covariance):
    vector = np.asarray(vector, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm) or covariance.shape != (len(vector), len(vector)):
        return np.nan
    unit = vector / norm
    variance = float(unit @ covariance @ unit)
    return np.sqrt(max(variance, 0.0))


def radius_mass_uncertainty(radius_m, mass_kg, log10_radius_std):
    if not np.isfinite(radius_m) or radius_m <= 0.0 or not np.isfinite(log10_radius_std):
        return np.nan, np.nan
    sigma_radius_m = np.log(10.0) * float(radius_m) * float(log10_radius_std)
    sigma_mass_kg = np.nan
    if np.isfinite(mass_kg):
        sigma_mass_kg = 3.0 * abs(float(mass_kg)) * sigma_radius_m / float(radius_m)
    return sigma_radius_m, sigma_mass_kg


def size_estimate_is_unconstrained(radius_fractional_uncertainty, mass_fractional_uncertainty):
    return (
        np.isfinite(radius_fractional_uncertainty)
        and radius_fractional_uncertainty > MAX_DEFAULT_RADIUS_FRACTIONAL_UNCERTAINTY
    ) or (
        np.isfinite(mass_fractional_uncertainty)
        and mass_fractional_uncertainty > MAX_DEFAULT_RADIUS_FRACTIONAL_UNCERTAINTY
    )


def fixed_initial_radius_fit_chi2(measured_total_paths_m, times_ns, sigma_m, keep_rows, rho_of_alt_m, p0_6, radius_m):
    measured = np.asarray(measured_total_paths_m, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.int64)
    sigma = np.asarray(sigma_m, dtype=np.float64)
    keep = np.asarray(keep_rows, dtype=bool)
    epoch_time_ns = int(times[0])
    times_fit = times[keep]
    measured_fit = measured[keep]
    sigma_fit = sigma[keep]
    t_rel_s = (times_fit.astype(np.float64) - float(epoch_time_ns)) / 1e9
    log10_radius = np.log10(float(radius_m))

    def residual(p6):
        params = np.concatenate([p6, [log10_radius]])
        pred, *_rest = ceplecha_fit.predict_paths(params, t_rel_s, times_fit, rho_of_alt_m)
        return ((pred - measured_fit) / sigma_fit).ravel()

    result = so.least_squares(
        residual,
        np.asarray(p0_6, dtype=np.float64),
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4]),
        loss="linear",
        max_nfev=90,
    )
    r = residual(result.x)
    return float(np.sum(r**2.0)), result.x


def constant_velocity_size_lower_bound(measured_total_paths_m, times_ns, sigma_m, keep_rows, initial_points_itrs_m, linear_params_6, chi2_constant):
    rho_of_alt_m, _meta = base_fit.density_interpolator(times_ns, np.asarray(initial_points_itrs_m, dtype=np.float64))
    good_radius = None
    bad_radius = None
    p0_6 = np.asarray(linear_params_6, dtype=np.float64)
    last_good_p0 = p0_6.copy()
    grid_radius_m = RADIUS_LOWER_BOUND_GRID_UM * 1e-6
    for radius_m in grid_radius_m:
        try:
            chi2, fitted_p0 = fixed_initial_radius_fit_chi2(
                measured_total_paths_m,
                times_ns,
                sigma_m,
                keep_rows,
                rho_of_alt_m,
                last_good_p0,
                radius_m,
            )
        except Exception:
            bad_radius = radius_m
            continue
        rejected = (chi2 - chi2_constant) > RADIUS_LOWER_BOUND_DELTA_CHI2
        if rejected:
            bad_radius = radius_m
        else:
            good_radius = radius_m
            last_good_p0 = fitted_p0
            break

    if good_radius is None:
        return {
            "radius_lower_bound_m": np.nan,
            "mass_lower_bound_kg": np.nan,
            "lower_bound_status": f">{grid_radius_m[-1] * 1e6:.2g} um",
            "delta_chi2_threshold": RADIUS_LOWER_BOUND_DELTA_CHI2,
        }
    if bad_radius is None:
        radius_bound_m = grid_radius_m[0]
        status = f"<={radius_bound_m * 1e6:.2g} um"
    else:
        lo = float(bad_radius)
        hi = float(good_radius)
        best_p0 = last_good_p0
        for _ in range(7):
            mid = np.sqrt(lo * hi)
            try:
                chi2, fitted_p0 = fixed_initial_radius_fit_chi2(
                    measured_total_paths_m,
                    times_ns,
                    sigma_m,
                    keep_rows,
                    rho_of_alt_m,
                    best_p0,
                    mid,
                )
            except Exception:
                lo = mid
                continue
            if (chi2 - chi2_constant) > RADIUS_LOWER_BOUND_DELTA_CHI2:
                lo = mid
            else:
                hi = mid
                best_p0 = fitted_p0
        radius_bound_m = hi
        status = "95% lower bound"
    mass_bound_kg = (4.0 / 3.0) * np.pi * ceplecha_fit.METEOROID_DENSITY_KG_M3 * radius_bound_m**3
    return {
        "radius_lower_bound_m": float(radius_bound_m),
        "mass_lower_bound_kg": float(mass_bound_kg),
        "lower_bound_status": status,
        "delta_chi2_threshold": RADIUS_LOWER_BOUND_DELTA_CHI2,
    }


def fit_linear_paths(
    measured_total_paths_m,
    times_ns,
    sigma_m,
    keep_rows,
    p0,
    initial_points_itrs_m=None,
    estimate_lower_bound=True,
):
    measured = np.asarray(measured_total_paths_m, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.int64)
    sigma = np.asarray(sigma_m, dtype=np.float64)
    keep = np.asarray(keep_rows, dtype=bool)
    epoch_time_ns = int(times[0])
    times_fit = times[keep]
    measured_fit = measured[keep]
    sigma_fit = sigma[keep]
    t_rel_fit_s = (times_fit.astype(np.float64) - float(epoch_time_ns)) / 1e9

    def predict(params, query_times_ns):
        query_times_ns = np.asarray(query_times_ns, dtype=np.int64)
        t_rel_s = (query_times_ns.astype(np.float64) - float(epoch_time_ns)) / 1e9
        x_gcrs = params[:3][None, :] + t_rel_s[:, None] * params[3:6][None, :]
        v_gcrs = np.repeat(params[3:6][None, :], len(t_rel_s), axis=0)
        x_itrs, v_itrs = base_fit.gcrs_state_samples_to_itrs(x_gcrs, v_gcrs, query_times_ns)
        total_paths_m, path_rates_mps = gfit.link_total_paths_and_rates_m(
            x_itrs,
            v_itrs,
            gfit.LINK_TX_POSITIONS_M,
            gfit.LINK_RX_POSITIONS_M,
        )
        predicted = total_paths_m + gfit.lfm_total_path_bias_m(path_rates_mps)
        return predicted, x_gcrs, v_gcrs, x_itrs, v_itrs

    def residual(params):
        predicted, *_rest = predict(params, times_fit)
        return ((predicted - measured_fit) / sigma_fit).ravel()

    result = so.least_squares(
        residual,
        np.asarray(p0, dtype=np.float64),
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4]),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=220,
    )
    predicted_fit, *_fit_state = predict(result.x, times_fit)
    residuals_fit = predicted_fit - measured_fit
    normalized_fit = residuals_fit / sigma_fit
    all_predicted, x_gcrs, v_gcrs, x_itrs, v_itrs = predict(result.x, times)
    all_residuals = all_predicted - measured
    all_normalized = all_residuals / sigma
    covariance = base_fit.linearized_covariance_summary(result, len(residual(result.x)))
    chi2_constant = float(np.sum(normalized_fit**2.0))
    lower_bound = {
        "radius_lower_bound_m": np.nan,
        "mass_lower_bound_kg": np.nan,
        "lower_bound_status": "not estimated",
        "delta_chi2_threshold": RADIUS_LOWER_BOUND_DELTA_CHI2,
    }
    if estimate_lower_bound and initial_points_itrs_m is not None and np.count_nonzero(keep) >= MIN_CONSTANT_VELOCITY_POINTS:
        try:
            lower_bound = constant_velocity_size_lower_bound(
                measured,
                times,
                sigma,
                keep,
                initial_points_itrs_m,
                result.x,
                chi2_constant,
            )
        except Exception as exc:
            lower_bound["lower_bound_status"] = f"failed: {exc}"
    return {
        "model": MODEL_LINEAR,
        "params": result.x,
        "parameter_covariance": covariance["parameter_covariance"],
        "parameter_std": covariance["parameter_std"],
        "position_std_m": covariance["position_std_m"],
        "velocity_std_mps": covariance["velocity_std_mps"],
        "covariance_available": covariance["covariance_available"],
        "covariance_degrees_of_freedom": covariance["degrees_of_freedom"],
        "covariance_residual_variance": covariance["residual_variance"],
        "fit_epoch_time_ns": epoch_time_ns,
        "keep_rows": keep,
        "time_ns": times_fit,
        "t_rel_s": (times_fit.astype(np.float64) - float(epoch_time_ns)) / 1e9,
        "all_time_ns": times,
        "all_t_rel_s": (times.astype(np.float64) - float(epoch_time_ns)) / 1e9,
        "predicted_total_paths_m": predicted_fit,
        "residuals_m": residuals_fit,
        "normalized_residuals": normalized_fit,
        "all_predicted_total_paths_m": all_predicted,
        "all_residuals_m": all_residuals,
        "all_normalized_residuals": all_normalized,
        "x_gcrs_m": x_gcrs[keep],
        "v_gcrs_mps": v_gcrs[keep],
        "x_itrs_m": x_itrs[keep],
        "v_itrs_mps": v_itrs[keep],
        "all_x_gcrs_m": x_gcrs,
        "all_v_gcrs_mps": v_gcrs,
        "all_x_itrs_m": x_itrs,
        "all_v_itrs_mps": v_itrs,
        "radius_m": np.full(np.count_nonzero(keep), np.nan),
        "mass_kg": np.full(np.count_nonzero(keep), np.nan),
        "all_radius_m": np.full(len(times), np.nan),
        "all_mass_kg": np.full(len(times), np.nan),
        "optimizer_success": bool(result.success),
        "optimizer_cost": float(result.cost),
        "rms_total_path_residual_m": float(np.sqrt(np.nanmean(residuals_fit**2.0))),
        "weighted_rms": float(np.sqrt(np.nanmean(normalized_fit**2.0))),
        "chi2": chi2_constant,
        **lower_bound,
    }


def fit_ceplecha_paths(measured_total_paths_m, times_ns, sigma_m, keep_rows, initial_points_itrs_m, p0_fallback=None):
    measured = np.asarray(measured_total_paths_m, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.int64)
    sigma = np.asarray(sigma_m, dtype=np.float64)
    keep = np.asarray(keep_rows, dtype=bool)
    rho_of_alt_m, _meta = base_fit.density_interpolator(times, np.asarray(initial_points_itrs_m, dtype=np.float64))
    guesses = ceplecha_fit.unique_initial_guesses(initial_points_itrs_m, times, reference_fit=None)
    if p0_fallback is not None:
        p0 = np.asarray(p0_fallback, dtype=np.float64)
        if len(p0) == 7 and np.all(np.isfinite(p0)):
            guesses.insert(0, p0)
    if not guesses:
        guesses = [ceplecha_fit.initial_ceplecha_guess(initial_points_itrs_m, times)]
    fit = ceplecha_fit.fit_ceplecha_multistart(
        measured,
        times,
        rho_of_alt_m,
        guesses,
        sigma_m=sigma,
        keep_rows=keep,
        epoch_time_ns=int(times[0]),
    )
    all_t_rel_s = (times.astype(np.float64) - float(fit["fit_epoch_time_ns"])) / 1e9
    (
        all_predicted,
        all_x_gcrs,
        all_v_gcrs,
        all_x_itrs,
        all_v_itrs,
        all_radius_m,
        all_mass_kg,
        _success,
        _message,
    ) = ceplecha_fit.predict_paths(fit["params"], all_t_rel_s, times, rho_of_alt_m)
    fit["model"] = MODEL_CEPLECHA
    fit["all_time_ns"] = times
    fit["all_t_rel_s"] = all_t_rel_s
    fit["all_predicted_total_paths_m"] = all_predicted
    fit["all_residuals_m"] = all_predicted - measured
    fit["all_normalized_residuals"] = fit["all_residuals_m"] / sigma
    fit["all_x_gcrs_m"] = all_x_gcrs
    fit["all_v_gcrs_mps"] = all_v_gcrs
    fit["all_x_itrs_m"] = all_x_itrs
    fit["all_v_itrs_mps"] = all_v_itrs
    fit["all_radius_m"] = all_radius_m
    fit["all_mass_kg"] = all_mass_kg
    return fit


def compute_refit_job(payload, model, manual_reject_mask):
    keep = ~np.asarray(manual_reject_mask, dtype=bool)
    retained_count = int(np.count_nonzero(keep))
    requested_model = model
    effective_model = requested_model
    min_points = base_fit.MIN_POINTS
    if requested_model == MODEL_LINEAR:
        min_points = MIN_CONSTANT_VELOCITY_POINTS
    elif retained_count < base_fit.MIN_POINTS:
        effective_model = MODEL_LINEAR
        min_points = MIN_CONSTANT_VELOCITY_POINTS
    if np.count_nonzero(keep) < min_points:
        raise RuntimeError(f"Need at least {min_points} retained pulses for {effective_model}")

    if effective_model == MODEL_LINEAR:
        fit = fit_linear_paths(
            payload["measured_total_paths_m"],
            payload["time_ns"],
            payload["sigma_m"],
            keep,
            payload["p0_linear"],
            initial_points_itrs_m=payload["x_itrs_initial_m"],
        )
    else:
        fit = fit_ceplecha_paths(
            payload["measured_total_paths_m"],
            payload["time_ns"],
            payload["sigma_m"],
            keep,
            payload["x_itrs_initial_m"],
            p0_fallback=payload["p0_ceplecha"],
        )
    fit["requested_model"] = requested_model
    fit["auto_model_fallback"] = bool(requested_model != effective_model)
    return fit


class ReviewStore:
    def __init__(self, path, source_h5, event_ids):
        self.path = path
        self.source_h5 = source_h5
        self.event_ids = np.asarray(event_ids, dtype=object)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(self.path, "a") as h:
            h.attrs["source_fit_h5"] = os.path.abspath(source_h5)
            h.attrs["format"] = "sanya tristatic manual fit review"
            h.attrs["quality_encoding"] = "-1 bad, 0 unknown, 1 good"
            h.require_group("reviews")
            if "event_id" not in h:
                h["event_id"] = np.asarray(event_ids, dtype=h5py.string_dtype(encoding="utf-8"))

    def load(self, event_id, n_points):
        with h5py.File(self.path, "a") as h:
            reviews = h.require_group("reviews")
            if event_id not in reviews:
                return {
                    "quality": QUALITY_UNKNOWN,
                    "model": MODEL_CEPLECHA,
                    "has_saved_review": False,
                    "manual_reject_mask": np.zeros(n_points, dtype=bool),
                }
            g = reviews[event_id]
            mask = np.asarray(g["manual_reject_mask"][:], dtype=bool) if "manual_reject_mask" in g else np.zeros(n_points, dtype=bool)
            if len(mask) != n_points:
                resized = np.zeros(n_points, dtype=bool)
                resized[: min(len(mask), n_points)] = mask[: min(len(mask), n_points)]
                mask = resized
            model = g.attrs.get("model", MODEL_CEPLECHA)
            if isinstance(model, bytes):
                model = model.decode("utf-8")
            return {
                "quality": int(g.attrs.get("quality", QUALITY_UNKNOWN)),
                "model": str(model) if str(model) in MODEL_LABELS else MODEL_CEPLECHA,
                "has_saved_review": True,
                "manual_reject_mask": mask,
            }

    def save(self, event_id, quality, model, manual_reject_mask, fit):
        with h5py.File(self.path, "a") as h:
            reviews = h.require_group("reviews")
            if event_id in reviews:
                del reviews[event_id]
            g = reviews.create_group(event_id)
            g.attrs["quality"] = int(quality)
            g.attrs["quality_label"] = QUALITY_LABELS.get(int(quality), "unknown")
            g.attrs["model"] = str(model)
            g.attrs["fit_model"] = str(fit.get("model", model))
            g.attrs["auto_model_fallback"] = bool(fit.get("auto_model_fallback", False))
            g.attrs["updated_utc"] = datetime.now(timezone.utc).isoformat()
            g.attrs["rms_total_path_residual_m"] = float(fit.get("rms_total_path_residual_m", np.nan))
            g.attrs["weighted_rms"] = float(fit.get("weighted_rms", np.nan))
            g.attrs["n_manual_reject"] = int(np.count_nonzero(manual_reject_mask))
            g.attrs["radius_lower_bound_m"] = float(fit.get("radius_lower_bound_m", np.nan))
            g.attrs["mass_lower_bound_kg"] = float(fit.get("mass_lower_bound_kg", np.nan))
            g.attrs["lower_bound_status"] = str(fit.get("lower_bound_status", ""))
            g.attrs["delta_chi2_threshold"] = float(fit.get("delta_chi2_threshold", np.nan))
            g["manual_reject_mask"] = np.asarray(manual_reject_mask, dtype=bool)
            if "params" in fit:
                g["params"] = np.asarray(fit["params"], dtype=np.float64)
            if "parameter_std" in fit:
                g["parameter_std"] = np.asarray(fit["parameter_std"], dtype=np.float64)
            if "parameter_covariance" in fit:
                g["parameter_covariance"] = np.asarray(fit["parameter_covariance"], dtype=np.float64)
            if "all_residuals_m" in fit:
                g["all_residuals_m"] = np.asarray(fit["all_residuals_m"], dtype=np.float64)
            if "all_predicted_total_paths_m" in fit:
                g["all_predicted_total_paths_m"] = np.asarray(fit["all_predicted_total_paths_m"], dtype=np.float64)
            for key in [
                "all_time_ns",
                "all_t_rel_s",
                "all_x_gcrs_m",
                "all_v_gcrs_mps",
                "all_x_itrs_m",
                "all_v_itrs_mps",
                "all_radius_m",
                "all_mass_kg",
                "velocity_std_mps",
            ]:
                if key in fit:
                    dtype = np.int64 if key == "all_time_ns" else np.float64
                    g[key] = np.asarray(fit[key], dtype=dtype)
            if "log10_radius_std" in fit:
                g.attrs["log10_radius_std"] = float(fit.get("log10_radius_std", np.nan))


class TristaticFitReviewer:
    def __init__(self, input_h5, review_h5, start_event_id=None):
        self.input_h5 = input_h5
        self.h = h5py.File(input_h5, "r")
        self.event_ids = decode_strings(self.h["event_id"][:])
        self.store = ReviewStore(review_h5, input_h5, self.event_ids)
        self.index = 0
        if start_event_id:
            matches = np.flatnonzero(self.event_ids == start_event_id)
            if len(matches) == 0:
                raise ValueError(f"Event {start_event_id} not found")
            self.index = int(matches[0])
        self.quality = QUALITY_UNKNOWN
        self.model = MODEL_CEPLECHA
        self.manual_reject_mask = None
        self.mask_changed_since_refit = False
        self.fit = None
        self.event_data = None
        self.pick_artists = {}
        self.colorbar = None
        self.refit_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.refit_jobs = {}
        self.refit_sequence = 0
        self.latest_refit_seq_by_event = {}
        self.refit_message = ""

        self.fig, self.axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
        plt.subplots_adjust(left=0.07, right=0.83, bottom=0.12, top=0.90, hspace=0.34, wspace=0.32)
        self.ax_buttons = {
            "prev": self.fig.add_axes([0.08, 0.025, 0.07, 0.045]),
            "next": self.fig.add_axes([0.16, 0.025, 0.07, 0.045]),
            "good": self.fig.add_axes([0.27, 0.025, 0.07, 0.045]),
            "bad": self.fig.add_axes([0.35, 0.025, 0.07, 0.045]),
            "unknown": self.fig.add_axes([0.43, 0.025, 0.09, 0.045]),
            "save": self.fig.add_axes([0.55, 0.025, 0.07, 0.045]),
            "reanalyze": self.fig.add_axes([0.63, 0.025, 0.10, 0.045]),
            "auto": self.fig.add_axes([0.74, 0.025, 0.08, 0.045]),
        }
        self.buttons = {
            "prev": Button(self.ax_buttons["prev"], "Prev"),
            "next": Button(self.ax_buttons["next"], "Save+Next"),
            "good": Button(self.ax_buttons["good"], "Good"),
            "bad": Button(self.ax_buttons["bad"], "Bad"),
            "unknown": Button(self.ax_buttons["unknown"], "Unknown"),
            "save": Button(self.ax_buttons["save"], "Save"),
            "reanalyze": Button(self.ax_buttons["reanalyze"], "Reanalyze"),
            "auto": Button(self.ax_buttons["auto"], "AutoOut"),
        }
        radio_ax = self.fig.add_axes([0.86, 0.68, 0.12, 0.17])
        self.radio = RadioButtons(radio_ax, MODEL_LABELS, active=0)
        self.status_ax = self.fig.add_axes([0.84, 0.05, 0.15, 0.59])
        self.status_ax.axis("off")
        self.status_text = self.status_ax.text(
            0.0,
            1.0,
            "",
            ha="left",
            va="top",
            fontsize=8.0,
            family="monospace",
            transform=self.status_ax.transAxes,
            clip_on=False,
        )

        self.buttons["prev"].on_clicked(lambda _event: self.goto(self.index - 1))
        self.buttons["next"].on_clicked(lambda _event: self.goto(self.index + 1))
        self.buttons["good"].on_clicked(lambda _event: self.set_quality(QUALITY_GOOD))
        self.buttons["bad"].on_clicked(lambda _event: self.set_quality(QUALITY_BAD))
        self.buttons["unknown"].on_clicked(lambda _event: self.set_quality(QUALITY_UNKNOWN))
        self.buttons["save"].on_clicked(lambda _event: self.save_current())
        self.buttons["reanalyze"].on_clicked(lambda _event: self.schedule_background_refit())
        self.buttons["auto"].on_clicked(lambda _event: self.auto_mark_outliers())
        self.radio.on_clicked(self.set_model)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.refit_timer = self.fig.canvas.new_timer(interval=250)
        self.refit_timer.add_callback(self.poll_refit_jobs)
        self.refit_timer.start()
        self.load_event(self.index)

    def read_event(self, idx):
        event_id = self.event_ids[idx]
        g = self.h["points"][event_id]
        if "all_time_ns" in g:
            time_ns = np.asarray(g["all_time_ns"][:], dtype=np.int64)
            measured = np.asarray(g["all_measured_total_paths_m"][:], dtype=np.float64)
            sigma = np.asarray(g["all_sigma_m"][:], dtype=np.float64) if "all_sigma_m" in g else np.ones_like(measured)
            snr_db = np.asarray(g["all_snr_db"][:], dtype=np.float64)
            x_itrs = np.asarray(g["all_x_itrs_m"][:], dtype=np.float64)
            v_itrs = np.asarray(g["all_v_itrs_mps"][:], dtype=np.float64)
            p0_ceplecha = np.asarray(g["params"][:], dtype=np.float64) if "params" in g else None
        else:
            time_ns = np.asarray(g["time_ns"][:], dtype=np.int64)
            measured = np.asarray(g["measured_total_paths_m"][:], dtype=np.float64)
            sigma = np.asarray(g["sigma_m"][:], dtype=np.float64) if "sigma_m" in g else np.ones_like(measured)
            snr_db = np.asarray(g["snr_db"][:], dtype=np.float64)
            x_itrs = np.asarray(g["x_itrs_m"][:], dtype=np.float64)
            v_itrs = np.asarray(g["v_itrs_mps"][:], dtype=np.float64)
            p0_ceplecha = np.asarray(g["params"][:], dtype=np.float64) if "params" in g else None
        input_radius_fractional_uncertainty = np.nan
        input_mass_fractional_uncertainty = np.nan
        if "initial_radius_m" in g.attrs and "log10_radius_std" in g.attrs:
            input_radius_fractional_uncertainty = np.log(10.0) * float(g.attrs["log10_radius_std"])
            input_mass_fractional_uncertainty = 3.0 * input_radius_fractional_uncertainty
        elif "initial_radius_m" in self.h and "log10_radius_std" in self.h:
            input_radius_fractional_uncertainty = np.log(10.0) * float(self.h["log10_radius_std"][idx])
            input_mass_fractional_uncertainty = 3.0 * input_radius_fractional_uncertainty
        if snr_db.ndim == 2:
            sanya_snr_db = snr_db[:, 0]
        else:
            sanya_snr_db = snr_db
        x_gcrs = ecef_to_gcrs(x_itrs, time_ns)
        t_rel_s = (time_ns.astype(np.float64) - float(time_ns[0])) / 1e9
        p0_linear = np.concatenate([x_gcrs[0], np.polyfit(t_rel_s, x_gcrs, 1)[0]])
        return {
            "event_id": event_id,
            "group": g,
            "time_ns": time_ns,
            "measured_total_paths_m": measured,
            "sigma_m": sigma,
            "sanya_snr_db": sanya_snr_db,
            "x_itrs_initial_m": x_itrs,
            "v_itrs_initial_mps": v_itrs,
            "p0_ceplecha": p0_ceplecha,
            "p0_linear": p0_linear,
            "input_radius_fractional_uncertainty": input_radius_fractional_uncertainty,
            "input_mass_fractional_uncertainty": input_mass_fractional_uncertainty,
        }

    def load_event(self, idx):
        self.index = int(np.clip(idx, 0, len(self.event_ids) - 1))
        self.event_data = self.read_event(self.index)
        review = self.store.load(self.event_data["event_id"], len(self.event_data["time_ns"]))
        self.quality = review["quality"]
        self.model = review["model"]
        if self.model == MODEL_CEPLECHA and size_estimate_is_unconstrained(
            self.event_data["input_radius_fractional_uncertainty"],
            self.event_data["input_mass_fractional_uncertainty"],
        ):
            self.model = MODEL_LINEAR
        self.manual_reject_mask = review["manual_reject_mask"]
        self.radio.set_active(MODEL_LABELS.index(self.model))
        self.mask_changed_since_refit = False
        self.fit = self.load_reviewed_fit(self.event_data["event_id"]) or self.load_stored_fit()
        self.redraw()

    def load_reviewed_fit(self, event_id):
        required = [
            "params",
            "all_time_ns",
            "all_t_rel_s",
            "all_predicted_total_paths_m",
            "all_residuals_m",
            "all_x_gcrs_m",
            "all_v_gcrs_mps",
            "all_x_itrs_m",
            "all_v_itrs_mps",
            "all_radius_m",
            "all_mass_kg",
        ]
        with h5py.File(self.store.path, "r") as h:
            if "reviews" not in h or event_id not in h["reviews"]:
                return None
            g = h["reviews"][event_id]
            if any(key not in g for key in required):
                return None
            fit = {
                "model": str(g.attrs.get("fit_model", g.attrs.get("model", self.model))),
                "requested_model": str(g.attrs.get("model", self.model)),
                "auto_model_fallback": bool(g.attrs.get("auto_model_fallback", False)),
                "params": np.asarray(g["params"][:], dtype=np.float64),
                "parameter_std": np.asarray(g["parameter_std"][:], dtype=np.float64)
                if "parameter_std" in g
                else np.full(7, np.nan),
                "parameter_covariance": np.asarray(g["parameter_covariance"][:], dtype=np.float64)
                if "parameter_covariance" in g
                else np.full((0, 0), np.nan),
                "velocity_std_mps": np.asarray(g["velocity_std_mps"][:], dtype=np.float64)
                if "velocity_std_mps" in g
                else np.full(3, np.nan),
                "all_time_ns": np.asarray(g["all_time_ns"][:], dtype=np.int64),
                "all_t_rel_s": np.asarray(g["all_t_rel_s"][:], dtype=np.float64),
                "all_predicted_total_paths_m": np.asarray(g["all_predicted_total_paths_m"][:], dtype=np.float64),
                "all_residuals_m": np.asarray(g["all_residuals_m"][:], dtype=np.float64),
                "all_x_gcrs_m": np.asarray(g["all_x_gcrs_m"][:], dtype=np.float64),
                "all_v_gcrs_mps": np.asarray(g["all_v_gcrs_mps"][:], dtype=np.float64),
                "all_x_itrs_m": np.asarray(g["all_x_itrs_m"][:], dtype=np.float64),
                "all_v_itrs_mps": np.asarray(g["all_v_itrs_mps"][:], dtype=np.float64),
                "all_radius_m": np.asarray(g["all_radius_m"][:], dtype=np.float64),
                "all_mass_kg": np.asarray(g["all_mass_kg"][:], dtype=np.float64),
                "rms_total_path_residual_m": float(g.attrs.get("rms_total_path_residual_m", np.nan)),
                "weighted_rms": float(g.attrs.get("weighted_rms", np.nan)),
                "radius_lower_bound_m": float(g.attrs.get("radius_lower_bound_m", np.nan)),
                "mass_lower_bound_kg": float(g.attrs.get("mass_lower_bound_kg", np.nan)),
                "lower_bound_status": str(g.attrs.get("lower_bound_status", "")),
                "delta_chi2_threshold": float(g.attrs.get("delta_chi2_threshold", np.nan)),
                "log10_radius_std": float(g.attrs.get("log10_radius_std", np.nan)),
            }
        if not np.isfinite(fit["rms_total_path_residual_m"]):
            fit["rms_total_path_residual_m"] = float(np.sqrt(np.nanmean(fit["all_residuals_m"] ** 2.0)))
        return fit

    def load_stored_fit(self):
        g = self.event_data["group"]
        fit = {
            "model": MODEL_CEPLECHA,
            "requested_model": self.model,
            "auto_model_fallback": False,
            "params": np.asarray(g["params"][:], dtype=np.float64),
            "parameter_std": np.asarray(g["parameter_std"][:], dtype=np.float64)
            if "parameter_std" in g
            else np.full(7, np.nan),
            "parameter_covariance": np.asarray(g["parameter_covariance"][:], dtype=np.float64)
            if "parameter_covariance" in g
            else np.full((0, 0), np.nan),
            "velocity_std_mps": np.asarray(g["velocity_std_mps"][:], dtype=np.float64)
            if "velocity_std_mps" in g
            else np.full(3, np.nan),
            "all_time_ns": np.asarray(g["all_time_ns"][:], dtype=np.int64),
            "all_t_rel_s": np.asarray(g["all_t_rel_s"][:], dtype=np.float64),
            "all_predicted_total_paths_m": np.asarray(g["all_predicted_total_paths_m"][:], dtype=np.float64),
            "all_residuals_m": np.asarray(g["all_residuals_m"][:], dtype=np.float64),
            "all_x_gcrs_m": np.asarray(g["all_x_gcrs_m"][:], dtype=np.float64),
            "all_v_gcrs_mps": np.asarray(g["all_v_gcrs_mps"][:], dtype=np.float64),
            "all_x_itrs_m": np.asarray(g["all_x_itrs_m"][:], dtype=np.float64),
            "all_v_itrs_mps": np.asarray(g["all_v_itrs_mps"][:], dtype=np.float64),
            "all_radius_m": np.asarray(g["all_radius_m"][:], dtype=np.float64),
            "all_mass_kg": np.asarray(g["all_mass_kg"][:], dtype=np.float64),
            "rms_total_path_residual_m": float(g.attrs.get("rms_total_path_residual_m", np.nan)),
            "weighted_rms": float(g.attrs.get("weighted_rms", np.nan)),
            "log10_radius_std": float(g.attrs.get("log10_radius_std", np.nan)),
        }
        if not np.isfinite(fit["rms_total_path_residual_m"]):
            fit["rms_total_path_residual_m"] = float(np.sqrt(np.nanmean(fit["all_residuals_m"] ** 2.0)))
        if not np.isfinite(fit["weighted_rms"]) and "all_sigma_m" in g:
            sigma = np.asarray(g["all_sigma_m"][:], dtype=np.float64)
            fit["weighted_rms"] = float(np.sqrt(np.nanmean((fit["all_residuals_m"] / sigma) ** 2.0)))
        return fit

    def current_refit_queue_count(self):
        return sum(1 for job in self.refit_jobs.values() if not job["future"].done())

    def event_payload_for_refit(self):
        keys = (
            "event_id",
            "time_ns",
            "measured_total_paths_m",
            "sigma_m",
            "x_itrs_initial_m",
            "p0_ceplecha",
            "p0_linear",
        )
        payload = {}
        for key in keys:
            value = self.event_data[key]
            payload[key] = np.asarray(value).copy() if isinstance(value, np.ndarray) else value
        return payload

    def schedule_background_refit(self, redraw=True, save_on_complete=True):
        if self.event_data is None:
            return
        event_id = self.event_data["event_id"]
        for seq, job in list(self.refit_jobs.items()):
            if job["event_id"] == event_id and not job["future"].running() and not job["future"].done():
                job["future"].cancel()
                del self.refit_jobs[seq]
        self.refit_sequence += 1
        seq = self.refit_sequence
        payload = self.event_payload_for_refit()
        mask = np.asarray(self.manual_reject_mask, dtype=bool).copy()
        future = self.refit_executor.submit(compute_refit_job, payload, self.model, mask)
        self.refit_jobs[seq] = {
            "future": future,
            "event_id": event_id,
            "model": self.model,
            "quality": self.quality,
            "mask": mask,
            "save_on_complete": bool(save_on_complete),
        }
        self.latest_refit_seq_by_event[event_id] = seq
        self.mask_changed_since_refit = True
        self.refit_message = "refit queued"
        if redraw:
            self.redraw()

    def poll_refit_jobs(self):
        changed = False
        for seq, job in list(self.refit_jobs.items()):
            future = job["future"]
            if not future.done():
                continue
            del self.refit_jobs[seq]
            if future.cancelled():
                changed = True
                continue
            event_id = job["event_id"]
            is_latest = self.latest_refit_seq_by_event.get(event_id) == seq
            try:
                fit = future.result()
            except Exception as exc:
                if self.event_data is not None and self.event_data["event_id"] == event_id and is_latest:
                    self.refit_message = f"refit failed: {exc}"
                    changed = True
                continue
            if job.get("save_on_complete", False):
                self.store.save(event_id, job["quality"], job["model"], job["mask"], fit)
            if self.event_data is not None and self.event_data["event_id"] == event_id and is_latest:
                self.fit = fit
                self.mask_changed_since_refit = False
                self.refit_message = "refit ready"
                changed = True
        if changed:
            self.redraw()
        return True

    def current_keep_rows(self, min_points):
        keep = ~np.asarray(self.manual_reject_mask, dtype=bool)
        if np.count_nonzero(keep) < min_points:
            return None
        return keep

    def refit(self):
        retained_count = int(np.count_nonzero(~self.manual_reject_mask))
        requested_model = self.model
        effective_model = requested_model
        min_points = base_fit.MIN_POINTS
        if requested_model == MODEL_LINEAR:
            min_points = MIN_CONSTANT_VELOCITY_POINTS
        elif retained_count < base_fit.MIN_POINTS:
            effective_model = MODEL_LINEAR
            min_points = MIN_CONSTANT_VELOCITY_POINTS

        keep = self.current_keep_rows(min_points)
        if keep is None:
            raise RuntimeError(f"Need at least {min_points} retained pulses for {effective_model}")
        d = self.event_data
        if effective_model == MODEL_LINEAR:
            fit = fit_linear_paths(
                d["measured_total_paths_m"],
                d["time_ns"],
                d["sigma_m"],
                keep,
                d["p0_linear"],
                initial_points_itrs_m=d["x_itrs_initial_m"],
            )
            fit["requested_model"] = requested_model
            fit["auto_model_fallback"] = bool(requested_model != effective_model)
            return fit
        fit = fit_ceplecha_paths(
            d["measured_total_paths_m"],
            d["time_ns"],
            d["sigma_m"],
            keep,
            d["x_itrs_initial_m"],
            p0_fallback=d["p0_ceplecha"],
        )
        fit["requested_model"] = requested_model
        fit["auto_model_fallback"] = False
        return fit

    def refit_and_redraw(self, save=True):
        try:
            self.fit = self.refit()
            self.mask_changed_since_refit = False
            self.status_text.set_text("")
            if save:
                self.save_current(draw=False)
        except Exception as exc:
            self.fit = None
            self.status_text.set_text(f"Refit failed:\n{exc}")
        self.redraw()

    def save_current(self, draw=True):
        if self.fit is not None:
            self.store.save(
                self.event_data["event_id"],
                self.quality,
                self.model,
                self.manual_reject_mask,
                self.fit,
            )
            self.status_text.set_text(f"Saved\n{os.path.basename(self.store.path)}")
        if draw:
            self.fig.canvas.draw_idle()

    def set_quality(self, quality):
        self.quality = int(quality)
        self.save_current(draw=False)
        self.redraw()

    def set_model(self, label):
        if label == self.model:
            return
        self.model = str(label)
        self.schedule_background_refit()

    def goto(self, idx):
        self.save_current(draw=False)
        self.load_event(idx)

    def on_key(self, event):
        if event.key in ("right", "n"):
            self.goto(self.index + 1)
        elif event.key in ("left", "p"):
            self.goto(self.index - 1)
        elif event.key == "a":
            self.auto_mark_outliers()
        elif event.key == "g":
            self.set_quality(QUALITY_GOOD)
        elif event.key == "x":
            self.set_quality(QUALITY_BAD)
        elif event.key == "u":
            self.set_quality(QUALITY_UNKNOWN)
        elif event.key == "s":
            self.save_current()
        elif event.key == "r":
            self.schedule_background_refit()
        elif event.key == "v":
            self.set_model(MODEL_LINEAR)
        elif event.key == "b":
            self.set_model(MODEL_CEPLECHA)

    def on_pick(self, event):
        if event.artist not in self.pick_artists or len(event.ind) == 0:
            return
        source_indices = self.pick_artists[event.artist]
        pulse_index = int(source_indices[int(event.ind[0])])
        self.manual_reject_mask[pulse_index] = ~self.manual_reject_mask[pulse_index]
        self.mask_changed_since_refit = True
        self.refit_message = "refit queued"
        self.redraw()
        self.schedule_background_refit(redraw=False)

    def add_pickable_scatter(self, ax, x, y, mask, **kwargs):
        if not np.any(mask):
            return None
        artist = ax.scatter(
            np.asarray(x)[mask],
            np.asarray(y)[mask],
            picker=6,
            **kwargs,
        )
        self.pick_artists[artist] = np.flatnonzero(mask)
        return artist

    def plot_pickable_scatter(self, ax, x, y, keep, snr_db):
        rejected = ~keep
        if np.any(rejected):
            self.add_pickable_scatter(
                ax,
                x,
                y,
                rejected,
                c="0.68",
                s=34,
                edgecolors="none",
                label="manual reject",
                zorder=3,
            )
        if np.any(keep):
            return self.add_pickable_scatter(
                ax,
                x,
                y,
                keep,
                c=np.asarray(snr_db)[keep],
                cmap="viridis",
                s=34,
                edgecolors="none",
                label="retained",
                zorder=4,
            )
        return None

    def residual_display_components(self):
        d = self.event_data
        fit = self.fit
        measured_itrs = event_plot.lfm_corrected_point_solutions(
            d["measured_total_paths_m"],
            fit["all_x_itrs_m"],
            fit["all_v_itrs_mps"],
        )
        measured_gcrs = ecef_to_gcrs(measured_itrs, d["time_ns"])
        measured_alt_km = np.asarray([jcoord.ecef2geodetic(*p)[2] / 1e3 for p in measured_itrs], dtype=np.float64)
        along_axis, cross_axis = event_plot.event_axes(fit["all_x_gcrs_m"], fit["all_v_gcrs_mps"])
        origin = fit["all_x_gcrs_m"][0]
        fit_along_m = ((fit["all_x_gcrs_m"] - origin) @ along_axis)
        fit_cross_m = ((fit["all_x_gcrs_m"] - origin) @ cross_axis)
        meas_along_m = ((measured_gcrs - origin) @ along_axis)
        meas_cross_m = ((measured_gcrs - origin) @ cross_axis)
        east_km, north_km = horizontal_offsets_km(measured_itrs)
        fit_east_km, fit_north_km = horizontal_offsets_km(fit["all_x_itrs_m"])
        fit_alt_km = np.asarray([jcoord.ecef2geodetic(*p)[2] / 1e3 for p in fit["all_x_itrs_m"]], dtype=np.float64)
        return {
            "measured_itrs": measured_itrs,
            "measured_alt_km": measured_alt_km,
            "along_resid_m": meas_along_m - fit_along_m,
            "cross_resid_m": meas_cross_m - fit_cross_m,
            "east_km": east_km,
            "north_km": north_km,
            "fit_east_km": fit_east_km,
            "fit_north_km": fit_north_km,
            "fit_alt_km": fit_alt_km,
        }

    def detect_obvious_outliers(self):
        if self.fit is None or self.event_data is None:
            return np.zeros(0, dtype=bool)
        d = self.event_data
        keep = ~np.asarray(self.manual_reject_mask, dtype=bool)
        min_points = MIN_CONSTANT_VELOCITY_POINTS if self.model == MODEL_LINEAR else base_fit.MIN_POINTS
        if np.count_nonzero(keep) <= min_points:
            return np.zeros(len(keep), dtype=bool)
        components = self.residual_display_components()
        sigma_m = np.nanmedian(np.asarray(d["sigma_m"], dtype=np.float64), axis=1)
        finite_sigma = sigma_m[np.isfinite(sigma_m) & (sigma_m > 0.0)]
        fallback_sigma = float(np.nanmedian(finite_sigma)) if len(finite_sigma) else 1.0
        sigma_m = np.where(np.isfinite(sigma_m) & (sigma_m > 0.0), sigma_m, fallback_sigma)
        along = np.asarray(components["along_resid_m"], dtype=np.float64)
        cross = np.asarray(components["cross_resid_m"], dtype=np.float64)
        residual_mag_m = np.sqrt(along**2.0 + cross**2.0)
        normalized_mag = residual_mag_m / sigma_m
        finite_keep = keep & np.isfinite(residual_mag_m) & np.isfinite(normalized_mag)
        if np.count_nonzero(finite_keep) <= min_points:
            return np.zeros(len(keep), dtype=bool)
        kept_mag = residual_mag_m[finite_keep]
        median_mag = float(np.nanmedian(kept_mag))
        mad_mag = float(np.nanmedian(np.abs(kept_mag - median_mag)))
        robust_scale_m = max(1.4826 * mad_mag, fallback_sigma, 1.0)
        obvious = finite_keep & (
            (normalized_mag >= GUI_AUTO_OUTLIER_NORM_THRESHOLD)
            | (residual_mag_m >= GUI_AUTO_OUTLIER_ABS_THRESHOLD_M)
            | (residual_mag_m >= median_mag + 6.0 * robust_scale_m)
        )
        max_new = max(0, int(np.floor(GUI_AUTO_OUTLIER_MAX_FRACTION * len(keep))))
        max_new = min(max_new, int(np.count_nonzero(keep) - min_points))
        candidate_indices = np.flatnonzero(obvious)
        if len(candidate_indices) > max_new:
            score = np.maximum(
                normalized_mag[candidate_indices] / GUI_AUTO_OUTLIER_NORM_THRESHOLD,
                residual_mag_m[candidate_indices] / GUI_AUTO_OUTLIER_ABS_THRESHOLD_M,
            )
            candidate_indices = candidate_indices[np.argsort(score)[::-1][:max_new]]
        new_reject = np.zeros(len(keep), dtype=bool)
        new_reject[candidate_indices] = True
        return new_reject

    def auto_mark_outliers(self):
        new_reject = self.detect_obvious_outliers()
        n_new = int(np.count_nonzero(new_reject))
        if n_new == 0:
            self.refit_message = "auto: none"
            self.redraw()
            return
        self.manual_reject_mask = np.asarray(self.manual_reject_mask, dtype=bool) | new_reject
        self.mask_changed_since_refit = True
        self.refit_message = f"auto rejected {n_new}"
        self.redraw()
        self.schedule_background_refit(redraw=False)

    def redraw(self):
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        for ax in self.axes.ravel():
            ax.clear()
        self.pick_artists = {}
        d = self.event_data
        keep = ~self.manual_reject_mask
        title = (
            f"{unix_ns_to_utc_label(d['time_ns'][0])}  "
            f"({self.index + 1}/{len(self.event_ids)})  {d['event_id']}"
        )
        self.fig.suptitle(title)
        if self.fit is None:
            self.axes[0, 0].text(0.5, 0.5, "Fit failed", ha="center", va="center", transform=self.axes[0, 0].transAxes)
            self.fig.canvas.draw_idle()
            return

        fit = self.fit
        components = self.residual_display_components()
        measured_alt_km = components["measured_alt_km"]
        along_resid_m = components["along_resid_m"]
        cross_resid_m = components["cross_resid_m"]
        fit_alt_km = components["fit_alt_km"]
        east_km = components["east_km"]
        north_km = components["north_km"]
        fit_east_km = components["fit_east_km"]
        fit_north_km = components["fit_north_km"]
        t_rel_s = fit["all_t_rel_s"]

        ax = self.axes[0, 0]
        sc = self.plot_pickable_scatter(ax, east_km, north_km, keep, d["sanya_snr_db"])
        ax.plot(fit_east_km, fit_north_km, color="#1b7837", lw=1.8)
        ax.set_aspect("auto")
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        ax.grid(True, color="0.88")
        if sc is not None:
            self.colorbar = self.fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            self.colorbar.set_label("Sanya SNR (dB)")
        ax.legend(loc="best", frameon=False)

        ax = self.axes[0, 1]
        ax.axhline(0.0, color="#1b7837", lw=1.5)
        ax.errorbar(t_rel_s[~keep], cross_resid_m[~keep], yerr=np.nanmedian(d["sigma_m"], axis=1)[~keep], fmt="o", color="0.68", ecolor="0.76", ms=4, lw=0.8, label="manual reject")
        ax.errorbar(t_rel_s[keep], cross_resid_m[keep], yerr=np.nanmedian(d["sigma_m"], axis=1)[keep], fmt="o", color="#2166ac", ecolor="0.45", ms=4, lw=0.8, label="retained")
        self.add_pickable_scatter(ax, t_rel_s, cross_resid_m, ~keep, c="0.68", s=46, alpha=0.01, edgecolors="none", zorder=5)
        self.add_pickable_scatter(ax, t_rel_s, cross_resid_m, keep, c="#2166ac", s=46, alpha=0.01, edgecolors="none", zorder=5)
        ax.set_xlabel("Time since first pulse (s)")
        ax.set_ylabel("Cross-track residual (m)")
        ax.grid(True, color="0.88")
        ax.legend(loc="best", frameon=False)

        ax = self.axes[1, 0]
        ax.plot(t_rel_s, fit_alt_km, color="#1b7837", lw=1.8, label="fit")
        ax.scatter(t_rel_s[~keep], measured_alt_km[~keep], color="0.68", s=24, label="manual reject")
        ax.scatter(t_rel_s[keep], measured_alt_km[keep], color="0.15", s=24, label="retained")
        ax.set_xlabel("Time since first pulse (s)")
        ax.set_ylabel("Altitude (km)")
        ax.grid(True, color="0.88")
        ax.legend(loc="best", frameon=False)

        ax = self.axes[1, 1]
        ax.axhline(0.0, color="#1b7837", lw=1.5)
        ax.errorbar(t_rel_s[~keep], along_resid_m[~keep], yerr=np.nanmedian(d["sigma_m"], axis=1)[~keep], fmt="o", color="0.68", ecolor="0.76", ms=4, lw=0.8, label="manual reject")
        ax.errorbar(t_rel_s[keep], along_resid_m[keep], yerr=np.nanmedian(d["sigma_m"], axis=1)[keep], fmt="o", color="#2166ac", ecolor="0.45", ms=4, lw=0.8, label="retained")
        ax.set_xlabel("Time since first pulse (s)")
        ax.set_ylabel("Along-track residual (m)")
        ax.grid(True, color="0.88")
        ax.legend(loc="best", frameon=False)

        speed_km_s = np.linalg.norm(fit["all_v_gcrs_mps"], axis=1) / 1e3
        velocity_cov = np.asarray(fit.get("parameter_covariance", np.full((0, 0), np.nan)), dtype=np.float64)
        if velocity_cov.shape[0] >= 6:
            sigma_v0_km_s = norm_uncertainty(fit["params"][3:6], velocity_cov[3:6, 3:6]) / 1e3
        else:
            velocity_std = np.asarray(fit.get("velocity_std_mps", np.full(3, np.nan)), dtype=np.float64)
            sigma_v0_km_s = norm_uncertainty(fit["all_v_gcrs_mps"][0], np.diag(velocity_std**2.0)) / 1e3
        position_cov = velocity_cov[:3, :3] if velocity_cov.shape[0] >= 3 else np.full((0, 0), np.nan)
        sigma_r0_m = norm_uncertainty(fit["params"][:3], position_cov)
        initial_mass = float(fit["all_mass_kg"][0]) if np.any(np.isfinite(fit["all_mass_kg"])) else np.nan
        initial_radius_um = float(fit["all_radius_m"][0] * 1e6) if np.any(np.isfinite(fit["all_radius_m"])) else np.nan
        sigma_radius_m, sigma_mass_kg = radius_mass_uncertainty(
            float(fit["all_radius_m"][0]) if np.any(np.isfinite(fit["all_radius_m"])) else np.nan,
            initial_mass,
            float(fit.get("log10_radius_std", np.nan)),
        )
        sigma_radius_um = sigma_radius_m * 1e6 if np.isfinite(sigma_radius_m) else np.nan
        params_text = [
            f"quality: {QUALITY_LABELS[self.quality]}",
            f"model: {self.model}",
            f"kept: {np.count_nonzero(keep)}/{len(keep)}",
            f"refit q: {self.current_refit_queue_count()}",
        ]
        if self.refit_message:
            params_text.append(self.refit_message)
        if self.model == MODEL_CEPLECHA:
            params_text += [
                f"radius: {initial_radius_um:.3g} ± {sigma_radius_um:.2g} um",
                f"mass: {initial_mass:.3g} ± {sigma_mass_kg:.2g} kg",
            ]
        else:
            radius_lower_m = float(fit.get("radius_lower_bound_m", np.nan))
            mass_lower_kg = float(fit.get("mass_lower_bound_kg", np.nan))
            delta_chi2 = float(fit.get("delta_chi2_threshold", np.nan))
            lower_status = str(fit.get("lower_bound_status", ""))
            if np.isfinite(radius_lower_m) and np.isfinite(mass_lower_kg):
                params_text += [
                    f"r0 > {radius_lower_m * 1e6:.3g} um",
                    f"m0 > {mass_lower_kg:.3g} kg",
                    f"95% 1-sided",
                    f"dchi2={delta_chi2:.2f}",
                    f"{lower_status}",
                ]
            else:
                params_text += [
                    "r0 lower n/a",
                    lower_status,
                ]
        params_text += [""]
        if self.mask_changed_since_refit:
            params_text.append("refit pending")
        params_text += [
            f"RMS {fit['rms_total_path_residual_m']:.1f} m",
            f"WRMS {fit['weighted_rms']:.2f}",
            f"v0 {speed_km_s[0]:.2f}±{sigma_v0_km_s:.2f}",
            f"vend {speed_km_s[-1]:.2f} km/s",
            f"h0 {fit_alt_km[0]:.2f} km",
            f"hend {fit_alt_km[-1]:.2f} km",
            f"|r0| unc {sigma_r0_m:.1f} m",
            "",
            "click: reject",
            "keys n/p g/x/u",
            "a/r/s v/b",
        ]
        self.status_text.set_text("\n".join(params_text))
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()
        self.refit_executor.shutdown(wait=False, cancel_futures=True)
        self.h.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", default=DEFAULT_INPUT_H5)
    parser.add_argument("--review-h5", default=DEFAULT_REVIEW_H5)
    parser.add_argument("--event-id", default=None)
    return parser.parse_args()


def main():
    reserve_review_keybindings()
    args = parse_args()
    gui = TristaticFitReviewer(args.input_h5, args.review_h5, start_event_id=args.event_id)
    gui.show()


if __name__ == "__main__":
    main()
