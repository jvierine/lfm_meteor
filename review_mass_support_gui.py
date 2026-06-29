"""GUI for reviewing whether tri-static events support mass estimation.

This reviewer compares a constant-velocity trajectory against the
shrinking-radius Ceplecha trajectory for each tri-static event.  It reports

    Delta BIC = BIC_constant_velocity - BIC_shrinking_radius

so positive values indicate that the additional shrinking-radius parameter is
supported by the retained measurements.
"""

import argparse
import os

import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

import plot_article_event_fit as event_plot
import review_tristatic_fits_gui as fit_review


DEFAULT_INPUT_H5 = "results/all_tristatic_ceplecha_snr_weighted_v20260616d.h5"
DEFAULT_REVIEW_H5 = "results/tristatic_fit_review.h5"
MODEL_CV = "constant velocity"
MODEL_SHRINKING = "shrinking radius"
MAX_RADIUS_FRACTIONAL_UNCERTAINTY = 0.50
MAX_RMS_TOTAL_PATH_RESIDUAL_M = 50.0
MAX_WEIGHTED_RMS = 1.25
MAX_ABS_TOTAL_PATH_RESIDUAL_M = 100.0
MAX_ABS_LINK_MEAN_RESIDUAL_M = 20.0
MAX_CLIP_FRACTION = 0.25


def reserve_keybindings():
    reserved = {"b", "g", "n", "p", "r", "s"}
    for key in list(plt.rcParams):
        if key.startswith("keymap."):
            plt.rcParams[key] = [value for value in plt.rcParams[key] if value not in reserved]


def bic_from_normalized_residuals(normalized_residuals, n_params):
    residuals = np.asarray(normalized_residuals, dtype=np.float64)
    finite = np.isfinite(residuals)
    n_obs = int(np.count_nonzero(finite))
    if n_obs <= 0:
        return np.nan, np.nan, n_obs
    chi2 = float(np.sum(residuals[finite] ** 2.0))
    return float(chi2 + int(n_params) * np.log(n_obs)), chi2, n_obs


def manual_mask_from_review(review_h5, event_id, n_points):
    if not review_h5 or not os.path.exists(review_h5):
        return np.zeros(n_points, dtype=bool)
    with h5py.File(review_h5, "r") as h:
        if "reviews" not in h or event_id not in h["reviews"]:
            return np.zeros(n_points, dtype=bool)
        g = h["reviews"][event_id]
        if "manual_reject_mask" not in g:
            return np.zeros(n_points, dtype=bool)
        mask = np.asarray(g["manual_reject_mask"][:], dtype=bool)
    if len(mask) != n_points:
        out = np.zeros(n_points, dtype=bool)
        out[: min(n_points, len(mask))] = mask[: min(n_points, len(mask))]
        return out
    return mask


def mass_candidate_mask(h):
    n = len(h["event_id"])
    mask = np.ones(n, dtype=bool)
    required = [
        "log10_radius_std",
        "rms_total_path_residual_m",
        "weighted_rms",
        "max_abs_total_path_residual_m",
        "max_abs_link_mean_residual_m",
        "clip_fraction",
        "clipping_disallowed",
        "clipping_is_isolated",
    ]
    if any(key not in h for key in required):
        return mask
    radius_frac_unc = np.log(10.0) * np.asarray(h["log10_radius_std"][:], dtype=float)
    mask &= np.isfinite(radius_frac_unc) & (radius_frac_unc < MAX_RADIUS_FRACTIONAL_UNCERTAINTY)
    mask &= np.asarray(h["rms_total_path_residual_m"][:], dtype=float) < MAX_RMS_TOTAL_PATH_RESIDUAL_M
    mask &= np.asarray(h["weighted_rms"][:], dtype=float) < MAX_WEIGHTED_RMS
    mask &= np.asarray(h["max_abs_total_path_residual_m"][:], dtype=float) < MAX_ABS_TOTAL_PATH_RESIDUAL_M
    mask &= np.asarray(h["max_abs_link_mean_residual_m"][:], dtype=float) < MAX_ABS_LINK_MEAN_RESIDUAL_M
    mask &= np.asarray(h["clip_fraction"][:], dtype=float) <= MAX_CLIP_FRACTION
    mask &= ~np.asarray(h["clipping_disallowed"][:], dtype=bool)
    mask &= np.asarray(h["clipping_is_isolated"][:], dtype=bool)
    return mask


def stored_ceplecha_fit(group):
    fit = {
        "model": MODEL_SHRINKING,
        "params": np.asarray(group["params"][:], dtype=np.float64),
        "fit_epoch_time_ns": int(group.attrs.get("fit_epoch_time_ns", group["time_ns"][0])),
        "time_ns": np.asarray(group["time_ns"][:], dtype=np.int64),
        "t_rel_s": np.asarray(group["t_rel_s"][:], dtype=np.float64),
        "predicted_total_paths_m": np.asarray(group["predicted_total_paths_m"][:], dtype=np.float64),
        "residuals_m": np.asarray(group["residuals_m"][:], dtype=np.float64),
        "normalized_residuals": np.asarray(group["normalized_residuals"][:], dtype=np.float64),
        "x_gcrs_m": np.asarray(group["x_gcrs_m"][:], dtype=np.float64),
        "v_gcrs_mps": np.asarray(group["v_gcrs_mps"][:], dtype=np.float64),
        "x_itrs_m": np.asarray(group["x_itrs_m"][:], dtype=np.float64),
        "v_itrs_mps": np.asarray(group["v_itrs_mps"][:], dtype=np.float64),
        "radius_m": np.asarray(group["radius_m"][:], dtype=np.float64),
        "mass_kg": np.asarray(group["mass_kg"][:], dtype=np.float64),
    }
    if "all_time_ns" in group:
        for key in [
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
        ]:
            fit[key] = np.asarray(group[key][:])
        sigma = np.asarray(group["all_sigma_m"][:], dtype=np.float64) if "all_sigma_m" in group else np.ones_like(fit["all_residuals_m"])
        fit["all_normalized_residuals"] = fit["all_residuals_m"] / sigma
        fit["keep_rows"] = np.asarray(group["all_keep_rows"][:], dtype=bool)
    else:
        fit["all_time_ns"] = fit["time_ns"]
        fit["all_t_rel_s"] = fit["t_rel_s"]
        fit["all_predicted_total_paths_m"] = fit["predicted_total_paths_m"]
        fit["all_residuals_m"] = fit["residuals_m"]
        fit["all_normalized_residuals"] = fit["normalized_residuals"]
        fit["all_x_gcrs_m"] = fit["x_gcrs_m"]
        fit["all_v_gcrs_mps"] = fit["v_gcrs_mps"]
        fit["all_x_itrs_m"] = fit["x_itrs_m"]
        fit["all_v_itrs_mps"] = fit["v_itrs_mps"]
        fit["all_radius_m"] = fit["radius_m"]
        fit["all_mass_kg"] = fit["mass_kg"]
        fit["keep_rows"] = np.ones(len(fit["time_ns"]), dtype=bool)
    return fit


class MassSupportReviewer:
    def __init__(self, input_h5, review_h5=None, event_id=None, candidates_only=True):
        self.input_h5 = input_h5
        self.review_h5 = review_h5
        self.h = h5py.File(input_h5, "r")
        all_event_ids = fit_review.decode_strings(self.h["event_id"][:])
        self.source_indices = np.arange(len(all_event_ids), dtype=int)
        if candidates_only:
            candidate_mask = mass_candidate_mask(self.h)
            self.source_indices = self.source_indices[candidate_mask]
            self.event_ids = all_event_ids[candidate_mask]
        else:
            self.event_ids = all_event_ids
        self.review_store = fit_review.ReviewStore(review_h5, input_h5, self.event_ids) if review_h5 else None
        self.index = 0
        if event_id:
            matches = np.flatnonzero(self.event_ids == event_id)
            if len(matches) == 0:
                raise ValueError(f"Event {event_id} not found")
            self.index = int(matches[0])

        self.cache = {}
        self.mass_supported = np.zeros(len(self.event_ids), dtype=bool)
        self.current_payload = None
        self.current_comparison = None
        self.pick_artists = {}
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
        plt.subplots_adjust(left=0.07, right=0.82, bottom=0.12, top=0.90, hspace=0.34, wspace=0.30)
        self.status_text = self.fig.text(0.845, 0.12, "", ha="left", va="bottom", fontsize=9, family="monospace")
        self.buttons = {
            "prev": Button(self.fig.add_axes([0.08, 0.025, 0.07, 0.045]), "Prev"),
            "next": Button(self.fig.add_axes([0.16, 0.025, 0.07, 0.045]), "Next"),
            "reanalyze": Button(self.fig.add_axes([0.27, 0.025, 0.10, 0.045]), "Reanalyze"),
            "good": Button(self.fig.add_axes([0.40, 0.025, 0.15, 0.045]), "Supports mass"),
            "bad": Button(self.fig.add_axes([0.56, 0.025, 0.17, 0.045]), "Does not support"),
        }
        self.buttons["prev"].on_clicked(lambda _event: self.goto(self.index - 1))
        self.buttons["next"].on_clicked(lambda _event: self.goto(self.index + 1))
        self.buttons["reanalyze"].on_clicked(lambda _event: self.analyze_current(force=True))
        self.buttons["good"].on_clicked(lambda _event: self.mark_supported(True))
        self.buttons["bad"].on_clicked(lambda _event: self.mark_supported(False))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.analyze_current(force=False)

    def read_payload(self, idx):
        event_id = self.event_ids[idx]
        group = self.h["points"][event_id]
        if "all_time_ns" in group:
            time_ns = np.asarray(group["all_time_ns"][:], dtype=np.int64)
            measured = np.asarray(group["all_measured_total_paths_m"][:], dtype=np.float64)
            sigma = np.asarray(group["all_sigma_m"][:], dtype=np.float64) if "all_sigma_m" in group else np.ones_like(measured)
            snr_db = np.asarray(group["all_snr_db"][:], dtype=np.float64)
            x_itrs = np.asarray(group["all_x_itrs_m"][:], dtype=np.float64)
            keep_stored = np.asarray(group["all_keep_rows"][:], dtype=bool)
        else:
            time_ns = np.asarray(group["time_ns"][:], dtype=np.int64)
            measured = np.asarray(group["measured_total_paths_m"][:], dtype=np.float64)
            sigma = np.asarray(group["sigma_m"][:], dtype=np.float64) if "sigma_m" in group else np.ones_like(measured)
            snr_db = np.asarray(group["snr_db"][:], dtype=np.float64)
            x_itrs = np.asarray(group["x_itrs_m"][:], dtype=np.float64)
            keep_stored = np.ones(len(time_ns), dtype=bool)
        manual_reject = manual_mask_from_review(self.review_h5, event_id, len(time_ns))
        keep = keep_stored & ~manual_reject
        if snr_db.ndim == 2:
            sanya_snr_db = snr_db[:, 0]
        else:
            sanya_snr_db = snr_db
        x_gcrs = fit_review.ecef_to_gcrs(x_itrs, time_ns)
        t_rel_s = (time_ns.astype(np.float64) - float(time_ns[0])) / 1e9
        p0_linear = np.concatenate([x_gcrs[0], np.polyfit(t_rel_s, x_gcrs, 1)[0]])
        return {
            "event_id": event_id,
            "group": group,
            "time_ns": time_ns,
            "measured_total_paths_m": measured,
            "sigma_m": sigma,
            "sanya_snr_db": sanya_snr_db,
            "x_itrs_initial_m": x_itrs,
            "p0_linear": p0_linear,
            "p0_ceplecha": np.asarray(group["params"][:], dtype=np.float64),
            "keep_rows": keep,
            "manual_reject_mask": manual_reject,
        }

    def compare_models(self, idx, force=False):
        event_id = self.event_ids[idx]
        if not force and event_id in self.cache:
            return self.cache[event_id]
        payload = self.read_payload(idx)
        keep = payload["keep_rows"]
        if np.count_nonzero(keep) < fit_review.MIN_CONSTANT_VELOCITY_POINTS:
            raise RuntimeError("Too few retained measurements for model comparison")
        cv_fit = fit_review.fit_linear_paths(
            payload["measured_total_paths_m"],
            payload["time_ns"],
            payload["sigma_m"],
            keep,
            payload["p0_linear"],
            initial_points_itrs_m=payload["x_itrs_initial_m"],
            estimate_lower_bound=False,
        )
        if np.array_equal(keep, stored_ceplecha_fit(payload["group"])["keep_rows"]):
            shrink_fit = stored_ceplecha_fit(payload["group"])
        else:
            shrink_fit = fit_review.fit_ceplecha_paths(
                payload["measured_total_paths_m"],
                payload["time_ns"],
                payload["sigma_m"],
                keep,
                payload["x_itrs_initial_m"],
                p0_fallback=payload["p0_ceplecha"],
            )
        cv_bic, cv_chi2, n_obs = bic_from_normalized_residuals(cv_fit["normalized_residuals"], 6)
        shrink_bic, shrink_chi2, _n_obs2 = bic_from_normalized_residuals(shrink_fit["normalized_residuals"], 7)
        comparison = {
            "payload": payload,
            "cv_fit": cv_fit,
            "shrink_fit": shrink_fit,
            "cv_bic": cv_bic,
            "shrink_bic": shrink_bic,
            "delta_bic": cv_bic - shrink_bic,
            "cv_chi2": cv_chi2,
            "shrink_chi2": shrink_chi2,
            "delta_chi2": cv_chi2 - shrink_chi2,
            "n_obs": n_obs,
        }
        self.cache[event_id] = comparison
        return comparison

    def analyze_current(self, force=False):
        try:
            comparison = self.compare_models(self.index, force=force)
        except Exception as exc:
            self.status_text.set_text(f"Analysis failed:\n{exc}")
            self.redraw_failed()
            return
        self.current_comparison = comparison
        self.current_payload = comparison["payload"]
        self.redraw(comparison)

    def goto(self, idx):
        self.index = int(np.clip(idx, 0, len(self.event_ids) - 1))
        self.analyze_current(force=False)

    def mark_supported(self, supported):
        self.mass_supported[self.index] = bool(supported)
        self.analyze_current(force=False)

    def save_manual_mask(self, payload, comparison):
        if self.review_store is None:
            return
        existing = self.review_store.load(payload["event_id"], len(payload["time_ns"]))
        self.review_store.save(
            payload["event_id"],
            existing["quality"],
            existing["model"],
            payload["manual_reject_mask"],
            comparison["shrink_fit"],
        )

    def save_manual_mask_only(self, payload):
        if self.review_h5 is None:
            return
        os.makedirs(os.path.dirname(self.review_h5), exist_ok=True)
        with h5py.File(self.review_h5, "a") as h:
            h.attrs["source_fit_h5"] = os.path.abspath(self.input_h5)
            h.attrs["format"] = "sanya tristatic manual fit review"
            reviews = h.require_group("reviews")
            event_id = payload["event_id"]
            g = reviews.require_group(event_id)
            if "manual_reject_mask" in g:
                del g["manual_reject_mask"]
            g["manual_reject_mask"] = np.asarray(payload["manual_reject_mask"], dtype=bool)
            g.attrs["n_manual_reject"] = int(np.count_nonzero(payload["manual_reject_mask"]))

    def on_key(self, event):
        if event.key in ("right", "n"):
            self.goto(self.index + 1)
        elif event.key in ("left", "p"):
            self.goto(self.index - 1)
        elif event.key == "r":
            self.analyze_current(force=True)
        elif event.key == "g":
            self.mark_supported(True)
        elif event.key == "b":
            self.mark_supported(False)

    def on_pick(self, event):
        if event.artist not in self.pick_artists or len(event.ind) == 0 or self.current_payload is None:
            return
        source_indices = self.pick_artists[event.artist]
        pulse_index = int(source_indices[int(event.ind[0])])
        payload = self.current_payload
        payload["manual_reject_mask"][pulse_index] = ~payload["manual_reject_mask"][pulse_index]
        payload["keep_rows"] = ~payload["manual_reject_mask"]
        if np.count_nonzero(payload["keep_rows"]) < fit_review.MIN_CONSTANT_VELOCITY_POINTS:
            payload["manual_reject_mask"][pulse_index] = ~payload["manual_reject_mask"][pulse_index]
            payload["keep_rows"] = ~payload["manual_reject_mask"]
            return
        event_id = payload["event_id"]
        self.save_manual_mask_only(payload)
        self.cache.pop(event_id, None)
        self.analyze_current(force=True)
        if self.current_comparison is not None:
            self.save_manual_mask(self.current_payload, self.current_comparison)

    def redraw_failed(self):
        for ax in self.axes.ravel():
            ax.clear()
        self.fig.canvas.draw_idle()

    def residual_products(self, payload, fit):
        measured_itrs = event_plot.lfm_corrected_point_solutions(
            payload["measured_total_paths_m"],
            fit["all_x_itrs_m"],
            fit["all_v_itrs_mps"],
        )
        measured_gcrs = fit_review.ecef_to_gcrs(measured_itrs, payload["time_ns"])
        along_axis, cross_axis = event_plot.event_axes(fit["all_x_gcrs_m"], fit["all_v_gcrs_mps"])
        origin = fit["all_x_gcrs_m"][0]
        fit_along_m = (fit["all_x_gcrs_m"] - origin) @ along_axis
        fit_cross_m = (fit["all_x_gcrs_m"] - origin) @ cross_axis
        meas_along_m = (measured_gcrs - origin) @ along_axis
        meas_cross_m = (measured_gcrs - origin) @ cross_axis
        return measured_itrs, meas_along_m - fit_along_m, meas_cross_m - fit_cross_m

    def redraw(self, comparison):
        for ax in self.axes.ravel():
            ax.clear()
        self.pick_artists = {}
        payload = comparison["payload"]
        keep = payload["keep_rows"]
        cv_fit = comparison["cv_fit"]
        shrink_fit = comparison["shrink_fit"]
        measured_itrs, cv_along_resid, cv_cross_resid = self.residual_products(payload, cv_fit)
        _measured_itrs2, sh_along_resid, sh_cross_resid = self.residual_products(payload, shrink_fit)
        east_km, north_km = fit_review.horizontal_offsets_km(measured_itrs)
        fit_east_km, fit_north_km = fit_review.horizontal_offsets_km(shrink_fit["all_x_itrs_m"])
        t_rel_s = (payload["time_ns"].astype(np.float64) - float(payload["time_ns"][0])) / 1e9
        sigma_m = np.nanmedian(payload["sigma_m"], axis=1)

        self.fig.suptitle(
            f"{fit_review.unix_ns_to_utc_label(payload['time_ns'][0])}  "
            f"({self.index + 1}/{len(self.event_ids)})  {payload['event_id']}"
        )

        ax = self.axes[0, 0]
        rejected = ~keep
        if np.any(rejected):
            rejected_artist = ax.scatter(
                east_km[rejected],
                north_km[rejected],
                c="0.70",
                s=30,
                edgecolors="none",
                label="rejected",
                picker=6,
            )
            self.pick_artists[rejected_artist] = np.flatnonzero(rejected)
        sc = None
        if np.any(keep):
            sc = ax.scatter(
                east_km[keep],
                north_km[keep],
                c=payload["sanya_snr_db"][keep],
                cmap="viridis",
                s=30,
                edgecolors="none",
                label="retained",
                picker=6,
            )
            self.pick_artists[sc] = np.flatnonzero(keep)
        ax.plot(fit_east_km, fit_north_km, color="#1b7837", lw=1.6, label="shrinking-radius fit")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        ax.grid(True, color="0.88")
        ax.legend(loc="best", frameon=False)

        ax = self.axes[0, 1]
        ax.axhline(0.0, color="0.3", lw=1.0)
        ax.errorbar(t_rel_s[keep], cv_cross_resid[keep], yerr=sigma_m[keep], fmt="o", color="#2166ac", ecolor="0.65", ms=4, lw=0.8, label="constant velocity")
        ax.errorbar(t_rel_s[keep], sh_cross_resid[keep], yerr=sigma_m[keep], fmt="o", color="#b2182b", ecolor="0.75", ms=3, lw=0.7, alpha=0.85, label="shrinking radius")
        if np.any(rejected):
            rejected_artist = ax.scatter(t_rel_s[rejected], sh_cross_resid[rejected], c="0.70", s=32, edgecolors="none", picker=6, label="rejected")
            self.pick_artists[rejected_artist] = np.flatnonzero(rejected)
        if np.any(keep):
            retained_artist = ax.scatter(t_rel_s[keep], sh_cross_resid[keep], c="#b2182b", s=18, edgecolors="none", picker=6)
            self.pick_artists[retained_artist] = np.flatnonzero(keep)
        ax.set_xlabel("Time since first pulse (s)")
        ax.set_ylabel("Cross-track residual (m)")
        ax.grid(True, color="0.88")
        ax.legend(loc="best", frameon=False)

        ax = self.axes[1, 0]
        ax.axhline(0.0, color="0.3", lw=1.0)
        ax.errorbar(t_rel_s[keep], cv_along_resid[keep], yerr=sigma_m[keep], fmt="o", color="#2166ac", ecolor="0.65", ms=4, lw=0.8, label="constant velocity")
        ax.errorbar(t_rel_s[keep], sh_along_resid[keep], yerr=sigma_m[keep], fmt="o", color="#b2182b", ecolor="0.75", ms=3, lw=0.7, alpha=0.85, label="shrinking radius")
        if np.any(rejected):
            rejected_artist = ax.scatter(t_rel_s[rejected], sh_along_resid[rejected], c="0.70", s=32, edgecolors="none", picker=6, label="rejected")
            self.pick_artists[rejected_artist] = np.flatnonzero(rejected)
        if np.any(keep):
            retained_artist = ax.scatter(t_rel_s[keep], sh_along_resid[keep], c="#b2182b", s=18, edgecolors="none", picker=6)
            self.pick_artists[retained_artist] = np.flatnonzero(keep)
        ax.set_xlabel("Time since first pulse (s)")
        ax.set_ylabel("Along-track residual (m)")
        ax.grid(True, color="0.88")
        ax.legend(loc="best", frameon=False)

        ax = self.axes[1, 1]
        names = [MODEL_CV, MODEL_SHRINKING]
        bics = [comparison["cv_bic"], comparison["shrink_bic"]]
        ax.bar(names, bics, color=["#2166ac", "#b2182b"])
        ax.set_ylabel("BIC")
        ax.grid(True, axis="y", color="0.88")
        ax.tick_params(axis="x", rotation=15)

        radius_um = float(shrink_fit["all_radius_m"][0] * 1e6)
        mass_kg = float(shrink_fit["all_mass_kg"][0])
        log10_radius_std = float(shrink_fit.get("log10_radius_std", np.nan))
        sigma_radius_m, sigma_mass_kg = fit_review.radius_mass_uncertainty(float(shrink_fit["all_radius_m"][0]), mass_kg, log10_radius_std)
        sigma_radius_um = sigma_radius_m * 1e6 if np.isfinite(sigma_radius_m) else np.nan
        verdict = "supports mass" if comparison["delta_bic"] >= 6.0 else "not a measured mass"
        if comparison["delta_bic"] > 10.0:
            strength = "strong"
        elif comparison["delta_bic"] > 6.0:
            strength = "moderate"
        elif comparison["delta_bic"] > 0.0:
            strength = "weak/marginal"
        else:
            strength = "none"
        self.status_text.set_text(
            "\n".join(
                [
                    f"Delta BIC: {comparison['delta_bic']:.2f}",
                    f"verdict: {verdict}",
                    f"evidence: {strength}",
                    f"N obs: {comparison['n_obs']}",
                    f"Delta chi2: {comparison['delta_chi2']:.2f}",
                    f"BIC cv: {comparison['cv_bic']:.2f}",
                    f"BIC shrink: {comparison['shrink_bic']:.2f}",
                    "",
                    f"r0: {radius_um:.3g} +/- {sigma_radius_um:.2g} um",
                    f"m0: {mass_kg:.3g} +/- {sigma_mass_kg:.2g} kg",
                    "",
                    "click map point: toggle reject",
                    "keys: n/p r g/b",
                ]
            )
        )
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()
        self.h.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", default=DEFAULT_INPUT_H5)
    parser.add_argument("--review-h5", default=DEFAULT_REVIEW_H5)
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--all-events", action="store_true", help="Review all fitted events instead of only mass-candidate events.")
    return parser.parse_args()


def main():
    reserve_keybindings()
    args = parse_args()
    gui = MassSupportReviewer(args.input_h5, args.review_h5, event_id=args.event_id, candidates_only=not args.all_events)
    gui.show()


if __name__ == "__main__":
    main()
