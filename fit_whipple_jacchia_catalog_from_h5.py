import argparse
import concurrent.futures
import hashlib
import os
from pathlib import Path

import h5py
import numpy as np

import fit_all_ballistic_snr_weighted as base
import fit_event_joint_delay_doppler_fft as fit


DEFAULT_INPUT_DIR = Path("results/tristatic_student_t_bootstrap_orbit100_20260630")
DEFAULT_OUTPUT_DIR = Path("results/tristatic_whipple_jacchia_20260701")
EVENT_PREFIX = "joint_delay_doppler_fft_"


def load_group(group):
    out = {}
    for key, value in group.attrs.items():
        out[key] = value.decode("utf-8") if isinstance(value, bytes) else value
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            arr = value[()]
            if getattr(arr, "dtype", None) is not None and arr.dtype.kind == "S":
                arr = np.asarray([item.decode("utf-8") for item in arr])
            out[key] = arr
    return out


def event_id_from_path(path):
    stem = Path(path).stem
    if stem.startswith(EVENT_PREFIX):
        return stem[len(EVENT_PREFIX) :]
    return stem


def whipple_seed_grid(source_joint, max_starts):
    params = np.asarray(source_joint.get("params", []), dtype=np.float64)
    if params.size < 6 or not np.all(np.isfinite(params[:6])):
        x = np.asarray(source_joint["x_gcrs_m"], dtype=np.float64)
        v = np.asarray(source_joint["v_gcrs_mps"], dtype=np.float64)
        base_state = np.concatenate([x[0], v[0]])
    else:
        base_state = params[:6]

    starts = []
    if str(source_joint.get("dynamical_model", "")) == "whipple_speed" and params.size >= 8:
        starts.append(params[:8].copy())
    for a_mps in (1.0, 10.0, 100.0, 300.0):
        for b_s_inv in (0.3, 1.0, 3.0, 10.0):
            starts.append(np.concatenate([base_state, [np.log10(a_mps), np.log10(b_s_inv)]]))
    unique = []
    seen = set()
    for start in starts:
        key = tuple(np.round(start[3:8], 5))
        if key in seen:
            continue
        seen.add(key)
        unique.append(start)
    return unique[: int(max_starts)]


def finite_percentile(values, q, axis=0):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.nanpercentile(arr, q, axis=axis)


def bic_from_joint_fit(joint_fit):
    path_keep = np.asarray(joint_fit.get("path_keep", []), dtype=bool)
    fft_keep = np.asarray(joint_fit.get("fft_keep", []), dtype=bool)
    path_resid = np.asarray(joint_fit.get("normalized_path_residuals", []), dtype=np.float64)
    fft_resid = np.asarray(joint_fit.get("normalized_fft_residuals", []), dtype=np.float64)
    values = []
    if path_resid.shape == path_keep.shape:
        values.append(path_resid[path_keep])
    if fft_resid.shape == fft_keep.shape:
        values.append(fft_resid[fft_keep])
    if not values:
        return np.nan, np.nan, 0, 0
    residuals = np.concatenate([np.ravel(v) for v in values])
    residuals = residuals[np.isfinite(residuals)]
    n_obs = int(residuals.size)
    k_params = int(np.asarray(joint_fit.get("full_params", joint_fit.get("params", [])), dtype=np.float64).size)
    if n_obs <= 0 or k_params <= 0:
        return np.nan, np.nan, n_obs, k_params
    chi2 = float(np.sum(residuals**2.0))
    bic = float(chi2 + k_params * np.log(n_obs))
    return bic, chi2, n_obs, k_params


def constant_velocity_seed(source_joint):
    params = np.asarray(source_joint.get("params", []), dtype=np.float64)
    if params.size >= 6 and np.all(np.isfinite(params[:6])):
        return params[:6]
    x = np.asarray(source_joint["x_gcrs_m"], dtype=np.float64)
    v = np.asarray(source_joint["v_gcrs_mps"], dtype=np.float64)
    return np.concatenate([x[0], v[0]])


def add_dynamic_mass_summary(best, times_ns, rho_of_alt_m):
    """Attach analytic dynamic-mass products derived from the WJ fit."""

    t_rel_s = np.asarray(best.get("t_rel_s", []), dtype=np.float64)
    mass_kg, radius_m, area_per_mass_m2_kg = fit.dynamic_mass_from_whipple_state(
        best.get("params", []),
        t_rel_s,
        best.get("x_itrs_m", []),
        best.get("v_gcrs_mps", []),
        rho_of_alt_m,
    )
    best["dynamic_mass_kg"] = mass_kg
    best["dynamic_radius_m"] = radius_m
    best["dynamic_area_per_mass_m2_kg"] = area_per_mass_m2_kg
    best["dynamic_mass_equation"] = "m/A = gamma rho_air v(t)^2 / |dv/dt|"
    best["dynamic_mass_speed_model"] = "v(t) = v_inf - a exp(b t); |dv/dt| = a b exp(b t)"
    best["dynamic_mass_epoch"] = "first_detected_pulse"
    best["dynamic_mass_area_convention"] = "A = pi r^2 for compact spherical meteoroid equivalent radius"
    best["dynamic_mass_drag_coefficient_gamma"] = float(fit.DYNAMIC_MASS_DRAG_COEFFICIENT)
    best["dynamic_mass_area_factor"] = float(fit.DYNAMIC_MASS_AREA_FACTOR)

    finite = np.isfinite(mass_kg) & (mass_kg > 0.0) & np.isfinite(radius_m) & (radius_m > 0.0)
    idx0 = int(np.flatnonzero(finite)[0]) if np.any(finite) else 0
    best["initial_dynamic_mass_kg"] = float(mass_kg[idx0]) if mass_kg.size else np.nan
    best["initial_dynamic_radius_m"] = float(radius_m[idx0]) if radius_m.size else np.nan
    best["initial_dynamic_area_per_mass_m2_kg"] = (
        float(area_per_mass_m2_kg[idx0]) if area_per_mass_m2_kg.size else np.nan
    )

    mass0_samples = []
    radius0_samples = []
    if (
        "bootstrap_params" in best
        and "bootstrap_x_gcrs_m" in best
        and "bootstrap_v_gcrs_mps" in best
    ):
        params = np.asarray(best["bootstrap_params"], dtype=np.float64)
        x_gcrs = np.asarray(best["bootstrap_x_gcrs_m"], dtype=np.float64)
        v_gcrs = np.asarray(best["bootstrap_v_gcrs_mps"], dtype=np.float64)
        success = np.asarray(best.get("bootstrap_optimizer_success", np.ones(params.shape[0], dtype=bool)), dtype=bool)
        n_boot = min(params.shape[0], x_gcrs.shape[0], v_gcrs.shape[0], success.shape[0])
        for sample_idx in range(n_boot):
            if not success[sample_idx]:
                continue
            try:
                sample_x_itrs, _sample_v_itrs = base.gcrs_state_samples_to_itrs(
                    x_gcrs[sample_idx],
                    v_gcrs[sample_idx],
                    times_ns,
                )
                sample_mass, sample_radius, _sample_area_per_mass = fit.dynamic_mass_from_whipple_state(
                    params[sample_idx],
                    t_rel_s,
                    sample_x_itrs,
                    v_gcrs[sample_idx],
                    rho_of_alt_m,
                )
            except Exception:
                continue
            if sample_mass.size <= idx0 or sample_radius.size <= idx0:
                continue
            if np.isfinite(sample_mass[idx0]) and sample_mass[idx0] > 0.0:
                mass0_samples.append(float(sample_mass[idx0]))
            if np.isfinite(sample_radius[idx0]) and sample_radius[idx0] > 0.0:
                radius0_samples.append(float(sample_radius[idx0]))

    best["initial_dynamic_mass_bootstrap_kg"] = np.asarray(mass0_samples, dtype=np.float64)
    best["initial_dynamic_radius_bootstrap_m"] = np.asarray(radius0_samples, dtype=np.float64)
    best["initial_dynamic_mass_bootstrap_samples_successful"] = int(len(mass0_samples))
    if len(mass0_samples):
        best["initial_dynamic_mass_median_kg"] = float(np.nanmedian(mass0_samples))
        best["initial_dynamic_mass_lo95_kg"] = float(np.nanpercentile(mass0_samples, 2.5))
        best["initial_dynamic_mass_hi95_kg"] = float(np.nanpercentile(mass0_samples, 97.5))
    if len(radius0_samples):
        best["initial_dynamic_radius_median_m"] = float(np.nanmedian(radius0_samples))
        best["initial_dynamic_radius_lo95_m"] = float(np.nanpercentile(radius0_samples, 2.5))
        best["initial_dynamic_radius_hi95_m"] = float(np.nanpercentile(radius0_samples, 97.5))

    # Make generic mass/radius fields point to the selected canonical mass product.
    best["initial_mass_kg"] = float(best["initial_dynamic_mass_kg"])
    best["initial_radius_m"] = float(best["initial_dynamic_radius_m"])
    return best


def add_whipple_bootstrap(
    best,
    measured,
    times_ns,
    rho_of_alt_m,
    sigma_m,
    fft_offset_hz,
    fft_keep,
    sigma_fft_hz,
    path_keep,
    fit_station_bias,
    fft_model,
    reference_chirp_rate_scale,
    residual_likelihood,
    student_t_nu_delay,
    student_t_nu_fft,
    n_samples,
    seed,
):
    n_samples = int(n_samples)
    if n_samples <= 0:
        best["bootstrap_enabled"] = False
        best["bootstrap_samples_requested"] = 0
        best["bootstrap_samples_successful"] = 0
        return best

    rng = np.random.default_rng(seed)
    measured = np.asarray(measured, dtype=np.float64)
    fft_offset_hz = np.asarray(fft_offset_hz, dtype=np.float64)
    sigma_m = np.asarray(sigma_m, dtype=np.float64)
    sigma_fft_hz = np.asarray(sigma_fft_hz, dtype=np.float64)
    path_keep = np.asarray(path_keep, dtype=bool)
    fft_keep = np.asarray(fft_keep, dtype=bool)

    base_apparent = np.asarray(best["apparent_path_length_m"], dtype=np.float64)
    base_fft = np.asarray(best["model_fft_peak_hz"], dtype=np.float64)
    params0 = np.asarray(best["params"], dtype=np.float64)

    param_samples = []
    full_param_samples = []
    x_samples = []
    v_samples = []
    speed_samples = []
    apparent_samples = []
    fft_model_samples = []
    measured_samples = []
    fft_observed_samples = []
    residual_summary = []
    optimizer_success = []
    failures = []

    for sample_idx in range(n_samples):
        if str(residual_likelihood) == "student_t":
            delay_noise = rng.standard_t(max(float(student_t_nu_delay), 1e-6), size=measured.shape) * sigma_m
            fft_noise = rng.standard_t(max(float(student_t_nu_fft), 1e-6), size=fft_offset_hz.shape) * sigma_fft_hz
        else:
            delay_noise = rng.normal(0.0, sigma_m, size=measured.shape)
            fft_noise = rng.normal(0.0, sigma_fft_hz, size=fft_offset_hz.shape)
        synthetic_measured = np.where(path_keep & np.isfinite(base_apparent), base_apparent + delay_noise, measured)
        synthetic_fft = np.where(fft_keep & np.isfinite(base_fft), base_fft + fft_noise, fft_offset_hz)
        try:
            sample_fit = fit.fit_joint_delay_doppler(
                synthetic_measured,
                times_ns,
                rho_of_alt_m,
                params0,
                sigma_m,
                synthetic_fft,
                fft_keep,
                sigma_fft_hz,
                keep_rows=np.ones(len(times_ns), dtype=bool),
                epoch_time_ns=int(times_ns[0]),
                fit_station_bias=bool(fit_station_bias),
                fft_model=str(fft_model),
                reference_chirp_rate_scale=float(reference_chirp_rate_scale),
                path_keep=path_keep,
                model_kind="whipple_speed",
                residual_likelihood=str(residual_likelihood),
                student_t_nu_delay=float(student_t_nu_delay),
                student_t_nu_fft=float(student_t_nu_fft),
            )
        except Exception as exc:
            failures.append(f"{sample_idx}:{str(exc)[:120]}")
            continue

        param_samples.append(np.asarray(sample_fit["params"], dtype=np.float64))
        full_param_samples.append(np.asarray(sample_fit["full_params"], dtype=np.float64))
        x_samples.append(np.asarray(sample_fit["x_gcrs_m"], dtype=np.float64))
        v_samples.append(np.asarray(sample_fit["v_gcrs_mps"], dtype=np.float64))
        speed_samples.append(np.asarray(sample_fit["speed_km_s"], dtype=np.float64))
        apparent_samples.append(np.asarray(sample_fit["apparent_path_length_m"], dtype=np.float64))
        fft_model_samples.append(np.asarray(sample_fit["model_fft_peak_hz"], dtype=np.float64))
        measured_samples.append(synthetic_measured)
        fft_observed_samples.append(synthetic_fft)
        residual_summary.append(
            [
                float(sample_fit["weighted_rms"]),
                float(sample_fit["rms_total_path_residual_m"]),
                float(sample_fit["rms_path_rate_residual_mps"]),
                float(sample_fit["rms_fft_residual_hz"]),
                float(sample_fit["mean_abs_total_path_residual_m"]),
                float(sample_fit["mean_abs_path_rate_residual_mps"]),
                float(sample_fit["mean_abs_fft_residual_hz"]),
            ]
        )
        optimizer_success.append(bool(sample_fit.get("optimizer_success", False)))

    if param_samples:
        param_arr = np.asarray(param_samples, dtype=np.float64)
        full_param_arr = np.asarray(full_param_samples, dtype=np.float64)
        x_arr = np.asarray(x_samples, dtype=np.float64)
        v_arr = np.asarray(v_samples, dtype=np.float64)
        speed_arr = np.asarray(speed_samples, dtype=np.float64)
        a_arr = 10.0 ** param_arr[:, 6]
        b_arr = 10.0 ** param_arr[:, 7]
        logab = param_arr[:, 6:8]
        state0_arr = np.column_stack([x_arr[:, 0, :], v_arr[:, 0, :]])
        eccentricity_arr = fit.heliocentric_eccentricity_from_gcrs_state(state0_arr, int(times_ns[0]))
        try:
            corr = float(np.corrcoef(logab[:, 0], logab[:, 1])[0, 1])
        except Exception:
            corr = np.nan
        best["bootstrap_enabled"] = True
        best["bootstrap_samples_requested"] = n_samples
        best["bootstrap_samples_successful"] = int(param_arr.shape[0])
        best["bootstrap_params"] = param_arr
        best["bootstrap_full_params"] = full_param_arr
        best["bootstrap_x_gcrs_m"] = x_arr
        best["bootstrap_v_gcrs_mps"] = v_arr
        best["bootstrap_speed_km_s"] = speed_arr
        best["bootstrap_apparent_path_length_m"] = np.asarray(apparent_samples, dtype=np.float64)
        best["bootstrap_model_fft_peak_hz"] = np.asarray(fft_model_samples, dtype=np.float64)
        best["bootstrap_sampled_measured_total_paths_m"] = np.asarray(measured_samples, dtype=np.float64)
        best["bootstrap_sampled_fft_beat_hz"] = np.asarray(fft_observed_samples, dtype=np.float64)
        best["bootstrap_residual_summary"] = np.asarray(residual_summary, dtype=np.float64)
        best["bootstrap_residual_summary_names"] = np.asarray(
            [
                "weighted_rms",
                "path_rms_m",
                "path_rate_rms_mps",
                "fft_rms_hz",
                "mean_abs_path_m",
                "mean_abs_path_rate_mps",
                "mean_abs_fft_hz",
            ],
            dtype=object,
        )
        best["bootstrap_optimizer_success"] = np.asarray(optimizer_success, dtype=bool)
        best["bootstrap_a_mps_median"] = float(np.nanmedian(a_arr))
        best["bootstrap_a_mps_lo95"] = float(np.nanpercentile(a_arr, 2.5))
        best["bootstrap_a_mps_hi95"] = float(np.nanpercentile(a_arr, 97.5))
        best["bootstrap_b_s_inv_median"] = float(np.nanmedian(b_arr))
        best["bootstrap_b_s_inv_lo95"] = float(np.nanpercentile(b_arr, 2.5))
        best["bootstrap_b_s_inv_hi95"] = float(np.nanpercentile(b_arr, 97.5))
        best["bootstrap_log10_a_log10_b_corr"] = corr
        best["bootstrap_eccentricity"] = eccentricity_arr
        best["bootstrap_eccentricity_median"] = float(np.nanmedian(eccentricity_arr))
        best["bootstrap_eccentricity_lo95"] = float(np.nanpercentile(eccentricity_arr, 2.5))
        best["bootstrap_eccentricity_hi95"] = float(np.nanpercentile(eccentricity_arr, 97.5))
        best["bootstrap_interstellar_fraction_e_gt_1"] = float(np.nanmean(eccentricity_arr > 1.0))
        best["bootstrap_start_speed_km_s_median"] = float(np.nanmedian(speed_arr[:, 0]))
        best["bootstrap_start_speed_km_s_lo95"] = float(np.nanpercentile(speed_arr[:, 0], 2.5))
        best["bootstrap_start_speed_km_s_hi95"] = float(np.nanpercentile(speed_arr[:, 0], 97.5))
        best["bootstrap_end_speed_km_s_median"] = float(np.nanmedian(speed_arr[:, -1]))
        best["bootstrap_end_speed_km_s_lo95"] = float(np.nanpercentile(speed_arr[:, -1], 2.5))
        best["bootstrap_end_speed_km_s_hi95"] = float(np.nanpercentile(speed_arr[:, -1], 97.5))
    else:
        best["bootstrap_enabled"] = False
        best["bootstrap_samples_requested"] = n_samples
        best["bootstrap_samples_successful"] = 0
    best["bootstrap_n_failures"] = int(len(failures))
    best["bootstrap_failures"] = ";".join(failures[:20])
    best["bootstrap_method"] = "parametric_measurement_student_t" if str(residual_likelihood) == "student_t" else "parametric_measurement_gaussian"
    return best


def fit_one(source_h5, output_dir, overwrite, max_starts, bootstrap_samples, bootstrap_seed):
    source_h5 = Path(source_h5)
    event_id = event_id_from_path(source_h5)
    output_base = Path(output_dir) / f"joint_delay_doppler_fft_{event_id}"
    output_h5 = output_base.with_suffix(".h5")
    if output_h5.exists() and not overwrite:
        try:
            with h5py.File(output_h5, "r") as h:
                jg = h["joint_fit"]
                return {
                    "event_id": event_id,
                    "status": "ok",
                    "output_base": str(output_base),
                    "loaded_existing": True,
                    "weighted_rms": float(jg.attrs.get("weighted_rms", np.nan)),
                    "path_rms_m": float(jg.attrs.get("rms_total_path_residual_m", np.nan)),
                    "path_rate_rms_mps": float(jg.attrs.get("rms_path_rate_residual_mps", np.nan)),
                    "fft_rms_hz": float(jg.attrs.get("rms_fft_residual_hz", np.nan)),
                    "n_points": float(jg.attrs.get("n_points", np.nan)),
                    "n_fft_observations": float(jg.attrs.get("n_fft_observations", np.nan)),
                    "start_speed_km_s": float(np.asarray(jg["speed_km_s"])[0]) if "speed_km_s" in jg else np.nan,
                    "end_speed_km_s": float(np.asarray(jg["speed_km_s"])[-1]) if "speed_km_s" in jg else np.nan,
                    "a_mps": float(10.0 ** np.asarray(jg["params"])[6])
                    if "params" in jg and np.asarray(jg["params"]).size >= 8
                    else np.nan,
                    "b_s_inv": float(10.0 ** np.asarray(jg["params"])[7])
                    if "params" in jg and np.asarray(jg["params"]).size >= 8
                    else np.nan,
                    "heliocentric_eccentricity": float(jg.attrs.get("heliocentric_eccentricity", np.nan)),
                    "bootstrap_eccentricity_lo95": float(jg.attrs.get("bootstrap_eccentricity_lo95", np.nan)),
                    "bootstrap_eccentricity_hi95": float(jg.attrs.get("bootstrap_eccentricity_hi95", np.nan)),
                    "bootstrap_interstellar_fraction_e_gt_1": float(jg.attrs.get("bootstrap_interstellar_fraction_e_gt_1", np.nan)),
                    "bootstrap_samples_successful": float(jg.attrs.get("bootstrap_samples_successful", np.nan)),
                    "bootstrap_log10_a_log10_b_corr": float(jg.attrs.get("bootstrap_log10_a_log10_b_corr", np.nan)),
                    "bic_whipple_jacchia": float(jg.attrs.get("bic_whipple_jacchia", np.nan)),
                    "bic_constant_velocity": float(jg.attrs.get("bic_constant_velocity", np.nan)),
                    "delta_bic_constant_minus_whipple_jacchia": float(jg.attrs.get("delta_bic_constant_minus_whipple_jacchia", np.nan)),
                    "initial_dynamic_mass_kg": float(jg.attrs.get("initial_dynamic_mass_kg", np.nan)),
                    "initial_dynamic_mass_lo95_kg": float(jg.attrs.get("initial_dynamic_mass_lo95_kg", np.nan)),
                    "initial_dynamic_mass_hi95_kg": float(jg.attrs.get("initial_dynamic_mass_hi95_kg", np.nan)),
                    "initial_dynamic_mass_bootstrap_samples_successful": float(jg.attrs.get("initial_dynamic_mass_bootstrap_samples_successful", np.nan)),
                }
        except Exception:
            pass

    try:
        with h5py.File(source_h5, "r") as h:
            source_joint = load_group(h["joint_fit"])
            source_fft = load_group(h["fft_observations"])
    except Exception as exc:
        return {"event_id": event_id, "status": "error", "output_base": str(output_base), "error": f"load_failed:{exc}"}
    source_fft.pop("link_names", None)

    required = (
        "measured_total_paths_m",
        "time_ns",
        "path_sigma_m",
        "observed_fft_beat_hz",
        "fft_keep",
        "fft_sigma_hz",
        "path_keep",
        "x_itrs_m",
    )
    missing = [key for key in required if key not in source_joint]
    if missing:
        return {"event_id": event_id, "status": "error", "output_base": str(output_base), "error": f"missing:{','.join(missing)}"}

    measured = np.asarray(source_joint["measured_total_paths_m"], dtype=np.float64)
    times_ns = np.asarray(source_joint["time_ns"], dtype=np.int64)
    sigma_m = np.asarray(source_joint["path_sigma_m"], dtype=np.float64)
    fft_offset_hz = np.asarray(source_joint["observed_fft_beat_hz"], dtype=np.float64)
    fft_keep = np.asarray(source_joint["fft_keep"], dtype=bool)
    sigma_fft_hz = np.asarray(source_joint["fft_sigma_hz"], dtype=np.float64)
    path_keep = np.asarray(source_joint["path_keep"], dtype=bool)
    reference_itrs = np.asarray(source_joint["x_itrs_m"], dtype=np.float64)
    rho_of_alt_m, _meta = base.density_interpolator(times_ns, reference_itrs)

    constant_fit = None
    try:
        constant_fit = fit.fit_joint_delay_doppler(
            measured,
            times_ns,
            rho_of_alt_m,
            constant_velocity_seed(source_joint),
            sigma_m,
            fft_offset_hz,
            fft_keep,
            sigma_fft_hz,
            keep_rows=np.ones(len(times_ns), dtype=bool),
            epoch_time_ns=int(times_ns[0]),
            fit_station_bias=bool(source_joint.get("fit_station_bias", True)),
            fft_model=str(source_joint.get("fft_model", "range_offset_corrected_beat")),
            reference_chirp_rate_scale=float(source_joint.get("reference_chirp_rate_scale", 1.0)),
            path_keep=path_keep,
            model_kind="constant_velocity",
            residual_likelihood=str(source_joint.get("residual_likelihood", "student_t")),
            student_t_nu_delay=float(source_joint.get("student_t_nu_delay", fit.DEFAULT_STUDENT_T_NU_DELAY)),
            student_t_nu_fft=float(source_joint.get("student_t_nu_fft", fit.DEFAULT_STUDENT_T_NU_FFT)),
        )
    except Exception as exc:
        constant_fit = {"constant_velocity_fit_error": str(exc)[:240]}

    best_wj = None
    failures = []
    for p0 in whipple_seed_grid(source_joint, max_starts=max_starts):
        try:
            candidate = fit.fit_joint_delay_doppler(
                measured,
                times_ns,
                rho_of_alt_m,
                p0,
                sigma_m,
                fft_offset_hz,
                fft_keep,
                sigma_fft_hz,
                keep_rows=np.ones(len(times_ns), dtype=bool),
                epoch_time_ns=int(times_ns[0]),
                fit_station_bias=bool(source_joint.get("fit_station_bias", True)),
                fft_model=str(source_joint.get("fft_model", "range_offset_corrected_beat")),
                reference_chirp_rate_scale=float(source_joint.get("reference_chirp_rate_scale", 1.0)),
                path_keep=path_keep,
                model_kind="whipple_speed",
                residual_likelihood=str(source_joint.get("residual_likelihood", "student_t")),
                student_t_nu_delay=float(source_joint.get("student_t_nu_delay", fit.DEFAULT_STUDENT_T_NU_DELAY)),
                student_t_nu_fft=float(source_joint.get("student_t_nu_fft", fit.DEFAULT_STUDENT_T_NU_FFT)),
            )
        except Exception as exc:
            failures.append(str(exc)[:160])
            continue
        if best_wj is None or candidate["weighted_rms"] < best_wj["weighted_rms"]:
            best_wj = candidate

    if best_wj is None:
        return {
            "event_id": event_id,
            "status": "error",
            "output_base": str(output_base),
            "error": "all_starts_failed:" + ";".join(failures[:5]),
        }

    bic_wj, chi2_wj, n_obs_wj, k_wj = bic_from_joint_fit(best_wj)
    if isinstance(constant_fit, dict) and "params" in constant_fit:
        bic_cv, chi2_cv, n_obs_cv, k_cv = bic_from_joint_fit(constant_fit)
    else:
        bic_cv, chi2_cv, n_obs_cv, k_cv = np.nan, np.nan, 0, 0
    delta_bic_constant_minus_wj = float(bic_cv - bic_wj) if np.all(np.isfinite([bic_cv, bic_wj])) else np.nan
    selected_model = "whipple_speed"
    if np.isfinite(delta_bic_constant_minus_wj) and delta_bic_constant_minus_wj < 0.0:
        selected_model = "constant_velocity"

    best = best_wj if selected_model == "whipple_speed" else constant_fit
    best["bic_whipple_jacchia"] = float(bic_wj)
    best["bic_constant_velocity"] = float(bic_cv)
    best["chi2_whipple_jacchia"] = float(chi2_wj)
    best["chi2_constant_velocity"] = float(chi2_cv)
    best["n_bic_observations"] = int(max(n_obs_wj, n_obs_cv))
    best["n_bic_params_whipple_jacchia"] = int(k_wj)
    best["n_bic_params_constant_velocity"] = int(k_cv)
    best["delta_bic_constant_minus_whipple_jacchia"] = float(delta_bic_constant_minus_wj)
    best["bic_selected_model"] = selected_model
    best["bic_selection_rule"] = (
        "Dynamic mass is estimated only when BIC favors Whipple-Jacchia over constant velocity "
        "(delta_bic_constant_minus_whipple_jacchia > 0)."
    )
    if selected_model == "constant_velocity":
        best["whipple_jacchia_weighted_rms"] = float(best_wj.get("weighted_rms", np.nan))
        best["whipple_jacchia_rms_total_path_residual_m"] = float(best_wj.get("rms_total_path_residual_m", np.nan))
        best["whipple_jacchia_rms_path_rate_residual_mps"] = float(best_wj.get("rms_path_rate_residual_mps", np.nan))
        best["dynamic_mass_not_estimated_reason"] = "BIC favors constant velocity over Whipple-Jacchia"
        best["initial_dynamic_mass_kg"] = np.nan
        best["initial_dynamic_radius_m"] = np.nan
        best["initial_dynamic_mass_lo95_kg"] = np.nan
        best["initial_dynamic_mass_hi95_kg"] = np.nan
        best["initial_dynamic_radius_lo95_m"] = np.nan
        best["initial_dynamic_radius_hi95_m"] = np.nan
        best["initial_dynamic_mass_bootstrap_samples_successful"] = 0
        best["initial_mass_kg"] = np.nan
        best["initial_radius_m"] = np.nan

    fit_station_bias = bool(source_joint.get("fit_station_bias", True))
    fft_model = str(source_joint.get("fft_model", "range_offset_corrected_beat"))
    reference_chirp_rate_scale = float(source_joint.get("reference_chirp_rate_scale", 1.0))
    residual_likelihood = str(source_joint.get("residual_likelihood", "student_t"))
    student_t_nu_delay = float(source_joint.get("student_t_nu_delay", fit.DEFAULT_STUDENT_T_NU_DELAY))
    student_t_nu_fft = float(source_joint.get("student_t_nu_fft", fit.DEFAULT_STUDENT_T_NU_FFT))
    event_seed_offset = int(hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:8], 16)
    sample_seed = None if bootstrap_seed is None else int(bootstrap_seed) + event_seed_offset
    if selected_model == "whipple_speed":
        best = add_whipple_bootstrap(
            best,
            measured,
            times_ns,
            rho_of_alt_m,
            sigma_m,
            fft_offset_hz,
            fft_keep,
            sigma_fft_hz,
            path_keep,
            fit_station_bias,
            fft_model,
            reference_chirp_rate_scale,
            residual_likelihood,
            student_t_nu_delay,
            student_t_nu_fft,
            bootstrap_samples,
            sample_seed,
        )
        best = add_dynamic_mass_summary(best, times_ns, rho_of_alt_m)
    else:
        best["bootstrap_enabled"] = False
        best["bootstrap_samples_requested"] = 0
        best["bootstrap_samples_successful"] = 0
    state0 = np.concatenate(
        [
            np.asarray(best["x_gcrs_m"], dtype=np.float64)[0],
            np.asarray(best["v_gcrs_mps"], dtype=np.float64)[0],
        ]
    )
    nominal_eccentricity = fit.heliocentric_eccentricity_from_gcrs_state(state0, int(times_ns[0]))
    best["heliocentric_eccentricity"] = float(nominal_eccentricity[0])
    best["orbit_state_epoch"] = "initial_detection"
    best["orbit_sampling_model"] = (
        "Instantaneous osculating heliocentric eccentricity from the Whipple-Jacchia "
        "GCRS position and velocity at the first detected pulse; no extrapolation to "
        "pre-atmospheric velocity or dark flight before radar detection."
    )

    best["surrogate_model_family"] = "Whipple-Jacchia fixed-direction exponential deceleration"
    best["surrogate_model_equation"] = "v(t) = (v0 - a exp(b t)) u0; x(t) is the analytic time integral"
    if selected_model == "constant_velocity":
        best["surrogate_model_family"] = "constant velocity"
        best["surrogate_model_equation"] = "x(t) = x0 + v0 t"
    best["source_event_h5"] = str(source_h5)
    best["source_catalog_dir"] = str(source_h5.parent)
    best["n_whipple_jacchia_starts"] = int(len(whipple_seed_grid(source_joint, max_starts=max_starts)))
    best["n_whipple_jacchia_failures"] = int(len(failures))
    best["whipple_jacchia_failures"] = ";".join(failures[:20])

    delay_fit = {
        "params": np.asarray(source_joint.get("params", np.full(7, np.nan)), dtype=np.float64),
        "rms_total_path_residual_m": float(source_joint.get("rms_total_path_residual_m", np.nan)),
        "weighted_rms": float(source_joint.get("weighted_rms", np.nan)),
        "initial_radius_m": float(source_joint.get("initial_radius_m", np.nan)),
        "initial_mass_kg": float(source_joint.get("initial_mass_kg", np.nan)),
    }
    os.makedirs(output_dir, exist_ok=True)
    fit.write_h5(
        str(output_base),
        event_id,
        delay_fit,
        best,
        source_fft,
        best["fft_sigma_hz"],
        512,
        "cached_joint_delay_doppler_h5",
        32,
        0.0,
        fit.DEFAULT_FFT_TIME_PAD_US,
        reference_chirp_rate_scale,
    )
    best_for_plot = dict(best)
    n_params = len(np.asarray(best["params"], dtype=np.float64))
    best_for_plot["parameter_covariance"] = np.full((n_params, n_params), np.nan)
    best_for_plot["parameter_std"] = np.full(n_params, np.nan)
    best_for_plot["covariance_available"] = False
    fit.plot_joint_fit(event_id, delay_fit, best_for_plot, str(output_base), rho_of_alt_m, snr_db=source_fft.get("fft_snr_db"))

    params = np.asarray(best["params"], dtype=np.float64)
    speed = np.asarray(best["speed_km_s"], dtype=np.float64)
    is_wj_selected = str(best.get("bic_selected_model", best.get("dynamical_model", ""))) == "whipple_speed"
    return {
        "event_id": event_id,
        "status": "ok",
        "output_base": str(output_base),
        "loaded_existing": False,
        "weighted_rms": float(best["weighted_rms"]),
        "path_rms_m": float(best["rms_total_path_residual_m"]),
        "path_rate_rms_mps": float(best["rms_path_rate_residual_mps"]),
        "fft_rms_hz": float(best["rms_fft_residual_hz"]),
        "mean_abs_path_m": float(best["mean_abs_total_path_residual_m"]),
        "mean_abs_path_rate_mps": float(best["mean_abs_path_rate_residual_mps"]),
        "mean_abs_fft_hz": float(best["mean_abs_fft_residual_hz"]),
        "n_points": float(best["n_points"]),
        "n_path_observations": float(best["n_path_observations"]),
        "n_fft_observations": float(best["n_fft_observations"]),
        "start_speed_km_s": float(speed[0]) if speed.size else np.nan,
        "end_speed_km_s": float(speed[-1]) if speed.size else np.nan,
        "a_mps": float(10.0 ** params[6]) if is_wj_selected and params.size >= 8 else np.nan,
        "b_s_inv": float(10.0 ** params[7]) if is_wj_selected and params.size >= 8 else np.nan,
        "heliocentric_eccentricity": float(best.get("heliocentric_eccentricity", np.nan)),
        "bic_whipple_jacchia": float(best.get("bic_whipple_jacchia", np.nan)),
        "bic_constant_velocity": float(best.get("bic_constant_velocity", np.nan)),
        "delta_bic_constant_minus_whipple_jacchia": float(best.get("delta_bic_constant_minus_whipple_jacchia", np.nan)),
        "selected_model_is_whipple_jacchia": float(1.0 if is_wj_selected else 0.0),
        "initial_dynamic_mass_kg": float(best.get("initial_dynamic_mass_kg", np.nan)),
        "initial_dynamic_mass_lo95_kg": float(best.get("initial_dynamic_mass_lo95_kg", np.nan)),
        "initial_dynamic_mass_hi95_kg": float(best.get("initial_dynamic_mass_hi95_kg", np.nan)),
        "initial_dynamic_mass_bootstrap_samples_successful": float(
            best.get("initial_dynamic_mass_bootstrap_samples_successful", np.nan)
        ),
        "n_starts": float(best["n_whipple_jacchia_starts"]),
        "n_failures": float(best["n_whipple_jacchia_failures"]),
        "bootstrap_samples_requested": float(best.get("bootstrap_samples_requested", np.nan)),
        "bootstrap_samples_successful": float(best.get("bootstrap_samples_successful", np.nan)),
        "bootstrap_n_failures": float(best.get("bootstrap_n_failures", np.nan)),
        "bootstrap_a_mps_lo95": float(best.get("bootstrap_a_mps_lo95", np.nan)),
        "bootstrap_a_mps_hi95": float(best.get("bootstrap_a_mps_hi95", np.nan)),
        "bootstrap_b_s_inv_lo95": float(best.get("bootstrap_b_s_inv_lo95", np.nan)),
        "bootstrap_b_s_inv_hi95": float(best.get("bootstrap_b_s_inv_hi95", np.nan)),
        "bootstrap_log10_a_log10_b_corr": float(best.get("bootstrap_log10_a_log10_b_corr", np.nan)),
        "bootstrap_eccentricity_lo95": float(best.get("bootstrap_eccentricity_lo95", np.nan)),
        "bootstrap_eccentricity_hi95": float(best.get("bootstrap_eccentricity_hi95", np.nan)),
        "bootstrap_interstellar_fraction_e_gt_1": float(best.get("bootstrap_interstellar_fraction_e_gt_1", np.nan)),
        "bootstrap_start_speed_km_s_lo95": float(best.get("bootstrap_start_speed_km_s_lo95", np.nan)),
        "bootstrap_start_speed_km_s_hi95": float(best.get("bootstrap_start_speed_km_s_hi95", np.nan)),
        "bootstrap_end_speed_km_s_lo95": float(best.get("bootstrap_end_speed_km_s_lo95", np.nan)),
        "bootstrap_end_speed_km_s_hi95": float(best.get("bootstrap_end_speed_km_s_hi95", np.nan)),
    }


def write_summary(path, rows, args):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    numeric_keys = (
        "weighted_rms",
        "path_rms_m",
        "path_rate_rms_mps",
        "fft_rms_hz",
        "mean_abs_path_m",
        "mean_abs_path_rate_mps",
        "mean_abs_fft_hz",
        "n_points",
        "n_path_observations",
        "n_fft_observations",
        "start_speed_km_s",
        "end_speed_km_s",
        "a_mps",
        "b_s_inv",
        "heliocentric_eccentricity",
        "bic_whipple_jacchia",
        "bic_constant_velocity",
        "delta_bic_constant_minus_whipple_jacchia",
        "selected_model_is_whipple_jacchia",
        "initial_dynamic_mass_kg",
        "initial_dynamic_mass_lo95_kg",
        "initial_dynamic_mass_hi95_kg",
        "initial_dynamic_mass_bootstrap_samples_successful",
        "n_starts",
        "n_failures",
        "bootstrap_samples_requested",
        "bootstrap_samples_successful",
        "bootstrap_n_failures",
        "bootstrap_a_mps_lo95",
        "bootstrap_a_mps_hi95",
        "bootstrap_b_s_inv_lo95",
        "bootstrap_b_s_inv_hi95",
        "bootstrap_log10_a_log10_b_corr",
        "bootstrap_eccentricity_lo95",
        "bootstrap_eccentricity_hi95",
        "bootstrap_interstellar_fraction_e_gt_1",
        "bootstrap_start_speed_km_s_lo95",
        "bootstrap_start_speed_km_s_hi95",
        "bootstrap_end_speed_km_s_lo95",
        "bootstrap_end_speed_km_s_hi95",
    )
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["input_dir"] = str(args.input_dir)
        h.attrs["output_dir"] = str(args.output_dir)
        h.attrs["model"] = "Whipple-Jacchia surrogate"
        h.attrs["model_equation"] = "v(t) = (v0 - a exp(b t)) u0"
        h.attrs["max_starts"] = int(args.max_starts)
        h.attrs["bootstrap_samples"] = int(args.bootstrap_samples)
        h.create_dataset("event_id", data=np.asarray([row.get("event_id", "") for row in rows], dtype=object), dtype=string_dtype)
        h.create_dataset("status", data=np.asarray([row.get("status", "") for row in rows], dtype=object), dtype=string_dtype)
        h.create_dataset("output_base", data=np.asarray([row.get("output_base", "") for row in rows], dtype=object), dtype=string_dtype)
        h.create_dataset("error", data=np.asarray([row.get("error", "") for row in rows], dtype=object), dtype=string_dtype)
        h.create_dataset("loaded_existing", data=np.asarray([bool(row.get("loaded_existing", False)) for row in rows], dtype=bool))
        for key in numeric_keys:
            h.create_dataset(key, data=np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=np.float64))


def main():
    parser = argparse.ArgumentParser(description="Fit cached tri-static events with a Whipple-Jacchia surrogate trajectory.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-h5", type=Path, default=None)
    parser.add_argument("--event-id", action="append", default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--max-starts", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260701)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_paths = sorted(Path(args.input_dir).glob("joint_delay_doppler_fft_tri_*.h5"))
    if args.event_id:
        wanted = set(args.event_id)
        input_paths = [path for path in input_paths if event_id_from_path(path) in wanted]
    if args.max_events is not None:
        input_paths = input_paths[: int(args.max_events)]
    summary_h5 = args.summary_h5 or Path(args.output_dir) / "whipple_jacchia_catalog_summary.h5"
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    if args.jobs <= 1:
        for idx, path in enumerate(input_paths, start=1):
            row = fit_one(path, args.output_dir, args.overwrite, args.max_starts, args.bootstrap_samples, args.bootstrap_seed)
            rows.append(row)
            print(
                f"[{idx}/{len(input_paths)}] {row.get('status')} {row.get('event_id')} "
                f"wrms={row.get('weighted_rms', np.nan):.3f} path={row.get('path_rms_m', np.nan):.2f} "
                f"rate={row.get('path_rate_rms_mps', np.nan):.1f}",
                flush=True,
            )
            write_summary(summary_h5, rows, args)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            futures = {
                executor.submit(
                    fit_one,
                    path,
                    args.output_dir,
                    args.overwrite,
                    args.max_starts,
                    args.bootstrap_samples,
                    args.bootstrap_seed,
                ): path
                for path in input_paths
            }
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                print(
                    f"[{idx}/{len(input_paths)}] {row.get('status')} {row.get('event_id')} "
                    f"wrms={row.get('weighted_rms', np.nan):.3f} path={row.get('path_rms_m', np.nan):.2f} "
                    f"rate={row.get('path_rate_rms_mps', np.nan):.1f}",
                    flush=True,
                )
                write_summary(summary_h5, rows, args)
    rows.sort(key=lambda row: row.get("event_id", ""))
    write_summary(summary_h5, rows, args)
    print(f"summary_h5={summary_h5}")
    print(f"n_ok={sum(1 for row in rows if row.get('status') == 'ok')}")
    print(f"n_error={sum(1 for row in rows if row.get('status') == 'error')}")


if __name__ == "__main__":
    main()
