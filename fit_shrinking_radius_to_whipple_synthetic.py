import argparse
import concurrent.futures
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import h5py
import numpy as np
import scipy.optimize as so

import fit_all_ballistic_snr_weighted as base
import fit_all_ceplecha_snr_weighted as cepl
import fit_event_joint_delay_doppler_fft as fit
import fit_gcrs_trajectories_lfm_ambiguity as gfit
from fit_whipple_jacchia_catalog_from_h5 import event_id_from_path, load_group


DEFAULT_SOURCE_DIR = Path("results/tristatic_student_t_bootstrap_orbit100_20260630")
DEFAULT_WHIPPLE_DIR = Path("results/tristatic_whipple_jacchia_bootstrap_orbit100_20260701")
DEFAULT_OUTPUT_DIR = Path("results/tristatic_shrinking_radius_to_whipple_synthetic_20260701")
RADIUS_GRID_UM = np.asarray([1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0], dtype=np.float64)
STATE_VECTOR_SIZE = 6


def mass_from_radius(radius_m):
    return (4.0 * np.pi / 3.0) * cepl.METEOROID_DENSITY_KG_M3 * np.asarray(radius_m, dtype=np.float64) ** 3.0


def forward(params, t_rel_s, times_ns, rho_of_alt_m):
    x_gcrs, v_gcrs, radius_m, mass_kg, success, message = cepl.propagate_ceplecha(params, t_rel_s, rho_of_alt_m)
    x_itrs, v_itrs = base.gcrs_state_samples_to_itrs(x_gcrs, v_gcrs, times_ns)
    path_length_m, path_rate_mps = gfit.link_total_paths_and_rates_m(
        x_itrs,
        v_itrs,
        gfit.LINK_TX_POSITIONS_M,
        gfit.LINK_RX_POSITIONS_M,
    )
    doppler_hz = gfit.doppler_from_path_length_rate_hz(path_rate_mps)
    apparent_path_length_m = path_length_m + gfit.lfm_total_path_bias_m(path_rate_mps)
    return {
        "x_gcrs_m": x_gcrs,
        "v_gcrs_mps": v_gcrs,
        "x_itrs_m": x_itrs,
        "v_itrs_mps": v_itrs,
        "radius_m": radius_m,
        "mass_kg": mass_kg,
        "path_length_m": path_length_m,
        "path_rate_mps": path_rate_mps,
        "doppler_hz": doppler_hz,
        "apparent_path_length_m": apparent_path_length_m,
        "ceplecha_success": success,
        "ceplecha_message": message,
    }


def bootstrap_state_covariance(
    whipple_joint,
    position_sigma_floor_m=1.0,
    velocity_sigma_floor_mps=5.0,
    covariance_shrinkage=0.05,
):
    """Return per-time whitening matrices from WJ bootstrap state covariance.

    The state vector is [x, y, z, vx, vy, vz] in GCRS coordinates.  Each
    whitening matrix W_k satisfies ||W_k delta_z_k||^2 ~= delta_z_k^T C_k^-1
    delta_z_k for the regularized empirical bootstrap covariance C_k.
    """
    x_boot = np.asarray(whipple_joint.get("bootstrap_x_gcrs_m", []), dtype=np.float64)
    v_boot = np.asarray(whipple_joint.get("bootstrap_v_gcrs_mps", []), dtype=np.float64)
    x_nom = np.asarray(whipple_joint["x_gcrs_m"], dtype=np.float64)
    n_times = x_nom.shape[0]
    floor_diag = np.asarray(
        [position_sigma_floor_m**2] * 3 + [velocity_sigma_floor_mps**2] * 3,
        dtype=np.float64,
    )
    fallback_inv_sigma = np.asarray(
        [1.0 / position_sigma_floor_m] * 3 + [1.0 / velocity_sigma_floor_mps] * 3,
        dtype=np.float64,
    )
    fallback = np.repeat(np.diag(fallback_inv_sigma)[None, :, :], n_times, axis=0)
    meta = {
        "state_weighting": "fixed_floor",
        "bootstrap_state_covariance_available": False,
        "bootstrap_state_covariance_samples": 0,
        "bootstrap_state_covariance_min_samples": 0,
        "bootstrap_state_covariance_median_condition": np.nan,
        "bootstrap_state_covariance_max_condition": np.nan,
        "state_covariance_position_sigma_floor_m": float(position_sigma_floor_m),
        "state_covariance_velocity_sigma_floor_mps": float(velocity_sigma_floor_mps),
        "state_covariance_shrinkage": float(covariance_shrinkage),
    }
    if x_boot.ndim != 3 or v_boot.ndim != 3 or x_boot.shape != v_boot.shape or x_boot.shape[1:] != x_nom.shape:
        return fallback, meta

    state = np.concatenate([x_boot, v_boot], axis=2)
    whitening = np.empty((n_times, STATE_VECTOR_SIZE, STATE_VECTOR_SIZE), dtype=np.float64)
    conditions = []
    sample_counts = []
    shrink = float(np.clip(covariance_shrinkage, 0.0, 1.0))
    for k in range(n_times):
        samples = state[:, k, :]
        finite = np.all(np.isfinite(samples), axis=1)
        samples = samples[finite]
        sample_counts.append(int(samples.shape[0]))
        if samples.shape[0] < STATE_VECTOR_SIZE + 2:
            whitening[k] = fallback[k]
            conditions.append(np.nan)
            continue
        cov = np.cov(samples, rowvar=False)
        cov = np.asarray(cov, dtype=np.float64)
        diag_cov = np.diag(np.clip(np.diag(cov), 0.0, np.inf))
        cov = (1.0 - shrink) * cov + shrink * diag_cov
        cov = cov + np.diag(floor_diag)
        try:
            chol = np.linalg.cholesky(cov)
            whitening[k] = np.linalg.solve(chol, np.eye(STATE_VECTOR_SIZE))
            conditions.append(float(np.linalg.cond(cov)))
        except np.linalg.LinAlgError:
            whitening[k] = fallback[k]
            conditions.append(np.nan)

    valid_conditions = np.asarray([c for c in conditions if np.isfinite(c)], dtype=np.float64)
    meta.update(
        {
            "state_weighting": "whipple_bootstrap_state_covariance",
            "bootstrap_state_covariance_available": True,
            "bootstrap_state_covariance_samples": int(x_boot.shape[0]),
            "bootstrap_state_covariance_min_samples": int(np.min(sample_counts)) if sample_counts else 0,
            "bootstrap_state_covariance_median_condition": float(np.nanmedian(valid_conditions))
            if valid_conditions.size
            else np.nan,
            "bootstrap_state_covariance_max_condition": float(np.nanmax(valid_conditions))
            if valid_conditions.size
            else np.nan,
        }
    )
    return whitening, meta


def fixed_state_whitening(n_times, position_sigma_m, velocity_sigma_mps):
    inv_sigma = np.asarray([1.0 / position_sigma_m] * 3 + [1.0 / velocity_sigma_mps] * 3, dtype=np.float64)
    whitening = np.repeat(np.diag(inv_sigma)[None, :, :], int(n_times), axis=0)
    meta = {
        "state_weighting": "fixed_diagonal",
        "bootstrap_state_covariance_available": False,
        "bootstrap_state_covariance_samples": 0,
        "bootstrap_state_covariance_min_samples": 0,
        "bootstrap_state_covariance_median_condition": np.nan,
        "bootstrap_state_covariance_max_condition": np.nan,
        "state_covariance_position_sigma_floor_m": float(position_sigma_m),
        "state_covariance_velocity_sigma_floor_mps": float(velocity_sigma_mps),
        "state_covariance_shrinkage": np.nan,
    }
    return whitening, meta


def fit_to_target(
    whipple_joint,
    x_target,
    v_target,
    x_itrs_for_density,
    position_sigma_m,
    velocity_sigma_mps,
    max_nfev,
    state_whitening=None,
    state_weighting_meta=None,
):
    t_rel_s = np.asarray(whipple_joint["t_rel_s"], dtype=np.float64)
    times_ns = np.asarray(whipple_joint["time_ns"], dtype=np.int64)
    x_target = np.asarray(x_target, dtype=np.float64)
    v_target = np.asarray(v_target, dtype=np.float64)
    rho_of_alt_m, _meta = base.density_interpolator(times_ns, np.asarray(x_itrs_for_density, dtype=np.float64))
    x0 = x_target[0]
    v0 = v_target[0]
    lower = np.concatenate([x0 - 2.0e4, np.full(3, -1.2e5), [np.log10(cepl.MIN_RADIUS_M)]])
    upper = np.concatenate([x0 + 2.0e4, np.full(3, 1.2e5), [np.log10(cepl.MAX_RADIUS_M)]])
    scale = np.asarray([1.0e3, 1.0e3, 1.0e3, 1.0e4, 1.0e4, 1.0e4, 1.0], dtype=np.float64)
    if state_whitening is None:
        state_whitening, state_weighting_meta = fixed_state_whitening(len(t_rel_s), position_sigma_m, velocity_sigma_mps)
    state_whitening = np.asarray(state_whitening, dtype=np.float64)

    def residual(params):
        try:
            model = forward(params, t_rel_s, times_ns, rho_of_alt_m)
        except Exception:
            return np.full(6 * len(t_rel_s), 1e6, dtype=np.float64)
        if not model["ceplecha_success"]:
            return np.full(6 * len(t_rel_s), 1e6, dtype=np.float64)
        delta = np.concatenate(
            [
                np.asarray(model["x_gcrs_m"], dtype=np.float64) - x_target,
                np.asarray(model["v_gcrs_mps"], dtype=np.float64) - v_target,
            ],
            axis=1,
        )
        return np.einsum("kij,kj->ki", state_whitening, delta).ravel()

    best = None
    for radius_um in RADIUS_GRID_UM:
        p0 = np.concatenate([x0, v0, [np.log10(radius_um * 1e-6)]])
        result = so.least_squares(
            residual,
            np.clip(p0, lower, upper),
            bounds=(lower, upper),
            x_scale=scale,
            max_nfev=int(max_nfev),
        )
        value = float(np.sum(residual(result.x) ** 2.0))
        if best is None or value < best["objective"]:
            best = {"result": result, "objective": value}
    result = best["result"]
    model = forward(result.x, t_rel_s, times_ns, rho_of_alt_m)
    dx = np.asarray(model["x_gcrs_m"], dtype=np.float64) - x_target
    dv = np.asarray(model["v_gcrs_mps"], dtype=np.float64) - v_target
    x_target_itrs, v_target_itrs = base.gcrs_state_samples_to_itrs(x_target, v_target, times_ns)
    target_path_length, target_path_rate = gfit.link_total_paths_and_rates_m(
        x_target_itrs,
        v_target_itrs,
        gfit.LINK_TX_POSITIONS_M,
        gfit.LINK_RX_POSITIONS_M,
    )
    whipple_path = target_path_length + gfit.lfm_total_path_bias_m(target_path_rate)
    whipple_rate = target_path_rate
    path_resid = np.asarray(model["apparent_path_length_m"], dtype=np.float64) - whipple_path
    rate_resid = np.asarray(model["path_rate_mps"], dtype=np.float64) - whipple_rate
    out = {
        "params": np.asarray(result.x, dtype=np.float64),
        "objective": float(best["objective"]),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "initial_radius_m": float(model["radius_m"][0]),
        "initial_mass_kg": float(model["mass_kg"][0]),
        "synthetic_position_rms_m": float(np.sqrt(np.nanmean(np.sum(dx**2.0, axis=1)))),
        "synthetic_velocity_rms_mps": float(np.sqrt(np.nanmean(np.sum(dv**2.0, axis=1)))),
        "synthetic_path_rms_m": float(np.sqrt(np.nanmean(path_resid**2.0))),
        "synthetic_path_mean_abs_m": float(np.nanmean(np.abs(path_resid))),
        "synthetic_path_rate_rms_mps": float(np.sqrt(np.nanmean(rate_resid**2.0))),
        "synthetic_path_rate_mean_abs_mps": float(np.nanmean(np.abs(rate_resid))),
        "path_residuals_m": path_resid,
        "path_rate_residuals_mps": rate_resid,
        "dynamical_model": "ceplecha_to_whipple_synthetic",
        **(state_weighting_meta or {}),
        **model,
    }
    return out


def fit_to_whipple(whipple_joint, position_sigma_m, velocity_sigma_mps, max_nfev, state_whitening=None, state_weighting_meta=None):
    return fit_to_target(
        whipple_joint,
        np.asarray(whipple_joint["x_gcrs_m"], dtype=np.float64),
        np.asarray(whipple_joint["v_gcrs_mps"], dtype=np.float64),
        np.asarray(whipple_joint["x_itrs_m"], dtype=np.float64),
        position_sigma_m,
        velocity_sigma_mps,
        max_nfev,
        state_whitening=state_whitening,
        state_weighting_meta=state_weighting_meta,
    )


def whipple_bootstrap_models(whipple_joint, n_samples):
    params0 = np.asarray(whipple_joint["params"], dtype=np.float64)
    samples = np.asarray(whipple_joint.get("bootstrap_params", []), dtype=np.float64)
    if n_samples <= 0 or samples.ndim != 2 or samples.shape[0] == 0:
        return []
    if samples.shape[0] > n_samples:
        idx = np.linspace(0, samples.shape[0] - 1, int(n_samples), dtype=np.int64)
        samples = samples[idx]
    t_rel_s = np.asarray(whipple_joint["t_rel_s"], dtype=np.float64)
    times_ns = np.asarray(whipple_joint["time_ns"], dtype=np.int64)
    rho_of_alt_m, _meta = base.density_interpolator(times_ns, np.asarray(whipple_joint["x_itrs_m"], dtype=np.float64))
    out = []
    for sample in samples:
        try:
            model = fit.forward_model_for_kind(
                np.asarray(sample[: len(params0)], dtype=np.float64),
                t_rel_s,
                times_ns,
                rho_of_alt_m,
                "whipple_speed",
            )
        except Exception:
            continue
        if bool(model.get("ceplecha_success", False)):
            out.append(model)
    return out


def add_bootstrap_uncertainty(
    fit_row,
    whipple_joint,
    n_samples,
    position_sigma_m,
    velocity_sigma_mps,
    max_nfev,
    state_whitening=None,
    state_weighting_meta=None,
):
    models = whipple_bootstrap_models(whipple_joint, int(n_samples))
    radius = []
    mass = []
    velocity_rms = []
    path_rms = []
    rate_rms = []
    for model in models:
        try:
            row = fit_to_target(
                whipple_joint,
                np.asarray(model["x_gcrs_m"], dtype=np.float64),
                np.asarray(model["v_gcrs_mps"], dtype=np.float64),
                np.asarray(model["x_itrs_m"], dtype=np.float64),
                position_sigma_m,
                velocity_sigma_mps,
                max_nfev,
                state_whitening=state_whitening,
                state_weighting_meta=state_weighting_meta,
            )
        except Exception:
            continue
        if bool(row.get("optimizer_success", False)) and np.isfinite(row.get("initial_radius_m", np.nan)):
            radius.append(float(row["initial_radius_m"]))
            mass.append(float(row["initial_mass_kg"]))
            velocity_rms.append(float(row["synthetic_velocity_rms_mps"]))
            path_rms.append(float(row["synthetic_path_rms_m"]))
            rate_rms.append(float(row["synthetic_path_rate_rms_mps"]))
    fit_row["bootstrap_samples_requested"] = int(n_samples)
    fit_row["bootstrap_samples_successful"] = int(len(radius))
    if radius:
        radius = np.asarray(radius, dtype=np.float64)
        mass = np.asarray(mass, dtype=np.float64)
        fit_row["bootstrap_initial_radius_samples_m"] = radius
        fit_row["bootstrap_initial_mass_samples_kg"] = mass
        fit_row["bootstrap_synthetic_velocity_rms_mps"] = np.asarray(velocity_rms, dtype=np.float64)
        fit_row["bootstrap_synthetic_path_rms_m"] = np.asarray(path_rms, dtype=np.float64)
        fit_row["bootstrap_synthetic_path_rate_rms_mps"] = np.asarray(rate_rms, dtype=np.float64)
        fit_row["bootstrap_initial_radius_median_m"] = float(np.nanmedian(radius))
        fit_row["bootstrap_initial_radius_lo95_m"] = float(np.nanpercentile(radius, 2.5))
        fit_row["bootstrap_initial_radius_hi95_m"] = float(np.nanpercentile(radius, 97.5))
        fit_row["bootstrap_initial_mass_median_kg"] = float(np.nanmedian(mass))
        fit_row["bootstrap_initial_mass_lo95_kg"] = float(np.nanpercentile(mass, 2.5))
        fit_row["bootstrap_initial_mass_hi95_kg"] = float(np.nanpercentile(mass, 97.5))
    return fit_row


def write_event(path, event_id, fit_row, source_h5, whipple_h5, position_sigma_m, velocity_sigma_mps):
    with h5py.File(path, "w") as h:
        h.attrs["event_id"] = event_id
        h.attrs["source_h5"] = str(source_h5)
        h.attrs["whipple_h5"] = str(whipple_h5)
        h.attrs["fit_target"] = "Whipple-Jacchia synthetic GCRS position and velocity samples"
        h.attrs["position_sigma_m"] = float(position_sigma_m)
        h.attrs["velocity_sigma_mps"] = float(velocity_sigma_mps)
        for key, value in fit_row.items():
            if isinstance(value, str):
                h.attrs[key] = value
            elif np.isscalar(value):
                h.attrs[key] = value
            else:
                h[key] = value


def fit_one(
    whipple_h5,
    source_dir,
    output_dir,
    position_sigma_m,
    velocity_sigma_mps,
    max_nfev,
    bootstrap_samples,
    bootstrap_max_nfev,
    overwrite,
    use_bootstrap_covariance,
    covariance_shrinkage,
):
    whipple_h5 = Path(whipple_h5)
    event_id = event_id_from_path(whipple_h5)
    source_h5 = Path(source_dir) / whipple_h5.name
    output_path = Path(output_dir) / f"shrinking_radius_to_whipple_{event_id}.h5"
    if output_path.exists() and not overwrite:
        with h5py.File(output_path, "r") as h:
            return event_id, "exists", int(h.attrs.get("optimizer_success", False)), float(h.attrs.get("synthetic_velocity_rms_mps", np.nan)), float(h.attrs.get("initial_radius_m", np.nan))
    with h5py.File(whipple_h5, "r") as h:
        whipple_joint = load_group(h["joint_fit"])
    if use_bootstrap_covariance:
        state_whitening, state_weighting_meta = bootstrap_state_covariance(
            whipple_joint,
            position_sigma_floor_m=position_sigma_m,
            velocity_sigma_floor_mps=velocity_sigma_mps,
            covariance_shrinkage=covariance_shrinkage,
        )
    else:
        state_whitening, state_weighting_meta = fixed_state_whitening(
            len(np.asarray(whipple_joint["t_rel_s"], dtype=np.float64)),
            position_sigma_m,
            velocity_sigma_mps,
        )
    fit_row = fit_to_whipple(
        whipple_joint,
        position_sigma_m,
        velocity_sigma_mps,
        max_nfev,
        state_whitening=state_whitening,
        state_weighting_meta=state_weighting_meta,
    )
    fit_row = add_bootstrap_uncertainty(
        fit_row,
        whipple_joint,
        int(bootstrap_samples),
        position_sigma_m,
        velocity_sigma_mps,
        int(bootstrap_max_nfev),
        state_whitening=state_whitening,
        state_weighting_meta=state_weighting_meta,
    )
    os.makedirs(output_dir, exist_ok=True)
    write_event(output_path, event_id, fit_row, source_h5, whipple_h5, position_sigma_m, velocity_sigma_mps)
    return event_id, "ok", int(fit_row["optimizer_success"]), float(fit_row["synthetic_velocity_rms_mps"]), float(fit_row["initial_radius_m"])


def write_summary(output_dir, rows):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    path = Path(output_dir) / "shrinking_radius_to_whipple_summary.h5"
    with h5py.File(path, "w") as h:
        h.attrs["script"] = Path(__file__).name
        h.create_dataset("event_id", data=np.asarray([r[0] for r in rows], dtype=object), dtype=string_dtype)
        h.create_dataset("status", data=np.asarray([r[1] for r in rows], dtype=object), dtype=string_dtype)
        h["optimizer_success"] = np.asarray([r[2] for r in rows], dtype=bool)
        h["synthetic_velocity_rms_mps"] = np.asarray([r[3] for r in rows], dtype=np.float64)
        h["initial_radius_m"] = np.asarray([r[4] for r in rows], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Fit a single shrinking-radius model to Whipple-Jacchia synthetic trajectories.")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--whipple-dir", default=DEFAULT_WHIPPLE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--position-sigma-m", type=float, default=10.0)
    parser.add_argument("--velocity-sigma-mps", type=float, default=50.0)
    parser.add_argument("--no-bootstrap-covariance", action="store_true")
    parser.add_argument("--covariance-shrinkage", type=float, default=0.05)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-max-nfev", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    whipple_paths = sorted(Path(args.whipple_dir).glob("joint_delay_doppler_fft_tri_*.h5"))
    if args.event_id:
        wanted = set(args.event_id)
        whipple_paths = [p for p in whipple_paths if event_id_from_path(p) in wanted]
    if not whipple_paths:
        raise SystemExit("No Whipple-Jacchia event HDF5 files matched.")
    rows = []
    if args.jobs <= 1:
        for idx, path in enumerate(whipple_paths, 1):
            row = fit_one(
                path,
                args.source_dir,
                args.output_dir,
                args.position_sigma_m,
                args.velocity_sigma_mps,
                args.max_nfev,
                args.bootstrap_samples,
                args.bootstrap_max_nfev,
                args.overwrite,
                not args.no_bootstrap_covariance,
                args.covariance_shrinkage,
            )
            rows.append(row)
            print(f"[{idx}/{len(whipple_paths)}] {row[1]} {row[0]} success={row[2]} vel_rms={row[3]:.1f} r0_um={row[4] * 1e6:.2f}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(
                    fit_one,
                    path,
                    args.source_dir,
                    args.output_dir,
                    args.position_sigma_m,
                    args.velocity_sigma_mps,
                    args.max_nfev,
                    args.bootstrap_samples,
                    args.bootstrap_max_nfev,
                    args.overwrite,
                    not args.no_bootstrap_covariance,
                    args.covariance_shrinkage,
                )
                for path in whipple_paths
            ]
            for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                row = fut.result()
                rows.append(row)
                print(f"[{idx}/{len(whipple_paths)}] {row[1]} {row[0]} success={row[2]} vel_rms={row[3]:.1f} r0_um={row[4] * 1e6:.2f}", flush=True)
    write_summary(args.output_dir, rows)


if __name__ == "__main__":
    main()
