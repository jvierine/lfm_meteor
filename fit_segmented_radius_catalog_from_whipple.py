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
from fit_whipple_jacchia_catalog_from_h5 import load_group


DEFAULT_SOURCE_DIR = Path("results/tristatic_student_t_bootstrap_orbit100_20260630")
DEFAULT_WHIPPLE_DIR = Path("results/tristatic_whipple_jacchia_bootstrap_orbit100_20260701")
DEFAULT_OUTPUT_DIR = Path("results/tristatic_segmented_radius_from_whipple_20260701")
RADIUS_GRID_UM = np.asarray([5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0], dtype=np.float64)


def event_id_from_path(path):
    stem = Path(path).stem
    prefix = "joint_delay_doppler_fft_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def mass_from_radius(radius_m):
    return (4.0 * np.pi / 3.0) * cepl.METEOROID_DENSITY_KG_M3 * np.asarray(radius_m, dtype=np.float64) ** 3.0


def segment_start_indices(t_rel_s, n_segments):
    n = len(t_rel_s)
    if n_segments <= 1:
        return np.asarray([0], dtype=np.int64)
    # Split by observed along-track sample count.  This keeps every segment tied
    # to actual pulse boundaries and avoids ambiguous state resets between pulses.
    starts = np.unique(np.floor(np.linspace(0, n, n_segments + 1)[:-1]).astype(np.int64))
    return starts[starts < n]


def segment_index_for_samples(n_samples, starts):
    idx = np.zeros(n_samples, dtype=np.int64)
    for seg, start in enumerate(starts):
        stop = starts[seg + 1] if seg + 1 < len(starts) else n_samples
        idx[start:stop] = seg
    return idx


def rk4_step(state, dt_s, x0_gcrs_m, direction, rho_of_alt_m):
    def deriv(y):
        s_m, speed_mps, radius_m = y
        x = x0_gcrs_m + s_m * direction
        alt_m = float(np.linalg.norm(x) - cepl.SPHERICAL_EARTH_RADIUS_M)
        rho = float(rho_of_alt_m(alt_m))
        radius = float(np.clip(radius_m, cepl.MIN_RADIUS_M, cepl.MAX_RADIUS_M))
        speed_abs = abs(float(speed_mps))
        dv_dt = -(3.0 / 4.0) * rho * speed_abs * speed_mps / (cepl.METEOROID_DENSITY_KG_M3 * radius)
        dr_dt = -rho * cepl.ABLATION_SIGMA_KG_J * speed_abs**3.0 / (8.0 * cepl.METEOROID_DENSITY_KG_M3)
        return np.asarray([speed_mps, dv_dt, dr_dt], dtype=np.float64)

    y = np.asarray(state, dtype=np.float64)
    if dt_s <= 0.0:
        return y
    k1 = deriv(y)
    k2 = deriv(y + 0.5 * dt_s * k1)
    k3 = deriv(y + 0.5 * dt_s * k2)
    k4 = deriv(y + dt_s * k3)
    out = y + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    out[2] = float(np.clip(out[2], cepl.MIN_RADIUS_M, cepl.MAX_RADIUS_M))
    return out


def segmented_radius_model(params, t_rel_s, times_ns, rho_of_alt_m, x0_ref_gcrs_m, direction, n_segments):
    starts = segment_start_indices(t_rel_s, n_segments)
    segment_idx = segment_index_for_samples(len(t_rel_s), starts)
    along_offset_m = float(params[0])
    speed0_mps = float(params[1])
    log_radius = np.asarray(params[2 : 2 + len(starts)], dtype=np.float64)
    radius0_m = 10.0 ** log_radius

    state = np.asarray([along_offset_m, speed0_mps, radius0_m[0]], dtype=np.float64)
    x_gcrs = np.zeros((len(t_rel_s), 3), dtype=np.float64)
    v_gcrs = np.zeros((len(t_rel_s), 3), dtype=np.float64)
    radius_m = np.zeros(len(t_rel_s), dtype=np.float64)
    last_t = float(t_rel_s[0])
    for i, t_s in enumerate(np.asarray(t_rel_s, dtype=np.float64)):
        seg = int(segment_idx[i])
        if i in starts:
            state[2] = radius0_m[seg]
        if i > 0:
            dt = float(t_s - last_t)
            n_step = max(1, int(np.ceil(abs(dt) / max(cepl.CEPLECHA_SAMPLE_DT_S, 1e-4))))
            for _ in range(n_step):
                state = rk4_step(state, dt / n_step, x0_ref_gcrs_m, direction, rho_of_alt_m)
            if i in starts:
                state[2] = radius0_m[seg]
        x_gcrs[i] = x0_ref_gcrs_m + state[0] * direction
        v_gcrs[i] = state[1] * direction
        radius_m[i] = state[2]
        last_t = float(t_s)

    mass_kg = mass_from_radius(radius_m)
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
        "segment_index": segment_idx,
        "segment_start_indices": starts,
        "segment_initial_radius_m": radius0_m,
        "apparent_path_length_m": apparent_path_length_m,
        "path_length_m": path_length_m,
        "path_rate_mps": path_rate_mps,
        "doppler_hz": doppler_hz,
    }


def fit_segment_count(source_joint, source_fft, whipple_joint, n_segments):
    measured = np.asarray(source_joint["measured_total_paths_m"], dtype=np.float64)
    times_ns = np.asarray(source_joint["time_ns"], dtype=np.int64)
    t_rel_s = np.asarray(source_joint["t_rel_s"], dtype=np.float64)
    path_keep = np.asarray(source_joint["path_keep"], dtype=bool)
    fft_keep = np.asarray(source_joint["fft_keep"], dtype=bool)
    fft_obs = np.asarray(source_joint["observed_fft_beat_hz"], dtype=np.float64)
    path_sigma = np.asarray(whipple_joint.get("path_fit_sigma_m", source_joint.get("path_sigma_m")), dtype=np.float64)
    fft_sigma = np.asarray(whipple_joint.get("fft_event_sigma_hz", source_joint.get("fft_sigma_hz")), dtype=np.float64)
    if fft_sigma.ndim == 0:
        fft_sigma = np.full_like(fft_obs, float(fft_sigma), dtype=np.float64)
    x0_ref = np.asarray(whipple_joint["x_gcrs_m"], dtype=np.float64)[0]
    v_ref = np.asarray(whipple_joint["v_gcrs_mps"], dtype=np.float64)[0]
    speed_ref = float(np.linalg.norm(v_ref))
    direction = v_ref / max(speed_ref, 1e-30)
    rho_of_alt_m, _ = base.density_interpolator(times_ns, np.asarray(whipple_joint["x_itrs_m"], dtype=np.float64))
    starts = segment_start_indices(t_rel_s, n_segments)
    n_radius = len(starts)
    chirp_rate = gfit.NOMINAL_CHIRP_RATE_HZ_PER_S * float(whipple_joint.get("reference_chirp_rate_scale", 1.0))

    def residual(x):
        dyn = x[: 2 + n_radius]
        station_bias = x[2 + n_radius : 2 + n_radius + 3]
        model = segmented_radius_model(dyn, t_rel_s, times_ns, rho_of_alt_m, x0_ref, direction, n_radius)
        apparent = model["apparent_path_length_m"]
        geo = model["path_length_m"]
        doppler = model["doppler_hz"]
        beat_model = doppler - (chirp_rate / gfit.C) * (measured - geo) + station_bias[None, :]
        path_resid = ((apparent - measured) / path_sigma)[path_keep]
        beat_resid = ((beat_model - fft_obs) / fft_sigma)[fft_keep]
        path_resid = fit.student_t_least_squares_residual(path_resid, float(whipple_joint.get("student_t_nu_delay", 1.5)))
        beat_resid = fit.student_t_least_squares_residual(beat_resid, float(whipple_joint.get("student_t_nu_fft", 3.0)))
        return np.concatenate([path_resid, beat_resid])

    best = None
    for radius_um in RADIUS_GRID_UM:
        log_r = np.full(n_radius, np.log10(radius_um * 1e-6), dtype=np.float64)
        x0 = np.concatenate([[0.0, speed_ref], log_r, np.zeros(3, dtype=np.float64)])
        lower = np.concatenate([[-5000.0, 5e3], np.full(n_radius, np.log10(cepl.MIN_RADIUS_M)), np.full(3, -500e3)])
        upper = np.concatenate([[5000.0, 90e3], np.full(n_radius, np.log10(cepl.MAX_RADIUS_M)), np.full(3, 500e3)])
        scale = np.concatenate([[500.0, 1e4], np.ones(n_radius), np.full(3, 5e4)])
        result = so.least_squares(residual, np.clip(x0, lower, upper), bounds=(lower, upper), x_scale=scale, max_nfev=240)
        value = float(np.sum(residual(result.x) ** 2.0))
        if best is None or value < best["objective"]:
            best = {"result": result, "objective": value}

    result = best["result"]
    dyn = result.x[: 2 + n_radius]
    station_bias = result.x[2 + n_radius : 2 + n_radius + 3]
    model = segmented_radius_model(dyn, t_rel_s, times_ns, rho_of_alt_m, x0_ref, direction, n_radius)
    beat_model = model["doppler_hz"] - (chirp_rate / gfit.C) * (measured - model["path_length_m"]) + station_bias[None, :]
    path_resid = model["apparent_path_length_m"] - measured
    fft_resid = beat_model - fft_obs
    fft_doppler_hz = fft_obs + (chirp_rate / gfit.C) * (measured - model["path_length_m"])
    fft_path_rate_mps = -gfit.RADAR_WAVELENGTH_M * fft_doppler_hz
    path_rate_resid_mps = model["path_rate_mps"] - fft_path_rate_mps
    n_scalar = int(np.count_nonzero(path_keep) + np.count_nonzero(fft_keep))
    k_params = int(result.x.size)
    bic = float(best["objective"] + k_params * np.log(max(n_scalar, 1)))
    return {
        "n_segments": int(n_radius),
        "params": result.x,
        "dynamic_params": dyn,
        "station_fft_bias_hz": station_bias,
        "objective": float(best["objective"]),
        "bic": bic,
        "n_scalar_measurements": n_scalar,
        "n_parameters": k_params,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "path_residuals_m": path_resid,
        "fft_residuals_hz": fft_resid,
        "path_rate_residuals_mps": path_rate_resid_mps,
        "rms_total_path_residual_m": float(np.sqrt(np.nanmean(path_resid[path_keep] ** 2.0))),
        "rms_fft_residual_hz": float(np.sqrt(np.nanmean(fft_resid[fft_keep] ** 2.0))),
        "rms_path_rate_residual_mps": float(np.sqrt(np.nanmean(path_rate_resid_mps[fft_keep] ** 2.0))),
        **model,
    }


def write_event(output_path, event_id, fits, best_idx, source_h5, whipple_h5):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as h:
        h.attrs["event_id"] = event_id
        h.attrs["source_h5"] = str(source_h5)
        h.attrs["whipple_h5"] = str(whipple_h5)
        h.attrs["selection_criterion"] = "BIC = student-t transformed weighted residual sum of squares + k ln N"
        h.attrs["best_n_segments"] = int(fits[best_idx]["n_segments"])
        h.attrs["best_bic"] = float(fits[best_idx]["bic"])
        for fit_idx, row in enumerate(fits):
            g = h.create_group(f"segments_{row['n_segments']}")
            for key, value in row.items():
                if isinstance(value, str):
                    g.attrs[key] = value
                elif np.isscalar(value):
                    g.attrs[key] = value
                else:
                    g[key] = value
        h["candidate_n_segments"] = np.asarray([r["n_segments"] for r in fits], dtype=np.int64)
        h["candidate_bic"] = np.asarray([r["bic"] for r in fits], dtype=np.float64)
        h["candidate_objective"] = np.asarray([r["objective"] for r in fits], dtype=np.float64)
        h["candidate_optimizer_success"] = np.asarray([r["optimizer_success"] for r in fits], dtype=bool)
        h.create_dataset("candidate_optimizer_message", data=np.asarray([r["optimizer_message"] for r in fits], dtype=object), dtype=string_dtype)


def fit_one(source_h5, whipple_dir, output_dir, max_segments, overwrite):
    source_h5 = Path(source_h5)
    event_id = event_id_from_path(source_h5)
    whipple_h5 = Path(whipple_dir) / source_h5.name
    output_path = Path(output_dir) / f"segmented_radius_{event_id}.h5"
    if output_path.exists() and not overwrite:
        return event_id, "exists", np.nan, -1
    with h5py.File(source_h5, "r") as h:
        source_joint = load_group(h["joint_fit"])
        source_fft = load_group(h["fft_observations"])
    with h5py.File(whipple_h5, "r") as h:
        whipple_joint = load_group(h["joint_fit"])
    fits = []
    for n_segments in range(1, int(max_segments) + 1):
        fits.append(fit_segment_count(source_joint, source_fft, whipple_joint, n_segments))
    best_idx = int(np.nanargmin([row["bic"] for row in fits]))
    os.makedirs(output_dir, exist_ok=True)
    write_event(output_path, event_id, fits, best_idx, source_h5, whipple_h5)
    return event_id, "ok", float(fits[best_idx]["bic"]), int(fits[best_idx]["n_segments"])


def write_summary(output_dir, rows):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    path = Path(output_dir) / "segmented_radius_catalog_summary.h5"
    with h5py.File(path, "w") as h:
        h.attrs["script"] = Path(__file__).name
        h.create_dataset("event_id", data=np.asarray([r[0] for r in rows], dtype=object), dtype=string_dtype)
        h.create_dataset("status", data=np.asarray([r[1] for r in rows], dtype=object), dtype=string_dtype)
        h["best_bic"] = np.asarray([r[2] for r in rows], dtype=np.float64)
        h["best_n_segments"] = np.asarray([r[3] for r in rows], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Fit segmented shrinking-radius mass models from Whipple-Jacchia trajectories.")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--whipple-dir", default=DEFAULT_WHIPPLE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = sorted(Path(args.source_dir).glob("joint_delay_doppler_fft_tri_*.h5"))
    rows = []
    if args.jobs <= 1:
        for idx, path in enumerate(paths, 1):
            row = fit_one(path, args.whipple_dir, args.output_dir, args.max_segments, args.overwrite)
            rows.append(row)
            print(f"[{idx}/{len(paths)}] {row[1]} {row[0]} best_segments={row[3]} bic={row[2]:.2f}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(fit_one, path, args.whipple_dir, args.output_dir, args.max_segments, args.overwrite)
                for path in paths
            ]
            for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                row = fut.result()
                rows.append(row)
                print(f"[{idx}/{len(paths)}] {row[1]} {row[0]} best_segments={row[3]} bic={row[2]:.2f}", flush=True)
    write_summary(args.output_dir, rows)


if __name__ == "__main__":
    main()
