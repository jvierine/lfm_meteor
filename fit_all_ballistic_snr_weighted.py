import json
import os

import h5py
import jcoord
import numpy as np
import scipy.optimize as so
from pymsis import msis, utils

import fit_gcrs_trajectories_lfm_ambiguity as gfit
import test_rank02_range_interpolation as interp
from grid_search_delays_beam_axis import DAN_PATTERN, SAN_PATTERN, WEN_PATTERN, gate_to_delay_us, load_events, pair_tristatic_events


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

if MPI is None:
    COMM = None
    RANK = 0
    SIZE = 1
else:
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()


SCRIPT_VERSION = "v20260611c"
OUTPUT_H5 = os.path.join("results", f"all_tristatic_ballistic_snr_weighted_{SCRIPT_VERSION}.h5")
OUTPUT_JSON = os.path.join("results", f"all_tristatic_ballistic_snr_weighted_{SCRIPT_VERSION}.json")
FIT_H5 = "results/gcrs_trajectory_fits_lfm_ambiguity_v20260610.h5"
UPSAMPLE_FACTOR = 4
MIN_POINTS = 8
MAX_LAT_DEG = gfit.MAX_LAT_DEG
MIN_B = 1e-4
MAX_B = 1e3
MSIS_ALT_GRID_KM = np.linspace(50.0, 130.0, 321)
SIGMA_CLIP_RMS = 4.0
SOURCE_TIMEZONE_OFFSET_NS = int(8.0 * 3600.0 * 1e9)


def is_root():
    return RANK == 0


def log(message):
    if is_root():
        print(message, flush=True)


def decode_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def load_reference_fits(path=FIT_H5):
    fits = {}
    with h5py.File(path, "r") as h:
        event_ids = [decode_string(x) for x in h["event_id"][:]]
        for idx, event_id in enumerate(event_ids):
            fits[event_id] = {
                "event_id": event_id,
                "r0_gcrs_m": h["r0_gcrs_m"][idx],
                "v0_gcrs_mps": h["v0_gcrs_mps"][idx],
                "t0_ns": int(h["t0_ns"][idx]),
                "speed_km_s": float(h["speed_km_s"][idx]),
                "rms_total_path_residual_m": float(h["rms_total_path_residual_m"][idx]),
            }
    return fits


def event_start_utc_ns(event):
    if bool(getattr(event, "times_ns_are_utc", False)):
        return int(event.t0_ns)
    return int(event.t0_ns) - SOURCE_TIMEZONE_OFFSET_NS


def match_reference_fit(san_event, ref_fit, tolerance_ms=200.0):
    target = event_start_utc_ns(san_event)
    if not ref_fit:
        return None
    best = min(ref_fit.values(), key=lambda fit: abs(int(fit["t0_ns"]) - target))
    if abs(int(best["t0_ns"]) - target) > int(tolerance_ms * 1e6):
        return None
    return best


def load_site_h5(path, fit, site):
    with h5py.File(path, "r") as h:
        times_ns = h["times_ns"][()].astype(np.int64)
        return {
            "path": path,
            "raw": h["raw"][()].astype(np.complex64),
            "times_ns": times_ns,
            "range_gate": h["range_gate"][()].astype(np.int32),
            "range_km": h["range_km"][()].astype(np.float64),
            "snr_peak_db": h["snr_peak_db"][()].astype(np.float64),
            "sr_mhz": float(h["sr_mhz"][()]),
            "bw_mhz": float(h["bw_mhz"][()]),
            "r0_km": float(h["r0"][()]),
            "az_deg": float(h["az"][()]),
            "el_deg": float(h["el"][()]),
            "doppler_hz": interp.fitted_doppler_hz(site, fit, times_ns),
        }


def refine_site(site_data):
    coarse = []
    for row, fd_hz in zip(site_data["raw"], site_data["doppler_hz"]):
        gate, _power_db = interp.doppler_matched_filter_peak(
            row,
            float(fd_hz),
            site_data["sr_mhz"],
            site_data["bw_mhz"],
            upsample_factor=1,
            coarse_center_gate=None,
        )
        coarse.append(gate)
    coarse = np.asarray(coarse, dtype=np.float64)
    fine = []
    power = []
    for row, fd_hz, center_gate in zip(site_data["raw"], site_data["doppler_hz"], coarse):
        gate, power_db = interp.doppler_matched_filter_peak(
            row,
            float(fd_hz),
            site_data["sr_mhz"],
            site_data["bw_mhz"],
            upsample_factor=UPSAMPLE_FACTOR,
            coarse_center_gate=float(center_gate),
        )
        fine.append(gate)
        power.append(power_db)
    fine = np.asarray(fine, dtype=np.float64)
    dr_km = gfit.C / (site_data["sr_mhz"] * 1e6) / 2.0 / 1e3
    return fine, site_data["r0_km"] + dr_km * fine, np.asarray(power, dtype=np.float64)


def matched_measurements_from_sites(san_event, dan_event, wen_event, site_data, refined):
    matches = gfit.match_pulses_with_time(san_event, dan_event, wen_event)
    measured = []
    times_ns = []
    snr_db = []
    source_indices = []
    for match in matches:
        si, di, wi = match["san_idx"], match["dan_idx"], match["wen_idx"]
        measured.append(
            [
                2.0 * refined["sanya_range_km"][si] * 1e3,
                float(gfit.delay_us_to_total_path_m(gfit.DAN_CENTER_US + refined["danzhou_gate"][di] / site_data["danzhou"]["sr_mhz"])),
                float(gfit.delay_us_to_total_path_m(gfit.WEN_CENTER_US + refined["wenchang_gate"][wi] / site_data["wenchang"]["sr_mhz"])),
            ]
        )
        times_ns.append(match["time_ns"])
        snr_db.append(
            [
                site_data["sanya"]["snr_peak_db"][si],
                site_data["danzhou"]["snr_peak_db"][di],
                site_data["wenchang"]["snr_peak_db"][wi],
            ]
        )
        source_indices.append([si, di, wi])
    return (
        np.asarray(measured, dtype=np.float64),
        np.asarray(times_ns, dtype=np.int64),
        np.asarray(snr_db, dtype=np.float64),
        np.asarray(source_indices, dtype=np.int32),
    )


def triangulate_points(measured_total_paths_m, san_az_deg=15.0, san_el_deg=75.0):
    x0 = gfit.initial_guess(san_az_deg, san_el_deg, float(np.nanmedian(measured_total_paths_m[:, 0]) / 2e3))
    points = []
    keep = []
    for idx, measured in enumerate(measured_total_paths_m):
        try:
            point = gfit.solve_position_from_total_paths_m(measured, x0)
        except Exception:
            keep.append(False)
            continue
        x0 = point
        llh = jcoord.ecef2geodetic(point[0], point[1], point[2])
        good = bool(np.all(np.isfinite(llh)) and float(llh[0]) <= MAX_LAT_DEG)
        keep.append(good)
        if good:
            points.append(point)
    return np.asarray(points, dtype=np.float64), np.asarray(keep, dtype=bool)


def density_interpolator(times_ns, points_ecef_m):
    mid = points_ecef_m[len(points_ecef_m) // 2]
    lat_deg, lon_deg, _alt_m = jcoord.ecef2geodetic(mid[0], mid[1], mid[2])
    date0 = np.datetime64(int(times_ns[len(times_ns) // 2]), "ns")
    try:
        data = msis.run([date0], [float(lon_deg)], [float(lat_deg)], MSIS_ALT_GRID_KM, geomagnetic_activity=1)
    except Exception:
        utils.download_f107_ap()
        data = msis.run([date0], [float(lon_deg)], [float(lat_deg)], MSIS_ALT_GRID_KM, geomagnetic_activity=1)
    rho = np.asarray(data[0, 0, 0, :, 0], dtype=np.float64)

    def rho_of_alt_m(alt_m):
        alt_km = np.clip(np.asarray(alt_m, dtype=np.float64) / 1e3, MSIS_ALT_GRID_KM[0], MSIS_ALT_GRID_KM[-1])
        return np.interp(alt_km, MSIS_ALT_GRID_KM, rho)

    return rho_of_alt_m, {"lat_deg": float(lat_deg), "lon_deg": float(lon_deg), "date_utc": str(date0)}


def initial_ballistic_guess(points_ecef_m, times_ns, log10_b=1.0):
    t_rel_s = (np.asarray(times_ns, dtype=np.float64) - float(times_ns[0])) / 1e9
    design = np.column_stack([np.ones_like(t_rel_s), t_rel_s])
    coeffs = np.linalg.lstsq(design, points_ecef_m, rcond=None)[0]
    return np.concatenate([coeffs[0], coeffs[1], [float(log10_b)]])


def rk4_step(state, dt_s, b_drag, rho_of_alt_m):
    def deriv(y):
        r = y[:3]
        v = y[3:]
        lat, lon, alt = jcoord.ecef2geodetic(r[0], r[1], r[2])
        rho = float(rho_of_alt_m(alt))
        speed = float(np.linalg.norm(v))
        return np.concatenate([v, -b_drag * rho * speed * v])

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt_s * k1)
    k3 = deriv(state + 0.5 * dt_s * k2)
    k4 = deriv(state + dt_s * k3)
    return state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate_ballistic(params, t_rel_s, rho_of_alt_m, dt_max_s=0.002):
    b_drag = float(np.clip(10.0 ** params[6], MIN_B, MAX_B))
    state = np.concatenate([params[:3], params[3:6]]).astype(np.float64)
    positions = []
    velocities = []
    t_prev = 0.0
    for t in np.asarray(t_rel_s, dtype=np.float64):
        while t_prev + 1e-12 < t:
            dt = min(dt_max_s, float(t - t_prev))
            state = rk4_step(state, dt, b_drag, rho_of_alt_m)
            t_prev += dt
        positions.append(state[:3].copy())
        velocities.append(state[3:6].copy())
    return np.asarray(positions), np.asarray(velocities), b_drag


def predict_paths(params, t_rel_s, rho_of_alt_m):
    x_itrs, v_itrs, b_drag = propagate_ballistic(params, t_rel_s, rho_of_alt_m)
    total_paths_m, path_rates_mps = gfit.link_total_paths_and_rates_m(x_itrs, v_itrs, gfit.LINK_TX_POSITIONS_M, gfit.LINK_RX_POSITIONS_M)
    return total_paths_m + gfit.lfm_total_path_bias_m(path_rates_mps), x_itrs, v_itrs, b_drag


def sigma_from_snr_db(snr_db, sigma_floor_m, sigma_0_m):
    snr_amp = 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 20.0)
    return np.sqrt(float(sigma_floor_m) ** 2.0 + (float(sigma_0_m) / np.maximum(snr_amp, 1e-6)) ** 2.0)


def fit_sigma_model(residuals_m, snr_db):
    r = np.asarray(residuals_m, dtype=np.float64).ravel()
    snr = np.asarray(snr_db, dtype=np.float64).ravel()
    good = np.isfinite(r) & np.isfinite(snr) & (np.abs(r) < np.nanpercentile(np.abs(r), 95.0))
    r = r[good]
    snr = snr[good]

    def nll(log_params):
        sigma = sigma_from_snr_db(snr, np.exp(log_params[0]), np.exp(log_params[1]))
        return np.sum(np.log(sigma) + 0.5 * (r / sigma) ** 2.0)

    result = so.minimize(nll, np.log([20.0, 300.0]), method="Nelder-Mead", options={"maxiter": 2000})
    sigma_floor, sigma_0 = np.exp(result.x)
    return {
        "sigma_floor_m": float(sigma_floor),
        "sigma_0_m": float(sigma_0),
        "optimizer_success": bool(result.success),
        "n_samples": int(len(r)),
    }


def linearized_covariance_summary(result, n_residuals):
    n_params = int(result.x.size)
    dof = int(n_residuals - n_params)
    empty_cov = np.full((n_params, n_params), np.nan, dtype=np.float64)
    if dof <= 0:
        return {
            "degrees_of_freedom": dof,
            "residual_variance": np.nan,
            "parameter_covariance": empty_cov,
            "parameter_std": np.full(n_params, np.nan, dtype=np.float64),
            "covariance_available": False,
        }
    jac = np.asarray(result.jac, dtype=np.float64)
    residual_variance = float(2.0 * result.cost / dof)
    try:
        covariance = np.linalg.pinv(jac.T @ jac) * residual_variance
        parameter_std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    except np.linalg.LinAlgError:
        covariance = empty_cov
        parameter_std = np.full(n_params, np.nan, dtype=np.float64)
        available = False
    else:
        available = bool(np.all(np.isfinite(parameter_std)))
    return {
        "degrees_of_freedom": dof,
        "residual_variance": residual_variance,
        "parameter_covariance": covariance,
        "parameter_std": parameter_std,
        "position_std_m": parameter_std[:3],
        "velocity_std_mps": parameter_std[3:6],
        "log10_b_std": float(parameter_std[6]) if len(parameter_std) > 6 else np.nan,
        "covariance_available": available,
    }


def fit_ballistic(measured_total_paths_m, times_ns, rho_of_alt_m, p0, sigma_m=None, keep_rows=None, robust_f_scale=2.0):
    measured = np.asarray(measured_total_paths_m, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.int64)
    if keep_rows is None:
        keep_rows = np.ones(len(times), dtype=bool)
    measured_fit = measured[keep_rows]
    times_fit = times[keep_rows]
    t_rel_s = (times_fit.astype(np.float64) - float(times_fit[0])) / 1e9
    if sigma_m is None:
        sigma = np.ones_like(measured_fit)
        f_scale = 50.0
    else:
        sigma = np.asarray(sigma_m, dtype=np.float64)[keep_rows]
        f_scale = robust_f_scale

    def residual(x):
        pred, _x, _v, _b = predict_paths(x, t_rel_s, rho_of_alt_m)
        return ((pred - measured_fit) / sigma).ravel()

    result = so.least_squares(
        residual,
        p0,
        bounds=(
            np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(MIN_B)]),
            np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(MAX_B)]),
        ),
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0]),
        loss="soft_l1",
        f_scale=f_scale,
        max_nfev=220,
    )
    pred, x_itrs, v_itrs, b_drag = predict_paths(result.x, t_rel_s, rho_of_alt_m)
    raw_resid = pred - measured_fit
    normalized = raw_resid / sigma
    llh = np.asarray([jcoord.ecef2geodetic(x[0], x[1], x[2]) for x in x_itrs], dtype=np.float64)
    covariance = linearized_covariance_summary(result, len(residual(result.x)))
    return {
        "params": result.x,
        "parameter_covariance": covariance["parameter_covariance"],
        "parameter_std": covariance["parameter_std"],
        "position_std_m": covariance["position_std_m"],
        "velocity_std_mps": covariance["velocity_std_mps"],
        "log10_b_std": covariance["log10_b_std"],
        "covariance_available": covariance["covariance_available"],
        "covariance_degrees_of_freedom": covariance["degrees_of_freedom"],
        "covariance_residual_variance": covariance["residual_variance"],
        "keep_rows": keep_rows,
        "time_ns": times_fit,
        "t_rel_s": t_rel_s,
        "measured_total_paths_m": measured_fit,
        "predicted_total_paths_m": pred,
        "residuals_m": raw_resid,
        "normalized_residuals": normalized,
        "x_itrs_m": x_itrs,
        "v_itrs_mps": v_itrs,
        "lat_deg": llh[:, 0],
        "lon_deg": llh[:, 1],
        "alt_km": llh[:, 2] / 1e3,
        "speed_km_s": np.linalg.norm(v_itrs, axis=1) / 1e3,
        "b_drag_m2_per_kg": float(b_drag),
        "rms_total_path_residual_m": float(np.sqrt(np.mean(raw_resid**2.0))),
        "median_abs_total_path_residual_m": float(np.median(np.abs(raw_resid))),
        "weighted_rms": float(np.sqrt(np.mean(normalized**2.0))),
        "n_points": int(len(times_fit)),
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "optimizer_cost": float(result.cost),
    }


def process_triplet(idx, triplet, ref_fit, sigma_model=None):
    san_event, dan_event, wen_event = triplet
    raw_event_id = f"tri_{idx:04d}_{san_event.t0_ns}"
    fit0 = match_reference_fit(san_event, ref_fit)
    if fit0 is None:
        return {"event_id": raw_event_id, "status": "missing_reference_fit"}
    event_id = fit0["event_id"]
    try:
        site_data = {
            "sanya": load_site_h5(san_event.path, fit0, "sanya"),
            "danzhou": load_site_h5(dan_event.path, fit0, "danzhou"),
            "wenchang": load_site_h5(wen_event.path, fit0, "wenchang"),
        }
        refined = {}
        for site in ("sanya", "danzhou", "wenchang"):
            gate, range_km, power_db = refine_site(site_data[site])
            refined[f"{site}_gate"] = gate
            refined[f"{site}_range_km"] = range_km
        measured, times_ns, snr_db, source_indices = matched_measurements_from_sites(san_event, dan_event, wen_event, site_data, refined)
        if len(times_ns) < MIN_POINTS:
            return {"event_id": event_id, "status": "too_few_points", "n_points": int(len(times_ns))}
        points, keep_geo = triangulate_points(measured, san_event.az_deg, san_event.el_deg)
        measured = measured[keep_geo]
        times_ns = times_ns[keep_geo]
        snr_db = snr_db[keep_geo]
        if len(times_ns) < MIN_POINTS:
            return {"event_id": event_id, "status": "too_few_geo_points", "n_points": int(len(times_ns))}
        rho_of_alt_m, msis_meta = density_interpolator(times_ns, points)
        p0 = initial_ballistic_guess(points, times_ns, log10_b=1.0)
        if sigma_model is None:
            fit = fit_ballistic(measured, times_ns, rho_of_alt_m, p0, sigma_m=None)
            sigma_used = None
        else:
            sigma_used = sigma_from_snr_db(snr_db, sigma_model["sigma_floor_m"], sigma_model["sigma_0_m"])
            fit1 = fit_ballistic(measured, times_ns, rho_of_alt_m, p0, sigma_m=sigma_used)
            per_pulse_norm = np.sqrt(np.mean(fit1["normalized_residuals"] ** 2.0, axis=1))
            keep_rows = np.ones(len(times_ns), dtype=bool)
            kept_indices = np.flatnonzero(fit1["keep_rows"])
            keep_rows[kept_indices] = per_pulse_norm < SIGMA_CLIP_RMS
            if np.sum(keep_rows) >= MIN_POINTS and np.sum(~keep_rows) > 0:
                fit = fit_ballistic(measured, times_ns, rho_of_alt_m, fit1["params"], sigma_m=sigma_used, keep_rows=keep_rows)
            else:
                fit = fit1
        return {
            "event_id": event_id,
            "status": "ok",
            "msis": msis_meta,
            "snr_db": snr_db,
            "sigma_m": sigma_used,
            **fit,
        }
    except Exception as exc:
        return {"event_id": event_id, "status": "error", "error": repr(exc)}


def local_process(triplets, ref_fit, sigma_model=None):
    outputs = []
    for idx in range(RANK, len(triplets), SIZE):
        outputs.append(process_triplet(idx, triplets[idx], ref_fit, sigma_model=sigma_model))
        if len(outputs) % 5 == 0:
            print(f"[rank {RANK}] processed {len(outputs)} local events", flush=True)
    return outputs


def gather_outputs(local_outputs):
    if COMM is None:
        return local_outputs
    gathered = COMM.gather(local_outputs, root=0)
    if not is_root():
        return None
    return [item for group in gathered for item in group]


def write_results(path, outputs, sigma_model):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    ok = [o for o in outputs if o["status"] == "ok"]
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["upsample_factor"] = UPSAMPLE_FACTOR
        h.attrs["sigma_floor_m"] = sigma_model["sigma_floor_m"]
        h.attrs["sigma_0_m"] = sigma_model["sigma_0_m"]
        h["event_id"] = np.asarray([o["event_id"] for o in ok], dtype=string_dtype)
        h["n_points"] = np.asarray([o["n_points"] for o in ok], dtype=np.int32)
        h["rms_total_path_residual_m"] = np.asarray([o["rms_total_path_residual_m"] for o in ok], dtype=np.float64)
        h["median_abs_total_path_residual_m"] = np.asarray([o["median_abs_total_path_residual_m"] for o in ok], dtype=np.float64)
        h["weighted_rms"] = np.asarray([o["weighted_rms"] for o in ok], dtype=np.float64)
        h["b_drag_m2_per_kg"] = np.asarray([o["b_drag_m2_per_kg"] for o in ok], dtype=np.float64)
        h["start_speed_km_s"] = np.asarray([o["speed_km_s"][0] for o in ok], dtype=np.float64)
        h["end_speed_km_s"] = np.asarray([o["speed_km_s"][-1] for o in ok], dtype=np.float64)
        h["start_alt_km"] = np.asarray([o["alt_km"][0] for o in ok], dtype=np.float64)
        h["end_alt_km"] = np.asarray([o["alt_km"][-1] for o in ok], dtype=np.float64)
        h["parameter_std"] = np.asarray([o["parameter_std"] for o in ok], dtype=np.float64)
        h["position_std_m"] = np.asarray([o["position_std_m"] for o in ok], dtype=np.float64)
        h["velocity_std_mps"] = np.asarray([o["velocity_std_mps"] for o in ok], dtype=np.float64)
        h["log10_b_std"] = np.asarray([o["log10_b_std"] for o in ok], dtype=np.float64)
        h["covariance_degrees_of_freedom"] = np.asarray([o["covariance_degrees_of_freedom"] for o in ok], dtype=np.int32)
        h["covariance_residual_variance"] = np.asarray([o["covariance_residual_variance"] for o in ok], dtype=np.float64)
        h["covariance_available"] = np.asarray([o["covariance_available"] for o in ok], dtype=bool)
        points = h.create_group("points")
        for o in ok:
            g = points.create_group(o["event_id"])
            for key in [
                "time_ns",
                "t_rel_s",
                "measured_total_paths_m",
                "predicted_total_paths_m",
                "residuals_m",
                "normalized_residuals",
                "x_itrs_m",
                "v_itrs_mps",
                "lat_deg",
                "lon_deg",
                "alt_km",
                "speed_km_s",
                "snr_db",
                "params",
                "parameter_std",
                "parameter_covariance",
                "position_std_m",
                "velocity_std_mps",
            ]:
                g[key] = o[key]
            g.attrs["log10_b_std"] = o["log10_b_std"]
            g.attrs["covariance_available"] = o["covariance_available"]
            g.attrs["covariance_degrees_of_freedom"] = o["covariance_degrees_of_freedom"]
            g.attrs["covariance_residual_variance"] = o["covariance_residual_variance"]
            if o["sigma_m"] is not None:
                g["sigma_m"] = o["sigma_m"]


def main():
    if is_root():
        log(f"loading triplets and reference fits; MPI ranks={SIZE}")
    ref_fit = load_reference_fits()
    triplets = pair_tristatic_events(load_events(SAN_PATTERN), load_events(DAN_PATTERN), load_events(WEN_PATTERN))
    if is_root():
        log(f"triplets={len(triplets)}; first pass unweighted robust ballistic fits")
    first_local = local_process(triplets, ref_fit, sigma_model=None)
    first_outputs = gather_outputs(first_local)

    if is_root():
        ok_first = [o for o in first_outputs if o["status"] == "ok"]
        residuals = np.concatenate([o["residuals_m"].ravel() for o in ok_first])
        snr = np.concatenate([o["snr_db"].ravel() for o in ok_first])
        sigma_model = fit_sigma_model(residuals, snr)
        log(
            "global sigma_path(SNR) = sqrt("
            f"{sigma_model['sigma_floor_m']:.2f}^2 + "
            f"({sigma_model['sigma_0_m']:.2f}/10^(SNR_dB/20))^2) m; "
            f"samples={sigma_model['n_samples']}"
        )
    else:
        sigma_model = None
    if COMM is not None:
        sigma_model = COMM.bcast(sigma_model, root=0)

    if is_root():
        log("second pass weighted robust ballistic fits with sigma clipping")
    second_local = local_process(triplets, ref_fit, sigma_model=sigma_model)
    second_outputs = gather_outputs(second_local)

    if is_root():
        ok = [o for o in second_outputs if o["status"] == "ok"]
        status_counts = {}
        for o in second_outputs:
            status_counts[o["status"]] = status_counts.get(o["status"], 0) + 1
        write_results(OUTPUT_H5, second_outputs, sigma_model)
        summary = {
            "script": os.path.basename(__file__),
            "script_version": SCRIPT_VERSION,
            "upsample_factor": UPSAMPLE_FACTOR,
            "n_triplets": len(triplets),
            "status_counts": status_counts,
            "sigma_model": sigma_model,
            "n_ok": len(ok),
            "rms_total_path_residual_m_median": float(np.nanmedian([o["rms_total_path_residual_m"] for o in ok])) if ok else np.nan,
            "rms_total_path_residual_m_range": [
                float(np.nanmin([o["rms_total_path_residual_m"] for o in ok])) if ok else np.nan,
                float(np.nanmax([o["rms_total_path_residual_m"] for o in ok])) if ok else np.nan,
            ],
            "b_drag_m2_per_kg_median": float(np.nanmedian([o["b_drag_m2_per_kg"] for o in ok])) if ok else np.nan,
            "position_std_m_median_xyz": np.nanmedian(np.asarray([o["position_std_m"] for o in ok], dtype=np.float64), axis=0).tolist()
            if ok
            else [np.nan, np.nan, np.nan],
            "velocity_std_mps_median_xyz": np.nanmedian(np.asarray([o["velocity_std_mps"] for o in ok], dtype=np.float64), axis=0).tolist()
            if ok
            else [np.nan, np.nan, np.nan],
            "log10_b_std_median": float(np.nanmedian([o["log10_b_std"] for o in ok])) if ok else np.nan,
            "covariance_available_count": int(np.count_nonzero([o["covariance_available"] for o in ok])) if ok else 0,
        }
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log(f"status_counts={status_counts}")
        log(f"median weighted ballistic RMS={summary['rms_total_path_residual_m_median']:.2f} m")
        log(f"median B={summary['b_drag_m2_per_kg_median']:.4g} m^2/kg")
        log(f"median position std xyz={summary['position_std_m_median_xyz']} m")
        log(f"median velocity std xyz={summary['velocity_std_mps_median_xyz']} m/s")
        log(f"wrote {OUTPUT_H5}")
        log(f"wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
