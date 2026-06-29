import json
import os

import astropy.units as u
import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import scipy.signal as sig
from astropy.coordinates import GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time

import sanya_opts as sc
from fit_gcrs_trajectories_lfm_ambiguity import (
    C,
    CHIRP_RATE_HZ_PER_S,
    DAN_CENTER_US,
    LINK_RX_POSITIONS_M,
    LINK_TX_POSITIONS_M,
    MAX_LAT_DEG,
    REFERENCE_CHIRP_RATE_SCALE,
    RADAR_WAVELENGTH_M,
    WEN_CENTER_US,
    delay_us_to_total_path_m,
    initial_guess,
    lfm_total_path_bias_m,
    solve_position_from_total_paths_m,
)

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

SCRIPT_VERSION = "v20260611b"
EVENT_ID_LOCAL = "tri_0134_1713850083054349899"
EVENT_ID_UTC = "tri_0134_1713821283054349899"
FIT_H5 = "results/gcrs_trajectory_fits_lfm_ambiguity_v20260613b.h5"
OUTPUT_BASE = os.path.join("results", f"rank02_range_interpolation_test_{SCRIPT_VERSION}")
UPSAMPLE_FACTORS = (1, 2, 4, 8, 16, 32)
SEARCH_HALF_WIDTH_GATES = 2.0
MIN_POINTS = 3
RANK02_FIT_FALLBACK = {
    "r0_gcrs_m": np.asarray([1370784.5385012818, -6000144.468170229, 2055985.9447967666], dtype=np.float64),
    "v0_gcrs_mps": np.asarray([-10530.004291651976, 26036.242280974377, -17797.52989150951], dtype=np.float64),
    "t0_ns": 1713821283127683328,
    "rms_total_path_residual_m": 48.10718448992003,
    "median_abs_total_path_residual_m": 29.561809513397748,
}

SITE_ORDER = ("sanya", "danzhou", "wenchang")
EVENT_PATHS = {
    "sanya": "results/tristatic_head_echoes/sanya/sanya_1713850083054349899.h5",
    "danzhou": "results/tristatic_head_echoes/danzhou/danzhou_1713850083119349957.h5",
    "wenchang": "results/tristatic_head_echoes/wenchang/wenchang_1713850083129349947.h5",
}
SITE_DELAY_US = {
    "danzhou": DAN_CENTER_US,
    "wenchang": WEN_CENTER_US,
}
SITE_RX_POSITIONS_M = {
    "sanya": np.asarray(sc.p_san, dtype=np.float64),
    "danzhou": np.asarray(sc.p_dan, dtype=np.float64),
    "wenchang": np.asarray(sc.p_wen, dtype=np.float64),
}
TX_POSITION_M = np.asarray(sc.p_san, dtype=np.float64)


def is_root():
    return RANK == 0


def log(message):
    if is_root():
        print(message, flush=True)


def lfm(length_us=199, sr_mhz=4.0, bandwidth_hz=4e6, chirp_rate_scale=REFERENCE_CHIRP_RATE_SCALE):
    t_s = np.arange(int(round(length_us * sr_mhz)), dtype=np.float64) / (sr_mhz * 1e6)
    sweep_rate = bandwidth_hz * 1e6 / length_us / 2.0 * float(chirp_rate_scale)
    code = np.exp(1j * 2.0 * np.pi * (t_s * bandwidth_hz / 2.0 - sweep_rate * t_s**2.0))
    return code.astype(np.complex64), t_s


def load_reference_fit():
    try:
        with h5py.File(FIT_H5, "r") as h:
            event_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h["event_id"][:]]
            idx = event_ids.index(EVENT_ID_UTC)
            return {
                "r0_gcrs_m": h["r0_gcrs_m"][idx],
                "v0_gcrs_mps": h["v0_gcrs_mps"][idx],
                "t0_ns": int(h["t0_ns"][idx]),
                "rms_total_path_residual_m": float(h["rms_total_path_residual_m"][idx]),
                "median_abs_total_path_residual_m": float(h["median_abs_total_path_residual_m"][idx]),
            }
    except Exception as exc:
        log(f"warning: could not read {FIT_H5}; using embedded rank02 fit fallback ({exc})")
        return dict(RANK02_FIT_FALLBACK)


def gcrs_state_to_itrs_general(r0_gcrs_m, v0_gcrs_mps, t_rel_s, times_ns, a0_gcrs_mps2=None):
    t_rel_s = np.asarray(t_rel_s, dtype=np.float64)
    times_ns = np.asarray(times_ns, dtype=np.int64)
    if a0_gcrs_mps2 is None:
        a0_gcrs_mps2 = np.zeros(3, dtype=np.float64)
    a0_gcrs_mps2 = np.asarray(a0_gcrs_mps2, dtype=np.float64)
    positions = r0_gcrs_m[None, :] + t_rel_s[:, None] * v0_gcrs_mps[None, :] + 0.5 * t_rel_s[:, None] ** 2.0 * a0_gcrs_mps2[None, :]
    velocities = v0_gcrs_mps[None, :] + t_rel_s[:, None] * a0_gcrs_mps2[None, :]
    obstime = Time(times_ns.astype(np.float64) / 1e9, format="unix", scale="utc")
    representation = CartesianRepresentation(
        positions[:, 0] * u.m,
        positions[:, 1] * u.m,
        positions[:, 2] * u.m,
        differentials=CartesianDifferential(
            velocities[:, 0] * u.m / u.s,
            velocities[:, 1] * u.m / u.s,
            velocities[:, 2] * u.m / u.s,
        ),
    )
    gcrs = GCRS(representation, obstime=obstime)
    itrs = gcrs.transform_to(ITRS(obstime=obstime))
    return (
        itrs.cartesian.without_differentials().xyz.to_value(u.m).T,
        itrs.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s).T,
    )


def fitted_doppler_hz(site, fit, times_ns):
    t_rel_s = (np.asarray(times_ns, dtype=np.float64) - float(fit["t0_ns"])) / 1e9
    positions, velocities = gcrs_state_to_itrs_general(fit["r0_gcrs_m"], fit["v0_gcrs_mps"], t_rel_s, times_ns)
    tx_vectors = positions - TX_POSITION_M[None, :]
    rx_vectors = positions - SITE_RX_POSITIONS_M[site][None, :]
    tx_unit = tx_vectors / np.linalg.norm(tx_vectors, axis=1)[:, None]
    rx_unit = rx_vectors / np.linalg.norm(rx_vectors, axis=1)[:, None]
    path_rate_mps = np.sum((tx_unit + rx_unit) * velocities, axis=1)
    return -path_rate_mps / RADAR_WAVELENGTH_M


def load_site(site, fit):
    with h5py.File(EVENT_PATHS[site], "r") as h:
        return {
            "raw": h["raw"][()].astype(np.complex64),
            "times_ns": h["times_ns"][()].astype(np.int64),
            "ranges_km": h["ranges_km_axis"][()].astype(np.float64),
            "range_gate": h["range_gate"][()].astype(np.int32),
            "range_km": h["range_km"][()].astype(np.float64),
            "snr_peak_db": h["snr_peak_db"][()].astype(np.float64),
            "sr_mhz": float(h["sr_mhz"][()]),
            "bw_mhz": float(h["bw_mhz"][()]),
            "r0_km": float(h["r0"][()]),
            "az_deg": float(h["az"][()]),
            "el_deg": float(h["el"][()]),
            "doppler_hz": fitted_doppler_hz(site, fit, h["times_ns"][()].astype(np.int64)),
        }


def doppler_matched_filter_peak(row, fd_hz, sr_mhz, bw_mhz, upsample_factor, coarse_center_gate=None):
    if upsample_factor == 1:
        raw_up = row
        sr_up_mhz = sr_mhz
    else:
        raw_up = sig.resample_poly(row, upsample_factor, 1).astype(np.complex64)
        sr_up_mhz = sr_mhz * upsample_factor
    code, t_s = lfm(sr_mhz=sr_up_mhz, bandwidth_hz=bw_mhz * 1e6)
    # Use the same received-chirp phase convention as the single-pulse ACF
    # Doppler diagnostic: the matched filter applies conj(doppler_code).
    doppler_code = code * np.exp(1j * 2.0 * np.pi * fd_hz * t_s).astype(np.complex64)
    corr = sig.fftconvolve(raw_up, np.conj(doppler_code), mode="same")
    power = np.abs(corr) ** 2.0
    if coarse_center_gate is None:
        idx0 = int(np.argmax(power))
    else:
        center = int(round(float(coarse_center_gate) * upsample_factor))
        half = int(np.ceil(SEARCH_HALF_WIDTH_GATES * upsample_factor))
        lo = max(0, center - half)
        hi = min(len(power), center + half + 1)
        idx0 = lo + int(np.argmax(power[lo:hi]))

    # Quadratic interpolation in power gives a sub-bin peak estimate after the
    # explicit raw-voltage upsampling. Keep it local and bounded.
    delta = 0.0
    if 0 < idx0 < len(power) - 1:
        ym1, y0, yp1 = float(power[idx0 - 1]), float(power[idx0]), float(power[idx0 + 1])
        denom = ym1 - 2.0 * y0 + yp1
        if denom < 0.0:
            delta = 0.5 * (ym1 - yp1) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
    fine_gate = (float(idx0) + delta) / float(upsample_factor)
    return fine_gate, float(10.0 * np.log10(max(power[idx0], 1e-30)))


def gather_indexed_values(n_items, local_items):
    if COMM is None:
        gathered = [local_items]
    else:
        gathered = COMM.gather(local_items, root=0)
    if not is_root():
        return None

    values = np.full(n_items, np.nan, dtype=np.float64)
    powers_db = np.full(n_items, np.nan, dtype=np.float64)
    for rank_items in gathered:
        for idx, value, power_db in rank_items:
            values[int(idx)] = float(value)
            powers_db[int(idx)] = float(power_db)
    if not np.all(np.isfinite(values)):
        missing = np.flatnonzero(~np.isfinite(values))
        raise RuntimeError(f"Missing MPI refined gates for indices {missing.tolist()}")
    return values, powers_db


def matched_filter_local_items(site_data, upsample_factor, coarse_gates=None):
    local_items = []
    n_pulses = int(site_data["raw"].shape[0])
    for idx in range(RANK, n_pulses, SIZE):
        center_gate = None if coarse_gates is None else float(coarse_gates[idx])
        gate, power_db = doppler_matched_filter_peak(
            site_data["raw"][idx],
            float(site_data["doppler_hz"][idx]),
            site_data["sr_mhz"],
            site_data["bw_mhz"],
            upsample_factor=upsample_factor,
            coarse_center_gate=center_gate,
        )
        local_items.append((idx, gate, power_db))
    return local_items


def precompute_coarse_gates(site_data_by_name):
    coarse = {}
    if is_root():
        log(f"precomputing 1x Doppler-corrected coarse gates using {SIZE} MPI rank(s)")
    for site in SITE_ORDER:
        local_items = matched_filter_local_items(site_data_by_name[site], upsample_factor=1, coarse_gates=None)
        gathered = gather_indexed_values(site_data_by_name[site]["raw"].shape[0], local_items)
        if is_root():
            coarse[site] = gathered[0]
    if COMM is not None:
        coarse = COMM.bcast(coarse if is_root() else None, root=0)
    return coarse


def refine_site_ranges(site_data, upsample_factor, coarse_gates):
    if upsample_factor == 1:
        fine_gates = np.asarray(coarse_gates, dtype=np.float64)
        local_items = matched_filter_local_items(site_data, upsample_factor=1, coarse_gates=None)
        gathered = gather_indexed_values(site_data["raw"].shape[0], local_items)
        powers_db = gathered[1] if is_root() else None
    else:
        local_items = matched_filter_local_items(site_data, upsample_factor=upsample_factor, coarse_gates=coarse_gates)
        gathered = gather_indexed_values(site_data["raw"].shape[0], local_items)
        if is_root():
            fine_gates, powers_db = gathered
        else:
            fine_gates = None
            powers_db = None

    if not is_root():
        return None, None, None

    dr_km = C / (site_data["sr_mhz"] * 1e6) / 2.0 / 1e3
    fine_ranges_km = site_data["r0_km"] + dr_km * fine_gates
    return fine_gates, fine_ranges_km, powers_db


def nearest_index(values, target):
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return None
    idx = int(np.argmin(np.abs(values - int(target))))
    return idx


def matched_measurements(site_data, refined_ranges):
    san = site_data["sanya"]
    dan = site_data["danzhou"]
    wen = site_data["wenchang"]
    tolerance_ns = int(7.5e6)
    measured_total_paths = []
    times_ns = []
    source_indices = []
    for san_idx, san_t in enumerate(san["times_ns"]):
        dan_idx = nearest_index(dan["times_ns"], san_t)
        wen_idx = nearest_index(wen["times_ns"], san_t)
        if dan_idx is None or wen_idx is None:
            continue
        dan_t = int(dan["times_ns"][dan_idx])
        wen_t = int(wen["times_ns"][wen_idx])
        if abs(dan_t - int(san_t)) > tolerance_ns or abs(wen_t - int(san_t)) > tolerance_ns:
            continue
        san_total = 2.0 * refined_ranges["sanya"][san_idx] * 1e3
        dan_total = delay_us_to_total_path_m(SITE_DELAY_US["danzhou"] + refined_ranges["danzhou_gate"][dan_idx] / dan["sr_mhz"])
        wen_total = delay_us_to_total_path_m(SITE_DELAY_US["wenchang"] + refined_ranges["wenchang_gate"][wen_idx] / wen["sr_mhz"])
        measured_total_paths.append([san_total, float(dan_total), float(wen_total)])
        times_ns.append(int(round((int(san_t) + dan_t + wen_t) / 3.0)))
        source_indices.append([san_idx, dan_idx, wen_idx])
    return (
        np.asarray(measured_total_paths, dtype=np.float64),
        np.asarray(times_ns, dtype=np.int64),
        np.asarray(source_indices, dtype=np.int32),
    )


def linear_initial_state(points_ecef_m, times_ns):
    points_gcrs_m = ecef_points_to_gcrs(points_ecef_m, times_ns)
    t_rel_s = (np.asarray(times_ns, dtype=np.float64) - float(times_ns[0])) / 1e9
    design = np.column_stack([np.ones_like(t_rel_s), t_rel_s])
    coeffs = np.linalg.lstsq(design, points_gcrs_m, rcond=None)[0]
    return coeffs[0], coeffs[1]


def ecef_points_to_gcrs(points_ecef_m, times_ns):
    obstime = Time(np.asarray(times_ns, dtype=np.float64) / 1e9, format="unix", scale="utc")
    itrs = ITRS(
        CartesianRepresentation(
            points_ecef_m[:, 0] * u.m,
            points_ecef_m[:, 1] * u.m,
            points_ecef_m[:, 2] * u.m,
        ),
        obstime=obstime,
    )
    gcrs = itrs.transform_to(GCRS(obstime=obstime))
    return gcrs.cartesian.xyz.to_value(u.m).T


def model_gcrs_positions(params, t_rel_s, acceleration=False):
    t_rel_s = np.asarray(t_rel_s, dtype=np.float64)
    r0 = np.asarray(params[:3], dtype=np.float64)
    v0 = np.asarray(params[3:6], dtype=np.float64)
    if acceleration:
        a0 = np.asarray(params[6:9], dtype=np.float64)
    else:
        a0 = np.zeros(3, dtype=np.float64)
    return r0[None, :] + t_rel_s[:, None] * v0[None, :] + 0.5 * t_rel_s[:, None] ** 2.0 * a0[None, :]


def total_paths_and_rates(positions_itrs_m, velocities_itrs_mps):
    tx_vectors = positions_itrs_m[:, None, :] - LINK_TX_POSITIONS_M[None, :, :]
    rx_vectors = positions_itrs_m[:, None, :] - LINK_RX_POSITIONS_M[None, :, :]
    tx_distances = np.linalg.norm(tx_vectors, axis=2)
    rx_distances = np.linalg.norm(rx_vectors, axis=2)
    tx_unit = tx_vectors / tx_distances[:, :, None]
    rx_unit = rx_vectors / rx_distances[:, :, None]
    total_paths_m = tx_distances + rx_distances
    path_rates_mps = np.sum((tx_unit + rx_unit) * velocities_itrs_mps[:, None, :], axis=2)
    return total_paths_m, path_rates_mps


def predict_paths(params, t_rel_s, times_ns, acceleration=False):
    if acceleration:
        r0 = np.asarray(params[:3], dtype=np.float64)
        v0 = np.asarray(params[3:6], dtype=np.float64)
        a0 = np.asarray(params[6:9], dtype=np.float64)
    else:
        r0 = np.asarray(params[:3], dtype=np.float64)
        v0 = np.asarray(params[3:6], dtype=np.float64)
        a0 = None
    x_itrs, v_itrs = gcrs_state_to_itrs_general(r0, v0, t_rel_s, times_ns, a0)
    total_paths_m, path_rates_mps = total_paths_and_rates(x_itrs, v_itrs)
    return total_paths_m + lfm_total_path_bias_m(path_rates_mps), x_itrs


def fit_trajectory(measured_total_paths_m, times_ns, san_az_deg, san_el_deg, san_median_range_km, acceleration=False):
    x0 = initial_guess(san_az_deg, san_el_deg, san_median_range_km)
    points = []
    valid_measurements = []
    valid_times = []
    for measured, t_ns in zip(measured_total_paths_m, times_ns):
        point = solve_position_from_total_paths_m(measured, x0)
        x0 = point
        llh = jcoord.ecef2geodetic(point[0], point[1], point[2])
        if not np.all(np.isfinite(llh)) or float(llh[0]) > MAX_LAT_DEG:
            continue
        points.append(point)
        valid_measurements.append(measured)
        valid_times.append(t_ns)
    if len(points) < MIN_POINTS:
        raise RuntimeError("Too few valid points for trajectory fit")

    points = np.asarray(points, dtype=np.float64)
    valid_measurements = np.asarray(valid_measurements, dtype=np.float64)
    valid_times = np.asarray(valid_times, dtype=np.int64)
    order = np.argsort(valid_times)
    points = points[order]
    valid_measurements = valid_measurements[order]
    valid_times = valid_times[order]
    t_rel_s = (valid_times.astype(np.float64) - float(valid_times[0])) / 1e9
    r0, v0 = linear_initial_state(points, valid_times)
    if acceleration:
        p0 = np.concatenate([r0, v0, np.zeros(3, dtype=np.float64)])
        x_scale = np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1e4, 1e4, 1e4])
    else:
        p0 = np.concatenate([r0, v0])
        x_scale = np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4])

    def residual(params):
        predicted, _ = predict_paths(params, t_rel_s, valid_times, acceleration=acceleration)
        return (predicted - valid_measurements).ravel()

    result = so.least_squares(residual, p0, method="trf", x_scale=x_scale, max_nfev=250)
    predicted, x_itrs = predict_paths(result.x, t_rel_s, valid_times, acceleration=acceleration)
    residuals = predicted - valid_measurements
    llh = np.asarray([jcoord.ecef2geodetic(x[0], x[1], x[2]) for x in x_itrs], dtype=np.float64)
    points_gcrs_m = ecef_points_to_gcrs(points, valid_times)
    model_points_gcrs_m = model_gcrs_positions(result.x, t_rel_s, acceleration=acceleration)
    out = {
        "params": result.x,
        "residuals_m": residuals,
        "time_ns": valid_times,
        "t_rel_s": t_rel_s,
        "points_ecef_m": points,
        "points_gcrs_m": points_gcrs_m,
        "model_points_gcrs_m": model_points_gcrs_m,
        "position_residuals_gcrs_m": points_gcrs_m - model_points_gcrs_m,
        "rms_total_path_residual_m": float(np.sqrt(np.mean(residuals**2.0))),
        "median_abs_total_path_residual_m": float(np.median(np.abs(residuals))),
        "n_points": int(len(valid_times)),
        "duration_s": float(t_rel_s[-1] - t_rel_s[0]),
        "speed_km_s": float(np.linalg.norm(result.x[3:6]) / 1e3),
        "start_alt_km": float(llh[0, 2] / 1e3),
        "end_alt_km": float(llh[-1, 2] / 1e3),
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "optimizer_cost": float(result.cost),
    }
    if acceleration:
        out["accel_mps2"] = float(np.linalg.norm(result.x[6:9]))
        out["along_track_accel_mps2"] = float(np.dot(result.x[6:9], result.x[3:6]) / np.linalg.norm(result.x[3:6]))
    return out


def plot_summary(rows):
    factors = np.asarray([row["upsample_factor"] for row in rows], dtype=np.float64)
    const_rms = np.asarray([row["constant_velocity"]["rms_total_path_residual_m"] for row in rows])
    accel_rms = np.asarray([row["constant_acceleration"]["rms_total_path_residual_m"] for row in rows])
    accel_mag = np.asarray([row["constant_acceleration"]["accel_mps2"] for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(factors, const_rms, "o-", label="constant velocity")
    axes[0].plot(factors, accel_rms, "s-", label="constant acceleration")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel("Total-path RMS residual (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(factors, accel_mag, "o-", color="#b23a48")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Raw-voltage interpolation factor")
    axes[1].set_ylabel("Fitted acceleration magnitude (m s$^{-2}$)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Rank02 Doppler-corrected range interpolation test, {SCRIPT_VERSION}")
    fig.savefig(f"{OUTPUT_BASE}.png", dpi=220)
    plt.close(fig)


def json_fit_summary(fit):
    keys = [
        "rms_total_path_residual_m",
        "median_abs_total_path_residual_m",
        "n_points",
        "duration_s",
        "speed_km_s",
        "start_alt_km",
        "end_alt_km",
        "optimizer_success",
        "optimizer_nfev",
        "optimizer_cost",
        "accel_mps2",
        "along_track_accel_mps2",
    ]
    return {key: fit[key] for key in keys if key in fit}


def main():
    if is_root():
        os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)
        log(f"rank02 interpolation test {SCRIPT_VERSION}; MPI ranks={SIZE}")
    fit = load_reference_fit()
    site_data = {site: load_site(site, fit) for site in SITE_ORDER}
    coarse_gates = precompute_coarse_gates(site_data)
    rows = []
    for upsample_factor in UPSAMPLE_FACTORS:
        if is_root():
            log(f"upsample {upsample_factor}: refining Doppler-corrected peaks")
        refined = {}
        range_shifts_m = {}
        power_stats = {}
        for site in SITE_ORDER:
            fine_gate, fine_range_km, power_db = refine_site_ranges(site_data[site], upsample_factor, coarse_gates[site])
            if is_root():
                refined[f"{site}_gate"] = fine_gate
                refined[site] = fine_range_km
                range_shifts_m[site] = {
                    "median_vs_original_m": float(np.median((fine_range_km - site_data[site]["range_km"]) * 1e3)),
                    "std_vs_original_m": float(np.std((fine_range_km - site_data[site]["range_km"]) * 1e3)),
                }
                power_stats[site] = {
                    "median_peak_power_db": float(np.median(power_db)),
                    "max_peak_power_db": float(np.max(power_db)),
                }

        if not is_root():
            continue

        measured, times_ns, source_indices = matched_measurements(site_data, refined)
        const_fit = fit_trajectory(
            measured,
            times_ns,
            site_data["sanya"]["az_deg"],
            site_data["sanya"]["el_deg"],
            float(np.median(refined["sanya"])),
            acceleration=False,
        )
        accel_fit = fit_trajectory(
            measured,
            times_ns,
            site_data["sanya"]["az_deg"],
            site_data["sanya"]["el_deg"],
            float(np.median(refined["sanya"])),
            acceleration=True,
        )
        row = {
            "upsample_factor": int(upsample_factor),
            "search_half_width_gates": SEARCH_HALF_WIDTH_GATES,
            "n_matched_measurements": int(len(times_ns)),
            "range_shift_stats": range_shifts_m,
            "power_stats": power_stats,
            "constant_velocity": json_fit_summary(const_fit),
            "constant_acceleration": json_fit_summary(accel_fit),
        }
        rows.append(row)
        log(
            f"up={upsample_factor:2d} "
            f"cv_rms={const_fit['rms_total_path_residual_m']:.2f} m "
            f"ca_rms={accel_fit['rms_total_path_residual_m']:.2f} m "
            f"accel={accel_fit['accel_mps2']:.1f} m/s^2 "
            f"n={const_fit['n_points']}"
        )

    if is_root():
        with open(f"{OUTPUT_BASE}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "script": os.path.basename(__file__),
                    "script_version": SCRIPT_VERSION,
                    "mpi_size": SIZE,
                    "event_id_local": EVENT_ID_LOCAL,
                    "event_id_utc": EVENT_ID_UTC,
                    "fit_h5": FIT_H5,
                    "reference_fit_rms_total_path_residual_m": fit["rms_total_path_residual_m"],
                    "reference_fit_median_abs_total_path_residual_m": fit["median_abs_total_path_residual_m"],
                    "rows": rows,
                },
                f,
                indent=2,
            )
        plot_summary(rows)
        log(f"wrote {OUTPUT_BASE}.json")
        log(f"wrote {OUTPUT_BASE}.png")


if __name__ == "__main__":
    main()
