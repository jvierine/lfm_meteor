import json
import os

import astropy.units as u
import h5py
import jcoord
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
from astropy.coordinates import GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time
from pymsis import msis, utils

import test_rank02_range_interpolation as interp


SCRIPT_VERSION = "v20260611b"
UPSAMPLE_FACTOR = 4
OUTPUT_BASE = os.path.join("results", f"rank02_ballistic_snr_fit_{SCRIPT_VERSION}")
MSIS_ALT_GRID_KM = np.linspace(50.0, 130.0, 321)
MIN_B = 1e-4
MAX_B = 1e3


def build_measurements():
    fit = interp.load_reference_fit()
    site_data = {site: interp.load_site(site, fit) for site in interp.SITE_ORDER}
    coarse_gates = interp.precompute_coarse_gates(site_data)
    refined = {}
    for site in interp.SITE_ORDER:
        fine_gate, fine_range_km, _power_db = interp.refine_site_ranges(site_data[site], UPSAMPLE_FACTOR, coarse_gates[site])
        if interp.is_root():
            refined[f"{site}_gate"] = fine_gate
            refined[site] = fine_range_km
    if not interp.is_root():
        return None

    measured, times_ns, source_indices = interp.matched_measurements(site_data, refined)
    snr_db = np.column_stack(
        [
            site_data["sanya"]["snr_peak_db"][source_indices[:, 0]],
            site_data["danzhou"]["snr_peak_db"][source_indices[:, 1]],
            site_data["wenchang"]["snr_peak_db"][source_indices[:, 2]],
        ]
    )
    return site_data, measured, times_ns, snr_db


def datetime64_from_ns(time_ns):
    return np.datetime64(int(time_ns), "ns")


def make_density_interpolator(times_ns, measured_total_paths_m):
    # Use the middle triangulated point as the representative MSIS location.
    points = triangulated_points_ecef(measured_total_paths_m)
    mid = points[len(points) // 2]
    lat_deg, lon_deg, _alt_m = jcoord.ecef2geodetic(mid[0], mid[1], mid[2])
    date0 = datetime64_from_ns(times_ns[len(times_ns) // 2])
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


def triangulated_points_ecef(measured_total_paths_m):
    x0 = interp.initial_guess(15.0, 75.0, float(np.nanmedian(measured_total_paths_m[:, 0]) / 2e3))
    points = []
    for measured in measured_total_paths_m:
        point = interp.solve_position_from_total_paths_m(measured, x0)
        points.append(point)
        x0 = point
    return np.asarray(points, dtype=np.float64)


def ecef_positions_to_gcrs(points_ecef_m, times_ns):
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


def itrs_state_to_gcrs_positions(positions_itrs_m, velocities_itrs_mps, times_ns):
    obstime = Time(np.asarray(times_ns, dtype=np.float64) / 1e9, format="unix", scale="utc")
    representation = CartesianRepresentation(
        positions_itrs_m[:, 0] * u.m,
        positions_itrs_m[:, 1] * u.m,
        positions_itrs_m[:, 2] * u.m,
        differentials=CartesianDifferential(
            velocities_itrs_mps[:, 0] * u.m / u.s,
            velocities_itrs_mps[:, 1] * u.m / u.s,
            velocities_itrs_mps[:, 2] * u.m / u.s,
        ),
    )
    itrs = ITRS(representation, obstime=obstime)
    gcrs = itrs.transform_to(GCRS(obstime=obstime))
    return gcrs.cartesian.without_differentials().xyz.to_value(u.m).T


def initial_ballistic_guess(measured_total_paths_m, times_ns, log10_b=-1.0):
    points = triangulated_points_ecef(measured_total_paths_m)
    t_rel_s = (np.asarray(times_ns, dtype=np.float64) - float(times_ns[0])) / 1e9
    design = np.column_stack([np.ones_like(t_rel_s), t_rel_s])
    coeffs = np.linalg.lstsq(design, points, rcond=None)[0]
    return np.concatenate([coeffs[0], coeffs[1], [float(log10_b)]])


def gcrs_params_to_itrs_initial(params, times_ns, acceleration=False):
    t_rel_s = (np.asarray(times_ns, dtype=np.float64) - float(times_ns[0])) / 1e9
    if acceleration:
        a0 = params[6:9]
    else:
        a0 = None
    x_itrs, v_itrs = interp.gcrs_state_to_itrs_general(params[:3], params[3:6], t_rel_s[:1], times_ns[:1], a0)
    return x_itrs[0], v_itrs[0]


def rk4_step(state, t_abs_s, dt_s, b_drag, rho_of_alt_m):
    def deriv(y):
        r = y[:3]
        v = y[3:]
        lat, lon, alt = jcoord.ecef2geodetic(r[0], r[1], r[2])
        rho = float(rho_of_alt_m(alt))
        speed = float(np.linalg.norm(v))
        a_drag = -b_drag * rho * speed * v
        return np.concatenate([v, a_drag])

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt_s * k1)
    k3 = deriv(state + 0.5 * dt_s * k2)
    k4 = deriv(state + dt_s * k3)
    return state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate_ballistic_itrs(params, t_rel_s, rho_of_alt_m, dt_max_s=0.002):
    b_drag = float(np.clip(10.0 ** params[6], MIN_B, MAX_B))
    state = np.concatenate([params[:3], params[3:6]]).astype(np.float64)
    positions = []
    velocities = []
    t_prev = 0.0
    for t in np.asarray(t_rel_s, dtype=np.float64):
        while t_prev + 1e-12 < t:
            dt = min(dt_max_s, float(t - t_prev))
            state = rk4_step(state, t_prev, dt, b_drag, rho_of_alt_m)
            t_prev += dt
        positions.append(state[:3].copy())
        velocities.append(state[3:6].copy())
    return np.asarray(positions), np.asarray(velocities), b_drag


def predict_ballistic_paths(params, t_rel_s, rho_of_alt_m):
    x_itrs, v_itrs, b_drag = propagate_ballistic_itrs(params, t_rel_s, rho_of_alt_m)
    total_paths_m, path_rates_mps = interp.total_paths_and_rates(x_itrs, v_itrs)
    return total_paths_m + interp.lfm_total_path_bias_m(path_rates_mps), x_itrs, v_itrs, b_drag


def sigma_from_snr_db(snr_db, sigma_floor_m, sigma_0_m):
    snr_amp = 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 20.0)
    return np.sqrt(float(sigma_floor_m) ** 2.0 + (float(sigma_0_m) / np.maximum(snr_amp, 1e-6)) ** 2.0)


def fit_sigma_model(residuals_m, snr_db):
    r = np.asarray(residuals_m, dtype=np.float64).ravel()
    snr = np.asarray(snr_db, dtype=np.float64).ravel()
    good = np.isfinite(r) & np.isfinite(snr)
    r = r[good]
    snr = snr[good]

    def nll(log_params):
        sigma_floor = np.exp(log_params[0])
        sigma_0 = np.exp(log_params[1])
        sigma = sigma_from_snr_db(snr, sigma_floor, sigma_0)
        return np.sum(np.log(sigma) + 0.5 * (r / sigma) ** 2.0)

    result = so.minimize(
        nll,
        np.log([8.0, 400.0]),
        method="Nelder-Mead",
        options={"maxiter": 2000},
    )
    sigma_floor, sigma_0 = np.exp(result.x)
    return {
        "sigma_floor_m": float(sigma_floor),
        "sigma_0_m": float(sigma_0),
        "optimizer_success": bool(result.success),
        "optimizer_fun": float(result.fun),
    }


def fit_ballistic(measured_total_paths_m, times_ns, rho_of_alt_m, p0_params, sigma_m=None):
    t_rel_s = (np.asarray(times_ns, dtype=np.float64) - float(times_ns[0])) / 1e9
    measured = np.asarray(measured_total_paths_m, dtype=np.float64)
    if sigma_m is None:
        sigma = np.ones_like(measured)
    else:
        sigma = np.asarray(sigma_m, dtype=np.float64)

    def residual(x):
        pred, _x_itrs, _v_itrs, _b = predict_ballistic_paths(x, t_rel_s, rho_of_alt_m)
        return ((pred - measured) / sigma).ravel()

    lower = np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(MIN_B)])
    upper = np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(MAX_B)])
    result = so.least_squares(
        residual,
        p0_params,
        bounds=(lower, upper),
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0]),
        max_nfev=250,
        verbose=0,
    )
    pred, x_itrs, v_itrs, b_drag = predict_ballistic_paths(result.x, t_rel_s, rho_of_alt_m)
    raw_resid = pred - measured
    llh = np.asarray([jcoord.ecef2geodetic(x[0], x[1], x[2]) for x in x_itrs], dtype=np.float64)
    return {
        "params": result.x,
        "b_drag_m2_per_kg": float(b_drag),
        "predicted_total_paths_m": pred,
        "residuals_m": raw_resid,
        "weighted_residuals": raw_resid / sigma,
        "x_itrs_m": x_itrs,
        "v_itrs_mps": v_itrs,
        "lat_deg": llh[:, 0],
        "lon_deg": llh[:, 1],
        "alt_km": llh[:, 2] / 1e3,
        "speed_km_s": np.linalg.norm(v_itrs, axis=1) / 1e3,
        "rms_total_path_residual_m": float(np.sqrt(np.mean(raw_resid**2.0))),
        "median_abs_total_path_residual_m": float(np.median(np.abs(raw_resid))),
        "weighted_rms": float(np.sqrt(np.mean((raw_resid / sigma) ** 2.0))),
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "optimizer_cost": float(result.cost),
    }


def plot_weighted_ballistic_2x2(times_ns, measured_total_paths_m, sigma_model, weighted_fit):
    os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)
    t_s = (np.asarray(times_ns, dtype=np.float64) - float(times_ns[0])) / 1e9
    measured_gcrs_m = ecef_positions_to_gcrs(triangulated_points_ecef(measured_total_paths_m), times_ns)
    model_gcrs_m = itrs_state_to_gcrs_positions(weighted_fit["x_itrs_m"], weighted_fit["v_itrs_mps"], times_ns)
    center_m = np.nanmean(measured_gcrs_m, axis=0)
    measured_km = (measured_gcrs_m - center_m[None, :]) / 1e3
    model_km = (model_gcrs_m - center_m[None, :]) / 1e3
    per_pulse_rms_m = np.sqrt(np.mean(weighted_fit["residuals_m"] ** 2.0, axis=1))

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    component_labels = ("GCRS x - mean (km)", "GCRS y - mean (km)", "GCRS z - mean (km)")
    for comp, ax in enumerate(axes.flat[:3]):
        ax.plot(t_s, measured_km[:, comp], "k.", ms=4, label="triangulated seed")
        ax.plot(t_s, model_km[:, comp], color="#2ca02c", lw=1.8, label="weighted ballistic fit")
        ax.set_xlabel("Time since first matched pulse (s)")
        ax.set_ylabel(component_labels[comp])
        ax.grid(True, alpha=0.28)
        if comp == 0:
            ax.legend(loc="best")

    ax = axes.flat[3]
    ax.plot(t_s, per_pulse_rms_m, "o-", ms=3.5, color="#2ca02c")
    ax.axhline(weighted_fit["rms_total_path_residual_m"], color="0.25", lw=1.2, ls="--", label="overall RMS")
    ax.set_xlabel("Time since first matched pulse (s)")
    ax.set_ylabel("Per-pulse total-path RMS (m)")
    ax.set_title(
        f"RMS={weighted_fit['rms_total_path_residual_m']:.2f} m, "
        f"B={weighted_fit['b_drag_m2_per_kg']:.2f} m$^2$ kg$^{{-1}}$"
    )
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")

    fig.suptitle(
        "Rank02 weighted ballistic trajectory fit "
        f"({UPSAMPLE_FACTOR}x interpolation; "
        f"$\\sigma=\\sqrt{{{sigma_model['sigma_floor_m']:.1f}^2 + "
        f"({sigma_model['sigma_0_m']:.0f}/10^{{\\mathrm{{SNR}}/20}})^2}}$ m)"
    )
    fig.savefig(f"{OUTPUT_BASE}.png", dpi=220)
    fig.savefig(f"{OUTPUT_BASE}.pdf")
    plt.close(fig)


def json_summary(fit):
    return {
        "rms_total_path_residual_m": fit["rms_total_path_residual_m"],
        "median_abs_total_path_residual_m": fit["median_abs_total_path_residual_m"],
        "weighted_rms": fit.get("weighted_rms"),
        "b_drag_m2_per_kg": fit.get("b_drag_m2_per_kg"),
        "start_speed_km_s": float(np.asarray(fit["speed_km_s"]).ravel()[0]),
        "end_speed_km_s": float(np.asarray(fit["speed_km_s"]).ravel()[-1]),
        "start_alt_km": float(np.asarray(fit["alt_km"]).ravel()[0]) if "alt_km" in fit else fit.get("start_alt_km"),
        "end_alt_km": float(np.asarray(fit["alt_km"]).ravel()[-1]) if "alt_km" in fit else fit.get("end_alt_km"),
        "optimizer_success": fit["optimizer_success"],
        "optimizer_nfev": fit["optimizer_nfev"],
    }


def main():
    data = build_measurements()
    if not interp.is_root():
        return
    site_data, measured, times_ns, snr_db = data
    rho_of_alt_m, msis_meta = make_density_interpolator(times_ns, measured)

    p0 = initial_ballistic_guess(measured, times_ns, log10_b=np.log10(40.0))
    # First pass estimates the SNR-dependent measurement scatter. The reported
    # fit below is the weighted ballistic fit only.
    first_pass_fit = fit_ballistic(measured, times_ns, rho_of_alt_m, p0, sigma_m=None)
    sigma_model = fit_sigma_model(first_pass_fit["residuals_m"], snr_db)
    sigma_m = sigma_from_snr_db(snr_db, sigma_model["sigma_floor_m"], sigma_model["sigma_0_m"])
    weighted_fit = fit_ballistic(measured, times_ns, rho_of_alt_m, first_pass_fit["params"], sigma_m=sigma_m)

    plot_weighted_ballistic_2x2(times_ns, measured, sigma_model, weighted_fit)
    out = {
        "script": os.path.basename(__file__),
        "script_version": SCRIPT_VERSION,
        "upsample_factor": UPSAMPLE_FACTOR,
        "msis": msis_meta,
        "sigma_model": sigma_model,
        "first_pass_ballistic_for_sigma": json_summary(first_pass_fit),
        "ballistic_snr_weighted": json_summary(weighted_fit),
    }
    with open(f"{OUTPUT_BASE}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"weighted ballistic RMS={weighted_fit['rms_total_path_residual_m']:.2f} m, B={weighted_fit['b_drag_m2_per_kg']:.4g} m^2/kg")
    print(
        "sigma_path(SNR) = sqrt("
        f"{sigma_model['sigma_floor_m']:.2f}^2 + ({sigma_model['sigma_0_m']:.2f}/10^(SNR_dB/20))^2) m"
    )
    print(f"wrote {OUTPUT_BASE}.json")
    print(f"wrote {OUTPUT_BASE}.png")
    print(f"wrote {OUTPUT_BASE}.pdf")


if __name__ == "__main__":
    main()
