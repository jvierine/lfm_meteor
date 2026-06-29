"""Sample Ceplecha shrinking-radius fits from varied starting points.

This is a diagnostic for local minima and practical parameter uncertainty.  It
refits one event many times with perturbed initial position, velocity, and
radius guesses, then reports the spread of the converged solutions.
"""

import argparse
import os

import h5py
import numpy as np
import scipy.optimize as so

import fit_all_ballistic_snr_weighted as base_fit
import fit_all_ceplecha_snr_weighted as ceplecha_fit
import plot_bayesian_model_selection_example as event_fit


DEFAULT_EVENTS = (
    "tri_0093_1713816477464351654",
    "tri_0102_1713818109279350281",
    "tri_0151_1713822392884349823",
    "tri_0182_1713824949124349594",
    "tri_0035_1713807353024349213",
)
DEFAULT_OUTPUT_H5 = "results/ceplecha_multistart_stability.h5"


def prepare_event(event_id):
    event = event_fit.refine_observation_keep(event_fit.read_event(event_id))
    measured = event["measured"]
    times = event["time_ns"]
    sigma = event["sigma"]
    keep_obs = event["keep_obs"]
    keep_rows = np.any(keep_obs, axis=1)
    rho_of_alt_m, _meta = base_fit.density_interpolator(times, event["x_itrs_initial"])
    times_fit = times[keep_rows]
    measured_fit = measured[keep_rows]
    sigma_fit = sigma[keep_rows]
    keep_obs_fit = keep_obs[keep_rows]
    t_rel_s = (times_fit.astype(np.float64) - float(times[0])) / 1e9

    def residual(params):
        predicted, *_rest = ceplecha_fit.predict_paths(params, t_rel_s, times_fit, rho_of_alt_m)
        return ((predicted - measured_fit) / sigma_fit)[keep_obs_fit]

    return event, residual


def fit_from_start(p0, residual, max_nfev=120):
    lower = np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(ceplecha_fit.MIN_RADIUS_M)])
    upper = np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(ceplecha_fit.MAX_RADIUS_M)])
    result = so.least_squares(
        residual,
        np.clip(np.asarray(p0, dtype=np.float64), lower, upper),
        bounds=(lower, upper),
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0]),
        loss=ceplecha_fit.ROBUST_LOSS,
        f_scale=ceplecha_fit.ROBUST_F_SCALE,
        max_nfev=max_nfev,
    )
    r = residual(result.x)
    radius_m = 10.0 ** float(result.x[6])
    mass_kg = (4.0 / 3.0) * np.pi * ceplecha_fit.METEOROID_DENSITY_KG_M3 * radius_m**3
    return {
        "params": result.x,
        "chi2": float(np.sum(r**2.0)),
        "cost": float(result.cost),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "radius_m": float(radius_m),
        "mass_kg": float(mass_kg),
        "speed_km_s": float(np.linalg.norm(result.x[3:6]) / 1e3),
    }


def starting_points(seed_params, rng, n_random):
    starts = [np.asarray(seed_params, dtype=np.float64).copy()]
    radius_grid_um = np.array([0.5, 1.0, 3.0, 7.0, 15.0, 30.0, 80.0, 200.0], dtype=np.float64)
    for radius_um in radius_grid_um:
        p = np.asarray(seed_params, dtype=np.float64).copy()
        p[6] = np.log10(radius_um * 1e-6)
        starts.append(p)
    for _idx in range(n_random):
        p = np.asarray(seed_params, dtype=np.float64).copy()
        p[:3] += rng.normal(0.0, 150.0, size=3)
        p[3:6] += rng.normal(0.0, 750.0, size=3)
        p[6] += rng.normal(0.0, 0.55)
        p[6] = np.clip(p[6], np.log10(ceplecha_fit.MIN_RADIUS_M), np.log10(ceplecha_fit.MAX_RADIUS_M))
        starts.append(p)
    return starts


def cluster_solutions(rows, chi2_tol):
    best = min(row["chi2"] for row in rows)
    near = [row for row in rows if row["chi2"] <= best + chi2_tol and row["success"]]
    radii_um = np.asarray([row["radius_m"] * 1e6 for row in near], dtype=np.float64)
    masses = np.asarray([row["mass_kg"] for row in near], dtype=np.float64)
    speeds = np.asarray([row["speed_km_s"] for row in near], dtype=np.float64)
    return best, near, radii_um, masses, speeds


def summarize(event_id, rows, chi2_tol):
    best, near, radii_um, masses, speeds = cluster_solutions(rows, chi2_tol)
    print(f"\n{event_id}")
    print(f"  fits={len(rows)} success={sum(row['success'] for row in rows)} best_chi2={best:.6g}")
    print(f"  near_best_delta_chi2<={chi2_tol:g}: {len(near)}")
    if len(near):
        print(
            "  radius_um near-best: "
            f"min={np.nanmin(radii_um):.6g} median={np.nanmedian(radii_um):.6g} max={np.nanmax(radii_um):.6g}"
        )
        print(
            "  mass_kg near-best: "
            f"min={np.nanmin(masses):.6g} median={np.nanmedian(masses):.6g} max={np.nanmax(masses):.6g}"
        )
        print(
            "  speed_km_s near-best: "
            f"min={np.nanmin(speeds):.6g} median={np.nanmedian(speeds):.6g} max={np.nanmax(speeds):.6g}"
        )
    by_chi2 = sorted(rows, key=lambda row: row["chi2"])[:8]
    print("  best solutions:")
    for idx, row in enumerate(by_chi2, start=1):
        print(
            f"    {idx:02d} chi2={row['chi2']:.6g} "
            f"r={row['radius_m'] * 1e6:.6g} um "
            f"m={row['mass_kg']:.6g} kg "
            f"v={row['speed_km_s']:.6g} km/s "
            f"nfev={row['nfev']} success={row['success']}"
        )


def write_h5(path, all_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as h:
        for event_id, rows in all_rows.items():
            group = h.create_group(event_id)
            group["params"] = np.asarray([row["params"] for row in rows], dtype=np.float64)
            group["chi2"] = np.asarray([row["chi2"] for row in rows], dtype=np.float64)
            group["cost"] = np.asarray([row["cost"] for row in rows], dtype=np.float64)
            group["radius_m"] = np.asarray([row["radius_m"] for row in rows], dtype=np.float64)
            group["mass_kg"] = np.asarray([row["mass_kg"] for row in rows], dtype=np.float64)
            group["speed_km_s"] = np.asarray([row["speed_km_s"] for row in rows], dtype=np.float64)
            group["success"] = np.asarray([row["success"] for row in rows], dtype=bool)
            group["nfev"] = np.asarray([row["nfev"] for row in rows], dtype=np.int32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", action="append", default=None, help="Event id to sample. May be repeated.")
    parser.add_argument("--n-random", type=int, default=32)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chi2-tol", type=float, default=1.0)
    parser.add_argument("--output-h5", default=DEFAULT_OUTPUT_H5)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    event_ids = args.event_id if args.event_id else list(DEFAULT_EVENTS)
    all_rows = {}
    for event_id in event_ids:
        event, residual = prepare_event(event_id)
        starts = starting_points(event["p0_shrinking"], rng, args.n_random)
        rows = [fit_from_start(p0, residual, max_nfev=args.max_nfev) for p0 in starts]
        all_rows[event_id] = rows
        summarize(event_id, rows, args.chi2_tol)
    write_h5(args.output_h5, all_rows)
    print(f"\nwrote {args.output_h5}")


if __name__ == "__main__":
    main()
