"""Physical extensions of the November voltage fits; Slurm workers only."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import traceback
import h5py
import numpy as np
from scipy.optimize import least_squares
from analyze import Native, save, STATIONS, C, GAMMA, WAVELENGTH


def one(task):
    folder, library = task
    target = folder / "physical.h5"
    if target.exists():
        return folder.name, "cached"
    try:
        import pymsis
        from astropy.coordinates import EarthLocation
        from astropy import units as u
        with h5py.File(folder / "input.h5") as h:
            d = {k: h[k][()] for k in h}
        with h5py.File(folder / "fit.h5") as h:
            seed = h["params"][:]
            position = h["position_ecef_m"][:]
        origin = EarthLocation.from_geocentric(*position[0], unit=u.m)
        height = np.arange(60., 181., 1.)
        dates = np.repeat(np.datetime64(int(d["time_ns"][0]), "ns"), len(height))
        # Explicit reference atmosphere: no claim of measured solar conditions.
        atmosphere = pymsis.calculate(dates, np.repeat(origin.lon.deg, len(height)),
            np.repeat(origin.lat.deg, len(height)), height,
            f107s=np.full(len(height), 150.), f107as=np.full(len(height), 150.),
            aps=np.full((len(height), 7), 4.))
        d.update(height_m=height*1000, density_kg_m3=atmosphere[:, 0].astype(float))
        model_dir = folder / "drag_model"
        save(model_dir / "input.h5", d, {"density_model": "NRLMSIS2.1 reference F107=150, F107a=150, Ap=4",
             "material_density_kg_m3": 3000., "Gamma": 1., "sigma_s2_m2": 1e-8})
        model = Native(library, model_dir / "input.h5", len(d["time_s"]))
        lo, hi = np.r_[[-30.]*3, [-50.]*3, -7.], np.r_[[30.]*3, [50.]*3, -2.]
        candidates = []
        for radius in (-5., -4., -3.):
            q = np.r_[seed[:6], radius]
            f = least_squares(model.residual, q, bounds=(lo, hi), x_scale="jac",
                              max_nfev=160, ftol=1e-8, xtol=1e-8, gtol=1e-6)
            candidates.append(f)
        fit = min(candidates, key=lambda f: f.cost)
        if not np.isfinite(fit.fun).all() or np.max(np.abs(fit.fun)) >= 1e9:
            raise ValueError("no valid physical trajectory among radius starts")
        q = fit.x
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac, rcond=1e-13)
        radius = 10**q[6]
        mass = 4*np.pi/3 * 3000 * radius**3
        width = 1.96*np.sqrt(max(covariance[6, 6], 0))
        interval = 10**(q[6] + np.array([-width, width]))
        # Local curvature is explicitly not a validated posterior or total error.
        resolved = bool(fit.success and -6.9 < q[6]-width and q[6]+width < -2.1 and
                        0.005 < width < 1/6)
        states = model.call("voltage_states", q, (len(d["time_s"]), 4))
        observable = model.call("voltage_observables", q, (len(d["time_s"]), 2))
        predicted = model.call("voltage_prediction", q, (len(d["time_s"]), d["real"].shape[1], 2))
        speed = np.linalg.norm(d["center"][3:] + 1000*q[3:6])
        direction = (d["center"][3:] + 1000*q[3:6])/speed
        gcrs = d["center"][:3] + 1000*q[:3] + states[:, :1]*direction
        frame = d["frames"]
        x = frame[:, :3] + np.einsum("nij,nj->ni", frame[:, 6:15].reshape(-1, 3, 3), gcrs)
        location = EarthLocation.from_geocentric(x[:, 0]*u.m, x[:, 1]*u.m, x[:, 2]*u.m)
        # The fitted waveform includes its complex coefficient; peak modeled
        # power gives |a|^2 for the unit-amplitude transmitted waveform.
        amplitude_power = np.max(predicted[:, :, 0]**2 + predicted[:, :, 1]**2, axis=1)
        save(target, dict(params=q, covariance=covariance, initial_radius_m=radius,
            initial_mass_kg=mass, radius_local95_m=interval,
            mass_local95_kg=4*np.pi/3*3000*interval**3,
            time_ns=d["time_ns"], time_s=d["time_s"], station=d["station"],
            speed_mps=states[:, 1], position_ecef_m=x, height_km=location.height.to_value(u.km),
            latitude_deg=location.lat.deg, longitude_deg=location.lon.deg,
            physical_path_m=observable[:, 0], physical_doppler_hz=observable[:, 1],
            modeled_amplitude_power_adc2=amplitude_power,
            initial_gcrs_position_m=d["center"][:3]+1000*q[:3],
            initial_gcrs_velocity_mps=d["center"][3:]+1000*q[3:6]),
            {"mass_resolved_local_curvature": resolved, "voltage_cost": fit.cost,
             "status": "provisional: reference atmosphere and uncalibrated hardware delays",
             "interval_method": "local Gaussian curvature only; no coverage validation",
             "solar_parameters": "F107=F107a=150 sfu; Ap=4 reference scenario",
             "density_kg_m3": 3000., "Gamma": 1., "sigma_s2_m2": 1e-8})
        model.close()
        return folder.name, "physical"
    except Exception:
        (folder / "physical_error.log").write_text(traceback.format_exc())
        return folder.name, "failed"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("/gpfs/pub/data_other/juha/20261117_tristatic"))
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    folders = sorted(x.parent for x in (a.output / "events").glob("tri_*/fit.h5"))
    if a.limit:
        folders = folders[:a.limit]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        result = list(pool.map(one, [(f, a.library.resolve()) for f in folders], chunksize=1))
    save(a.output / "physical_status.h5", {"event": np.array([r[0] for r in result], dtype="S32"),
         "status": np.array([r[1] for r in result], dtype="S16")})
    print("PHYSICAL", {s: sum(r[1] == s for r in result) for s in ("physical", "cached", "failed")}, flush=True)


if __name__ == "__main__":
    main()
