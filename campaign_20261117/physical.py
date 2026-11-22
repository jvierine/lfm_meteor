"""Physical extensions of the November voltage fits; Slurm workers only."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import traceback
import time
import h5py
import numpy as np
from scipy.optimize import least_squares
from analyze import Native, save, STATIONS, C, GAMMA, WAVELENGTH
from fit_numerics import optimize, profile_interval
from quality import path_quality

FIT_VERSION = "three-station-four-ipp-profile-v1"
ROWS = ('time_s','time_ns','station','frames','rx','real','imag','noise_power',
        'sample_time_s','whitener_real','whitener_imag')


def mass_fit_mask(position):
    from derived import beam, beam_frame
    from analyze import ENU
    relative = (position-STATIONS[0]) @ ENU.T
    direction = relative/np.linalg.norm(relative, axis=1)[:, None]
    axis = beam_frame()[0]
    aperture_offset = 40/WAVELENGTH*(direction-axis)
    gain = beam(position)[0]
    mainlobe = (direction @ axis > 0) & np.all(np.abs(aperture_offset[:, :2]) < 1, axis=1)
    return mainlobe & np.isfinite(gain) & (gain >= -10.), gain


def three_station_mask(times, station, window_s=.020, tolerance_s=.0001):
    """Union of windows spanning at most four IPPs and containing all sites."""
    times=np.asarray(times);station=np.asarray(station)
    order=np.argsort(times);t=times[order];s=station[order]
    changes=np.zeros(len(t)+1,dtype=int)
    for left in range(len(t)):
        right=int(np.searchsorted(t,t[left]+window_s+tolerance_s,side='right'))
        if all(np.any(s[left:right]==site) for site in range(3)):
            changes[left]+=1;changes[right]-=1
    result=np.zeros(len(t),dtype=bool)
    result[order]=np.cumsum(changes[:-1])>0
    return result


def one(task):
    folder, library = task
    target = folder / "physical.h5"
    previous=None
    if target.exists():
        with h5py.File(target) as h:
            if h.attrs.get("fit_method_version","")==FIT_VERSION and h.attrs.get('path_quality_limit_m',0)==40.:
                return folder.name,"cached"
            previous=h["params"][:]
    try:
        import pymsis
        from astropy.coordinates import EarthLocation
        from astropy import units as u
        with h5py.File(folder / "input.h5") as h:
            d = {k: h[k][()] for k in h}
        with h5py.File(folder / "fit.h5") as h:
            seed = h["params"][:]
            position = h["position_ecef_m"][:]
            observed_path=h["apparent_path_m"][:]
        _, selection_gain = mass_fit_mask(position)
        mask = three_station_mask(d['time_s'],d['station'])
        save(folder / "dynamic_mass_selection.h5",
             dict(retained=mask, one_way_gain_db=selection_gain, time_ns=d['time_ns'], station=d['station']),
             dict(window_s=.020, timing_tolerance_s=.0001,
                  geometry='all three stations detected within a four-IPP window; no beam-gain cut',
                  fit_method_version=FIT_VERSION))
        if any(np.count_nonzero(mask & (d['station']==s)) < 6 for s in range(3)):
            # Keep the existing trajectory, but never retain its old mass selection.
            if target.exists():
                import shutil
                backup=folder/'superseded_mass'/str(time.time_ns())
                backup.mkdir(parents=True)
                shutil.copy2(target, backup/'physical.h5')
                with h5py.File(target, 'r+') as h:
                    for key, value in [('dynamic_mass_fit_mask', mask), ('mass_fit_one_way_gain_db', selection_gain)]:
                        if key in h: del h[key]
                        h[key]=value
                    h.attrs['mass_resolved_profile']=False
                    h.attrs['mass_resolved_local_curvature']=False
                    h.attrs['dynamic_mass_fit_status']='insufficient three-station four-IPP coverage; previous trajectory retained'
                    h.attrs['fit_method_version']=FIT_VERSION
            return folder.name, 'insufficient'
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
        selected = {k: (v[mask] if k in ROWS else v) for k,v in d.items()}
        save(model_dir / 'three_station_input.h5', selected, {'window_s': .020})
        model = Native(library, model_dir / "three_station_input.h5", int(mask.sum()))
        lo, hi = np.r_[[-30.]*3, [-50.]*3, -7.], np.r_[[30.]*3, [50.]*3, -2.]
        candidates = []
        if previous is not None:candidates.append(optimize(model.residual,previous,(lo,hi)))
        for radius in (-5., -4., -3.):
            q = np.r_[seed[:6], radius]
            f = optimize(model.residual,q,(lo,hi))
            candidates.append(f)
        fit = min(candidates, key=lambda f: f.cost)
        if not np.isfinite(fit.fun).all() or np.max(np.abs(fit.fun)) >= 1e9:
            raise ValueError("no valid physical trajectory among radius starts")
        q = fit.x
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac, rcond=1e-13)
        radius = 10**q[6]
        mass = 4*np.pi/3 * 3000 * radius**3
        profiles={float(q[6]):(fit.cost,q[:6].copy())}
        def cost_at(radius):
            if radius not in profiles:
                nearest=min(profiles,key=lambda x:abs(x-radius))
                fun=lambda x:model.residual(np.r_[x,radius])
                trial=optimize(fun,profiles[nearest][1],(lo[:6],hi[:6]),max_nfev=140)
                profiles[radius]=(trial.cost,trial.x)
            return profiles[radius][0]
        interval_log=profile_interval(cost_at,q[6],fit.cost)
        interval=10**interval_log
        # A singular or noisy Hessian cannot certify a measured radius.
        resolved=bool(fit.success and np.isfinite(interval_log).all() and
            interval_log[0]<q[6]<interval_log[1] and
            interval_log[1]-interval_log[0]<1/3 and
            min(v[0] for v in profiles.values())>=fit.cost-.1)
        model.close()
        # Full-arc predictions remain available; only the masked voltages were fitted.
        model = Native(library, model_dir / 'input.h5', len(d['time_s']))
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
        path_residual=observed_path-observable[:,0]-C*observable[:,1]/GAMMA
        quality_pass,max_residual,bad_count=path_quality(path_residual,mask)
        interval_resolved=resolved
        resolved=resolved and quality_pass
        updated=folder/"physical-refined.h5"
        save(updated, dict(params=q, covariance=covariance, initial_radius_m=radius,
            dynamic_mass_fit_mask=mask, mass_fit_one_way_gain_db=selection_gain,
            initial_mass_kg=mass, radius_local95_m=interval,
            mass_local95_kg=4*np.pi/3*3000*interval**3,
            time_ns=d["time_ns"], time_s=d["time_s"], station=d["station"],
            speed_mps=states[:, 1], position_ecef_m=x, height_km=location.height.to_value(u.km),
            latitude_deg=location.lat.deg, longitude_deg=location.lon.deg,
            physical_path_m=observable[:, 0], physical_doppler_hz=observable[:, 1],
            model_apparent_path_m=observable[:,0]+C*observable[:,1]/GAMMA,
            path_residual_m=path_residual,
            multistart_params=np.array([f.x for f in candidates]),
            multistart_cost=np.array([f.cost for f in candidates]),
            profile_log10_radius=np.array(sorted(profiles)),
            profile_cost=np.array([profiles[r][0] for r in sorted(profiles)]),
            optimizer_optimality=fit.optimality,
            modeled_amplitude_power_adc2=amplitude_power,
            initial_gcrs_position_m=d["center"][:3]+1000*q[:3],
            initial_gcrs_velocity_mps=d["center"][3:]+1000*q[3:6]),
            {"mass_resolved_local_curvature": resolved, "mass_resolved_profile":resolved,
             "mass_interval_resolved_profile":interval_resolved,
             "path_quality_pass":quality_pass,"path_quality_limit_m":40.,
             "max_used_path_residual_m":max_residual,"bad_used_path_measurements":bad_count,
             "fit_method_version":FIT_VERSION,"voltage_cost": fit.cost,
             "dynamic_mass_fit_status":("fitted where all three sites detect within four IPPs" if quality_pass else "rejected: fitted path residual reaches or exceeds 40 m; review required"),
             "mass_selection_geometry":"20 ms window plus 100 us timing tolerance; no beam-gain cut; original epoch preserved",
             "status": "reference atmosphere; common calibrated delay",
             "interval_method": "conditional thermal-noise profile likelihood, Delta chi2=3.84; no total-error coverage claim",
             "solar_parameters": "F107=F107a=150 sfu; Ap=4 reference scenario",
             "density_kg_m3": 3000., "Gamma": 1., "sigma_s2_m2": 1e-8})
        if target.exists():
            backup=folder/"superseded_mass"/str(time.time_ns());backup.mkdir(parents=True)
            for name in ("physical.h5","rcs.h5","orbits.h5"):
                old=folder/name
                if old.exists():os.replace(old,backup/name)
        os.replace(updated,target)
        model.close()
        return folder.name, "physical"
    except Exception:
        (folder / "physical_error.log").write_text(traceback.format_exc())
        if target.exists():
            import shutil
            backup=folder/'superseded_mass'/str(time.time_ns())
            backup.mkdir(parents=True)
            shutil.copy2(target, backup/'physical.h5')
            with h5py.File(target, 'r+') as h:
                h.attrs['mass_resolved_profile']=False
                h.attrs['mass_resolved_local_curvature']=False
                h.attrs['dynamic_mass_fit_status']='main-lobe refit failed; previous trajectory retained'
        return folder.name, "failed"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("/gpfs/pub/data_other/juha/20261117_tristatic"))
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    folders = sorted(x.parent for x in (a.output / "events").glob("tri_*/fit.h5"))
    if (a.output/"association_selection.h5").exists():
        with h5py.File(a.output/"association_selection.h5") as h:
            accepted={n.decode() for n,k in zip(h["event_id"][:],h["accepted"][:]) if k}
        folders=[f for f in folders if f.name in accepted]
    if a.limit:
        folders = folders[:a.limit]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        result = list(pool.map(one, [(f, a.library.resolve()) for f in folders], chunksize=1))
    save(a.output / "physical_status.h5", {"event": np.array([r[0] for r in result], dtype="S32"),
         "status": np.array([r[1] for r in result], dtype="S16")})
    print("PHYSICAL", {s: sum(r[1] == s for r in result) for s in ("physical", "cached", "insufficient", "failed")}, flush=True)


if __name__ == "__main__":
    main()
