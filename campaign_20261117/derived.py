"""Per-event RCS diagnostics and backward heliocentric propagation.

Uses the existing scattering table; never recomputes FDTD. The provisional
receiver-temperature and atmosphere scenarios are saved, not hidden.
"""
import argparse
import concurrent.futures
from pathlib import Path
import traceback
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from analyze import C, FS, WAVELENGTH, STATIONS, ENU, save

RATE_FACTOR = 4*np.sqrt(2*np.pi)*951.*3.8e-26
SANYA_AZ_EL = (14.996337890625, 74.9981689453125)


def beam_frame(az=SANYA_AZ_EL[0], el=SANYA_AZ_EL[1]):
    a, e = np.deg2rad([az, el])
    axis = np.array([np.sin(a)*np.cos(e), np.cos(a)*np.cos(e), np.sin(e)])
    east = np.array([1., 0., 0.])-axis[0]*axis
    east /= np.linalg.norm(east)
    north = np.cross(axis, east)
    return axis, east, north


def beam_power(direction, axis):
    """Horizontal east/north aperture, evaluated in ENU direction cosines."""
    delta = np.asarray(direction)-axis
    return np.sinc(40/WAVELENGTH*delta[..., 0])**2 * np.sinc(40/WAVELENGTH*delta[..., 1])**2


def directions_from_offsets(east_deg, north_deg, az=SANYA_AZ_EL[0], el=SANYA_AZ_EL[1]):
    axis, east, north = beam_frame(az, el)
    direction = axis + np.tan(np.deg2rad(east_deg))[..., None]*east + np.tan(np.deg2rad(north_deg))[..., None]*north
    return direction/np.linalg.norm(direction, axis=-1, keepdims=True)


def angular_offsets(direction, az=SANYA_AZ_EL[0], el=SANYA_AZ_EL[1]):
    axis, east, north = beam_frame(az, el)
    denominator = np.asarray(direction) @ axis
    if np.any(denominator <= 0):
        raise ValueError("beam offsets require directions in the forward hemisphere")
    return (np.rad2deg(np.arctan2(np.asarray(direction) @ east, denominator)),
            np.rad2deg(np.arctan2(np.asarray(direction) @ north, denominator)))


def beam(points, az=SANYA_AZ_EL[0], el=SANYA_AZ_EL[1]):
    axis, east, north = beam_frame(az, el)
    e = np.deg2rad(el)
    relative = (points-STATIONS[0]) @ ENU.T
    distance = np.linalg.norm(relative, axis=1)
    direction = relative/distance[:, None]
    power = beam_power(direction, axis)
    gain_db = 10*np.log10(np.maximum(power, 1e-30))
    gain = 10**((46 + 25*np.log10(np.sin(e)) + gain_db)/10)
    return gain_db, gain, distance, relative @ east, relative @ north


def source_root(curve, power, source):
    roots = []
    for j in range(len(source)-1):
        if not np.isfinite(curve[j:j+2]).all() or curve[j] == curve[j+1]:
            continue
        w = (power-curve[j])/(curve[j+1]-curve[j])
        if 0 <= w <= 1 and not (j and w == 0):
            roots.append(source[j] + w*(source[j+1]-source[j]))
    return roots[0] if len(roots) == 1 else np.nan


def orbit_product(folder, state, covariance, epoch_ns):
    import rebound
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time
    from astropy import units as u
    if np.dot(state[3:],state[3:])/2 - 3.986004418e14/np.linalg.norm(state[:3]) <= 0:
        raise ValueError("fitted geocentric state is not an incoming escaping meteoroid orbit")
    time = Time(np.longdouble(epoch_ns)/1e9, format="unix", scale="utc")
    obliquity = np.deg2rad(23.4392911)
    rotation = np.array([[1, 0, 0], [0, np.cos(obliquity), np.sin(obliquity)],
                         [0, -np.sin(obliquity), np.cos(obliquity)]])
    sim = rebound.Simulation()
    sim.units = ("AU", "day", "Msun")
    masses = (1., 1.6601e-7, 2.4478e-6, 3.0035e-6, 3.2272e-7,
              9.5479e-4, 2.8589e-4, 4.3662e-5, 5.1514e-5)
    names = ("sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune")
    earth = None
    for name, mass in zip(names, masses):
        p, v = get_body_barycentric_posvel(name, time)
        xyz = rotation @ p.xyz.to_value(u.au)
        vel = rotation @ v.xyz.to_value(u.au/u.day)
        sim.add(m=mass, x=xyz[0], y=xyz[1], z=xyz[2], vx=vel[0], vy=vel[1], vz=vel[2])
        if name == "earth":
            earth = np.r_[xyz, vel]
    sim.N_active = len(names)
    rng = np.random.default_rng(int(epoch_ns) % (2**32))
    cov = covariance[:6, :6]*1e6
    eigen, vectors = np.linalg.eigh(cov)
    factor = vectors @ np.diag(np.sqrt(np.maximum(eigen, 0)))
    draws = np.vstack([state, state + rng.normal(size=(63, 6)) @ factor.T])
    for draw in draws:
        p = earth[:3] + rotation @ draw[:3]/u.au.to(u.m)
        v = earth[3:] + rotation @ draw[3:]*86400/u.au.to(u.m)
        sim.add(m=0., x=p[0], y=p[1], z=p[2], vx=v[0], vy=v[1], vz=v[2])
    sim.move_to_com()
    sim.integrator = "ias15"
    sim.integrate(-1000., exact_finish_time=1)
    fields = {k: [] for k in ("a_au", "e", "i_deg", "q_au", "Q_au", "specific_energy_j_kg")}
    for particle in list(sim.particles)[len(names):]:
        o = particle.orbit(primary=sim.particles[0])
        for key, value in zip(fields, (o.a, o.e, np.rad2deg(o.inc), o.a*(1-o.e), o.a*(1+o.e),
                                      -1.32712440018e20/(2*o.a*u.au.to(u.m)))):
            fields[key].append(value)
    save(folder / "orbits.h5", fields, {"propagation_days": -1000.,
        "integrator": "REBOUND IAS15; Sun and eight planets; massless meteor ensemble",
        "uncertainty": "64 local-curvature draws; conditional, not validated posterior",
        "status": "provisional hardware timing and reference-atmosphere state"})


def one(task):
    folder, lookup = task
    try:
        with h5py.File(folder / "physical.h5") as h:
            d = {k: h[k][()] for k in h}
        with h5py.File(folder / "input.h5") as h:
            noise = h["noise_power"][:]
        keep = d["station"] == 0
        gain_db, gain, distance, east, north = beam(d["position_ecef_m"][keep])
        received = d["modeled_amplitude_power_adc2"][keep]/noise[keep] * 1.380649e-23*120.*FS
        rcs = received*(4*np.pi)**3 * distance**4 / (4.7e6*gain**2*WAVELENGTH**2)
        rcs_db = 10*np.log10(rcs)
        with h5py.File(lookup) as h:
            height, velocity = h["altitude_km"][:], h["velocity_km_s"][:]
            source = np.log10(h["source_n0_rM2_m_inv"][:])
            values = h["rcs_dbsm"][:]
        interp = RegularGridInterpolator((height, velocity), values, bounds_error=False, fill_value=np.nan)
        curves = interp(np.c_[d["height_km"][keep], d["speed_mps"][keep]/1000])
        roots = np.array([source_root(c, r, source) for c, r in zip(curves, rcs_db)])
        rate = RATE_FACTOR*10**roots
        times = d["time_s"][keep]
        valid = (gain_db >= -10) & np.isfinite(rate)
        pairs = valid[:-1] & valid[1:] & (np.diff(times) <= .050001)
        mass = np.sum(np.diff(times)[pairs]*(rate[:-1][pairs]+rate[1:][pairs])/2) if pairs.any() else np.nan
        save(folder / "rcs.h5", dict(time_s=times, time_ns=d["time_ns"][keep],
            rcs_dbsm=rcs_db, relative_gain_db=gain_db, height_km=d["height_km"][keep],
            speed_km_s=d["speed_mps"][keep]/1000, east_m=east, north_m=north,
            inferred_source_log10_m_inv=roots, inferred_mass_loss_rate_kg_s=rate,
            source_in_domain=np.isfinite(roots), in_beam=valid, integrated_mass_loss_kg=mass),
            {"system_temperature_k": 120., "calibration": "reference temperature, not November sky-noise calibration",
             "beam": "40 m square aperture; 46 dBi normal gain with cos^2.5 scan loss",
             "lookup": str(lookup), "mass_rate_factor": RATE_FACTOR,
             "inversion": "unique within-table root only; no boundary clipping or extrapolation"})
        state = np.r_[d["initial_gcrs_position_m"], d["initial_gcrs_velocity_mps"]]
        orbit_product(folder, state, d["covariance"], int(d["time_ns"][0]))
        return folder.name, "derived"
    except Exception:
        (folder / "derived_error.log").write_text(traceback.format_exc())
        return folder.name, "failed"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("/gpfs/pub/data_other/juha/20261117_tristatic"))
    p.add_argument("--lookup", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    folders = sorted(p.parent for p in (a.output/"events").glob("tri_*/physical.h5"))
    if a.limit:
        folders = folders[:a.limit]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(one, [(f, a.lookup) for f in folders], chunksize=1))
    save(a.output/"derived_status.h5", {"event": np.array([r[0] for r in results], dtype="S32"),
        "status": np.array([r[1] for r in results], dtype="S16")})
    print("DERIVED", {s:sum(r[1]==s for r in results) for s in ("derived", "failed")}, flush=True)


if __name__ == "__main__":
    main()
