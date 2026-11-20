"""November station cuts -> independent, provisional tri-static products.

All numeric arrays and processing provenance are stored in HDF5. No plotting
backend is imported here; Rustplotlib renders the scientific products.
"""
import argparse
import concurrent.futures
import ctypes as ct
import hashlib
import os
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
from scipy.optimize import least_squares

C = 299792458.0
FS = 4e6
B = 4e6
T = 200e-6
GAMMA = B / T
WAVELENGTH = C / 430e6
SITES = ("sanya", "danzhou", "wenchang")
LLH = np.array([[18.3492, 109.6222, 50.0], [19.5281, 109.1322, 99.9],
                [19.5982, 110.7908, 24.9]])
PTR = ct.POINTER(ct.c_double)


def ecef(llh):
    lat, lon = np.deg2rad(llh[:2])
    n = 6378137.0 / np.sqrt(1 - 6.69437999014e-3 * np.sin(lat)**2)
    return np.array([(n + llh[2]) * np.cos(lat) * np.cos(lon),
                     (n + llh[2]) * np.cos(lat) * np.sin(lon),
                     (n * (1 - 6.69437999014e-3) + llh[2]) * np.sin(lat)])


STATIONS = np.array([ecef(x) for x in LLH])


def enu_matrix(lat, lon):
    a, b = np.deg2rad([lat, lon])
    return np.array([[-np.sin(b), np.cos(b), 0],
                     [-np.sin(a)*np.cos(b), -np.sin(a)*np.sin(b), np.cos(a)],
                     [np.cos(a)*np.cos(b), np.cos(a)*np.sin(b), np.sin(a)]])


ENU = enu_matrix(*LLH[0, :2])


def save(path, fields, attrs=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with h5py.File(temp, "w") as h:
        for key, value in fields.items():
            h.create_dataset(key, data=value, compression="gzip" if np.ndim(value) else None)
        h.attrs.update(attrs or {})
        h.attrs["created_unix"] = time.time()
        h.attrs["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    os.replace(temp, path)


def associate(root, out):
    catalogues = []
    for site in SITES:
        with h5py.File(root / ("20261117_" + site) / "catalogue.h5") as h:
            catalogues.append({k: h[k][:] for k in (
                "event_start_utc_ns", "event_stop_utc_ns", "event_peak_snr_db")})
    matches, ambiguous = [], []
    s = catalogues[0]
    for i, (start, stop) in enumerate(zip(s["event_start_utc_ns"], s["event_stop_utc_ns"])):
        ids = []
        for remote in catalogues[1:]:
            ids.append(np.flatnonzero((remote["event_start_utc_ns"] <= stop + 50000000) &
                                      (remote["event_stop_utc_ns"] >= start - 50000000)))
        if len(ids[0]) == 1 and len(ids[1]) == 1:
            matches.append([i, int(ids[0][0]), int(ids[1][0])])
        elif len(ids[0]) and len(ids[1]):
            ambiguous.append(i)
    matches = np.asarray(matches, dtype=np.int64).reshape(-1, 3)
    # A remote event cannot be assigned to two Sanya events without review.
    keep = np.ones(len(matches), bool)
    for j in (1, 2):
        values, count = np.unique(matches[:, j], return_counts=True)
        keep &= ~np.isin(matches[:, j], values[count > 1])
    save(out / "associations.h5", {"station_event_indices": matches[keep],
         "nonunique_station_event_indices": matches[~keep],
         "ambiguous_sanya_indices": np.asarray(ambiguous, dtype=np.int64),
         "station_counts": np.array([len(x["event_start_utc_ns"]) for x in catalogues])},
         {"method": "unique event interval overlap within 50 ms; geometry not yet validated",
          "hardware_delay_status": "recorded gate timing; independent calibration not yet established"})
    print("ASSOCIATIONS", len(matches[keep]), "ambiguous", len(ambiguous) + np.sum(~keep), flush=True)


def frames_at(ns):
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential
    from astropy.time import Time
    from astropy import units as u
    from astropy.utils import iers
    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "warn"
    times = Time(ns.astype(np.longdouble) / 1e9, format="unix", scale="utc")
    zero = np.zeros((len(ns), 3))

    def transform(x, v):
        rep = CartesianRepresentation(x.T * u.m, differentials=CartesianDifferential(v.T * u.m/u.s))
        state = GCRS(rep, obstime=times).transform_to(ITRS(obstime=times))
        return state.cartesian.xyz.to_value(u.m).T, state.cartesian.differentials["s"].d_xyz.to_value(u.m/u.s).T

    x0, v0 = transform(zero, zero)
    matrices = [np.empty((len(ns), 3, 3)) for _ in range(4)]
    for j in range(3):
        basis = zero.copy()
        basis[:, j] = 1.0
        x, v = transform(basis, zero)
        matrices[0][:, :, j], matrices[1][:, :, j] = x - x0, v - v0
        x, v = transform(zero, basis)
        matrices[2][:, :, j], matrices[3][:, :, j] = x - x0, v - v0
    return np.concatenate([x0, v0] + [m.reshape(len(ns), 9) for m in matrices], axis=1)


def to_gcrs(x, v, ns):
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential
    from astropy.time import Time
    from astropy import units as u
    t = Time(np.longdouble(ns)/1e9, format="unix", scale="utc")
    rep = CartesianRepresentation(x*u.m, differentials=CartesianDifferential(v*u.m/u.s))
    g = ITRS(rep, obstime=t).transform_to(GCRS(obstime=t))
    return np.r_[g.cartesian.xyz.to_value(u.m), g.cartesian.differentials["s"].d_xyz.to_value(u.m/u.s)]


def load_measurements(root, indices):
    rows = []
    for station, (site, index) in enumerate(zip(SITES, indices)):
        path = root / ("20261117_" + site) / "event_cuts" / ("event_%06d.h5" % index)
        with h5py.File(path) as h:
            p = h["pulses"]
            offsets = h["raw/pulse_offsets"][:]
            words = h["raw/iq_words_iq"][:].reshape(-1, 2)
            keep = (h["ransac_inlier"][:] == 1) & (p["peak_snr_db"][:] >= 15)
            for k in np.flatnonzero(keep):
                lag = int(p["echo_start_sample"][k]) - int(h["raw/first_sample"][k])
                first = lag - 240
                count = int(offsets[k+1] - offsets[k])
                if first < 0 or first + 1280 > count:
                    continue
                iq = words[int(offsets[k])+first:int(offsets[k])+first+1280]
                raw = iq[:, 0].astype(float) + 1j*iq[:, 1]
                sample = h["raw/first_delay_us"][k]*1e-6 + (first + np.arange(1280))/FS
                noise = float(p["raw_noise_power_adc2"][k])
                if not np.isfinite(noise) or noise <= 0:
                    continue
                rows.append(dict(station=station, ns=int(p["utc_ns"][k]), raw=raw,
                    sample=sample, noise=noise, snr=float(p["peak_snr_db"][k]),
                    apparent=C*float(p["apparent_delay_us"][k])*1e-6,
                    az=float(p["tx_azimuth_deg"][k]), el=float(p["tx_elevation_deg"][k]),
                    rxaz=float(p["rx_azimuth_deg"][k]), rxel=float(p["rx_elevation_deg"][k])))
    rows.sort(key=lambda r: (r["ns"], r["station"]))
    if any(sum(r["station"] == s for r in rows) < 6 for s in range(3)):
        raise ValueError("fewer than six usable voltage pulses at one receiver")
    if len({(r["station"], r["ns"]) for r in rows}) != len(rows):
        raise ValueError("duplicate station/pulse measurement")
    return rows


def initial_trajectory(rows):
    ns = np.array([r["ns"] for r in rows], np.int64)
    dt = (ns - ns.min())*1e-9
    station = np.array([r["station"] for r in rows])
    observed = np.array([r["apparent"] for r in rows])
    az, el = np.deg2rad([rows[0]["az"], rows[0]["el"]])
    axis = ENU.T @ np.array([np.sin(az)*np.cos(el), np.cos(az)*np.cos(el), np.sin(el)])
    anchor = STATIONS[0] + axis*100000/np.sin(el)

    def residual(q):
        x = anchor + q[:3]*1000 + dt[:, None]*q[3:]*1000
        a, b = x - STATIONS[0], x - STATIONS[station]
        la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        rate = np.sum((a/la[:, None] + b/lb[:, None])*q[3:]*1000, axis=1)
        return (la + lb - C*rate/(WAVELENGTH*GAMMA) - observed)/100

    candidates = []
    for speed in (20., 40., 65.):
        q = np.r_[np.zeros(3), ENU.T @ np.array([0., 0., -speed])]
        fit = least_squares(residual, q, bounds=(np.r_[[-80.]*3, [-100.]*3], np.r_[[80.]*3, [100.]*3]),
                            loss="soft_l1", max_nfev=250, x_scale="jac")
        candidates.append(fit)
    fit = min(candidates, key=lambda f: np.sum(residual(f.x)**2))
    position, velocity = anchor + 1000*fit.x[:3], 1000*fit.x[3:]
    if np.linalg.norm(velocity) < 3000 or np.linalg.norm(velocity) > 100000:
        raise ValueError("initial trajectory speed outside physical search interval")
    return to_gcrs(position, velocity, ns.min()), ns, dt, station, observed, np.std(residual(fit.x))*100


class Native:
    def __init__(self, library, input_path, nrows):
        self.lib = ct.CDLL(str(library))
        self.lib.voltage_load.argtypes = [ct.c_char_p]
        self.lib.voltage_load.restype = ct.c_void_p
        self.lib.voltage_len.argtypes = [ct.c_void_p]
        self.lib.voltage_len.restype = ct.c_size_t
        self.lib.voltage_free.argtypes = [ct.c_void_p]
        for name in ("voltage_residual", "voltage_prediction", "voltage_observables", "voltage_states"):
            fn = getattr(self.lib, name)
            fn.argtypes = [ct.c_void_p, PTR, PTR]
            fn.restype = ct.c_int
        self.handle = self.lib.voltage_load(os.fsencode(input_path))
        if not self.handle:
            raise ValueError("native voltage input rejected")
        self.size = self.lib.voltage_len(self.handle)
        self.nrows = nrows

    def call(self, name, q, shape):
        q = np.ascontiguousarray(q, dtype=np.float64)
        out = np.empty(shape)
        if not getattr(self.lib, name)(self.handle, q.ctypes.data_as(PTR), out.ctypes.data_as(PTR)):
            raise ValueError("invalid native trajectory")
        return out

    def residual(self, q):
        try:
            return self.call("voltage_residual", q, self.size)
        except ValueError:
            return np.full(self.size, 1e10)

    def close(self):
        self.lib.voltage_free(self.handle)


def fit_one(task):
    root, out, library, event, indices = task
    folder = out / ("tri_%06d" % event)
    if (folder / "fit.h5").exists():
        return event, "cached"
    try:
        rows = load_measurements(root, indices)
        center, ns, dt, station, apparent, seed_rms = initial_trajectory(rows)
        frame = frames_at(ns)
        real = np.array([r["raw"].real for r in rows])
        imag = np.array([r["raw"].imag for r in rows])
        fields = dict(center=center, time_s=dt, time_ns=ns, frames=frame,
            tx=STATIONS[0], rx=STATIONS[station], station=station, real=real, imag=imag,
            noise_power=np.array([r["noise"] for r in rows]),
            sample_time_s=np.array([r["sample"] for r in rows]),
            fs=FS, bandwidth=B, duration=T, wavelength=WAVELENGTH,
            whitener_real=np.ones((len(rows), 1)), whitener_imag=np.zeros((len(rows), 1)))
        save(folder / "input.h5", fields, {"calibration": "recorded gate timing; no April hardware correction", "whitening": "none"})
        # First establish a kinematic solution. The eight-parameter kernel tends
        # to constant velocity as its exponential deceleration amplitude -> 0.
        model = Native(library, folder / "input.h5", len(rows))
        q = np.r_[np.zeros(6), -4., -2.]
        lo, hi = np.r_[[-30.]*3, [-50.]*3, -5., -4.], np.r_[[30.]*3, [50.]*3, 4., 3.]
        fit = least_squares(model.residual, q, bounds=(lo, hi), x_scale="jac", max_nfev=180,
                            diff_step=1e-4, ftol=1e-8, xtol=1e-8, gtol=1e-6)
        q = fit.x
        prediction = model.call("voltage_prediction", q, (len(rows), 1280, 2))
        observables = model.call("voltage_observables", q, (len(rows), 2))
        states = model.call("voltage_states", q, (len(rows), 4))
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac, rcond=1e-12)
        speed = np.linalg.norm(center[3:] + 1000*q[3:6])
        direction = (center[3:] + 1000*q[3:6])/speed
        gcrs = center[:3] + 1000*q[:3] + states[:, :1]*direction
        position = frame[:, :3] + np.einsum("nij,nj->ni", frame[:, 6:15].reshape(-1, 3, 3), gcrs)
        native_v = np.repeat((center[3:] + 1000*q[3:6])[None, :], len(rows), axis=0)
        position += np.einsum("nij,nj->ni", frame[:, 24:33].reshape(-1, 3, 3), native_v)
        # Range residuals and residual beats are diagnostics, not extra likelihood terms.
        path_model = observables[:, 0] + C*observables[:, 1]/GAMMA
        residual = apparent - path_model
        raw = real + 1j*imag
        predicted = prediction[:, :, 0] + 1j*prediction[:, :, 1]
        fft = np.fft.fftfreq(8192, 1/FS)
        beat = []
        for row, pred in zip(raw, predicted):
            phase = np.divide(pred, np.abs(pred), out=np.zeros_like(pred), where=np.abs(pred) > .2*np.max(np.abs(pred)))
            power = np.abs(np.fft.fft(row*np.conj(phase)*np.hanning(1280), 8192))**2
            beat.append(fft[np.argmax(power)])
        save(folder / "fit.h5", dict(params=q, covariance=covariance, time_ns=ns, time_s=dt,
            station=station, apparent_path_m=apparent, model_apparent_path_m=path_model,
            path_residual_m=residual, residual_beat_hz=np.array(beat),
            position_ecef_m=position, position_gcrs_m=gcrs, speed_mps=states[:, 1],
            initial_gcrs_position_m=center[:3]+1000*q[:3], initial_gcrs_velocity_mps=center[3:]+1000*q[3:6],
            snr_db=np.array([r["snr"] for r in rows]),
            predicted_real=prediction[:, :, 0], predicted_imag=prediction[:, :, 1],
            station_event_indices=indices),
            {"optimizer_success": bool(fit.success), "seed_path_rms_m": seed_rms,
             "path_rms_m": np.sqrt(np.mean(residual**2)), "voltage_cost": float(fit.cost),
             "model": "kinematic exponential deceleration; physical drag fit pending",
             "status": "provisional: uncalibrated hardware delays", "mass_available": False})
        model.close()
        return event, "fit"
    except Exception:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "error.log").write_text(traceback.format_exc())
        return event, "failed"


def fit_catalogue(root, out, library, workers, limit):
    with h5py.File(out / "associations.h5") as h:
        indices = h["station_event_indices"][:]
    if limit:
        indices = indices[:limit]
    tasks = [(root, out / "events", library, i, row) for i, row in enumerate(indices)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fit_one, tasks, chunksize=1))
    save(out / "fit_status.h5", {"event_index": np.array([x[0] for x in results]),
         "status": np.array([x[1] for x in results], dtype="S16")})
    print("FITS", {s: sum(x[1] == s for x in results) for s in ("fit", "cached", "failed")}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["associate", "fit"])
    p.add_argument("--root", type=Path, default=Path("/gpfs/pub/data_other/juha"))
    p.add_argument("--output", type=Path, default=Path("/gpfs/pub/data_other/juha/20261117_tristatic"))
    p.add_argument("--library", type=Path, default=Path("tristatic_native/target/release/libtristatic_native.so"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.workers < 1:
        p.error("workers must be positive")
    if a.stage == "associate":
        associate(a.root, a.output)
    else:
        fit_catalogue(a.root, a.output, a.library.resolve(), a.workers, a.limit)


if __name__ == "__main__":
    main()
