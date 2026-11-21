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
import toml

import h5py
import numpy as np
from scipy.optimize import least_squares

C = 299792458.0
FS = 4e6
B = 4e6
T = 200e-6
GAMMA = B / T
WAVELENGTH = C / 430e6
# Common additive propagation-delay calibration for all three receivers.
DELAY_CORRECTION_US = -5.976391135593515
SITES = ("sanya", "danzhou", "wenchang")
LLH = np.array([[18.3492, 109.6222, 50.0], [19.5281, 109.1322, 99.9],
                [19.5982, 110.7908, 24.9]])
PTR = ct.POINTER(ct.c_double)


def remaining_delay_correction(h):
    """Apply only the part of the calibration absent from the stored product."""
    config = toml.loads(bytes(h["config_toml_utf8"][:]).decode("utf-8"))
    return DELAY_CORRECTION_US - float(config["delay_correction_us"])


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
    from overlap import measure_overlap
    sufficient = np.zeros(len(matches),bool)
    for n in np.flatnonzero(keep):
        times=[]
        for site,index in zip(SITES,matches[n]):
            with h5py.File(root/("20261117_"+site)/"event_cuts"/("event_%06d.h5"%index)) as h:
                valid=(h["ransac_inlier"][:]==1)&(h["pulses/peak_snr_db"][:]>=14)
                times.append(h["pulses/utc_ns"][:][valid])
        sufficient[n]=measure_overlap(times)[0]
    save(out / "associations.h5", {"station_event_indices": matches[keep & sufficient],
         "insufficient_overlap_station_event_indices": matches[keep & ~sufficient],
         "nonunique_station_event_indices": matches[~keep],
         "ambiguous_sanya_indices": np.asarray(ambiguous, dtype=np.int64),
         "station_counts": np.array([len(x["event_start_utc_ns"]) for x in catalogues])},
         {"method": "unique candidates; >=6 shared pulses, >=25 ms, >=50 percent of every station; 100 us matching tolerance",
          "hardware_delay_status": "common calibrated delay at all three sites",
          "delay_correction_us": DELAY_CORRECTION_US})
    print("ASSOCIATIONS", np.sum(keep & sufficient), "insufficient overlap",np.sum(keep & ~sufficient), "ambiguous", len(ambiguous) + np.sum(~keep), flush=True)


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
            correction_us = remaining_delay_correction(h)
            p = h["pulses"]
            offsets = h["raw/pulse_offsets"][:]
            words = h["raw/iq_words_iq"][:].reshape(-1, 2)
            keep = (h["ransac_inlier"][:] == 1) & (p["peak_snr_db"][:] >= 14)
            for k in np.flatnonzero(keep):
                lag = int(p["echo_start_sample"][k]) - int(h["raw/first_sample"][k])
                first = lag - 240
                count = int(offsets[k+1] - offsets[k])
                if first < 0 or first + 1280 > count:
                    continue
                iq = words[int(offsets[k])+first:int(offsets[k])+first+1280]
                # Preserve the station reader's Q + i I convention, not I + i Q.
                raw = iq[:, 1].astype(float) + 1j*iq[:, 0]
                sample = (h["raw/first_delay_us"][k] + correction_us)*1e-6 + (first + np.arange(1280))/FS
                noise = float(p["raw_noise_power_adc2"][k])
                if not np.isfinite(noise) or noise <= 0:
                    continue
                rows.append(dict(station=station, ns=int(p["utc_ns"][k]), raw=raw,
                    sample=sample, noise=noise, snr=float(p["peak_snr_db"][k]),
                    apparent=C*(float(p["apparent_delay_us"][k]) + correction_us)*1e-6,
                    az=float(p["tx_azimuth_deg"][k]), el=float(p["tx_elevation_deg"][k]),
                    rxaz=float(p["rx_azimuth_deg"][k]), rxel=float(p["rx_elevation_deg"][k])))
    rows.sort(key=lambda r: (r["ns"], r["station"]))
    if any(sum(r["station"] == s for r in rows) < 3 for s in range(3)):
        raise ValueError("fewer than three usable voltage pulses at one receiver")
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
        with h5py.File(folder / "fit.h5") as h:
            compatible = h.attrs.get("script_sha256") == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if compatible:
            return event, "cached"
        archive = out / "superseded" / (folder.name + "_" + str(time.time_ns()))
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.rename(folder, archive)
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
        save(folder / "input.h5", fields, {"calibration": "common additive delay correction at all three sites",
             "delay_correction_us": DELAY_CORRECTION_US, "whitening": "none"})
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
        raw = real + 1j*imag
        predicted = prediction[:, :, 0] + 1j*prediction[:, :, 1]
        fft = np.fft.fftfreq(131072, 1/FS)
        beat, reference_beat, beat_sigma = [], [], []
        for i, (row, pred) in enumerate(zip(raw, predicted)):
            phase = np.divide(pred, np.abs(pred), out=np.zeros_like(pred), where=np.abs(pred) > .2*np.max(np.abs(pred)))
            power = np.abs(np.fft.fft(row*np.conj(phase)*np.hanning(1280), len(fft)))**2
            beat.append(fft[np.argmax(power)])
            tau = fields["sample_time_s"][i] - apparent[i]/C
            reference = np.exp(2j*np.pi*(B/2*tau - GAMMA/2*tau*tau))
            power = np.abs(np.fft.fft(row*np.conj(reference)*np.hanning(1280), len(fft)))**2
            reference_beat.append(fft[np.argmax(power)])
            support = np.abs(pred) > .2*np.max(np.abs(pred))
            t = fields["sample_time_s"][i, support]
            per_sample_snr = np.max(np.abs(pred)**2)/fields["noise_power"][i]
            beat_sigma.append(1/np.sqrt(8*np.pi**2*max(per_sample_snr, 1e-12)*np.sum((t-t.mean())**2)))
        corrected_apparent = apparent + C/GAMMA*np.array(reference_beat)
        residual = corrected_apparent - path_model
        save(folder / "fit.h5", dict(params=q, covariance=covariance, time_ns=ns, time_s=dt,
            station=station, apparent_path_m=corrected_apparent, matched_filter_path_m=apparent,
            measured_gate_beat_hz=np.array(reference_beat), conditional_beat_std_hz=np.array(beat_sigma),
            model_apparent_path_m=path_model,
            path_residual_m=residual, residual_beat_hz=np.array(beat),
            position_ecef_m=position, position_gcrs_m=gcrs, speed_mps=states[:, 1],
            initial_gcrs_position_m=center[:3]+1000*q[:3], initial_gcrs_velocity_mps=center[3:]+1000*q[3:6],
            snr_db=np.array([r["snr"] for r in rows]),
            predicted_real=prediction[:, :, 0], predicted_imag=prediction[:, :, 1],
            station_event_indices=indices),
            {"optimizer_success": bool(fit.success), "seed_path_rms_m": seed_rms,
             "path_rms_m": np.sqrt(np.mean(residual**2)), "voltage_cost": float(fit.cost),
             "model": "kinematic exponential deceleration; physical drag fit pending",
             "status": "common delay calibration applied", "delay_correction_us": DELAY_CORRECTION_US,
             "mass_available": False})
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
