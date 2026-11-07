"""Single-event archive audit; no active analysis-tree imports or saved-fit seed."""
from pathlib import Path
import argparse
import ctypes as ct
import hashlib
import platform
import time

import h5py
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
EVENT = 'tri_0012_1713798481419350028'
SOURCE = ROOT / 'L1/events' / EVENT
OUT = ROOT / 'reproduced'
PTR = ct.POINTER(ct.c_double)


def ptr(x):
    return x.ctypes.data_as(PTR)


def main():
    global ROOT, SOURCE, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-cuts', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='Directory containing the extracted L0, L1 and source folders')
    parser.add_argument('--library', type=Path,
                        help='Optional path to the rebuilt native library')
    args = parser.parse_args()
    ROOT = args.root.resolve()
    SOURCE = ROOT / 'L1/events' / EVENT
    OUT = ROOT / 'reproduced'
    if args.raw_cuts:
        OUT = ROOT / 'reproduced-raw-cuts'
    OUT.mkdir(exist_ok=True)
    began = time.monotonic()
    with h5py.File(SOURCE / 'input.h5') as h:
        d = {k: h[k][()] for k in h}
        source_files = str(h.attrs['source_event_files']).split(';')
    with h5py.File(SOURCE / 'fit.h5') as h:
        initial = h['initial'][:]
        reference = h['params'][:]
    comparisons = []
    raw_rows = np.empty((len(d['time_s']), 1000), complex)
    raw_sample_times = np.empty(raw_rows.shape)
    for station, name in enumerate(('sanya', 'danzhou', 'wenchang')):
        path = ROOT / 'L0' / name / Path(source_files[station]).name
        with h5py.File(path) as h:
            gates = h['range_gate_index'][:]
            times = h['time_ns_utc'][:]
            raw = h['raw_voltage'][:]
        for row in np.flatnonzero(d['station'] == station):
            columns = np.flatnonzero(times == d['time_ns'][row])
            assert len(columns) == 1, (name, int(d['time_ns'][row]))
            start = int(d['gate'][row]) - 640
            idx = gates.astype(int) - start
            keep = (idx >= 0) & (idx < 1280)
            archived = d['real'][row, idx[keep]] + 1j*d['imag'][row, idx[keep]]
            err = float(np.max(np.abs(archived - raw[keep, columns[0]])))
            comparisons.append((station, row, keep.sum(), err))
            lower = max(0, int(gates[0])-start)
            upper = min(1280, int(gates[-1])+1-start)
            offset = min(max(140, lower), upper-1000)
            assert offset >= lower and offset+1000 <= upper
            raw_offset = start+offset-int(gates[0])
            raw_rows[row] = raw[raw_offset:raw_offset+1000, columns[0]]
            raw_sample_times[row] = d['sample_time_s'][row, offset:offset+1000]
    comparisons = np.array(comparisons)
    print('RAW OVERLAP: rows, samples, maximum absolute difference:',
          len(comparisons), int(comparisons[:, 2].sum()), comparisons[:, 3].max(), flush=True)
    assert comparisons[:, 3].max() == 0, 'Raw-cut/prepared-input disagreement'
    input_path = SOURCE / 'input.h5'
    if args.raw_cuts:
        d['real'], d['imag'] = raw_rows.real, raw_rows.imag
        d['sample_time_s'] = raw_sample_times
        input_path = OUT / 'input.h5'
        with h5py.File(input_path, 'w') as h:
            for k, value in d.items():
                h[k] = value
            h.attrs['reconstruction'] = '1000 voltage samples per pulse from L0 raw cuts; archived frames, atmosphere, noise, calibration and pre-fit guess'

    suffix = {'Darwin': '.dylib', 'Linux': '.so'}[platform.system()]
    libpath = args.library or (ROOT / 'source/lfm_meteor/tristatic_native/target/release'
                              / ('libtristatic_native' + suffix))
    lib = ct.CDLL(str(libpath))
    lib.voltage_load.argtypes = [ct.c_char_p]
    lib.voltage_load.restype = ct.c_void_p
    lib.voltage_free.argtypes = [ct.c_void_p]
    lib.voltage_len.argtypes = [ct.c_void_p]
    lib.voltage_len.restype = ct.c_size_t
    lib.voltage_residual.argtypes = [ct.c_void_p, PTR, PTR]
    lib.voltage_residual.restype = ct.c_int
    lib.voltage_states.argtypes = [ct.c_void_p, PTR, PTR]
    lib.voltage_states.restype = ct.c_int
    lib.voltage_logp.argtypes = [ct.c_void_p, PTR]
    lib.voltage_logp.restype = ct.c_double
    lib.voltage_cache_build.argtypes = [ct.c_void_p, PTR, ct.c_size_t, ct.c_double, ct.c_char_p]
    lib.voltage_cache_build.restype = ct.c_void_p
    lib.voltage_cache_logp.argtypes = [ct.c_void_p, ct.c_void_p, PTR, ct.c_int]
    lib.voltage_cache_logp.restype = ct.c_double
    lib.voltage_cache_free.argtypes = [ct.c_void_p]
    handle = lib.voltage_load(str(input_path).encode())
    assert handle
    cache = None
    try:
        size = lib.voltage_len(handle)

        def residual(q):
            q = np.ascontiguousarray(q)
            r = np.empty(size)
            assert lib.voltage_residual(handle, ptr(q), ptr(r))
            return r

        # The archived driver's Fisher-scaled Powell optimization, starting from
        # the archived pre-fit guess, not the published optimum.
        step = np.r_[np.full(3, 1e-5), np.full(3, 1e-4), 1e-4]
        eye = np.diag(step)
        jac = np.array([(residual(initial+e)-residual(initial-e))/2 for e in eye])
        w, u = np.linalg.eigh(jac @ jac.T)
        inv = (u/np.maximum(w, max(np.max(w)*1e-12, 1e-12))) @ u.T
        cov = step[:, None]*inv*step[None, :]
        limit = np.r_[np.full(3, 1.), np.full(3, 10.), 1.]
        cov *= min(1., float(np.min(limit/np.sqrt(np.diag(cov)))))**2
        transform = np.linalg.cholesky(cov)
        cache = lib.voltage_cache_build(handle, ptr(initial), 0, 2.,
                                       str(OUT / 'likelihood_cache.h5').encode())
        assert cache

        def objective(z):
            q = np.ascontiguousarray(initial + transform @ z)
            value = lib.voltage_cache_logp(cache, handle, ptr(q), 0)
            return -value if np.isfinite(value) else 1e30

        fit = minimize(objective, np.zeros(7), method='Powell',
                       options=dict(maxiter=100, xtol=1e-5, ftol=1e-9))
        q = np.ascontiguousarray(initial + transform @ fit.x)
        print('FIT:', fit.success, fit.message, 'evaluations', fit.nfev, flush=True)
        states = np.empty((len(d['time_s']), 4))
        assert lib.voltage_states(handle, ptr(q), ptr(states))
        v = d['center'][3:] + 1000*q[3:6]
        direction = v/np.linalg.norm(v)
        xyz_gcrs = d['center'][:3]+1000*q[:3]+states[:, 0, None]*direction
        vel_gcrs = states[:, 1, None]*direction
        f = d['frames']
        xyz = (f[:, :3] + np.einsum('nij,nj->ni', f[:, 6:15].reshape(-1,3,3), xyz_gcrs)
               + np.einsum('nij,nj->ni', f[:, 24:33].reshape(-1,3,3), vel_gcrs))
        with h5py.File(SOURCE / 'physical_states.h5') as h:
            position_difference = np.linalg.norm(xyz-h['position_ecef_m'][:], axis=1)
            speed_difference = abs(states[:, 1]/1000-h['speed_km_s'][:])*1000
        radius = 10**q[6]
        reference_radius = 10**reference[6]
        mass = 4*np.pi/3*3000*radius**3
        raw_lp = lib.voltage_logp(handle, ptr(q))
        ref_lp = lib.voltage_logp(handle, ptr(reference))
        passed = (bool(fit.success) and position_difference.max() < 1.
                  and speed_difference.max() < 5.
                  and abs(radius/reference_radius-1) < .01
                  and raw_lp-ref_lp > -.01)
        print('RESULT: radius_um', radius*1e6, 'reference_um', reference_radius*1e6,
              'mass_kg', mass, 'max_position_difference_m', position_difference.max(),
              'max_speed_difference_m_s', speed_difference.max(),
              'raw_logp_difference', raw_lp-ref_lp, flush=True)
        with h5py.File(OUT / 'replication.h5', 'w') as h:
            for k, value in dict(initial=initial, params=q, reference_params=reference,
                                 raw_overlap_audit=comparisons, position_ecef_m=xyz,
                                 speed_m_s=states[:, 1], time_ns=d['time_ns'], station=d['station'],
                                 position_difference_m=position_difference,
                                 speed_difference_m_s=speed_difference).items():
                h[k] = value
            h.attrs.update(event=EVENT, optimizer_success=bool(fit.success),
                           optimizer_evaluations=fit.nfev, elapsed_seconds=time.monotonic()-began,
                           radius_m=radius, reference_radius_m=reference_radius, mass_kg=mass,
                           raw_logp=raw_lp, reference_raw_logp=ref_lp,
                           raw_cuts_used=bool(args.raw_cuts),
                           acceptance_passed=passed,
                           acceptance_criteria='optimizer success; max position difference <1 m; max speed difference <5 m/s; radius difference <1%; raw log likelihood no worse by >0.01',
                           python_version=platform.python_version(), numpy_version=np.__version__,
                           script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           limitations='Archived frames, atmosphere, noise, calibration and pre-fit guess reused. No posterior rerun. L0 cuts lack outer noise padding.',
                           library_sha256=hashlib.sha256(libpath.read_bytes()).hexdigest())
        assert passed, 'Replication tolerance failed; inspect replication.h5'
    finally:
        if cache:
            lib.voltage_cache_free(cache)
        lib.voltage_free(handle)


if __name__ == '__main__':
    main()
