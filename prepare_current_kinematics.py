"""Freeze the accepted direct physical fits for height, radiant and area products.

No historical trajectory fallback. The orbit manifest fixes membership and fit
identity; the Rust physical model evaluates only retained station/pulse rows.
"""
import ctypes as ct
import hashlib
import os
import sys
from pathlib import Path

import h5py
import numpy as np
if not hasattr(np, 'mat'):
    np.mat = np.asmatrix
import jcoord

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / 'results/tristatic_cached_shrinking_20261029'
MANIFEST = ROOT / 'results/tristatic_shrinking_orbits_current_20261107/orbits.h5'
FLAGS = ROOT / 'results/review_flags_scatter_snapshot'
OUTPUT = ROOT / 'results/current_kinematics_20261107/kinematics.h5'
AREA = OUTPUT.parent / 'beam_area.h5'
START = '2024-04-22T12:01:26.659'
STOP = '2024-04-22T23:49:53.159'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accepted_ids(manifest=None):
    manifest = MANIFEST if manifest is None else manifest
    with h5py.File(manifest) as h:
        ids = list(h['event_id'].asstr()[:])
    if not len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Expected a nonempty unique audited event manifest')
    return ids


def check_export(path=None):
    path = OUTPUT if path is None else path
    with h5py.File(path) as h:
        if h.attrs['manifest_sha256'] != sha(MANIFEST):
            raise ValueError('Kinematic export needs refresh: orbit manifest changed')
        np.testing.assert_array_equal(h['event_id'].asstr()[:], accepted_ids())
        if h.attrs['model'] != 'direct shrinking radius':
            raise ValueError('Not a direct physical fit export')
    return Path(path)


def main():
    library = Path(os.environ.get('TRISTATIC_LIBRARY',
                    ROOT / 'tristatic_native/target/debug/libtristatic_native.dylib'))
    lib = ct.CDLL(str(library))
    ptr = ct.POINTER(ct.c_double)
    lib.voltage_load.argtypes = [ct.c_char_p]
    lib.voltage_load.restype = ct.c_void_p
    lib.voltage_states.argtypes = [ct.c_void_p, ptr, ptr]
    lib.voltage_states.restype = ct.c_int
    lib.voltage_free.argtypes = [ct.c_void_p]
    ids = accepted_ids()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT.with_suffix('.h5.partial')
    with h5py.File(MANIFEST) as manifest, h5py.File(partial, 'w') as out:
        out.attrs.update(model='direct shrinking radius', manifest=str(MANIFEST),
                         manifest_sha256=sha(MANIFEST), library_sha256=sha(library),
                         source_program=__file__, source_sha256=sha(__file__),
                         observation_start_utc=START, observation_stop_utc=STOP,
                         selection='Current audited trajectory manifest; retained station/pulse rows; no mass cut',
                         position_convention='nominal pulse-time GCRS state transformed to ITRS',
                         sample_convention='one sample per distinct retained UTC pulse time')
        out['event_id'] = np.asarray(ids, dtype=h5py.string_dtype())
        for event in ids:
            folder = CATALOG / event
            audit = manifest['source_audit'][event].attrs
            if sha(folder/'fit.h5') != audit['fit_sha256']:
                raise ValueError(f'{event}: fit differs from accepted manifest')
            with h5py.File(folder/'input.h5') as h:
                d = {k:h[k][()] for k in ('center','time_s','time_ns','station','frames',
                                         'initial_detection_time_ns','initial_detection_time_s')}
                input_revision = int(h.attrs['manual_flag_revision'])
                if h.attrs['model'] != 'shrinking_radius_voltage':
                    raise ValueError(f'{event}: wrong physical model')
            with h5py.File(folder/'fit.h5') as h:
                q = np.ascontiguousarray(h['params'][:])
                if len(q) != 7 or not h.attrs['optimizer_success']:
                    raise ValueError(f'{event}: invalid physical fit')
            flag = FLAGS/(event+'.h5')
            plot_path = Path(audit['current_plot'])
            with h5py.File(plot_path) as plot:
                revision = int(plot.attrs['fit_flag_revision'])
            if flag.exists():
                with h5py.File(flag) as h:
                    if h.attrs['bad'] or int(h.attrs['version']) != revision:
                        raise ValueError(f'{event}: rejected or stale review flags')
                    excluded = set(h['excluded'].asstr()[:])
            else:
                if revision != 0:
                    raise ValueError(f'{event}: missing review revision')
                excluded = set()
            with h5py.File(plot_path) as plot:
                if plot.attrs['fit_flag_revision'] != revision or revision != audit['flag_revision']:
                    raise ValueError(f'{event}: inconsistent revisions')
                input_index = {(int(s),int(t)):j for j,(s,t) in enumerate(zip(d['station'],d['time_ns']))}
                retained = [(int(s),int(t)) for s,t in zip(plot['station'][:],plot['time_ns'][:])]
                if len(set(retained)) != len(retained):
                    raise ValueError(f'{event}: duplicate retained identity')
                keep = np.asarray([input_index[key] for key in retained])
                residual = plot['modeled_path_m'][:] - plot['seed_apparent_path_m'][:]
            if any(f'{s}:{t}' in excluded for s,t in retained):
                raise ValueError(f'{event}: excluded pulse survived')
            if abs(d['initial_detection_time_s']) > 1e-10:
                raise ValueError(f'{event}: unsupported time origin')
            state = d['center'] + 1000*q[:6]
            handle = lib.voltage_load(str(folder/'input.h5').encode())
            if not handle:
                raise ValueError(f'{event}: Rust load failed')
            evolution = np.empty((len(d['time_s']),4))
            try:
                if not lib.voltage_states(handle,q.ctypes.data_as(ptr),evolution.ctypes.data_as(ptr)):
                    raise ValueError(f'{event}: Rust integration failed')
            finally:
                lib.voltage_free(handle)
            direction = state[3:]/np.linalg.norm(state[3:])
            x = state[:3]+evolution[:,0,None]*direction
            v = evolution[:,1,None]*direction
            frames = d['frames']
            xyz = (frames[:,:3]+np.einsum('nij,nj->ni',frames[:,6:15].reshape(-1,3,3),x)
                   +np.einsum('nij,nj->ni',frames[:,24:33].reshape(-1,3,3),v))
            if not np.isfinite(xyz).all():
                raise ValueError(f'{event}: nonfinite physical state')
            xyz = xyz[keep]
            for key in ('time_ns','time_s','station'):
                d[key] = d[key][keep]
            # Full-state diagnostics can contain excluded rows; compare by identity.
            with h5py.File(folder/'physical_states.h5') as old:
                index = {(int(s),int(t)):j for j,(s,t) in enumerate(zip(old['station'][:],old['time_ns'][:]))}
                saved = old['position_ecef_m'][:][[index[(int(s),int(t))] for s,t in zip(d['station'],d['time_ns'])]]
                delta = float(np.max(np.linalg.norm(xyz-saved,axis=1)))
            _, unique = np.unique(d['time_ns'],return_index=True)
            xyz = xyz[unique]
            g = out.create_group('events/'+event)
            g.attrs.update(fit_sha256=sha(folder/'fit.h5'), input_sha256=sha(folder/'input.h5'),
                           flag_sha256=sha(flag) if flag.exists() else '', flag_revision=revision,
                           input_flag_revision=input_revision,
                           current_plot_sha256=sha(plot_path),
                           saved_position_max_difference_m=delta, retained_station_pulses=len(residual),
                           rms_total_path_residual_m=float(np.sqrt(np.mean(residual**2))))
            g['state_gcrs'] = state
            g['epoch_ns'] = d['initial_detection_time_ns']
            g['time_ns'] = d['time_ns'][unique]
            g['position_ecef_m'] = xyz
            g['speed_km_s'] = evolution[keep][unique,1]/1000
            g['height_km'] = np.array([jcoord.ecef2geodetic(*p)[2]/1000 for p in xyz])
            g['retained_time_ns'] = d['time_ns']
            g['retained_station'] = d['station']
    partial.replace(OUTPUT)
    print(OUTPUT, flush=True)
    prepare_area()


def prepare_area():
    from estimate_sanya_gain_containment_area import sanya_geometry, offsets_for_positions, contour_area
    geom = sanya_geometry()
    ids = accepted_ids()
    gains = []
    with h5py.File(check_export()) as h:
        for event in ids:
            _,_,gain = offsets_for_positions(h['events/'+event+'/position_ecef_m'][:],geom)
            gains.append(float(np.max(gain)))
    # The order statistic guarantees at least 95% event containment.
    cutoff = float(np.sort(gains)[len(ids)-int(np.ceil(.95*len(ids)))])
    grids = [901,1801,3601]
    areas = []
    for n in grids:
        *_,perp,horizontal = contour_area(cutoff,geom,95.,20.,n)
        areas.append(horizontal)
    with h5py.File(AREA,'w') as h:
        h.attrs.update(model='direct shrinking radius', source=str(OUTPUT), source_sha256=sha(OUTPUT),
                       manifest_sha256=sha(MANIFEST), source_program=__file__,
                       cutoff_relative_gain_db=cutoff, containment_fraction=.95,
                       achieved_containment_fraction=float(np.mean(np.asarray(gains)>=cutoff)),
                       area_horizontal_km2=horizontal, area_perpendicular_km2=perp,
                       common_volume_altitude_km=95., grid_half_width_deg=20.,
                       area_method='legacy tangent-plane area convention; grid refinement recorded')
        h['event_id'] = np.asarray(ids,dtype=h5py.string_dtype())
        h['max_gain_db'] = gains
        h['grid_n'] = grids
        h['area_horizontal_by_grid_km2'] = areas
    print('AREA',cutoff,horizontal,areas,flush=True)


def refresh_manifest():
    """Rebind an unchanged kinematic snapshot after an uncertainty-only update."""
    with h5py.File(MANIFEST) as manifest, h5py.File(OUTPUT,'r+') as h:
        np.testing.assert_array_equal(h['event_id'].asstr()[:],accepted_ids())
        for event in accepted_ids():
            g = h['events/'+event]
            audit = manifest['source_audit/'+event].attrs
            if g.attrs['fit_sha256'] != audit['fit_sha256'] or sha(CATALOG/event/'fit.h5') != audit['fit_sha256']:
                raise ValueError(f'{event}: changed best fit; full regeneration required')
            if g.attrs['input_sha256'] != sha(CATALOG/event/'input.h5'):
                raise ValueError(f'{event}: changed physical input; full regeneration required')
            path = Path(audit['current_plot'])
            with h5py.File(path) as new, h5py.File(CATALOG/event/'plot.h5') as old:
                for key in ('station','time_ns','time_s','modeled_path_m','speed_km_s'):
                    np.testing.assert_array_equal(new[key][:],old[key][:])
                np.testing.assert_array_equal(new['station'][:],g['retained_station'][:])
                np.testing.assert_array_equal(new['time_ns'][:],g['retained_time_ns'][:])
                if new.attrs['fit_flag_revision'] != g.attrs['flag_revision']:
                    raise ValueError(f'{event}: changed flags; full regeneration required')
            g.attrs['current_plot_sha256'] = sha(path)
        h.attrs.update(manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),source_sha256=sha(__file__),
                       manifest_refresh='uncertainty-only update; all fit/input hashes and retained kinematic arrays unchanged')
    prepare_area()


if __name__ == '__main__':
    if '--refresh-manifest' in sys.argv:
        refresh_manifest()
    else:
        main()
