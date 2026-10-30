"""Propagate current direct shrinking-radius state ensembles, never WJ states.

MPI distributes events. Input snapshots and exclusions remain auditable in HDF5.
The orbit propagation is gravitational only, from first retained detection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
from astropy.coordinates import solar_system_ephemeris
from mpi4py import MPI

from propagate_tristatic_orbits_1000d_classification import (
    propagate_event, summarize_classification, write_outputs,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(folder, flags, output, nsamples):
    with h5py.File(folder / 'input.h5') as h:
        center = h['center'][:]
        epoch = int(h['initial_detection_time_ns'][()])
        revision = int(h.attrs['manual_flag_revision'])
        assert abs(float(h['initial_detection_time_s'][()])) < 1e-10
    with h5py.File(folder / 'fit.h5') as h:
        assert 'shrinking' in str(h.attrs['model']).lower(), 'Not a physical-model fit'
        best = h['params'][:]
    with h5py.File(folder / 'plot.h5') as h:
        method = str(h.attrs.get('uncertainty_method', 'conditional physical-model posterior'))
        assert bool(h.attrs.get('uncertainty_converged', False)), 'Unresolved state uncertainty'
        assert int(h.attrs['fit_flag_revision']) == revision, 'Fit/input flags disagree'
        radius_interval = h['initial_radius_m_quantiles'][:]
    flag = flags / (folder.name + '.h5')
    current_revision = 0
    if flag.exists():
        with h5py.File(flag) as h:
            assert not bool(h.attrs.get('bad',False)), 'Excluded from scientific catalogue by review'
            current_revision = int(h.attrs['version'])
    assert current_revision == revision, 'Manual flags await refit'
    if 'random-deletion' in method:
        sources = [folder / 'random_deletion.h5']
        with h5py.File(sources[0]) as h:
            assert np.all(h['optimizer_converged'][:]), 'Subset optimizers failed'
            draws = h['params'][:][np.isclose(h['drop_fraction'][:], .35)]
    elif 'bootstrap' in method:
        sources = [folder / 'block_bootstrap_states.h5']
        with h5py.File(sources[0]) as h:
            draws = h['params'][:]
    else:
        sources = sorted(folder.glob('chain_*.h5'))
        assert len(sources) == 4, 'Four posterior chains required'
        chains = []
        for path in sources:
            with h5py.File(path) as h:
                chains.append(h['samples'][:])
        length = min(map(len, chains))
        draws = np.asarray([c[:length] for c in chains]).reshape(-1, 7)
    assert draws.ndim == 2 and draws.shape[1] == 7 and np.isfinite(draws).all()
    assert np.allclose(np.quantile(10**draws[:, 6], [.025, .5, .975]),
                       radius_interval, rtol=.01, atol=0), 'State samples do not match published intervals'
    # No cut on radius uncertainty: weak mass constraints need not imply weak orbits.
    rng = np.random.default_rng(int(hashlib.sha256(folder.name.encode()).hexdigest()[:8], 16))
    sample = draws[rng.choice(len(draws), min(nsamples, len(draws)), replace=False)]
    states = center[None, :] + 1000 * sample[:, :6]
    best_state = center + 1000 * best[:6]
    path = output / 'inputs' / (folder.name + '.h5')
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path) + '.partial', 'w') as h:
        h.attrs.update(event_id=folder.name, model='direct shrinking radius',
                       uncertainty_method=method, flag_revision=revision,
                       input_sha256=digest(folder/'input.h5'), fit_sha256=digest(folder/'fit.h5'),
                       source_files=json.dumps([str(p) for p in sources]),
                       source_sha256=json.dumps([digest(p) for p in sources]),
                       source_draws=len(draws), sampling_seed=int(hashlib.sha256(folder.name.encode()).hexdigest()[:8],16))
        h['best_state_gcrs'] = best_state
        h['sample_params'] = sample
        g = h.create_group('joint_fit')
        g['time_ns'] = np.array([epoch], dtype=np.int64)
        g['bootstrap_x_gcrs_m'] = states[:, None, :3]
        g['bootstrap_v_gcrs_mps'] = states[:, None, 3:]
        g['bootstrap_eccentricity'] = np.full(len(states), np.nan)
    os.replace(str(path) + '.partial', path)
    return path, method


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--flags', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--days-back', type=float, default=1000)
    a = p.parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    a.output.mkdir(parents=True, exist_ok=True)
    solar_system_ephemeris.set('builtin')
    folders = sorted(a.input_dir.glob('tri_*'))
    rows, failures = [], []
    for folder in folders[rank::size]:
        started = time.monotonic()
        try:
            source, method = prepare(folder, a.flags, a.output, a.samples)
            ev = propagate_event(source, a.days_back)
            assert np.isfinite(ev['e']).all(), 'Nonfinite propagated elements'
            rows.append((ev, method))
            target = a.output / (folder.name + '.h5')
            with h5py.File(str(target) + '.partial', 'w') as h:
                for k, v in ev.items():
                    if k != 'event_id':
                        h[k] = v
                h.attrs.update(event_id=folder.name, uncertainty_method=method,
                               elapsed_seconds=time.monotonic()-started)
            os.replace(str(target) + '.partial', target)
            print('DONE', folder.name, method, round(time.monotonic()-started, 1), flush=True)
        except Exception as exc:
            failures.append((folder.name, f'{type(exc).__name__}: {exc}'))
            print('EXCLUDED', folder.name, failures[-1][1], flush=True)
    gathered = comm.gather((rows, failures), root=0)
    if rank == 0:
        rows = sorted([r for group, _ in gathered for r in group], key=lambda x:x[0]['event_id'])
        failures = sorted([r for _, group in gathered for r in group])
        assert rows, 'No current state ensembles available'
        events = [r[0] for r in rows]
        # The asteroidal/cometary criteria describe bound orbits; do not force
        # hyperbolic solutions into the cometary class.
        classified = []
        for ev in events:
            bound = np.asarray(ev['e']) < 1.
            if np.any(bound):
                classified.append({k: (v[bound] if isinstance(v, np.ndarray) else v)
                                   for k, v in ev.items()})
        summary = summarize_classification(classified)
        for row in summary:
            row['cometary_sample_fraction'] = row['cometary_event_mean_fraction']
            row['asteroidal_sample_fraction'] = row['asteroidal_event_mean_fraction']
        a.output_h5 = a.output / 'orbits.h5'
        a.output_tex = a.output / 'classification.tex'
        write_outputs(events, summary, a)
        with h5py.File(a.output_h5, 'a') as h:
            h.attrs.update(model='direct shrinking radius', samples_per_event=a.samples,
                source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
                created_unix=time.time(), input_events=len(folders), included_events=len(events),
                uncertainty='Propagated current saved state ensembles; methods recorded per event; physical systematics excluded',
                ephemeris='astropy builtin', forces='Gravity only; no radiation pressure or atmospheric drag before detection',
                classification_selection='Bound samples only; equal weight per event',
                classification_events=len(classified))
            dtype = h5py.string_dtype()
            h.create_dataset('uncertainty_method',data=[r[1] for r in rows],dtype=dtype)
            h.create_dataset('excluded_event_id',data=[r[0] for r in failures],dtype=dtype)
            h.create_dataset('exclusion_reason',data=[r[1] for r in failures],dtype=dtype)
        print('COMPLETE',len(events),'excluded',len(failures),flush=True)
        for row in summary:
            print(row['key'],row['cometary_event_mean_fraction'],flush=True)


if __name__ == '__main__':
    main()
