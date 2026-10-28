"""Detached validation gate followed by the full cached-observation catalogue.

Three injections are recovery checks, not a frequentist coverage certificate.
The validation catalogue never replaces the published catalogue automatically.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import h5py
import numpy as np

from make_tristatic_posterior_injection import make_injection
from refit_tristatic_mass_posterior import process_event, GIT_REVISION, VERSION


CHAIN_FLAGS = {
    'chain_too_short_for_autocorrelation_time', 'mass_chain_nonstationary',
    'autocorrelation_estimate_unstable', 'low_walker_acceptance',
}
WEAK = 'tri_0010_1713798090639350891'
STRONG = 'tri_0162_1713823274259349823'


def write_status(root, phase, **details):
    row = dict(phase=phase, unix_time=time.time(), git_revision=GIT_REVISION, **details)
    print(json.dumps(row), flush=True)
    with h5py.File(root / 'pipeline_status.partial.h5', 'w') as h:
        for key, value in row.items():
            h.attrs[key] = value
    (root / 'pipeline_status.partial.h5').replace(root / 'pipeline_status.h5')


def read_result(path):
    with h5py.File(path) as h:
        p = h['mass_posterior']
        return p['mass_quantiles_kg'][:], set(p.attrs['flags'].split(';'))


def evaluate_validation(root):
    reasons, records = [], []
    injection_sources = [root / 'injections' / f'joint_delay_doppler_fft_tri_synthetic_{r}um.h5' for r in (20, 30, 60)]
    for source in injection_sources:
        if not source.exists():
            reasons.append(f'missing injection input: {source.stem}')
            continue
        result = root / 'validation' / 'injections' / source.name
        if not result.exists():
            reasons.append(f'missing injection result: {source.stem}')
            continue
        q, flags = read_result(result)
        with h5py.File(source) as h:
            truth = float(h.attrs['true_mass_kg'])
        recovered = bool(q[0] <= truth <= q[2])
        stable = not bool(flags & CHAIN_FLAGS)
        records.append((source.stem, truth, *q, recovered, stable))
        if not recovered or not stable:
            reasons.append(f'injection recovery/convergence check failed: {source.stem}')
    agreement = {}
    for event in (STRONG, WEAK):
        paths = [root / 'validation' / f'replica_{i}' / f'joint_delay_doppler_fft_{event}.h5' for i in (1, 2)]
        if not all(p.exists() for p in paths):
            reasons.append(f'missing independent fit: {event}')
            continue
        results = [read_result(p) for p in paths]
        delta = float(np.max(np.abs(np.log10(results[0][0] / results[1][0]))))
        stable = all(not bool(flags & CHAIN_FLAGS) for _, flags in results)
        agreement[event] = (delta, stable)
        # A weak event is allowed to remain unconstrained; never call its mass
        # validated merely because the better-conditioned controls passed.
        if event == STRONG and (delta > .15 or not stable):
            reasons.append(f'independent strong-event posterior check failed: {event}')
    with h5py.File(root / 'validation_report.h5', 'w') as h:
        h.attrs['passed'] = not reasons
        h.attrs['failures'] = '; '.join(reasons)
        h.attrs['scope'] = 'Three synthetic recovery cases and independent-ensemble checks; not a coverage certification.'
        h.attrs['maximum_independent_quantile_difference_dex'] = .15
        h.create_dataset('injection_id', data=[r[0] for r in records], dtype=h5py.string_dtype())
        h['injection_truth_and_mass_quantiles_kg'] = np.asarray([r[1:5] for r in records])
        h['injection_recovered_and_chain_stable'] = np.asarray([r[5:] for r in records], dtype=bool)
        for event, (delta, stable) in agreement.items():
            g = h.create_group(event)
            g.attrs['maximum_quantile_difference_dex'] = delta
            g.attrs['both_chains_stable'] = stable
    return reasons


def summarize_catalog(root, sources):
    rows = []
    for source in sources:
        path = root / 'catalog' / source.name
        if path.exists():
            q, flags = read_result(path)
            with h5py.File(path) as h:
                p = h['mass_posterior']
                rms = (p.attrs['rms_path_residual_m'], p.attrs['rms_beat_residual_hz'])
            complete = all(path.with_suffix(s).exists() for s in ('.pdf', '.png'))
            rows.append((source.stem, q, ';'.join(sorted(flags - {''})), complete, rms))
    with h5py.File(root / 'catalog_summary.h5', 'w') as h:
        h.attrs['n_requested'] = len(sources)
        h.attrs['n_results'] = len(rows)
        h.attrs['version'] = VERSION
        h.attrs['git_revision'] = GIT_REVISION
        h.create_dataset('event_id', data=[r[0] for r in rows], dtype=h5py.string_dtype())
        h['mass_quantiles_kg'] = np.asarray([r[1] for r in rows]).reshape(-1, 3)
        h.create_dataset('flags', data=[r[2] for r in rows], dtype=h5py.string_dtype())
        h['plots_complete'] = np.asarray([r[3] for r in rows], dtype=bool)
        h['rms_path_m_and_beat_hz'] = np.asarray([r[4] for r in rows]).reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=100)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--max-steps', type=int, default=32000)
    args = parser.parse_args()
    if args.jobs < 1 or args.steps < 100 or args.max_steps < args.steps:
        parser.error('Require jobs >= 1 and max-steps >= steps >= 100')
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    try:
        sources = sorted(args.input_dir.glob('joint_delay_doppler_fft_tri_*.h5'))
        if not sources:
            raise RuntimeError('No cached events')
        write_status(root, 'validation_running', n_events=len(sources), n_validation=7)
        source = args.input_dir / f'joint_delay_doppler_fft_{WEAK}.h5'
        tasks = []
        for radius, seed in ((20., 34521), (30., 34522), (60., 34523)):
            injected = root / 'injections' / f'joint_delay_doppler_fft_tri_synthetic_{int(radius)}um.h5'
            if not injected.exists():
                make_injection(source, injected, radius, seed)
            tasks.append((injected, root / 'validation' / 'injections', seed))
        for replica, seed in ((1, 20261028), (2, 592734)):
            for event in (STRONG, WEAK):
                tasks.append((args.input_dir / f'joint_delay_doppler_fft_{event}.h5',
                              root / 'validation' / f'replica_{replica}', seed))
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
            futures = []
            for source, destination, seed in tasks:
                destination.mkdir(parents=True, exist_ok=True)
                futures.append(pool.submit(process_event, source, destination, args.steps, 40, seed, args.max_steps))
            for i, future in enumerate(as_completed(futures), 1):
                print(json.dumps(future.result()), flush=True)
                write_status(root, 'validation_running', n_events=len(sources), n_validation_finished=i, n_validation=7)
        reasons = evaluate_validation(root)
        if reasons:
            write_status(root, 'validation_needs_review', reason='; '.join(reasons))
            return 2
        destination = root / 'catalog'
        destination.mkdir(exist_ok=True)
        write_status(root, 'catalog_running', n_events=len(sources), n_finished=0)
        failures = 0
        with ProcessPoolExecutor(max_workers=min(args.jobs, 100)) as pool:
            futures = [pool.submit(process_event, source, destination, args.steps, 40, 20261028, args.max_steps)
                       for source in sources]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                failures += row['status'] == 'error'
                print(json.dumps(row), flush=True)
                write_status(root, 'catalog_running', n_events=len(sources), n_finished=i, n_failed=failures)
        summarize_catalog(root, sources)
        write_status(root, 'complete', n_events=len(sources), n_finished=len(sources), n_failed=failures)
        return 0
    except Exception as exc:
        write_status(root, 'failed', reason=str(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
