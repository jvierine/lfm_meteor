"""Freeze review masks and run the native seven-parameter posterior catalogue."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

from refit_voltage_reviewed import ROW_KEYS
from fit_shrinking_voltage import save


def prepare(source, flags, target):
    target.mkdir(parents=True, exist_ok=True)
    with h5py.File(source / 'input.h5') as h:
        fields = {k: h[k][()] for k in h}
        attrs = dict(h.attrs)
    old = {x.decode() if isinstance(x, bytes) else str(x)
           for x in attrs.get('manually_excluded_ids', [])}
    excluded, revision = set(), 0
    if flags.exists():
        with h5py.File(flags) as h:
            excluded = set(h['excluded'].asstr()[:])
            revision = int(h.attrs.get('version', 0))
    if old - excluded:
        raise ValueError(f'{source.name}: restored measurements require original input')
    ids = [f'{int(s)}:{int(t)}' for s, t in zip(fields['station'], fields['time_ns'])]
    if len(set(ids)) != len(ids) or excluded - (set(ids) | old):
        raise ValueError(f'{source.name}: unmatched or duplicate measurement IDs')
    keep = np.array([x not in excluded for x in ids])
    for key in ROW_KEYS:
        fields[key] = fields[key][keep]
    attrs.update(manual_flag_revision=revision, manually_excluded_ids=sorted(excluded),
                 manually_excluded_count=len(excluded), posterior_source=str(source))
    save(target / 'input.h5', fields, attrs)
    shutil.copy2(source / 'fit.h5', target / 'fit.h5')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'flags', 'output', 'library'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cores', type=int, default=128)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    frozen, snapshot = a.output / 'inputs', a.output / 'flags_snapshot'
    snapshot.mkdir(exist_ok=True)
    events = sorted(x.parent.name for x in a.source.glob('tri_*/input.h5'))
    for event in events:
        flag = a.flags / (event + '.h5')
        if flag.exists():
            shutil.copy2(flag, snapshot / flag.name)
        prepare(a.source / event, snapshot / flag.name, frozen / event)
    save(a.output / 'launch.h5', {'events': np.asarray(events, dtype='S')},
         {'started_unix': time.time(), 'cores': a.cores, 'event_count': len(events),
          'method': 'seven-parameter native independent importance posterior',
          'paper_products_unchanged': True})
    batch_size = a.cores // 4
    failures = []
    for start in range(0, len(events), batch_size):
        batch = events[start:start + batch_size]
        print('BATCH', start, len(batch), flush=True)
        command = ['mpirun', '--bind-to', 'core', '-np', str(4 * len(batch)),
                   sys.executable, '-u', 'sample_seven_parameter_importance.py',
                   '--source', str(frozen), '--flags', str(snapshot),
                   '--output', str(a.output / 'posterior'), '--library', str(a.library),
                   '--optimize-seconds', '5', '--events', *batch]
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.extend(batch)
        save(a.output / 'progress.h5', {'failed_batch_events': np.asarray(failures, dtype='S')},
             {'processed_events': min(start + len(batch), len(events)),
              'total_events': len(events), 'updated_unix': time.time()})
    verified, unresolved = [], []
    for event in events:
        path = a.output / 'posterior' / event / 'posterior.h5'
        good = False
        if path.exists():
            with h5py.File(path) as h:
                good = bool(h.attrs['convergence_verified'])
        (verified if good else unresolved).append(event)
    save(a.output / 'complete.h5',
         {'verified_events': np.asarray(verified, dtype='S'),
          'unresolved_events': np.asarray(unresolved, dtype='S')},
         {'finished_unix': time.time(), 'event_count': len(events),
          'verified_count': len(verified), 'unresolved_count': len(unresolved)})
    print('FINISHED', len(verified), 'verified;', len(unresolved), 'unresolved', flush=True)


if __name__ == '__main__':
    main()
