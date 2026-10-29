"""Refit an immutable snapshot of exact manually rejected station/pulse IDs."""
import argparse
import hashlib
import os
from pathlib import Path
import traceback

import h5py
import numpy as np
from mpi4py import MPI
import run_native_voltage_catalog as driver

ROW_KEYS = ('time_s', 'time_ns', 'frames', 'rx', 'station', 'real', 'imag',
            'noise_power', 'whitener_real', 'whitener_imag', 'sample_time_s', 'gate')


def masked_input(source, flags, target):
    with h5py.File(source) as h:
        data = {k: h[k][()] for k in h}
        attrs = dict(h.attrs)
    excluded, revision = set(), 0
    if Path(flags).exists():
        with h5py.File(flags) as h:
            excluded = set(h['excluded'].asstr()[:])
            revision = int(h.attrs['version'])
    ids = [f'{s}:{t}' for s, t in zip(data['station'], data['time_ns'])]
    if len(set(ids)) != len(ids) or excluded - set(ids):
        raise ValueError('Duplicate or unmatched manual measurement IDs')
    keep = np.array([i not in excluded for i in ids])
    first = int(np.argmin(data['time_s']))
    data['initial_detection_time_s'] = data['time_s'][first]
    data['initial_detection_time_ns'] = data['time_ns'][first]
    data['initial_detection_frame'] = data['frames'][first].copy()
    for s in range(3):
        times = data['time_s'][keep & (data['station'] == s)]
        if len(times) < 3 or np.ptp(times) <= 0:
            raise ValueError(f'Insufficient retained measurements at station {s}')
    for key in ROW_KEYS:
        data[key] = data[key][keep]
    with h5py.File(target, 'w') as h:
        for k, v in data.items():
            h[k] = v
        h.attrs.update(attrs)
        h.attrs.update(manual_flag_revision=revision,
                       manual_flag_sha256=hashlib.sha256(Path(flags).read_bytes()).hexdigest() if Path(flags).exists() else 'no manual exclusions',
                       manually_excluded_count=len(excluded), original_measurement_count=len(ids),
                       original_input=str(source))
        h.attrs['manually_excluded_ids'] = sorted(excluded)
    return data


def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'flags', 'output', 'library'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--max-nfev', type=int, default=150)
    p.add_argument('--warmup', type=int, default=4000)
    p.add_argument('--draws', type=int, default=8000)
    p.add_argument('--all-events',action='store_true')
    p.add_argument('--sample-all',action='store_true')
    p.add_argument('--review-products',action='store_true')
    p.add_argument('--pending-only',action='store_true',help='Distribute only unfinished events across MPI workers')
    args = p.parse_args()
    comm = MPI.COMM_WORLD
    events = []
    for flag in sorted(Path(args.flags).glob('tri_*.h5')):
        with h5py.File(flag) as h:
            if len(h['excluded']):
                events.append(flag.stem)
    if args.all_events:
        events=sorted(p.parent.name for p in Path(args.source).glob('tri_*/input.h5'))
    if args.pending_only:
        events=[event for event in events if not (Path(args.output)/event/'complete.h5').exists()]
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if comm.rank == 0:
        print(f'Refitting {len(events)} events on {comm.size} workers; sample_all={args.sample_all}', flush=True)
    lib = driver.load_library(args.library)
    for event in events[comm.rank::comm.size]:
        def prepare(path, unused_events, unused_reference):
            return masked_input(Path(args.source)/event/'input.h5',
                                Path(args.flags)/(event+'.h5'), path)
        driver.prepare = prepare
        try:
            print('START '+event, flush=True)
            result = driver.run_one(args, event, [], {}, lib)
            if args.review_products:
                from voltage_path_uncertainty import add_intervals
                from export_voltage_exclusions import export as excluded_export
                add_intervals(Path(args.output)/event)
                excluded_export(Path(args.output)/event,lib)
            print('DONE '+event+' '+result, flush=True)
        except Exception:
            error = traceback.format_exc()
            folder = Path(args.output)/event
            folder.mkdir(exist_ok=True)
            (folder/'error.log').write_text(error)
            print('ERROR '+event+' '+error, flush=True)
    comm.Barrier()
    if comm.rank == 0:
        with h5py.File(Path(args.output)/'run_complete.h5', 'w') as h:
            h.attrs.update(requested_events=len(events), provisional=True)
            h['events'] = np.asarray(events, dtype=h5py.string_dtype())
        print('ALL WORKERS FINISHED', flush=True)


if __name__ == '__main__':
    main()
