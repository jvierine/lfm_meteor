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
                       manual_flag_sha256=hashlib.sha256(Path(flags).read_bytes()).hexdigest(),
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
    args = p.parse_args()
    comm = MPI.COMM_WORLD
    events = []
    for flag in sorted(Path(args.flags).glob('tri_*.h5')):
        with h5py.File(flag) as h:
            if len(h['excluded']):
                events.append(flag.stem)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if comm.rank == 0:
        print(f'Refitting {len(events)} manually flagged events on {comm.size} workers', flush=True)
    lib = driver.load_library(args.library)
    for event in events[comm.rank::comm.size]:
        def prepare(path, unused_events, unused_reference):
            return masked_input(Path(args.source)/event/'input.h5',
                                Path(args.flags)/(event+'.h5'), path)
        driver.prepare = prepare
        try:
            print('START '+event, flush=True)
            result = driver.run_one(args, event, [], {}, lib)
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
