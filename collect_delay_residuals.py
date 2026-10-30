"""Collect current, reviewed physical-fit residuals without a mass-quality cut."""
import argparse
from pathlib import Path
import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('flags', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
rows, excluded = [], []
for folder in sorted(a.root.glob('tri_*')):
    try:
        flag = a.flags / (folder.name + '.h5')
        revision, bad, omitted = 0, False, set()
        if flag.exists():
            with h5py.File(flag) as h:
                revision = int(h.attrs['version'])
                bad = bool(h.attrs.get('bad', False))
                omitted = set(h['excluded'].asstr()[:])
        if bad:
            raise ValueError('review-rejected event')
        with h5py.File(folder / 'plot.h5') as h:
            if 'shrinking' not in str(h.attrs.get('trajectory_model', '')).lower():
                raise ValueError('not a direct shrinking-radius fit')
            if int(h.attrs.get('fit_flag_revision', 0)) != revision:
                raise ValueError('fit predates current review flags')
            station, times = h['station'][:], h['time_ns'][:]
            observed, model = h['seed_apparent_path_m'][:], h['modeled_path_m'][:]
            keep = np.array([f'{int(s)}:{int(t)}' not in omitted for s,t in zip(station,times)])
            keep &= np.isfinite(observed) & np.isfinite(model)
            rows.append((folder.name, station[keep], observed[keep], model[keep]))
    except (OSError, KeyError, ValueError) as exc:
        excluded.append((folder.name, str(exc)))
assert rows
a.output.mkdir(parents=True, exist_ok=True)
with h5py.File(a.output/'plot.h5', 'w') as h:
    h['station'] = np.concatenate([r[1] for r in rows])
    h['seed_apparent_path_m'] = np.concatenate([r[2] for r in rows])
    h['modeled_path_m'] = np.concatenate([r[3] for r in rows])
    for key, value in [('events', [r[0] for r in rows]), ('excluded_event', [r[0] for r in excluded]),
                       ('exclusion_reason', [r[1] for r in excluded])]:
        h.create_dataset(key, data=value, dtype=h5py.string_dtype())
    h.attrs.update(n_events=len(rows), source_root=str(a.root), flags_root=str(a.flags),
                   source_program=Path(__file__).read_text(), residual_convention='model minus observed apparent path; metres',
                   clipping='none', mass_quality_cut=False)
    r = h['modeled_path_m'][:]-h['seed_apparent_path_m'][:]
    print('Events', len(rows), 'measurements',len(r),'RMS',np.sqrt(np.mean(r*r)), 'range',np.min(r),np.max(r))
    print('Excluded', excluded)
