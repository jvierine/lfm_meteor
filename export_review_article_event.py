"""Archive the live review's scientific data for a native six-panel figure."""
from pathlib import Path
import argparse
import json
import urllib.request
import h5py
import numpy as np
from voltage_review.server import Store

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--event', default='tri_0012_1713798481419350028')
parser.add_argument('--catalog', type=Path, default=ROOT/'results/tristatic_cached_shrinking_20261029')
parser.add_argument('--flags', type=Path, default=ROOT/'results/review_flags_scatter_snapshot')
args = parser.parse_args()
EVENT = args.event
OUT = ROOT/f'results/article_review_event_{EVENT.split("_")[1]}.h5'
d = Store(args.catalog, args.flags, latest=args.catalog).data(EVENT)
assert d['annotation']['version'] == d['fit_flag_revision']
assert d['rcs_mass']
with h5py.File(OUT, 'w') as h:
    h.attrs.update(event=EVENT, source_program=str(Path(__file__).resolve()), density_kg_m3=3000.)
    from prepare_current_kinematics import sha
    with h5py.File(args.catalog/EVENT/'plot.h5') as source:
        for key, value in source.attrs.items():
            if key.startswith('uncertainty_'):
                h.attrs[key] = value
    h.attrs['fit_sha256'] = sha(args.catalog/EVENT/'fit.h5')
    h.attrs['plot_sha256'] = sha(args.catalog/EVENT/'plot.h5')
    for key, value in d.items():
        if isinstance(value, list) and value and isinstance(value[0], (float, int)):
            h[key] = value
        elif isinstance(value, (float, int, str, bool)):
            h.attrs[key] = value
    for key, value in d['beam'].items():
        if key == 'contours':
            for i, contour in enumerate(value):
                g = h.require_group(f'beam/contours/{i}')
                g.attrs['gain_db'] = contour['level']
                for j, line in enumerate(contour['lines']): g[str(j)] = line
        elif isinstance(value, list):
            h['beam/'+key] = np.asarray([np.nan if x is None else float(x) for x in value])
        else: h.require_group('beam').attrs[key] = value
    for key, value in d['rcs_mass'].items():
        if key != 'curves': h.require_group('rcs_mass').attrs[key] = value
    h['excluded_ids'] = np.asarray(d['annotation']['excluded'], dtype=h5py.string_dtype())
print(OUT)
