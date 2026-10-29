"""Export the current reviewed masses for a native Figure 6 preview."""
from pathlib import Path
from collections import Counter
import time
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/tristatic_cached_shrinking_20261029'
FLAGS = ROOT / 'results/review_flags_scatter_snapshot'
OUT = ROOT / 'results/current_dynamic_mass_distribution'

rows = []
omitted = []
for folder in sorted(SOURCE.glob('tri_*')):
    try:
        with h5py.File(folder / 'plot.h5') as f:
            revision = 0
            flag = FLAGS / (folder.name + '.h5')
            if flag.exists():
                with h5py.File(flag) as g:
                    revision = int(g.attrs['version'])
            if int(f.attrs.get('fit_flag_revision', -1)) != revision:
                raise ValueError('new flags awaiting refit')
            if not f.attrs.get('uncertainty_converged', False):
                raise ValueError('uncertainty unresolved')
            lo, median, hi = f['initial_mass_kg_quantiles'][:]
            mass = float(f.attrs['best_initial_mass_kg'])
            speed = float(f['speed_km_s'][0])
            radius = float(f.attrs['best_initial_radius_m'])
            radius_lo, _, radius_hi = f['initial_radius_m_quantiles'][:]
            # Current direct-fit log10(radius/m) prior spans [-7, -2].
            if radius >= 0.01 * 0.999 or radius <= 1e-7 * 1.001 or radius_hi >= 0.01 * 0.99 or radius_lo <= 1e-7 * 1.01:
                raise ValueError('radius estimate or interval reaches prior boundary')
            if not np.all(np.isfinite([lo, hi, mass, speed])) or min(lo, hi, mass) <= 0:
                raise ValueError('invalid mass or interval')
            selected = np.log10(hi / lo) < 1
            rows.append((folder.name, mass, speed, lo, hi, selected,
                         str(f.attrs.get('uncertainty_method', 'saved conditional posterior'))))
    except (OSError, KeyError, ValueError) as exc:
        omitted.append((folder.name, str(exc)))
OUT.mkdir(exist_ok=True)
rcs_rows = []
with h5py.File(ROOT / 'results/current_review_mass_comparison/scatter.h5') as comparison:
    index = {event: i for i, event in enumerate(comparison['event_id'].asstr()[:])}
    for row in rows:
        if row[5] and row[0] in index:
            i = index[row[0]]
            rcs_rows.append((row[0], comparison['rcs_mass_kg'][i], row[2],
                             comparison['delta_mass_lo95_noise_only_kg'][i],
                             comparison['delta_mass_hi95_noise_only_kg'][i]))
with h5py.File(OUT / 'figure6.h5', 'w') as f:
    for i, key in enumerate(['rcs_event_id', 'rcs_mass_kg', 'rcs_speed_km_s', 'rcs_lo95_kg', 'rcs_hi95_kg']):
        f[key] = np.asarray([r[i] for r in rcs_rows], dtype=h5py.string_dtype() if i == 0 else float)
    f.attrs['rcs_meaning'] = 'Observed in-beam mass loss; same dynamic-mass-selected events with valid matched RCS products'
    for i, name in enumerate(['event_id', 'initial_mass_kg', 'initial_speed_km_s',
                              'mass_95_lo_kg', 'mass_95_hi_kg', 'selected', 'uncertainty_method']):
        vals = [r[i] for r in rows]
        f[name] = np.asarray(vals, dtype=h5py.string_dtype() if i in (0, 6) else bool if i == 5 else float)
    for i, key in enumerate(['omitted_event', 'omission_reason']):
        f[key] = np.asarray([r[i] for r in omitted], dtype=h5py.string_dtype())
    f.attrs.update(created_unix=time.time(), source_program=str(Path(__file__).resolve()),
                   source_catalogue=str(SOURCE), density_kg_m3=3000., bin_width_dex=0.48,
                   selection='Finite current 95% intervals narrower than one dex; uncertainty diagnostics passed; exclude radius-boundary estimates',
                   mass_estimator='Best fit; saved event-specific intervals, mixed interim catalogue')
print('Available', len(rows), 'selected', sum(r[5] for r in rows), 'omitted', Counter(r[1] for r in omitted))
print('Selected bootstrap', sum(r[5] and 'bootstrap' in r[6] for r in rows))
print('RCS matched', len(rcs_rows))
