"""Snapshot matched direct-fit and RCS masses; render with native Rustplotlib."""
from pathlib import Path
import hashlib
import os
import subprocess
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/tristatic_cached_shrinking_20261029'
FLAGS = ROOT / 'results/review_flags_scatter_snapshot'
OUT = ROOT / 'results/current_review_mass_comparison'


def main():
    rows, skipped = [], []
    for folder in sorted(SOURCE.glob('tri_*')):
        try:
            with h5py.File(folder/'plot.h5') as p, h5py.File(folder/'rcs_mass.h5') as r:
                flag = FLAGS/(folder.name+'.h5')
                for key, file in [('fit_sha256', folder/'fit.h5'), ('beam_sha256', folder/'beam_rcs.h5'), ('flag_sha256', flag)]:
                    digest = hashlib.sha256(file.read_bytes() if file.exists() else b'').hexdigest()
                    if r.attrs.get(key) != digest:
                        raise ValueError('RCS product awaiting refresh')
                revision = 0
                if flag.exists():
                    with h5py.File(flag) as f: revision = int(f.attrs['version'])
                if int(p.attrs.get('fit_flag_revision', -1)) != revision:
                    raise ValueError('new flags awaiting refit')
                if not p.attrs.get('uncertainty_converged', False):
                    raise ValueError('uncertainty unresolved')
                lo, median, hi = p['initial_mass_kg_quantiles'][:]
                row = [p.attrs['best_initial_mass_kg'], r.attrs['mass_kg'], lo, hi,
                       r.attrs['lo95_scatter_kg'], r.attrs['hi95_scatter_kg'],
                       r.attrs['rcs_rms_db'], r.attrs.get('uncertainty_clamped_fraction', 0)]
                if not np.all(np.isfinite(row)) or min(row[:6]) <= 0:
                    raise ValueError('nonfinite or nonpositive interval')
                rows.append((folder.name, row))
        except (OSError, KeyError, ValueError) as e:
            skipped.append((folder.name, str(e)))
    OUT.mkdir(exist_ok=True)
    with h5py.File(OUT/'scatter.h5', 'w') as h:
        h['event_id'] = np.asarray([r[0] for r in rows], dtype=h5py.string_dtype())
        for i, key in enumerate(['dynamic_mass_kg', 'rcs_mass_kg', 'dynamic_lo95_kg', 'dynamic_hi95_kg',
                                 'delta_mass_lo95_noise_only_kg', 'delta_mass_hi95_noise_only_kg', 'rcs_rms_db', 'boundary']):
            h[key] = np.asarray([r[1][i] for r in rows])
        h['omitted_event'] = np.asarray([r[0] for r in skipped], dtype=h5py.string_dtype())
        h['omission_reason'] = np.asarray([r[1] for r in skipped], dtype=h5py.string_dtype())
        h.attrs.update(marginal_initial_mass=True, source_program=str(Path(__file__).resolve()),
                       dynamic_uncertainty='Saved direct shrinking-radius marginal posterior 95% interval',
                       rcs_uncertainty='Event dBsm RMS shared-offset 95% interval; legacy dataset names retained for renderer',
                       rcs_meaning='Observed in-beam mass loss, not initial mass')
    print('Matched', len(rows), 'omitted', len(skipped), flush=True)
    binary = ROOT.parent/'rustplotlib/target/release/native-paper-figures'
    subprocess.run([str(binary), 'memo-rcs-dynamic', str(OUT/'scatter.h5'), str(OUT/'scatter')],
                   env=dict(os.environ, DYLD_LIBRARY_PATH='/opt/anaconda3/lib'), check=True)


if __name__ == '__main__':
    main()
