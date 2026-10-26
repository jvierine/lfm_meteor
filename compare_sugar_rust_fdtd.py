"""Collect independently computed native Maxwell results beside Sugar's archive."""
import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.runs.glob('wp*.h5')):
        with h5py.File(path) as h:
            if h.attrs['normalization'] != 'archive_wp':
                continue
            rows.append((path, dict(h.attrs), int(h['n'][()]), float(h['dx_m'][()])))
    if not rows:
        raise RuntimeError('No completed native FDTD results')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.archive) as source, h5py.File(args.output, 'w') as output:
        for name in source:
            source.copy(name, output)
        for name, value in source.attrs.items():
            output.attrs[name] = value
        g = output.create_group('native_rust')
        g.attrs['status'] = 'Independent Maxwell pilots, not convergence-certified'
        g.attrs['source_program'] = Path(__file__).read_text()
        for name, key in [('ablation_rate_particles_s', 'archive_ablation_rate_particles_s'),
                          ('rcs_m2', 'computed_rcs_m2'),
                          ('difference_from_archive_db', 'difference_from_archive_db')]:
            g[name] = np.array([r[1][key] for r in rows])
        g['n'] = np.array([r[2] for r in rows])
        g['dx_m'] = np.array([r[3] for r in rows])
        for i, (path, attrs, n, dx) in enumerate(rows):
            with h5py.File(path) as h:
                case = g.create_group(f'case_{i:03d}')
                for name, value in attrs.items():
                    case.attrs[name] = value
                for name in h:
                    h.copy(name, case)
            print(f'{path.name}: {10*np.log10(attrs["computed_rcs_m2"]):.3f} dBsm; '
                  f'difference {attrs["difference_from_archive_db"]:+.3f} dB')


if __name__ == '__main__':
    main()
