"""Event-sampling uncertainty in the existing 95%-containment collecting area."""
import ast
from pathlib import Path
import hashlib

import h5py
import numpy as np
import plot_memo09_antenna_gain_patterns as gain_model

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/current_kinematics_20261107/beam_area.h5'
OUTPUT = ROOT / 'results/collecting_area_uncertainty_20261108.h5'


def main():
    # Reuse only the numerical contour function; no plotting dependencies.
    path = ROOT / 'estimate_sanya_gain_containment_area.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'contour_area']
    namespace = dict(np=np, gain_model=gain_model)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    site = gain_model.SITES[0]
    geom = dict(site=site, pointing_el_deg=site.pointing_el_deg)
    with h5py.File(SOURCE) as source:
        gains = source['max_gain_db'][:]
        nominal = float(source.attrs['area_horizontal_km2'])
        cutoff = float(source.attrs['cutoff_relative_gain_db'])
        ngrid = 1801
        grid = namespace['contour_area'](cutoff, geom, 95., 20., ngrid)
        relative = np.sort(grid[2].ravel())
        cell_area = grid[-1] / np.count_nonzero(grid[3])
        rng = np.random.default_rng(20261108)
        index = len(gains) - int(np.ceil(.95 * len(gains)))
        resampled = rng.choice(gains, (20000, len(gains)))
        thresholds = np.partition(resampled, index, axis=1)[:, index]
        area = (len(relative)-np.searchsorted(relative, thresholds, side='left')) * cell_area
        # Anchor the approximation to the archived, finer-grid nominal area.
        area *= nominal / grid[-1]
        earth_area = 4 * np.pi * (6371. + 95.)**2
        with h5py.File(OUTPUT, 'w') as output:
            output.attrs.update(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                                source_program=Path(__file__).read_text(), seed=20261108,
                                n_events=len(gains), grid_n=ngrid, nominal_area_km2=nominal,
                                earth_area_km2=earth_area,
                                interpretation='Event-resampling uncertainty of empirical 95%-containment boundary; fixed beam model and radiant symmetry.',
                                nominal_grid_area_km2=grid[-1])
            output['area_draws_km2'] = area
            output['area_interval95_km2'] = np.quantile(area, [.025, .975])
            output['area_factor_draws'] = earth_area / area
            output['fourfold_area_factor_draws'] = 4 * earth_area / area
            output['fourfold_area_factor_interval95'] = np.quantile(4 * earth_area / area, [.025, .975])
            output['relative_rate_factor_draws'] = nominal / area
        print('Area:', nominal, '95%', np.quantile(area, [.025,.975]))
        print('Earth/area:', earth_area/nominal, 'fourfold:', 4*earth_area/nominal,
              '95%',np.quantile(4*earth_area/area,[.025,.975]))
        print('Grid check:', grid[-1], 'vs', nominal)


if __name__ == '__main__':
    main()
