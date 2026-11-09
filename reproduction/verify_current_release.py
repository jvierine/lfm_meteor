"""Check an extracted reviewer release without importing active project code."""
import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    for inventory in ('release-inventory.h5', 'source/inventory.h5'):
        with h5py.File(root / inventory) as h:
            for path, expected in zip(h['path'].asstr()[:], h['sha256'].asstr()[:]):
                assert digest(root / path) == expected, path
    data = root / 'results/tristatic_calibrated_likelihood_20261109'
    with h5py.File(data / 'mass_selection.h5') as h:
        events = h['events'].asstr()[:]
        selected = events[h['dynamic_mass_included'][:]]
        assert len(events) == len(set(events)) == 234
        assert len(selected) == 135
        assert all((root / 'L1/events' / e / 'input.h5').exists() for e in events)
    with h5py.File(data / 'products/mass_distribution.h5') as h:
        np.testing.assert_array_equal(selected, h['event_id'].asstr()[:])
        mass, lo, hi = (h[k][:] for k in ('initial_mass_kg', 'mass_95_lo_kg', 'mass_95_hi_kg'))
        assert np.all((lo > 0) & (lo <= mass) & (mass <= hi) & (hi < 10*lo))
        assert len(h['rcs_event_id']) == 115
        speed = h['initial_speed_km_s'][:]
    with (h5py.File(data / 'products/mass_influx.h5') as h,
          h5py.File(data / 'products/imputed_mass_influx.h5') as result):
        w = h['beam_sampling/weight_ton_day_per_kg'][:]
        bounds = np.column_stack((lo, hi))
        values = []
        for p in (3., 3.5):
            contributions = bounds * (w*(72/speed)**(p-3))[:, None]
            values.append(contributions.sum(axis=0) + 99*np.median(contributions, axis=0))
        np.testing.assert_allclose(values, result['beam_sampling/exponent_sensitivity/endpoint_envelopes_ton_day'][:])
        dynamic_range = np.array([values[0][0], values[1][1]])
        np.testing.assert_allclose(dynamic_range, [33.74410330, 169.13269675], rtol=1e-7)
        with h5py.File(data / 'current_mass_influx.h5') as old:
            rcs = old['rcs']
            names = rcs['event_id'].asstr()[:]
            assert len(names) == 197
            bounds = np.array([np.quantile(rcs['input_mass_draws/'+e][:], [.025, .975]) for e in names])
            speed = rcs['speed_km_s'][:]
            w = rcs['plus_altitude_heuristic/weight_per_day'][:]/1000
            values = []
            for p in (3., 3.5):
                contributions = bounds * (w*(72/speed)**(p-3))[:, None]
                values.append(contributions.sum(axis=0) + 37*np.median(contributions, axis=0))
            np.testing.assert_allclose(values, result['rcs/beam_sampling/exponent_sensitivity/endpoint_envelopes_ton_day'][:])
            rcs_range = [values[0][0], values[1][1]]
    with h5py.File(root / 'orbits/orbits.h5') as h:
        assert len(h['event_id']) == 216
        assert int(h['valid'][:].sum()) == 21600
        assert np.all(h['e'][:][h['valid'][:].astype(bool)] < 1)
    print('Verified: 234 trajectories; 135 selected dynamic masses; 197 RCS masses; 115 pairs.')
    print('Median-contribution sensitivity ranges (tonnes/day):', dynamic_range, rcs_range)
    print('Verified: 216 orbital ensembles, 21600 bound samples; all inventory hashes match.')


if __name__ == '__main__':
    main()
