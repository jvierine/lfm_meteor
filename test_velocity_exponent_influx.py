"""Sensitivity of the current influx to the velocity-weight exponent."""
from pathlib import Path
import hashlib

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/submission_mass_remote_20261107/current_mass_influx.h5'
OUTPUT = ROOT / 'results/velocity_exponent_influx_20261108.h5'
EXPONENTS = np.array([3., 3.1, 3.5])


def weights_at_exponent(reference_weights, speed, exponent):
    return reference_weights * (72. / speed) ** (exponent - 3.)


def main():
    rng = np.random.default_rng(20261108)
    with h5py.File(SOURCE) as source, h5py.File(OUTPUT, 'w') as output:
        output.attrs.update(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                            source_program=Path(__file__).read_text(), seed=20261108,
                            reference_speed_km_s=72., n_realizations=20000,
                            interpretation='Effective-exponent sensitivity, not equivalent calibrations of head-echo detection probability.',
                            fixed='Mass selection, saved mass distributions, area, radiant exposure, altitude factor; same mass and Poisson draws across exponents.')
        output['exponent'] = EXPONENTS
        for kind in ['dynamic', 'rcs']:
            g = source[kind]
            names = g['event_id'].asstr()[:]
            speed = g['speed_km_s'][:]
            mass = g['mass_kg'][:]
            reference = g['plus_altitude_heuristic/weight_per_day'][:]
            np.testing.assert_allclose(np.sum(reference * mass) / 1000.,
                                       g['plus_altitude_heuristic'].attrs['rate_ton_day'])
            draws = np.array([rng.choice(g['input_mass_draws'][name][:], 20000) for name in names])
            counts = rng.poisson(1., draws.shape)
            weights = np.array([weights_at_exponent(reference, speed, p) for p in EXPONENTS])
            rates = weights @ mass / 1000.
            samples = weights @ (draws * counts) / 1000.
            np.testing.assert_allclose(weights[0], reference)
            assert np.all(np.diff(rates) > 0)
            target = output.create_group(kind)
            target['event_id'] = np.asarray(names, dtype=h5py.string_dtype())
            target['weight_per_day'] = weights
            target['rate_ton_day'] = rates
            target['interval95_ton_day'] = np.quantile(samples, [.025, .975], axis=1).T
            target['combined_draws_ton_day'] = samples
            factor = len(source['catalogue_event_id']) / len(names)
            target.attrs['missing_mass_count_factor'] = factor
            target['count_scaled_rate_ton_day'] = rates * factor
            for p, rate, interval in zip(EXPONENTS, rates, target['interval95_ton_day'][:]):
                print(kind, 'exponent', p, 'rate', round(rate, 3), '95%', np.round(interval, 3),
                      'count-scaled', round(rate * factor, 3))


if __name__ == '__main__':
    main()
