"""Sensitivity of the current influx to the velocity-weight exponent."""
from pathlib import Path
import argparse
import hashlib

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/submission_mass_remote_20261107/current_mass_influx.h5'
OUTPUT = ROOT / 'results/velocity_exponent_influx_20261108.h5'
AREA = ROOT / 'results/collecting_area_uncertainty_20261108.h5'
EXPONENTS = np.array([3., 3.1, 3.5])


def weights_at_exponent(reference_weights, speed, exponent):
    return reference_weights * (72. / speed) ** (exponent - 3.)


def influx_ton_day(weights, masses):
    return np.asarray(weights) @ np.asarray(masses) / 1000.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimum-exponent', type=float, default=3.)
    parser.add_argument('--maximum-exponent', type=float, default=3.5)
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--realizations', type=int, default=20000)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--area-uncertainty', type=Path, default=AREA)
    args = parser.parse_args()
    if not (np.isfinite(args.minimum_exponent) and np.isfinite(args.maximum_exponent)
            and 3. <= args.minimum_exponent < args.maximum_exponent <= 6.
            and args.realizations > 0):
        parser.error('Require 3 <= minimum < maximum <= 6 and positive realizations')
    exponents = np.unique(np.r_[np.linspace(args.minimum_exponent, args.maximum_exponent,
                              int(np.ceil((args.maximum_exponent-args.minimum_exponent)*10))+1),
                              EXPONENTS[(EXPONENTS>=args.minimum_exponent)&(EXPONENTS<=args.maximum_exponent)],
                              5.3 if args.minimum_exponent<=5.3<=args.maximum_exponent else args.minimum_exponent])
    nrep = args.realizations
    rng = np.random.default_rng(args.seed)
    exponent_draws = np.random.default_rng(args.seed+1).uniform(args.minimum_exponent, args.maximum_exponent, nrep)
    with h5py.File(args.area_uncertainty) as area:
        area_factors = np.random.default_rng(args.seed+2).choice(area['relative_rate_factor_draws'][:], nrep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source) as source, h5py.File(args.output, 'w') as output:
        output.attrs.update(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                            source_program=Path(__file__).read_text(), seed=args.seed,
                            reference_speed_km_s=72., n_realizations=nrep,
                            minimum_exponent=args.minimum_exponent, maximum_exponent=args.maximum_exponent,
                            interpretation='Effective-exponent sensitivity, not equivalent calibrations of head-echo detection probability.',
                            fixed='Mass selection, saved mass distributions, radiant exposure, altitude factor; same mass and Poisson draws across exponents; common area uncertainty applied to final joint draws.')
        output['exponent'] = exponents
        output['uncertainty_exponent_draws'] = exponent_draws
        output['uncertainty_area_factors'] = area_factors
        output.attrs['area_uncertainty_sha256'] = hashlib.sha256(args.area_uncertainty.read_bytes()).hexdigest()
        output.attrs['exponent_uncertainty'] = f'One uniform({args.minimum_exponent},{args.maximum_exponent}) exponent shared by all events and both mass estimators per realization.'
        for kind in ['dynamic', 'rcs']:
            g = source[kind]
            names = g['event_id'].asstr()[:]
            speed = g['speed_km_s'][:]
            mass = g['mass_kg'][:]
            reference = g['plus_altitude_heuristic/weight_per_day'][:]
            np.testing.assert_allclose(np.sum(reference * mass) / 1000.,
                                       g['plus_altitude_heuristic'].attrs['rate_ton_day'])
            draws = np.array([rng.choice(g['input_mass_draws'][name][:], nrep) for name in names])
            counts = rng.poisson(1., draws.shape)
            weights = np.array([weights_at_exponent(reference, speed, p) for p in exponents])
            rates = influx_ton_day(weights, mass)
            samples = influx_ton_day(weights, draws * counts)
            if exponents[0] == 3.:
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
            sampled_weights = reference[:, None] * (72. / speed[:, None]) ** (exponent_draws[None, :] - 3.)
            joint = np.sum(sampled_weights * draws * counts, axis=0) / 1000.
            target['mass_count_exponent_draws_ton_day'] = joint
            target['mass_count_exponent_interval95_ton_day'] = np.quantile(joint, [.025, .975])
            target['count_scaled_joint_interval95_ton_day'] = factor * np.quantile(joint, [.025, .975])
            print(kind, 'combined exponent uncertainty 95%', np.quantile(joint, [.025,.975]),
                  'count-scaled', factor*np.quantile(joint, [.025,.975]), flush=True)
            with_area = joint * area_factors
            target['mass_count_exponent_area_draws_ton_day'] = with_area
            target['mass_count_exponent_area_interval95_ton_day'] = np.quantile(with_area, [.025,.975])
            target['count_scaled_area_interval95_ton_day'] = factor*np.quantile(with_area, [.025,.975])
            print(kind, 'including area 95%', np.quantile(with_area, [.025,.975]),
                  'count-scaled', factor*np.quantile(with_area, [.025,.975]), flush=True)
            for p, rate, interval in zip(exponents, rates, target['interval95_ton_day'][:]):
                print(kind, 'exponent', p, 'rate', round(rate, 3), '95%', np.round(interval, 3),
                      'count-scaled', round(rate * factor, 3))


if __name__ == '__main__':
    main()
