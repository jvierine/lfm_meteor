"""Count-ratio sensitivity estimate for missing masses in the 232-event sample."""
from pathlib import Path
import hashlib

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/submission_mass_remote_20261107/current_mass_influx.h5'
OUTPUT = ROOT / 'results/missing_mass_count_approximation_20261108.h5'


def main():
    with h5py.File(SOURCE) as source, h5py.File(OUTPUT, 'w') as output:
        ids = source['catalogue_event_id'].asstr()[:]
        assert len(ids) == len(set(ids)) == 232
        assert np.all(source['usable_weight'][:])
        output.attrs.update(
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            source_program=Path(__file__).read_text(),
            assumption='Missing events have the same average weighted mass contribution as measured events.',
            limitation='A fixed count-ratio approximation, not a calibrated correction for mass-dependent missingness.',
            n_catalogue=len(ids))
        for kind in ['dynamic', 'rcs']:
            group = source[kind]
            measured = group['event_id'].asstr()[:]
            assert len(measured) == len(set(measured))
            assert set(measured).issubset(ids)
            factor = len(ids) / len(measured)
            result = output.create_group(kind)
            result.attrs.update(n_measured=len(measured), n_missing=len(ids)-len(measured), factor=factor)
            result['missing_event_id'] = np.asarray(sorted(set(ids)-set(measured)), dtype=h5py.string_dtype())
            for stage in ['area_exposure', 'plus_velocity_heuristic', 'plus_altitude_heuristic']:
                original = group[stage]
                target = result.create_group(stage)
                target.attrs['measured_rate_ton_day'] = original.attrs['rate_ton_day']
                target.attrs['approximate_rate_ton_day'] = factor * original.attrs['rate_ton_day']
            print(kind, 'N=', len(measured), 'factor=', factor,
                  'measured=', result['plus_altitude_heuristic'].attrs['measured_rate_ton_day'],
                  'approximate=', result['plus_altitude_heuristic'].attrs['approximate_rate_ton_day'])


if __name__ == '__main__':
    main()
