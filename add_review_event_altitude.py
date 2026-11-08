"""Attach fitted altitude to beam points, allowing float UTC rounding only."""
import argparse
from pathlib import Path
import h5py
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', nargs='+', type=Path)
    parser.add_argument('--kinematics', type=Path,
                        default=Path('results/current_kinematics_20261107/kinematics.h5'))
    args = parser.parse_args()
    with h5py.File(args.kinematics) as source:
        for path in args.events:
            with h5py.File(path, 'r+') as target:
                event = str(target.attrs['event'])
                states = source['events/' + event]
                if 'fit_sha256' in target.attrs:
                    assert target.attrs['fit_sha256'] == states.attrs['fit_sha256']
                else:
                    with h5py.File(Path('results/tristatic_cached_shrinking_20261029') / event / 'plot.h5') as plot:
                        np.testing.assert_allclose(target['modeled_path_m'][:],
                                                   plot['modeled_path_m'][:], rtol=0, atol=1e-6)
                    import hashlib
                    fit = Path('results/tristatic_cached_shrinking_20261029') / event / 'fit.h5'
                    assert hashlib.sha256(fit.read_bytes()).hexdigest() == states.attrs['fit_sha256']
                times = states['time_ns'][:]
                heights = states['height_km'][:]
                # Beam products also include pulses excluded from the fit.
                # Use the saved evaluation of that same trajectory at all pulses.
                import jcoord
                physical = Path('results/tristatic_cached_shrinking_20261029') / event / 'physical_states.h5'
                with h5py.File(physical) as full:
                    all_times = full['time_ns'][:]
                    positions = full['position_ecef_m'][:]
                    match = {int(t): i for i, t in enumerate(all_times)}
                    np.testing.assert_allclose(positions[[match[int(t)] for t in times]],
                                               states['position_ecef_m'][:], rtol=0, atol=1e-3)
                    times = all_times
                    heights = np.array([jcoord.ecef2geodetic(*p)[2]/1000 for p in positions])
                beam = target['beam']
                indices = []
                for t in beam['time_ns'][:]:
                    delta = np.abs(times - int(t))
                    nearest = int(np.argmin(delta))
                    # The review export stores UTC nanoseconds as float64.
                    assert delta[nearest] <= 256, 'No matching fitted pulse'
                    indices.append(nearest)
                altitude = heights[indices]
                assert altitude.shape == beam['east_m'].shape
                assert np.all(np.isfinite(altitude))
                if 'height_km' in beam:
                    np.testing.assert_allclose(beam['height_km'][:], altitude)
                else:
                    beam['height_km'] = altitude
                beam['height_km'].attrs.update(units='km', source=str(args.kinematics),
                                               matching='nearest UTC within 256 ns float rounding, verified fit SHA256')
                print(path, altitude.min(), altitude.max())

if __name__ == '__main__':
    main()
