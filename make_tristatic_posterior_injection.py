"""One end-to-end synthetic recovery case, not a coverage certification."""
from pathlib import Path
import argparse
import h5py
import numpy as np
from scipy import stats

import fit_all_ballistic_snr_weighted as base
import fit_event_joint_delay_doppler_fft as fit
from fit_whipple_jacchia_catalog_from_h5 import load_group
from refit_tristatic_mass_posterior import attach_geodetic_height, save_group


def make_injection(source_path, out, radius_um=30., seed=34522):
    with h5py.File(source_path) as h:
        source = load_group(h['joint_fit'])
        fft = load_group(h['fft_observations'])
    times = source['time_ns']
    t = (times - times[0]).astype(float) * 1e-9
    density, _ = base.density_interpolator(times, source['x_itrs_m'])
    attach_geodetic_height(density, times)
    density.ceplecha_sigma_kg_j = 1e-8
    radius = radius_um * 1e-6
    p = np.r_[source['params'][:3], source['v_gcrs_mps'][0], np.log10(radius)]
    model = fit.forward_model_for_kind(p, t, times, density, 'ceplecha')
    assert model['ceplecha_success']
    source['params'] = p
    source['dynamical_model'] = 'ceplecha'
    source['x_itrs_m'] = model['x_itrs_m']
    source['x_gcrs_m'] = model['x_gcrs_m']
    source['v_gcrs_mps'] = model['v_gcrs_mps']
    source['path_sigma_m'] = np.full((len(t), 3), 4.)
    source['fft_sigma_hz'] = np.full((len(t), 3), 300.)
    source['path_keep'] = np.ones((len(t), 3), bool)
    source['fft_keep'] = np.ones((len(t), 3), bool)
    rng = np.random.default_rng(seed)
    gaussian = np.empty((len(t), 3, 2))
    chol = np.linalg.cholesky([[1., .8], [.8, 1.]])
    gaussian[0] = rng.normal(size=(3, 2)) @ chol.T
    for i in range(1, len(t)):
        gaussian[i] = .4 * gaussian[i-1] + np.sqrt(1-.4**2) * (rng.normal(size=(3, 2)) @ chol.T)
    noise = stats.t.ppf(stats.norm.cdf(gaussian), [1.5, 3.])
    measured = model['apparent_path_length_m'] + noise[:, :, 0] * source['path_sigma_m']
    alpha = fit.gfit.NOMINAL_CHIRP_RATE_HZ_PER_S / fit.gfit.C
    # Recompute the beat at each sampled delay anchor, not the original anchor.
    beat = model['doppler_hz'] - alpha * (measured - model['path_length_m']) + noise[:, :, 1] * source['fft_sigma_hz']
    source['measured_total_paths_m'] = measured
    source['observed_fft_beat_hz'] = beat
    source['fft_model'] = 'range_offset_corrected_beat'
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, 'w') as h:
        save_group(h.create_group('joint_fit'), source)
        save_group(h.create_group('fft_observations'), fft)
        h['truth_params'] = p
        h.attrs['true_radius_m'] = radius
        h.attrs['true_mass_kg'] = 4 * np.pi * 3000 * radius**3 / 3
        h.attrs['injection_seed'] = seed
        h.attrs['purpose'] = 'Single recovery test; not frequentist coverage validation'
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius-um', type=float, default=30.)
    parser.add_argument('--seed', type=int, default=34522)
    args = parser.parse_args()
    make_injection(args.source, args.output, args.radius_um, args.seed)


if __name__ == '__main__':
    main()
