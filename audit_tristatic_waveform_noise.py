"""Monte Carlo dependence of matched delay and dechirped beat from one pulse."""
from pathlib import Path
import h5py
import numpy as np

import fit_event_joint_delay_doppler_fft as fit
import simulate_fractional_delay_beat_frequency as sim


def main():
    sim.PULSE_LENGTH_US = fit.CANONICAL_PULSE_LENGTH_US
    rng = np.random.default_rng(20261028)
    alpha = fit.gfit.NOMINAL_CHIRP_RATE_HZ_PER_S / fit.gfit.C
    raw, _ = sim.simulate_doppler_measurement(0., 60000.)
    output = Path('results/tristatic_mass_waveform_audit_20261028.h5')
    with h5py.File(output, 'w') as h:
        h.attrs['source_program'] = str(Path(__file__).resolve())
        h.attrs['pulse_length_us'] = sim.PULSE_LENGTH_US
        h.attrs['interpretation'] = 'Controlled synthetic noise test, not an empirical calibration for real echoes'
        for snr in (15., 20., 30.):
            measurements = []
            for _ in range(128):
                row = sim.add_complex_voltage_noise(raw, rng, snr)
                coarse = sim.matched_filter_gate(row)
                gate, _ = fit.interp.doppler_matched_filter_peak(
                    row, 0., sim.MEASUREMENT_SR_MHZ, sim.BANDWIDTH_MHZ, 32, coarse)
                # Beat dechirping rounds its reference to the upsampled grid.
                reference_gate = np.rint(gate * 32) / 32
                beat, *_ = fit.dechirped_fft_offset_hz(row, reference_gate, 4., 4.,
                                                     sim.PULSE_LENGTH_US, 32, gate_upsample_factor=32)
                path = fit.gfit.C * (reference_gate - sim.REFERENCE_GATE) / 4e6
                measurements.append([path, beat, path + beat / alpha])
            values = np.asarray(measurements)
            residual = np.c_[-values[:, 0], -alpha * values[:, 0] - values[:, 1]]
            group = h.create_group(f'snr{int(snr)}')
            group['measured_path_beat_refined_path'] = values
            group['residual_covariance'] = np.cov(residual.T)
            group.attrs['paired_residual_correlation'] = np.corrcoef(residual.T)[0, 1]
            group.attrs['delay_std_m'] = np.std(values[:, 0], ddof=1)
            group.attrs['refined_delay_std_m'] = np.std(values[:, 2], ddof=1)
            print(snr, dict(group.attrs), flush=True)


if __name__ == '__main__':
    main()
