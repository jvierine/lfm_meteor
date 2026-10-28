import h5py
import numpy as np

from run_validated_tristatic_mass_catalog import evaluate_validation, STRONG, WEAK


def fake_result(path, mass=1e-9, flags=''):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as h:
        p = h.create_group('mass_posterior')
        p['mass_quantiles_kg'] = np.array([.7, 1., 1.4]) * mass
        p.attrs['flags'] = flags


def complete_validation(root):
    for radius in (20, 30, 60):
        name = f'joint_delay_doppler_fft_tri_synthetic_{radius}um.h5'
        source = root / 'injections' / name
        source.parent.mkdir(exist_ok=True)
        with h5py.File(source, 'w') as h:
            h.attrs['true_mass_kg'] = 1e-9
        fake_result(root / 'validation' / 'injections' / name)
    for replica in (1, 2):
        for event in (STRONG, WEAK):
            fake_result(root / 'validation' / f'replica_{replica}' / f'joint_delay_doppler_fft_{event}.h5')


def test_gate_rejects_missing_injections(tmp_path):
    assert evaluate_validation(tmp_path)


def test_gate_accepts_complete_stable_recoveries(tmp_path):
    complete_validation(tmp_path)
    assert not evaluate_validation(tmp_path)


def test_gate_rejects_short_chains(tmp_path):
    complete_validation(tmp_path)
    fake_result(tmp_path / 'validation/injections/joint_delay_doppler_fft_tri_synthetic_30um.h5',
                flags='chain_too_short_for_autocorrelation_time')
    assert evaluate_validation(tmp_path)


def test_gate_rejects_disagreeing_independent_runs(tmp_path):
    complete_validation(tmp_path)
    fake_result(tmp_path / 'validation/replica_2' / f'joint_delay_doppler_fft_{STRONG}.h5', mass=2e-9)
    assert evaluate_validation(tmp_path)


def test_weak_unidentified_mass_is_recorded_not_called_validated(tmp_path):
    complete_validation(tmp_path)
    fake_result(tmp_path / 'validation/replica_2' / f'joint_delay_doppler_fft_{WEAK}.h5',
                flags='chain_too_short_for_autocorrelation_time')
    assert not evaluate_validation(tmp_path)
    with h5py.File(tmp_path / 'validation_report.h5') as h:
        assert not h[WEAK].attrs['both_chains_stable']
