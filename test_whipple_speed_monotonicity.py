import numpy as np

import fit_event_joint_delay_doppler_fft as fit


def test_whipple_speed_starts_at_vinf_minus_a_and_decreases():
    epoch_ns = 1_713_823_274_259_349_823
    t_rel_s = np.linspace(0.0, 0.15, 20)
    times_ns = epoch_ns + np.rint(t_rel_s * 1e9).astype(np.int64)
    params = np.array(
        [
            6.42e6,
            1.0e4,
            2.0e4,
            -3.0e3,
            31.0e3,
            4.0e3,
            np.log10(20.0),
            np.log10(4.0),
        ],
        dtype=np.float64,
    )

    model = fit.forward_model_whipple_speed_link_observables(params, t_rel_s, times_ns)
    speed_mps = np.linalg.norm(model["v_gcrs_mps"], axis=1)

    assert model["ceplecha_success"], model["ceplecha_message"]
    expected_start_mps = np.linalg.norm(params[3:6]) - 10.0 ** params[6]
    assert abs(speed_mps[0] - expected_start_mps) < 1e-6
    assert np.all(np.diff(speed_mps) <= 1e-6)
    assert speed_mps[-1] < speed_mps[0]


def test_whipple_speed_rejects_nonpositive_terminal_speed():
    epoch_ns = 1_713_823_274_259_349_823
    t_rel_s = np.linspace(0.0, 1.0, 20)
    times_ns = epoch_ns + np.rint(t_rel_s * 1e9).astype(np.int64)
    params = np.array(
        [
            6.42e6,
            1.0e4,
            2.0e4,
            -3.0e3,
            31.0e3,
            4.0e3,
            np.log10(2.0e4),
            np.log10(5.0),
        ],
        dtype=np.float64,
    )

    model = fit.forward_model_whipple_speed_link_observables(params, t_rel_s, times_ns)

    assert not model["ceplecha_success"]
    assert "nonpositive" in model["ceplecha_message"]
