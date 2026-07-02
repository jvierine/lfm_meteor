"""Synthetic checks for the shrinking-radius meteoroid model.

These tests exercise the same Ceplecha/shrinking-radius propagation and
path-observable fitting code used by the Sanya processing scripts.  The goal is
to verify that a planted radius is recoverable when the data are actually drawn
from the model.
"""

import numpy as np

import fit_all_ceplecha_snr_weighted as cepl


R_EARTH_M = cepl.SPHERICAL_EARTH_RADIUS_M
RHO_AIR_KG_M3 = 8.0e-7


def constant_density(_height_m):
    return RHO_AIR_KG_M3


def synthetic_state(radius_um=25.0):
    position0_m = np.array([R_EARTH_M + 96.0e3, 0.0, 0.0], dtype=np.float64)
    velocity0_mps = np.array([-3200.0, 31_000.0, 4200.0], dtype=np.float64)
    return np.concatenate([position0_m, velocity0_mps, [np.log10(radius_um * 1e-6)]])


def synthetic_times(n=26, dt_s=0.008):
    epoch_ns = 1_713_823_274_259_349_823
    return epoch_ns + np.rint(np.arange(n, dtype=np.float64) * dt_s * 1e9).astype(np.int64)


def fit_from_radius_grid(measured_paths_m, times_ns, radius_grid_um):
    guesses = []
    true_like = synthetic_state(radius_um=25.0)
    for radius_um in radius_grid_um:
        guess = true_like.copy()
        guess[:3] += np.array([20.0, -30.0, 15.0])
        guess[3:6] += np.array([50.0, -80.0, 30.0])
        guess[6] = np.log10(radius_um * 1e-6)
        guesses.append(guess)

    best = None
    sigma_m = np.full_like(measured_paths_m, 0.5, dtype=np.float64)
    keep_rows = np.ones(len(times_ns), dtype=bool)
    for guess in guesses:
        fit = cepl.fit_ceplecha(
            measured_paths_m,
            times_ns,
            constant_density,
            guess,
            sigma_m=sigma_m,
            keep_rows=keep_rows,
            epoch_time_ns=int(times_ns[0]),
            loss="linear",
        )
        if best is None or fit["rms_total_path_residual_m"] < best["rms_total_path_residual_m"]:
            best = fit
    return best


def test_ceplecha_smaller_radius_decelerates_more():
    times_ns = synthetic_times()
    t_rel_s = (times_ns.astype(np.float64) - float(times_ns[0])) / 1e9
    params_small = synthetic_state(radius_um=10.0)
    params_large = synthetic_state(radius_um=100.0)

    _x_s, v_s, r_s, m_s, success_s, message_s = cepl.propagate_ceplecha(params_small, t_rel_s, constant_density)
    _x_l, v_l, r_l, m_l, success_l, message_l = cepl.propagate_ceplecha(params_large, t_rel_s, constant_density)

    assert success_s, message_s
    assert success_l, message_l
    assert r_s[-1] < r_s[0]
    assert m_s[-1] < m_s[0]
    assert np.linalg.norm(v_s[-1]) < np.linalg.norm(v_l[-1])
    assert (np.linalg.norm(v_s[0]) - np.linalg.norm(v_s[-1])) > 2.0 * (
        np.linalg.norm(v_l[0]) - np.linalg.norm(v_l[-1])
    )


def test_ceplecha_fit_recovers_radius_from_noiseless_synthetic_paths():
    true_radius_um = 25.0
    times_ns = synthetic_times()
    t_rel_s = (times_ns.astype(np.float64) - float(times_ns[0])) / 1e9
    true_params = synthetic_state(radius_um=true_radius_um)
    measured_paths_m, *_ = cepl.predict_paths(true_params, t_rel_s, times_ns, constant_density)

    fit = fit_from_radius_grid(
        measured_paths_m,
        times_ns,
        radius_grid_um=np.asarray([5.0, 15.0, 40.0, 120.0], dtype=np.float64),
    )

    recovered_radius_um = fit["initial_radius_m"] * 1e6
    assert fit["ceplecha_success"], fit["ceplecha_message"]
    assert fit["rms_total_path_residual_m"] < 0.02
    assert abs(recovered_radius_um - true_radius_um) / true_radius_um < 0.02


def test_ceplecha_fit_radius_is_identifiable_with_small_path_noise():
    rng = np.random.default_rng(20260701)
    true_radius_um = 25.0
    times_ns = synthetic_times()
    t_rel_s = (times_ns.astype(np.float64) - float(times_ns[0])) / 1e9
    true_params = synthetic_state(radius_um=true_radius_um)
    measured_paths_m, *_ = cepl.predict_paths(true_params, t_rel_s, times_ns, constant_density)
    noisy_paths_m = measured_paths_m + rng.normal(0.0, 0.5, size=measured_paths_m.shape)

    fit = fit_from_radius_grid(
        noisy_paths_m,
        times_ns,
        radius_grid_um=np.asarray([5.0, 15.0, 40.0, 120.0], dtype=np.float64),
    )

    recovered_radius_um = fit["initial_radius_m"] * 1e6
    assert fit["ceplecha_success"], fit["ceplecha_message"]
    assert fit["rms_total_path_residual_m"] < 0.8
    assert abs(recovered_radius_um - true_radius_um) / true_radius_um < 0.15
