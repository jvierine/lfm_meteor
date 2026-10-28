import numpy as np
from scipy import stats

from refit_tristatic_mass_posterior import (
    copula_loglike, correlation_cholesky, normal_scores, posterior_flags,
)


def test_independent_copula_reduces_to_student_t_likelihood():
    residual = np.array([-4., 1., 12., -.5])
    scale = np.array([2., 3., 4., .2])
    nu = np.array([1.5, 3., 1.5, 3.])
    expected = np.sum(stats.t.logpdf(residual / scale, nu) - np.log(scale))
    np.testing.assert_allclose(copula_loglike(residual, scale, nu, np.eye(4)), expected)


def test_copula_recovers_injected_dependence_with_heavy_tails():
    rng = np.random.default_rng(52)
    corr = np.array([[1., .8], [.8, 1.]])
    gaussian = rng.multivariate_normal([0., 0.], corr, size=2000)
    nu = np.array([1.5, 3.])
    t_draw = stats.t.ppf(stats.norm.cdf(gaussian), nu)
    np.testing.assert_allclose(normal_scores(t_draw, nu), gaussian, atol=1e-7)
    dependent = sum(copula_loglike(v, np.ones(2), nu, np.linalg.cholesky(corr)) for v in t_draw)
    independent = sum(copula_loglike(v, np.ones(2), nu, np.eye(2)) for v in t_draw)
    assert dependent > independent


def test_gapped_correlation_is_positive_definite():
    t = np.array([0., .005, .010, .025, .040])
    mask = np.array([[1, 1], [1, 0], [0, 1], [1, 1], [0, 1]], bool)
    chol = correlation_cholesky(t, mask, .85, .8)
    np.testing.assert_allclose(np.diag(chol @ chol.T), 1., atol=1e-8)
    assert np.all(np.diag(chol) > 0)


def test_prior_limited_tail_is_not_reported_as_a_precise_mass():
    chain = np.zeros((20, 40, 16))
    radius = np.linspace(-2.2, -2.001, 400)
    flags = posterior_flags(chain, np.ones(16), 10 ** (3 * radius), radius)
    assert 'upper_mass_tail_prior_limited' in flags
    assert 'chain_too_short_for_autocorrelation_time' in flags


def test_free_molecular_drag_convention_in_actual_ode():
    from meteor_trajectory_models import integrate_ceplecha
    rho, rho_m, r, v = 1e-7, 3000., 1e-4, 30000.
    dt = 1e-5
    result = integrate_ceplecha(
        np.array([6.47e6, 0., 0.]), np.array([0., v, 0.]), r,
        lambda h: rho, meteoroid_density_kg_m3=rho_m, ablation_sigma_kg_j=0.,
        t_span_s=(0., dt), sample_dt_s=dt,
    )
    acceleration = (np.linalg.norm(result.velocity_mps[-1]) - v) / dt
    np.testing.assert_allclose(acceleration, -3 * rho * v**2 / (4 * rho_m * r), rtol=1e-4)


def test_msis_height_matches_geodetic_not_spherical_height():
    import jcoord
    import fit_all_ballistic_snr_weighted as base
    from refit_tristatic_mass_posterior import attach_geodetic_height
    times = np.array([1713798090639350891], dtype=np.int64)
    xyz = np.array([jcoord.geodetic2ecef(18.5, 109.7, 95000.)])
    inertial = base.gfit.ecef_to_gcrs(xyz, times)
    density = attach_geodetic_height(lambda h: 1e-7, times)
    np.testing.assert_allclose(density.height_from_gcrs(inertial[0]), 95000., atol=.1)
    assert abs(np.linalg.norm(inertial[0]) - 6371000. - 95000.) > 4000.


def test_small_mass_ode_is_tolerance_converged():
    from meteor_trajectory_models import integrate_ceplecha
    rho_m, r = 3000., 2e-6
    m = 4 * np.pi * rho_m * r**3 / 3
    args = (np.array([6.47e6, 0., 0.]), np.array([0., 30000., 0.]), r, lambda h: 1e-7)
    kwargs = dict(meteoroid_density_kg_m3=rho_m, ablation_sigma_kg_j=1e-8,
                  t_span_s=(0., .02), sample_dt_s=.0005)
    a = integrate_ceplecha(*args, **kwargs, rtol=1e-9, atol=np.r_[np.full(3, 1e-4), np.full(3, 1e-5), m * 1e-10])
    b = integrate_ceplecha(*args, **kwargs, rtol=1e-11, atol=np.r_[np.full(3, 1e-6), np.full(3, 1e-7), m * 1e-12])
    assert a.success and b.success
    np.testing.assert_allclose(a.mass_kg, b.mass_kg, rtol=1e-5)
    np.testing.assert_allclose(a.velocity_mps, b.velocity_mps, rtol=1e-6, atol=1e-4)


def test_ceplecha_sigma_mass_speed_integral():
    import fit_all_ceplecha_snr_weighted as cepl
    density = lambda h: 8e-7
    density.ceplecha_sigma_kg_j = 1e-8
    p = np.array([6.47e6, 0., 0., 0., 30000., 0., -4.5])
    t = np.linspace(0., .03, 15)
    _, v, _, mass, success, message = cepl.propagate_ceplecha(p, t, density)
    assert success, message
    expected = .5 * density.ceplecha_sigma_kg_j * ((v*v).sum(axis=1) - (v[0]*v[0]).sum())
    np.testing.assert_allclose(np.log(mass/mass[0]), expected, atol=1e-5)


def test_beat_and_delay_share_the_lfm_ambiguity_combination():
    import fit_event_joint_delay_doppler_fft as f
    geometric = np.array([200000., 230000., 290000.])
    doppler = np.array([-20000., -10000., 4000.])
    alpha = f.gfit.NOMINAL_CHIRP_RATE_HZ_PER_S / f.gfit.C
    apparent = geometric + doppler / alpha
    measured = apparent + np.array([3., -2., 5.])
    beat = doppler - alpha * (measured - geometric)
    np.testing.assert_allclose(beat, alpha * (apparent - measured), atol=1e-7)
