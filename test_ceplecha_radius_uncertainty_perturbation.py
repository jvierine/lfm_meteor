"""Sanity tests for Ceplecha radius uncertainty estimates.

These tests deliberately perturb the fitted radius after the fit while keeping
the fitted trajectory state fixed.  If the objective does not change by about
one chi-square unit, the radius uncertainty cannot be smaller than that
perturbation scale.
"""

import numpy as np
import scipy.optimize as so

import fit_all_ballistic_snr_weighted as base_fit
import fit_all_ceplecha_snr_weighted as ceplecha_fit
import plot_bayesian_model_selection_example as selection_plot


RADIUS_TEST_FRACTIONS = (0.05, 0.10, 0.30, 0.50, 1.00)
DELTA_CHI2_ONE_SIGMA = 1.0


def fitted_radius_residual(event_id):
    event = selection_plot.refine_observation_keep(selection_plot.read_event(event_id))
    measured = event["measured"]
    times = event["time_ns"]
    sigma = event["sigma"]
    keep_obs = event["keep_obs"]
    keep_rows = np.any(keep_obs, axis=1)
    rho_of_alt_m, _meta = base_fit.density_interpolator(times, event["x_itrs_initial"])
    times_fit = times[keep_rows]
    measured_fit = measured[keep_rows]
    sigma_fit = sigma[keep_rows]
    keep_obs_fit = keep_obs[keep_rows]
    t_rel_s = (times_fit.astype(np.float64) - float(times[0])) / 1e9

    def residual(params):
        predicted, *_rest = ceplecha_fit.predict_paths(params, t_rel_s, times_fit, rho_of_alt_m)
        return ((predicted - measured_fit) / sigma_fit)[keep_obs_fit]

    result = so.least_squares(
        residual,
        event["p0_shrinking"],
        bounds=(
            np.array([-np.inf, -np.inf, -np.inf, -8e4, -8e4, -8e4, np.log10(ceplecha_fit.MIN_RADIUS_M)]),
            np.array([np.inf, np.inf, np.inf, 8e4, 8e4, 8e4, np.log10(ceplecha_fit.MAX_RADIUS_M)]),
        ),
        x_scale=np.array([6.4e6, 6.4e6, 6.4e6, 5e4, 5e4, 5e4, 1.0]),
        loss=ceplecha_fit.ROBUST_LOSS,
        f_scale=ceplecha_fit.ROBUST_F_SCALE,
        max_nfev=220,
    )
    r0 = residual(result.x)
    chi2_0 = float(np.sum(r0**2.0))
    return result.x, residual, chi2_0


def radius_delta_chi2(event_id, fractions=RADIUS_TEST_FRACTIONS):
    params, residual, chi2_0 = fitted_radius_residual(event_id)
    out = {}
    for fraction in fractions:
        values = {}
        if fraction < 1.0:
            minus = params.copy()
            minus[6] += np.log10(1.0 - fraction)
            r_minus = residual(minus)
            values["minus"] = float(np.sum(r_minus**2.0) - chi2_0)
        plus = params.copy()
        plus[6] += np.log10(1.0 + fraction)
        r_plus = residual(plus)
        values["plus"] = float(np.sum(r_plus**2.0) - chi2_0)
        out[float(fraction)] = values
    return out


def radius_uncertainty_floor_fraction(delta_chi2_by_fraction, threshold=DELTA_CHI2_ONE_SIGMA):
    floor = 0.0
    for fraction, directions in delta_chi2_by_fraction.items():
        if any(delta_chi2 < threshold for delta_chi2 in directions.values()):
            floor = max(floor, float(fraction))
    return floor


def test_weak_radius_case_requires_large_upper_error_bar():
    delta = radius_delta_chi2("tri_0234_1713829501934350014")
    assert delta[1.00]["plus"] < DELTA_CHI2_ONE_SIGMA
    assert radius_uncertainty_floor_fraction(delta) >= 1.0


def test_marginal_radius_case_requires_at_least_fifty_percent_error_bar():
    delta = radius_delta_chi2("tri_0150_1713822354419349670")
    assert delta[0.50]["plus"] < DELTA_CHI2_ONE_SIGMA
    assert radius_uncertainty_floor_fraction(delta) >= 0.50


def test_better_case_rejects_large_radius_perturbations():
    delta = radius_delta_chi2("tri_0202_1713826450419349670")
    assert delta[0.10]["minus"] > DELTA_CHI2_ONE_SIGMA
    assert delta[0.10]["plus"] > DELTA_CHI2_ONE_SIGMA
    assert radius_uncertainty_floor_fraction(delta) < 0.10


if __name__ == "__main__":
    for event_id in [
        "tri_0234_1713829501934350014",
        "tri_0150_1713822354419349670",
        "tri_0202_1713826450419349670",
    ]:
        delta_chi2 = radius_delta_chi2(event_id)
        floor = radius_uncertainty_floor_fraction(delta_chi2)
        print(event_id, f"floor_fraction={floor:.2g}", delta_chi2)
