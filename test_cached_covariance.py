from types import SimpleNamespace
import numpy as np
from fit_cached_shrinking_voltage import covariance


def test_fisher_proposal_retains_correlated_measurement_information():
    rng=np.random.default_rng(64)
    design=rng.normal(size=(40,7))/np.array([.001,.002,.001,.01,.02,.01,.02])
    design[:,1]=.9*design[:,0]+.1*design[:,1]
    q=np.r_[np.zeros(6),-4.]
    cache=SimpleNamespace(model=SimpleNamespace(residual=lambda p:design@(p-q)))
    actual=covariance(cache,q)
    expected=np.linalg.inv(design.T@design)
    np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-12)


def test_fisher_proposal_handles_radius_prior_edge():
    q=np.r_[np.zeros(6),-2.]
    design=np.diag(np.arange(1.,8.)*100)
    cache=SimpleNamespace(model=SimpleNamespace(residual=lambda p:design@(p-q)))
    np.testing.assert_allclose(covariance(cache,q),np.linalg.inv(design.T@design),rtol=1e-7,atol=1e-12)
