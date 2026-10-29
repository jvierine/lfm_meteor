import numpy as np
from scipy.signal import lfilter
from voltage_noise_covariance import fit_noise, colored_noise


def test_complex_ar_recovery_and_heldout_whitening():
    rng=np.random.default_rng(53);truth=np.array([.45+.2j])
    rows=np.array([colored_noise(rng,4096,truth,2.) for _ in range(40)])
    result=fit_noise(rows,np.full(40,2000))
    assert result['order']==1
    np.testing.assert_allclose(result['coeff'],truth,atol=.015)
    assert abs(result['heldout_variance']-2.)<.08
    assert result['whiteness_pass']


def test_white_noise_does_not_require_ar_terms():
    rng=np.random.default_rng(99)
    rows=np.array([colored_noise(rng,4096,[],1.) for _ in range(40)])
    result=fit_noise(rows,np.full(40,2000))
    assert result['order']==0
    assert result['whiteness_pass']


def test_gls_amplitude_matches_dense_covariance():
    n=30;a=.4+.2j;w=np.eye(n,dtype=complex)
    w[np.arange(1,n),np.arange(n-1)]=-a
    w[0,0]=np.sqrt(1-abs(a)**2)
    covariance=np.linalg.inv(w.conj().T@w)
    rng=np.random.default_rng(5);s=rng.normal(size=n)+1j*rng.normal(size=n)
    y=(2+3j)*s+rng.normal(size=n)+1j*rng.normal(size=n)
    expected=np.vdot(s,np.linalg.solve(covariance,y))/np.vdot(s,np.linalg.solve(covariance,s))
    actual=np.vdot(w@s,w@y)/np.vdot(w@s,w@s)
    np.testing.assert_allclose(actual,expected,rtol=1e-12)
