import numpy as np
from diagnose_pulse_phase import unwrap_guided


def test_guided_wraps():
    predicted = -np.arange(30)*37.3
    residual = .1*np.arange(30)
    observed = np.angle(np.exp(1j*(predicted+residual)))
    recovered, corrected = unwrap_guided(observed, predicted)
    np.testing.assert_allclose(corrected, residual, atol=1e-12)
    np.testing.assert_allclose(np.diff(recovered), np.diff(predicted+residual), atol=1e-12)

