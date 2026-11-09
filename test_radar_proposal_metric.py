import unittest
import numpy as np
from radar_proposal_metric import cap_covariance


class ProposalMetricTests(unittest.TestCase):
    def test_weak_radius_does_not_freeze_position_velocity(self):
        covariance=np.diag([1e-6]*6+[1e12])
        capped=cap_covariance(covariance,np.ones(7))
        np.testing.assert_allclose(np.diag(capped)[:6],1e-6)
        self.assertAlmostEqual(capped[6,6],1.)

    def test_correlation_and_positive_definiteness_are_preserved(self):
        factor=np.array([[.01,0.,0.],[.004,.02,0.],[100.,50.,1000.]])
        covariance=factor@factor.T
        capped=cap_covariance(covariance,[1.,1.,1.])
        correlation=lambda c:c/np.sqrt(np.diag(c)[:,None]*np.diag(c)[None,:])
        np.testing.assert_allclose(correlation(capped),correlation(covariance))
        self.assertGreater(np.linalg.eigvalsh(capped).min(),0.)

    def test_invalid_metric_is_rejected(self):
        with self.assertRaises(ValueError):cap_covariance(np.diag([1.,np.nan]),[1.,1.])


if __name__=='__main__':unittest.main()
