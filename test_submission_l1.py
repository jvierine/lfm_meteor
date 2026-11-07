import unittest
import numpy as np
from export_submission_l1 import state_vectors, track_rates


class L1Tests(unittest.TestCase):
    def test_quadratic_rate(self):
        times = np.arange(9)*10000000 + 1713791855379348755
        t = np.arange(9)*.01
        rate, error, degree = track_rates(times, 120000 + 3000*t - 500*t*t)
        np.testing.assert_allclose(rate, 3000-1000*t, atol=1e-6)
        self.assertEqual(degree, 2)
        self.assertLess(error.max(), 1e-6)

    def test_undefined_rate(self):
        rate, _, degree = track_rates([1,1], [1,2])
        self.assertTrue(np.isnan(rate).all())
        self.assertEqual(degree, 0)

    def test_frame_velocity(self):
        center = np.array([100.,200.,300.,400.,0.,0.])
        frames = np.zeros((3,42))
        frames[:,6:15] = np.eye(3).ravel()
        frames[:,33:42] = np.eye(3).ravel()
        frames[:,15:24] = np.array([[0,-.1,0],[.1,0,0],[0,0,0]]).ravel()
        pos = np.array([[100.,200.,300.],[104.,200.,300.],[108.,200.,300.]])
        x,v,ve,error = state_vectors(center, np.zeros(7), frames, pos, np.full(3,400.))
        np.testing.assert_equal(x,pos)
        np.testing.assert_equal(v,np.tile([400.,0.,0.],(3,1)))
        np.testing.assert_allclose(ve,np.c_[np.full(3,380.),pos[:,0]*.1,np.zeros(3)])
        self.assertEqual(error,0.)


if __name__ == "__main__":
    unittest.main()
