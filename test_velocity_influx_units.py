import unittest
import numpy as np
from test_velocity_exponent_influx import influx_ton_day, weights_at_exponent


class VelocityInfluxTests(unittest.TestCase):
    def test_kilograms_to_tonnes(self):
        self.assertEqual(influx_ton_day([1000.], [1.]), 1.)

    def test_reference_and_upper_endpoint(self):
        reference = np.array([2., 3.])
        speed = np.array([36., 72.])
        np.testing.assert_allclose(weights_at_exponent(reference, speed, 3.), reference)
        np.testing.assert_allclose(weights_at_exponent(reference, speed, 6.), [16., 3.])

    def test_matrix_matches_scalar_event_sum(self):
        speed = np.array([20., 40., 60.])
        reference = np.array([1e8, 2e8, 3e8])
        mass = np.array([1e-9, 2e-9, 3e-9])
        for p in np.linspace(3., 6., 31):
            expected = sum(w*m*(72./v)**(p-3.) for w,m,v in zip(reference,mass,speed))/1000.
            self.assertAlmostEqual(influx_ton_day(weights_at_exponent(reference,speed,p),mass), expected)


if __name__ == '__main__':
    unittest.main()
