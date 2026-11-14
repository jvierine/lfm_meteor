import unittest
import numpy as np
from estimate_source_mass_influx import partition, calculate


class SourceInfluxTests(unittest.TestCase):
    def test_partition_conserves_unknown_and_fractional_membership(self):
        np.testing.assert_allclose(partition([10, 20, 30], [1, .25, np.nan]), [15, 15, 30])

    def test_imputation_uses_global_median_and_preserves_exponent(self):
        ids, votes, rates, bounds, measured = calculate(
            ["a", "b"], np.array([2., 4.]), np.array([[1., 3.], [2., 6.]]),
            np.ones(2), np.array([36., 72.]), ["a", "b", "c", "d"],
            {"a": 1., "b": 0., "c": .5}, [3., 3.5])
        np.testing.assert_allclose(rates[0], [3.5, 5.5, 3.])
        np.testing.assert_allclose(rates.sum(axis=1),
                                  [12., 2 * (2 * np.sqrt(2) + 4)])
        self.assertEqual(ids, ["a", "b", "c", "d"])
        self.assertTrue(np.isnan(votes[-1]))
        self.assertTrue(np.all(bounds[:, 0] <= rates))
        self.assertTrue(np.all(rates <= bounds[:, 1]))

    def test_invalid_membership_rejected(self):
        with self.assertRaises(ValueError):
            partition([1], [1.1])


if __name__ == "__main__":
    unittest.main()
