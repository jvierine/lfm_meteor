import unittest
import numpy as np
from radiant_symmetry import symmetry_members, daily_weight_from_summed_hours


class FourfoldSymmetryTests(unittest.TestCase):
    def test_reflected_members_have_identical_family(self):
        lon, lat = symmetry_members(213., 34.)
        expected = sorted(zip(lon, lat))
        for x, y in zip(lon, lat):
            a, b = symmetry_members(x, y)
            self.assertEqual(sorted(zip(a, b)), expected)

    def test_uneven_and_missing_quadrants_are_unbiased(self):
        for hours in ([12.,12.,12.,12.], [12.,0.,0.,0.], [1.,7.,0.,.25]):
            rate_per_quadrant = 2.3
            expected_count = rate_per_quadrant*np.sum(hours)
            recovered = expected_count*daily_weight_from_summed_hours(sum(hours))
            self.assertAlmostEqual(recovered, 4*24*rate_per_quadrant)

    def test_uniform_coverage_has_no_extra_factor_four(self):
        self.assertEqual(daily_weight_from_summed_hours(4*12),24/12)

    def test_completely_unobserved_family_is_not_identifiable(self):
        with self.assertRaises(ValueError):daily_weight_from_summed_hours(0)


if __name__ == '__main__':unittest.main()
