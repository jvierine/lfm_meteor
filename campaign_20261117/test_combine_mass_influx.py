import unittest
import numpy as np
from combine_mass_influx import rate_curves


class CombinationTests(unittest.TestCase):
    def test_reference_exponent_and_missing_contributions(self):
        result=rate_curves([1,2,3],[2,2,2],[36,36,36],3,2,np.array([3,3.5]))
        np.testing.assert_allclose(result['measured'],[12,12*np.sqrt(2)])
        np.testing.assert_allclose(result['median_adjusted'],[20,20*np.sqrt(2)])
        np.testing.assert_allclose(result['omit_top_two'],[2,2*np.sqrt(2)])

    def test_identical_campaigns_are_not_added(self):
        rate=np.array([12.,20.])
        np.testing.assert_allclose((11.8*rate+11.7*rate)/(11.8+11.7),rate)

    def test_reference_speed_has_no_exponent_dependence(self):
        result=rate_curves([1,2,3],[2,2,2],[72,72,72],3,0,np.linspace(3,3.5,101))
        np.testing.assert_allclose(result['measured'],12)


if __name__=='__main__':unittest.main()
