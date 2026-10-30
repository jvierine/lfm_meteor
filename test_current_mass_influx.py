import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from estimate_current_mass_influx import summarize


class InfluxTests(unittest.TestCase):
    def test_fixed_masses_and_poisson_variance(self):
        rows = {str(i): (float(i+1)*1e-10, np.full(20, float(i+1)*1e-10), 'fixed', 0., 'test') for i in range(4)}
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory)/'test.h5', 'w') as h:
                summarize(h, list(rows), rows, np.full(4,24.), np.full(4,72.), 10., np.random.default_rng(1))
                expected = 4*np.pi*6466**2/10 * 4 * (1258/234)*1e-9/1000
                self.assertAlmostEqual(h.attrs['corrected_ton_day'],expected)
                np.testing.assert_allclose(h['mass_only_interval95_ton_day'][:], expected)
                self.assertAlmostEqual(np.std(h['combined_draws_ton_day'][:])/h.attrs['weighted_poisson_sigma_ton_day'],1.,delta=.03)
                self.assertAlmostEqual(h.attrs['top_two_fraction'],.7)
                self.assertAlmostEqual(h.attrs['omit_two_largest_contributions_ton_day'],expected*.3)


if __name__ == '__main__':
    unittest.main()
