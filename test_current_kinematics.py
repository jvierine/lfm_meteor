"""Selection, geometry and scientific-array checks for the submission export."""
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from prepare_current_kinematics import OUTPUT, AREA, CATALOG, MANIFEST, accepted_ids, check_export, sha


class CurrentKinematics(unittest.TestCase):
    def test_identical_accepted_ids(self):
        ids = accepted_ids()
        for path in (OUTPUT, AREA, OUTPUT.parent/'radiants.h5', OUTPUT.parent/'height.h5',
                     OUTPUT.parent/'altitude_correction.h5'):
            with h5py.File(path) as h:
                np.testing.assert_array_equal(h['event_id'].asstr()[:], ids)

    def test_no_stale_manifest_fallback(self):
        with patch('prepare_current_kinematics.sha', return_value='changed'):
            with self.assertRaises(ValueError):
                check_export()

    def test_state_identity_and_unique_pulses(self):
        with h5py.File(OUTPUT) as h:
            for name,g in h['events'].items():
                with h5py.File(CATALOG/name/'fit.h5') as f, h5py.File(CATALOG/name/'input.h5') as inp:
                    self.assertEqual(sha(CATALOG/name/'fit.h5'),g.attrs['fit_sha256'])
                    np.testing.assert_array_equal(g['state_gcrs'][:],inp['center'][:]+1000*f['params'][:6])
                self.assertEqual(len(g['time_ns']),len(set(g['time_ns'][:])))
                self.assertTrue(np.isfinite(g['position_ecef_m'][:]).all())
                self.assertTrue(np.all(g['speed_km_s'][:]>0))

    def test_exposure_and_area(self):
        with h5py.File(OUTPUT.parent/'radiants.h5') as h, h5py.File(AREA) as area:
            family = h['family_effective_exposure_h'][:]
            self.assertEqual(family.shape,(232,4))
            self.assertTrue(np.all((family>=0)&(family<=h.attrs['observed_hours'])))
            np.testing.assert_allclose(family.sum(axis=1),h['family_effective_exposure_sum_h'][:])
            self.assertTrue(np.all(family.sum(axis=1)>0))
            self.assertFalse(h.attrs['altitude_correction_included'])
            self.assertFalse(h.attrs['velocity_correction_included'])
            self.assertEqual(h.attrs['source_sha256'],sha(OUTPUT))
            gains = area['max_gain_db'][:]
            self.assertGreaterEqual(np.mean(gains>=area.attrs['cutoff_relative_gain_db']),.95)
            self.assertEqual(h.attrs['area_horizontal_km2'],area.attrs['area_horizontal_km2'])
            grid = area['area_horizontal_by_grid_km2'][:]
            self.assertLess(abs(grid[-1]/grid[-2]-1),.003)

    def test_height_and_altitude_counts(self):
        with h5py.File(OUTPUT.parent/'height.h5') as h, h5py.File(OUTPUT) as k:
            self.assertEqual(len(h['tristatic_height_km']),sum(len(g['time_ns']) for g in k['events'].values()))
            self.assertEqual(h.attrs['source_sha256'],sha(OUTPUT))
            self.assertEqual(len(h['monostatic_height_km']),57300)
        with h5py.File(OUTPUT.parent/'altitude_correction.h5') as h:
            a = h.attrs
            self.assertAlmostEqual(a['altitude_sampling_factor'],a['monostatic_layer_events']*a['tristatic_common_events']/a['monostatic_common_events']/232)

    def test_rcs_membership_and_selection(self):
        with h5py.File(OUTPUT.parent/'rcs_pulses.h5') as h:
            p = h['pulses']
            self.assertEqual(h.attrs['source_sha256'],sha(OUTPUT))
            self.assertTrue(set(p['event_id'].asstr()[:]).issubset(accepted_ids()))
            self.assertTrue(np.all(p['sanya_relative_gain_db'][:]>=-3))
            self.assertTrue(np.isfinite(p['rcs_dbsm'][:]).all())
            np.testing.assert_allclose(p['rcs_dbsm'][:],10*np.log10(p['rcs_m2'][:]))


if __name__=='__main__':
    unittest.main()
