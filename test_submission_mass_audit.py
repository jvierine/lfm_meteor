"""Regression checks on the isolated submission audit, not mutable paper files."""
import unittest
import h5py
import numpy as np
from audit_submission_mass_selection import OUT, ARCHIVE, PANELS


class AuditTests(unittest.TestCase):
    def test_influx_reproduction(self):
        with h5py.File(OUT/'audit.h5') as a, h5py.File(ARCHIVE) as b:
            for name in ['dynamic','rcs','dynamic_matched','rcs_matched']:
                for key in a[name]:
                    if isinstance(a[name][key],h5py.Dataset):
                        np.testing.assert_array_equal(a[name][key][:],b[name][key][:])
                for key in a[name].attrs:
                    np.testing.assert_equal(a[name].attrs[key],b[name].attrs[key])

    def test_figure_identity_and_cuts(self):
        with h5py.File(OUT/'figure7.h5') as a, h5py.File(OUT/'figure8.h5') as b, h5py.File(PANELS) as old:
            names = a['event_id'].asstr()[:].tolist()
            paired = a['rcs_event_id'].asstr()[:]
            self.assertEqual(len(names),118)
            self.assertEqual(len(paired),101)
            self.assertEqual(len(set(names)),len(names))
            self.assertTrue(np.all(a['selected'][:]))
            self.assertTrue(np.all(np.log10(a['mass_95_hi_kg'][:]/a['mass_95_lo_kg'][:])<1))
            p = b['mass_comparison']
            np.testing.assert_array_equal(p['event_id'].asstr()[:],paired)
            for key,src in [('dynamic_mass_kg','initial_mass_kg'),('dynamic_lo95_kg','mass_95_lo_kg'),('dynamic_hi95_kg','mass_95_hi_kg')]:
                np.testing.assert_array_equal(p[key][:],a[src][:][[names.index(n) for n in paired]])
            for key,src in [('rcs_mass_kg','rcs_mass_kg'),('delta_mass_lo95_noise_only_kg','rcs_lo95_kg'),('delta_mass_hi95_noise_only_kg','rcs_hi95_kg')]:
                np.testing.assert_array_equal(p[key][:],a[src][:])
            for key in old['pulses']:
                np.testing.assert_array_equal(old['pulses'][key][:],b['pulses'][key][:])

    def test_size_and_precision(self):
        with h5py.File(OUT/'audit.h5') as a:
            s=a['current_sizes']
            np.testing.assert_allclose(s['mass_kg'][:],np.pi/6*3000*(s['diameter_um'][:]*1e-6)**3)
            self.assertEqual(len(a['removed_from_frozen_dynamic']),76)
            self.assertEqual(len(a['added_to_frozen_dynamic']),1)
        with h5py.File(OUT/'precision/catalogue_statistics.h5') as c, h5py.File(OUT/'precision/trajectory_precision.h5') as p:
            self.assertEqual(len(c['event_id']),233)
            np.testing.assert_allclose(np.sqrt(np.average(c['event_apparent_path_rms_m'][:]**2,weights=c['event_measurement_count'][:])),p.attrs['apparent_path_rms_m'])


if __name__ == '__main__':
    unittest.main()
