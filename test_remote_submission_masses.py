"""End-to-end identities and numerical normalization for remote-source products."""
from pathlib import Path
import hashlib
import unittest
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/submission_mass_remote_20261107'


class RemoteTests(unittest.TestCase):
    def test_hash_manifest(self):
        with h5py.File(OUT/'remote_hash_manifest.h5') as h:
            self.assertTrue(h.attrs['all_match'])
            self.assertEqual(len(h['accepted_event_id']),232)
            for path,digest,before,after in zip(h['relative_path'].asstr()[:],h['local_sha256'].asstr()[:],h['remote_before_sha256'].asstr()[:],h['remote_after_sha256'].asstr()[:]):
                self.assertEqual(hashlib.sha256((OUT/'snapshot'/path).read_bytes()).hexdigest(),digest)
                self.assertEqual(digest,before);self.assertEqual(digest,after)

    def test_exact_mass_pairs_and_flux_cohorts(self):
        with h5py.File(OUT/'joint_fft_mass_distribution.h5') as f,h5py.File(OUT/'rcs_mass_comparison.h5') as p,h5py.File(OUT/'current_mass_influx.h5') as flux:
            names=f['event_id'].asstr()[:].tolist();paired=f['rcs_event_id'].asstr()[:]
            self.assertEqual(len(names),154);self.assertEqual(len(paired),133)
            np.testing.assert_array_equal(paired,p['mass_comparison/event_id'].asstr()[:])
            idx=[names.index(n) for n in paired]
            for key,src in [('dynamic_mass_kg','initial_mass_kg'),('dynamic_lo95_kg','mass_95_lo_kg'),('dynamic_hi95_kg','mass_95_hi_kg')]:
                np.testing.assert_array_equal(p['mass_comparison'][key][:],f[src][:][idx])
            for key,src in [('rcs_mass_kg','rcs_mass_kg'),('delta_mass_lo95_noise_only_kg','rcs_lo95_kg'),('delta_mass_hi95_noise_only_kg','rcs_hi95_kg')]:
                np.testing.assert_array_equal(p['mass_comparison'][key][:],f[src][:])
            np.testing.assert_array_equal(names,flux['dynamic/event_id'].asstr()[:])
            np.testing.assert_array_equal(paired,flux['dynamic_matched/event_id'].asstr()[:])
            np.testing.assert_array_equal(paired,flux['rcs_matched/event_id'].asstr()[:])
            self.assertEqual(len(flux['rcs/event_id']),195)
            self.assertTrue(np.all(flux['usable_weight'][:]))

    def test_flux_units_and_corrections(self):
        with h5py.File(OUT/'current_mass_influx.h5') as h:
            for name in ['dynamic','rcs','dynamic_matched','rcs_matched']:
                g=h[name];base=4*np.pi*6466**2/h.attrs['collecting_area_km2']*4*24/g['effective_hours'][:]
                v=(72/g['speed_km_s'][:])**3
                for stage,w in [('area_exposure',base),('plus_velocity_heuristic',base*v),('plus_altitude_heuristic',base*v*h.attrs['altitude_sampling_factor'])]:
                    a=g[stage]
                    np.testing.assert_allclose(a['weight_per_day'][:],w)
                    np.testing.assert_allclose(a.attrs['rate_ton_day'],np.dot(w,g['mass_kg'][:])/1000)
                    np.testing.assert_array_equal(a['combined_interval95_ton_day'][:],np.quantile(a['combined_draws_ton_day'][:],[.025,.975]))
                np.testing.assert_allclose(g['plus_altitude_heuristic/combined_draws_ton_day'][:],g['plus_velocity_heuristic/combined_draws_ton_day'][:]*h.attrs['altitude_sampling_factor'])

    def test_precision_and_pr(self):
        with h5py.File(OUT/'mass_size_pr_precision.h5') as h:
            self.assertEqual(h.attrs['n_catalogue'],232)
            self.assertEqual(h.attrs['n_measurements'],13743)
            np.testing.assert_allclose(h.attrs['pooled_apparent_path_rms_m'],np.sqrt(np.average(h['event_apparent_path_rms_m'][:]**2,weights=h['event_measurement_count'][:])))
            np.testing.assert_array_equal(h['pr_years_event_median'][:],np.median(h['pr_years_orbit_draws_fixed_mass'][:],axis=1))


if __name__=='__main__':unittest.main()
