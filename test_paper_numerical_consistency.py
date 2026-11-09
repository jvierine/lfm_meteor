"""Check that the manuscript uses the selected catalogue's current numbers."""
from pathlib import Path
import unittest
import h5py
import numpy as np


ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results/tristatic_calibrated_likelihood_20261109/products'
PAPER=ROOT.parent/'sanya_tristatic_paper'


class PaperNumbers(unittest.TestCase):
    def test_selected_mass_numbers(self):
        text=(PAPER/'main.tex').read_text()
        with h5py.File(DATA/'mass_distribution.h5') as h:
            self.assertEqual(len(h['event_id']),135)
            self.assertEqual(len(h['rcs_event_id']),115)
            self.assertTrue(np.all(h['mass_95_hi_kg'][:]<10*h['mass_95_lo_kg'][:]))
        for value in ['135 dynamic masses and 197 RCS mass losses','115 events have both estimates',
                      '1.52\\times10^{-6}','2.35\\times10^{-7}',
                      'of \\(2.8\\times10^{-9}']:
            self.assertIn(value,text)

    def test_no_obsolete_result_generation(self):
        text=(PAPER/'main.tex').read_text()
        for value in ['194 selected dynamic','for 194 and 169','169 Sanya events',
                      '154 selected dynamic','23\\,200 sampled states','73.08 & 4.499',
                      '94--1290','100\\,000 realizations']:
            self.assertNotIn(value,text)

    def test_current_orbit_counts(self):
        source=ROOT/'results/tristatic_seven_parameter_catalogue_20261108/orbits/orbits.h5'
        with h5py.File(source) as h:
            self.assertEqual(len(h['event_id']),216)
            self.assertEqual(np.sum(h['valid'][:]),21600)
            self.assertTrue(np.all(h['e'][:][h['valid'][:]]<1))
        text=(PAPER/'main.tex').read_text()
        self.assertIn('21\\,600 sampled states from the 216 meteors',text)

    def test_influx_ranges_match_saved_products(self):
        with h5py.File(DATA/'imputed_mass_influx.h5') as h:
            np.testing.assert_allclose(h['beam_sampling/total_estimate_and_envelope_ton_day'][:],
                [51.15378488,33.7441033,99.93720446],rtol=1e-8)
            np.testing.assert_allclose(h['rcs/beam_sampling/total_endpoint_envelope_ton_day'][:],
                [2.54154311,9.80708243],rtol=1e-8)
            self.assertEqual(h.attrs['imputation_statistic'],'median weighted contribution')
            np.testing.assert_allclose(h['beam_sampling/exponent_sensitivity/range_ton_day'][:],
                [33.7441033,169.13269675],rtol=1e-8)
            np.testing.assert_allclose(h['rcs/beam_sampling/exponent_sensitivity/range_ton_day'][:],
                [2.54154311,15.72696152],rtol=1e-8)
            self.assertTrue(h['beam_sampling/exponent_sensitivity'].attrs['exponent_shared_by_all_events'])

    def test_classification_table_matches_current_ensembles(self):
        path=ROOT/'results/tristatic_seven_parameter_catalogue_20261108/orbits/orbits.h5'
        table=(PAPER/'tables/tristatic_orbit_classification_minus1000d.tex').read_text()
        with h5py.File(path) as h:
            g=h['classification_summary']
            for symbol,cometary,asteroidal in zip(g['symbol'].asstr()[:],
                    g['cometary_event_mean_fraction'][:],g['asteroidal_event_mean_fraction'][:]):
                line=next(x for x in table.splitlines() if x.startswith(symbol+' &'))
                self.assertIn(f'& {100*cometary:.0f}\\% & {100*asteroidal:.0f}\\%',line)

    def test_approved_opening_and_drag_equation_are_preserved(self):
        text=(PAPER/'main.tex').read_text()
        self.assertIn('Although our sample is relatively small',text)
        self.assertIn('\\int_0^e \\frac{y^{3/5}}{(1-y^2)^{3/2}}',text)


if __name__=='__main__':unittest.main()
