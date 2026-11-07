"""Check PR integral normalization and bound-orbit domain independently."""
import unittest
import numpy as np
from scipy.integrate import quad
from astropy.constants import L_sun, GM_sun, c, au
from finalize_submission_mass_figures import pr_years,YEAR_S


class PRTests(unittest.TestCase):
    def test_integral(self):
        radius=60e-6; a=1.7
        for e in [.001,.1,.5,.9,.98,.999]:
            integral=quad(lambda y:y**.6/(1-y*y)**1.5,0,e,epsabs=1e-10,epsrel=1e-10)[0]
            expected=8*np.pi*(4/3*3000*radius)*c.value**2*(a*au.value)**2/(5*L_sun.value)*e**(-1.6)*(1-e*e)**2*integral/YEAR_S
            np.testing.assert_allclose(pr_years(radius,a,e),expected,rtol=1e-9)

    def test_circular_and_scaling(self):
        r=60e-6
        beta=3*L_sun.value/(16*np.pi*GM_sun.value*c.value*3000*r)
        np.testing.assert_allclose(pr_years(r,1.,0.),c.value*au.value**2/(4*beta*GM_sun.value)/YEAR_S)
        np.testing.assert_allclose(pr_years(2*r,2.,.4),8*pr_years(r,1.,.4))

    def test_unbound_rejected(self):
        for a,e in [(1.,1.),(-1.,1.1),(1.,np.nan)]:
            with self.assertRaises(ValueError):
                pr_years(60e-6,a,e)


if __name__=='__main__':
    unittest.main()
