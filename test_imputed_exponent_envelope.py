import unittest
import numpy as np
from impute_calibrated_mass_influx import exponent_envelope


class ExponentEnvelopeTests(unittest.TestCase):
    def test_shared_exponent_endpoints_enclose_dense_sweep(self):
        bounds=np.array([[1.,2.],[3.,7.],[2.,9.]])
        weight=np.array([.4,2.,1.]);speed=np.array([20.,50.,72.])
        endpoints=exponent_envelope(bounds,weight,speed,4)
        for p in np.linspace(3,3.5,101):
            values=bounds*(weight*(72/speed)**(p-3))[:,None]
            total=values.sum(axis=0)+4*np.median(values,axis=0)
            self.assertGreaterEqual(total[0],endpoints[0,0]-1e-12)
            self.assertLessEqual(total[1],endpoints[-1,1]+1e-12)

    def test_reference_p_three_unchanged(self):
        b=np.array([[1.,4.],[2.,3.]])
        expected=(b*np.array([2.,3.])[:,None]).sum(axis=0)
        np.testing.assert_allclose(exponent_envelope(b,[2.,3.],[30.,60.],0)[0],expected)

    def test_unsupported_speed_rejected(self):
        with self.assertRaises(ValueError):exponent_envelope([[1.,2.]],[1.],[73.],0)


if __name__=='__main__':unittest.main()
