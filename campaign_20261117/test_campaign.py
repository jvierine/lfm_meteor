import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from analyze import load_measurements, SITES, ENU, STATIONS
from derived import source_root, beam


class CampaignTests(unittest.TestCase):
    def test_raw_cut_replays_q_plus_i_i_not_conjugated_waveform(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for site in SITES:
                path=root/("20261117_"+site)/"event_cuts/event_000000.h5"
                path.parent.mkdir(parents=True)
                with h5py.File(path,"w") as h:
                    p=h.create_group("pulses")
                    for key,value in dict(echo_start_sample=[400]*3,peak_snr_db=[20.]*3,
                        utc_ns=[1000000000,1005000000,1010000000],raw_noise_power_adc2=[1.]*3,
                        apparent_delay_us=[566.]*3,tx_azimuth_deg=[15.]*3,tx_elevation_deg=[75.]*3,
                        rx_azimuth_deg=[15.]*3,rx_elevation_deg=[75.]*3).items():p[key]=value
                    h["ransac_inlier"]=np.ones(3,np.uint8)
                    g=h.create_group("raw")
                    g["pulse_offsets"]=np.arange(4)*2000
                    g["first_sample"]=np.zeros(3,np.uint64)
                    g["first_delay_us"]=np.full(3,466.)
                    g["iq_words_iq"]=np.tile([3,7],6000).astype(np.int32)
            rows=load_measurements(root,[0,0,0])
            self.assertEqual(len(rows),9)
            np.testing.assert_array_equal(rows[0]["raw"],np.full(1280,7+3j))
            self.assertAlmostEqual(rows[0]["sample"][0],466e-6+160/4e6)

    def test_lookup_inversion_does_not_hide_out_of_domain_or_multiple_roots(self):
        source=np.array([1.,2.,3.,4.])
        self.assertEqual(source_root(np.array([-90.,-70.,-50.,-30.]),-60.,source),2.5)
        self.assertTrue(np.isnan(source_root(np.array([-90.,-70.,-50.,-30.]),-100.,source)))
        self.assertTrue(np.isnan(source_root(np.array([-80.,-50.,-80.,-40.]),-60.,source)))

    def test_beam_axis_has_unit_relative_gain(self):
        az,el=np.deg2rad([14.996337890625,74.9981689453125])
        axis=ENU.T@np.array([np.sin(az)*np.cos(el),np.cos(az)*np.cos(el),np.sin(el)])
        gain,absolute,distance,east,north=beam((STATIONS[0]+100000*axis)[None,:])
        self.assertAlmostEqual(gain[0],0.,places=10)
        self.assertAlmostEqual(distance[0],100000.,places=6)
        self.assertAlmostEqual(east[0],0.,places=6)
        self.assertAlmostEqual(north[0],0.,places=6)


if __name__=="__main__":unittest.main()
