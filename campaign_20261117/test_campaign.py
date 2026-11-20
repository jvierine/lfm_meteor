import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from analyze import load_measurements, SITES, ENU, STATIONS
from derived import source_root, beam, beam_frame, beam_power, directions_from_offsets, angular_offsets


class CampaignTests(unittest.TestCase):
    def test_raw_cut_replays_q_plus_i_i_not_conjugated_waveform(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for site in SITES:
                path=root/("20261117_"+site)/"event_cuts/event_000000.h5"
                path.parent.mkdir(parents=True)
                with h5py.File(path,"w") as h:
                    p=h.create_group("pulses")
                    for key,value in dict(echo_start_sample=[800]*3,peak_snr_db=[20.]*3,
                        utc_ns=[1000000000,1005000000,1010000000],raw_noise_power_adc2=[1.]*3,
                        apparent_delay_us=[566.]*3,tx_azimuth_deg=[15.]*3,tx_elevation_deg=[75.]*3,
                        rx_azimuth_deg=[15.]*3,rx_elevation_deg=[75.]*3).items():p[key]=value
                    h["ransac_inlier"]=np.ones(3,np.uint8)
                    g=h.create_group("raw")
                    g["pulse_offsets"]=np.arange(4)*2000
                    g["first_sample"]=np.full(3,250,np.uint64)
                    g["first_delay_us"]=np.full(3,466.+250/4.)
                    g["iq_words_iq"]=np.tile([3,7],6000).astype(np.int32)
            rows=load_measurements(root,[0,0,0])
            self.assertEqual(len(rows),9)
            np.testing.assert_array_equal(rows[0]["raw"],np.full(1280,7+3j))
            # The cut's start delay already contains its nonzero sample index.
            self.assertAlmostEqual(rows[0]["sample"][0],466e-6+(800-240)/4e6)

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

    def test_beam_histogram_and_gain_use_the_same_sky_frame(self):
        axis,east,north=beam_frame()
        np.testing.assert_allclose(np.array([east,north,axis])@np.array([east,north,axis]).T,np.eye(3),atol=1e-14)
        self.assertGreater(east[0],0)
        self.assertGreater(north[1],0)
        x,y=np.meshgrid(np.linspace(-2,2,11),np.linspace(-2,2,11))
        directions=directions_from_offsets(x,y)
        dx,dy=angular_offsets(directions)
        np.testing.assert_allclose(dx,x,atol=1e-12)
        np.testing.assert_allclose(dy,y,atol=1e-12)
        xyz=STATIONS[0]+100000*directions.reshape(-1,3)@ENU
        gain,_,_,_,_=beam(xyz)
        np.testing.assert_allclose(gain,10*np.log10(beam_power(directions,axis)).ravel(),atol=1e-8)

    def test_elevation_offset_is_not_geographic_north(self):
        az,el=np.deg2rad([14.996337890625,75.9981689453125])
        d=np.array([np.sin(az)*np.cos(el),np.cos(az)*np.cos(el),np.sin(el)])
        east,north=angular_offsets(d)
        self.assertLess(north,-.9)
        self.assertLess(east,-.2)


if __name__=="__main__":unittest.main()
