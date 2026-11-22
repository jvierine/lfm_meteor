import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from analyze import associate, load_measurements, SITES, ENU, STATIONS, C, DELAY_CORRECTION_US, remaining_delay_correction
from derived import source_root, beam, beam_frame, beam_power, directions_from_offsets, angular_offsets
from overlap import measure_overlap
from fit_numerics import jacobian, optimize, profile_interval
from review_extremes import multi_receiver_mask


class CampaignTests(unittest.TestCase):
    def test_monostatic_selection_checks_height_and_slow_long_tracks(self):
        from monostatic_selection import meteor_mask
        keep,_,_=meteor_mask([100.,74.,121.,100.,100.,100.],
                            [.1,.1,.1,.6,.6,.1],[0.,20.,20.,2.,20.,2.])
        np.testing.assert_array_equal(keep,[True,False,False,False,True,True])
    def test_outlier_rejection_is_limited_to_isolated_measurements(self):
        from quality import isolated_outlier_mask
        t=np.tile(np.arange(30)*.005,3);s=np.repeat(np.arange(3),30)
        mask=np.ones(90,dtype=bool);r=np.zeros(90);r[10]=50.
        proposal,_=isolated_outlier_mask(t,s,r,mask,mask)
        self.assertEqual(proposal.sum(),89)
        r[11:13]=50.
        self.assertIsNone(isolated_outlier_mask(t,s,r,mask,mask)[0])
        r[:]=0.;r[[2,8,15,22]]=50.
        self.assertIsNone(isolated_outlier_mask(t,s,r,mask,mask)[0])
    def test_quality_requires_every_used_residual_below_40_m(self):
        from quality import path_quality
        self.assertTrue(path_quality([39.99,-39.99,1000],[True,True,False])[0])
        self.assertFalse(path_quality([0.,40.],[True,True])[0])
        self.assertFalse(path_quality([np.nan],[True])[0])
        self.assertFalse(path_quality([],[])[0])
    def test_three_station_four_ipp_windows_do_not_admit_single_site_tails(self):
        from physical import three_station_mask
        times=np.array([0.,.01,.02,.025,.05,.06,.10])
        station=np.array([0,1,2,0,0,1,2])
        np.testing.assert_array_equal(three_station_mask(times,station),
                                     [True,True,True,True,False,False,False])
        np.testing.assert_array_equal(three_station_mask(times[::-1],station[::-1]),
                                     [False,False,False,True,True,True,True])
        self.assertFalse(three_station_mask([0.,.02,.04],[0,1,2]).any())

    def test_calibration_is_not_applied_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            with h5py.File(Path(temp)/"test.h5", "w") as h:
                h["config_toml_utf8"] = np.frombuffer(
                    ("delay_correction_us = %.16g\n" % DELAY_CORRECTION_US).encode(), dtype=np.uint8)
                self.assertAlmostEqual(remaining_delay_correction(h), 0., places=12)

    def test_mass_selection_excludes_nulls_and_sidelobes(self):
        from physical import mass_fit_mask
        from analyze import WAVELENGTH
        axis = beam_frame()[0]
        directions = np.repeat(axis[None, :], 4, axis=0)
        directions[:, 0] += np.array([0., .5, 1., 1.43])*WAVELENGTH/40
        directions[:, 2] = np.sqrt(1-np.sum(directions[:, :2]**2, axis=1))
        positions = STATIONS[0]+120000*directions@ENU
        mask, gain = mass_fit_mask(positions)
        np.testing.assert_array_equal(mask, [True, True, False, False])
        self.assertGreater(gain[1], -10.)

    def test_review_support_uses_pulses_not_interval_envelopes(self):
        times=np.array([0.,.005,.010,.000025,.010025,.020])
        station=np.array([0,0,0,1,1,2])
        np.testing.assert_array_equal(multi_receiver_mask(times,station),[True,False,True,True,True,False])
    def test_explicit_jacobian_steps_work_for_zero_and_nonzero_parameters(self):
        a=np.arange(28,dtype=float).reshape(4,7)/7
        for x in [np.zeros(7),np.arange(7,dtype=float)]:
            np.testing.assert_allclose(jacobian(lambda q:a@q,x),a,atol=1e-9)

    def test_optimizer_escapes_radius_seeds_despite_finite_model_precision(self):
        truth=np.r_[np.linspace(.01,.06,6),-4.63]
        bounds=(np.r_[[-30.]*3,[-50.]*3,-7.],np.r_[[30.]*3,[50.]*3,-2.])
        for radius in [-5.,-4.,-3.]:
            fit=optimize(lambda q:np.round(q-truth,6),np.r_[np.zeros(6),radius],bounds)
            np.testing.assert_allclose(fit.x,truth,atol=2e-6)

    def test_profile_intervals_reject_flat_and_boundary_mass_fits(self):
        expected=np.sqrt(2*1.920729410347062)*.03
        result=profile_interval(lambda r:.5*((r+4)/.03)**2,-4.,0.)
        np.testing.assert_allclose(result,[-4-expected,-4+expected],atol=.001)
        self.assertTrue(np.isnan(profile_interval(lambda r:0.,-3.,0.)).all())
        self.assertTrue(np.isnan(profile_interval(lambda r:0.,-2.,0.)).all())

    def test_profile_jump_is_not_a_confidence_limit(self):
        result=profile_interval(lambda r:0. if abs(r+4)<.002 else 11000.,-4.,0.)
        self.assertTrue(np.isnan(result).all())
    def test_association_stage_excludes_nearby_but_disjoint_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for s,site in enumerate(SITES):
                folder=root/("20261117_"+site)
                (folder/"event_cuts").mkdir(parents=True)
                times=np.arange(20,dtype=np.int64)*5000000+(125000000 if s else 0)
                with h5py.File(folder/"catalogue.h5","w") as h:
                    h["event_start_utc_ns"]=[times[0]]
                    h["event_stop_utc_ns"]=[times[-1]]
                    h["event_peak_snr_db"]=[20.]
                with h5py.File(folder/"event_cuts/event_000000.h5","w") as h:
                    h["ransac_inlier"]=np.ones(20,np.uint8)
                    h["pulses/peak_snr_db"]=np.full(20,20.)
                    h["pulses/utc_ns"]=times
            associate(root,root/"out")
            with h5py.File(root/"out/associations.h5") as h:
                self.assertEqual(len(h["station_event_indices"]),0)
                self.assertEqual(len(h["insufficient_overlap_station_event_indices"]),1)

    def test_disjoint_station_arcs_cannot_form_tristatic_event(self):
        t=np.arange(20,dtype=np.int64)*5000000
        self.assertFalse(measure_overlap([t,t+125000000,t+125000000])[0])

    def test_shared_pulses_allow_small_clock_offsets_not_neighbouring_ipps(self):
        t=np.arange(12,dtype=np.int64)*5000000
        self.assertTrue(measure_overlap([t,t+25000,t-25000])[0])
        self.assertFalse(measure_overlap([t,t+1000000,t])[0])
        self.assertFalse(measure_overlap([t,t[:5],t[:5]])[0])

    def test_shared_arc_must_cover_half_of_each_station(self):
        t=np.arange(40,dtype=np.int64)*5000000
        self.assertFalse(measure_overlap([t,t[:8],t[:8]])[0])
        self.assertTrue(measure_overlap([t,t[:20],t[:20]])[0])
    def test_raw_cut_replays_q_plus_i_i_not_conjugated_waveform(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for site in SITES:
                path=root/("20261117_"+site)/"event_cuts/event_000000.h5"
                path.parent.mkdir(parents=True)
                with h5py.File(path,"w") as h:
                    h["config_toml_utf8"]=np.frombuffer(b"delay_correction_us = 0.0\n",dtype=np.uint8)
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
            for row in rows:
                self.assertAlmostEqual(row["sample"][0],
                    (466.+DELAY_CORRECTION_US)*1e-6+(800-240)/4e6, places=12)
                self.assertAlmostEqual(row["apparent"],
                    C*(566.+DELAY_CORRECTION_US)*1e-6, places=8)

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
