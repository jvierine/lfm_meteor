import numpy as np
from export_voltage_rcs_mass import integrate, invert_scatter
from export_voltage_rcs_mass import RATE_FACTOR

def run(rcs,gain=None,rejected=None):
    t=np.arange(len(rcs),dtype=float)
    source=np.array([12.,13.,14.,15.])
    table=np.broadcast_to([0.,10.,20.,30.],(2,2,4)).copy()
    return integrate(t,np.array(rcs),np.full(len(t),95.),np.full(len(t),40.),
        np.zeros(len(t)) if gain is None else np.array(gain),
        np.zeros(len(t),bool) if rejected is None else np.array(rejected),
        (np.array([90.,100.]),np.array([30.,50.]),source,table),
        dict(snr=np.full(len(t),80.),generator=np.random.default_rng(1)))

def test_gap_interpolation_and_altitude_speed():
    p,s,_=run([10.,np.nan,10.])
    np.testing.assert_allclose(s['mass_kg'],2*RATE_FACTOR*1e13)
    np.testing.assert_allclose(p[0]['interpolated_curves_dbsm'],np.tile([0.,10.,20.,30.],(len(p[0]['time_s']),1)))

def test_no_bridge_outside_beam():
    _,s,_=run([10.,10.,10.],gain=[0.,-11.,0.])
    assert np.isnan(s['mass_kg'])

def test_rejected_value_not_used():
    _,a,_=run([10.,29.,10.],rejected=[False,True,False])
    _,b,_=run([10.,np.nan,10.])
    assert a['mass_kg']==b['mass_kg']

def test_lookup_boundary_clamping():
    _,s,_=run([-10.,-10.])
    np.testing.assert_allclose(s['mass_kg'],RATE_FACTOR*1e12)

def test_scatter_tails_extend_edge_slopes():
    root,extended=invert_scatter(np.array([0.,10.,20.,30.]),np.array([-10.,15.,40.]),np.array([12.,13.,14.,15.]))
    np.testing.assert_allclose(root,[12.,13.5,15.])
    np.testing.assert_array_equal(extended,[True,False,True])

def test_large_scatter_retains_interval():
    _,s,_=run([1.,29.])
    assert s['rcs_rms_db']==14.
    assert np.isfinite(s['lo95_scatter_kg']+s['hi95_scatter_kg'])
    assert s['uncertainty_clamped_fraction']>0

def test_event_rms_shared_scatter():
    _,s,draws=run([9.,11.])
    np.testing.assert_allclose(s['rcs_rms_db'],1.)
    assert s['lo95_scatter_kg']<s['mass_kg']<s['hi95_scatter_kg']
    # A shared offset maps to a common multiplicative rate error for this table.
    np.testing.assert_allclose(np.std(np.log10(draws/s['mass_kg'])),.1,rtol=.05)

def test_bilinear_altitude_velocity_values():
    values=np.array([[[0.,10.,20.,30.],[10.,20.,30.,40.]],
                     [[20.,30.,40.,50.],[30.,40.,50.,60.]]])
    pieces,summary,_=integrate(np.array([0.,1.]),np.array([25.,25.]),
        np.array([95.,95.]),np.array([40.,40.]),np.zeros(2),np.zeros(2,bool),
        (np.array([90.,100.]),np.array([30.,50.]),np.array([12.,13.,14.,15.]),values),
        dict(snr=np.full(2,80.),generator=np.random.default_rng(1)))
    np.testing.assert_allclose(pieces[0]['interpolated_curves_dbsm'][0],[15.,25.,35.,45.])
    np.testing.assert_allclose(summary['mass_kg'],RATE_FACTOR*1e13)
