"""Combine independently corrected campaign rates, preserving their identities."""
import argparse
import hashlib
from pathlib import Path
import h5py
import numpy as np


def rate_curves(mass, reference_weight, speed, reference_p, missing, p):
    mass=np.asarray(mass); speed=np.asarray(speed);reference_weight=np.asarray(reference_weight)
    assert mass.shape==speed.shape==reference_weight.shape
    assert np.all(np.isfinite(mass)&(mass>0))
    assert np.all(np.isfinite(speed)&(speed>0))
    assert np.all(np.isfinite(reference_weight)&(reference_weight>0))
    contribution=mass[None,:]*reference_weight[None,:]*(72/speed[None,:])**(p[:,None]-reference_p)
    measured=contribution.sum(axis=1)
    adjusted=measured+missing*np.median(contribution,axis=1)
    # Remove two largest contributions separately at each common exponent.
    omit=np.sort(contribution,axis=1)[:,:-2].sum(axis=1)
    return dict(measured=measured,median_adjusted=adjusted,omit_top_two=omit,
                contribution=contribution,mass=mass,speed=speed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--april',type=Path,required=True)
    parser.add_argument('--november',type=Path,required=True)
    parser.add_argument('--revised',type=Path,required=True)
    parser.add_argument('--april-radiants',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();p=np.linspace(3,3.5,101)
    curves={};counts={}
    with h5py.File(a.april_radiants) as h:april_h=float(h.attrs['observed_hours'])
    with h5py.File(a.april/'products/mass_influx.h5') as h,h5py.File(a.april/'products/mass_distribution.h5') as d:
        np.testing.assert_array_equal(h['event_id'].asstr()[:],d['event_id'].asstr()[:])
        curves['april_dynamic']=rate_curves(h['mass_kg'][:],h['beam_sampling/weight_ton_day_per_kg'][:],d['initial_speed_km_s'][:],3,99,p)
        counts['april_dynamic']=len(h['mass_kg'])
    with h5py.File(a.april/'current_mass_influx.h5') as h:
        g=h['rcs'];n=len(g['mass_kg']);counts['april_rcs']=n
        curves['april_rcs']=rate_curves(g['mass_kg'][:],g['plus_altitude_heuristic/weight_per_day'][:]/1000,g['speed_km_s'][:],3,234-n,p)
    with h5py.File(a.november) as h,h5py.File(a.revised) as r:
        november_h=float(h.attrs['live_hours'])
        np.testing.assert_array_equal(h['dynamic/event_id'][:],r['event_id'][:])
        names=h['event_id'][:];lookup={name:i for i,name in enumerate(names)}
        for kind in ('dynamic','rcs'):
            g=h[kind];ids=g['event_id'][:]
            speed=np.array([h['speed_km_s'][lookup[name]] for name in ids])
            mass=r['revised_mass_kg'][:] if kind=='dynamic' else g['mass_kg'][:]
            curves['november_'+kind]=rate_curves(mass,g['weight_tonne_day_per_kg'][:],speed,3.25,int(g.attrs['missing_events']),p)
            counts['november_'+kind]=len(mass)
        np.testing.assert_allclose(curves['november_dynamic']['measured'],r['measured_rate_tonne_day'][:])
        np.testing.assert_allclose(curves['november_rcs']['measured'],h['rcs/measured_rate_tonne_day'][:])
    live=np.array([april_h,november_h]);alpha=live/live.sum()
    assert np.isclose(alpha.sum(),1)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(a.output,'w') as h:
        h.attrs.update(combination='observing-time-weighted mean of two independently corrected daily rates',
            interpretation='two-campaign snapshot average, not an annual mean or a pooled selection-function fit',
            uncertainty='common p=3..3.5 sensitivity only; no calibrated global confidence interval',
            april_date='2024-04-22',november_date='2026-11-17',
            november_mass_revision='411 and 563 use corrected-pointing minus-10-dB main-lobe refits; other masses and survey weights unchanged',
            physical_model='campaign physical assumptions and uncertainty methods differ; no homogenized refit performed')
        h['p']=p;h['live_hours']=live;h['campaign_fraction']=alpha
        for name,c in curves.items():
            g=h.create_group(name);g.attrs['measured_events']=counts[name]
            for key,value in c.items():g[key]=value
            print(name,'n',counts[name],'measured',c['measured'][[0,-1]],'median-adjusted',c['median_adjusted'][[0,-1]])
        for kind in ('dynamic','rcs'):
            g=h.create_group('combined_'+kind)
            for mode in ('measured','median_adjusted','omit_top_two'):
                value=alpha[0]*curves['april_'+kind][mode]+alpha[1]*curves['november_'+kind][mode]
                g[mode]=value
                np.testing.assert_array_less(np.minimum(curves['april_'+kind][mode],curves['november_'+kind][mode])-1e-10,value)
                np.testing.assert_array_less(value,np.maximum(curves['april_'+kind][mode],curves['november_'+kind][mode])+1e-10)
                print('combined',kind,mode,value[[0,-1]])
        sources=h.create_group('provenance')
        for path in (a.april/'products/mass_influx.h5',a.april/'products/mass_distribution.h5',a.april/'current_mass_influx.h5',a.november,a.revised,a.april_radiants,Path(__file__)):
            sources.attrs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    print('LIVE HOURS',live,'combined',live.sum(),'fractions',alpha)


if __name__=='__main__':main()
