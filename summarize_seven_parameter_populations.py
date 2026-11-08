"""Recalculate RCS-drop, size and radiation-drag summaries from current products."""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import quad

from survey_rcs_steps import abrupt_drop, self_test


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    a=p.parse_args()
    root=a.root
    self_test()
    with h5py.File(root/'products/current_mass_influx.h5') as h:
        dynamic=dict(zip(h['dynamic/event_id'].asstr()[:],h['dynamic/mass_kg'][:]))
        rcs=dict(zip(h['rcs/event_id'].asstr()[:],h['rcs/mass_kg'][:]))
        ids=list(h['catalogue_event_id'].asstr()[:])
    with h5py.File(root/'products/radiants.h5') as h:
        speed=dict(zip(h['event_id'].asstr()[:],h['speed_km_s'][:]))
    with h5py.File(root/'orbits/orbits.h5') as h:
        orbit_ids=list(h['event_id'].asstr()[:])
        probability=dict(zip(orbit_ids,h['cometary_probability_bound'][:]))
        ae={e:(h['a_au'][i],h['e'][i]) for i,e in enumerate(orbit_ids)}
    cases=[]
    for event in ids:
        folder=root/'catalog'/event
        with h5py.File(folder/'plot.h5') as h:
            kept=set(h['time_ns'][:][h['station'][:]==0])
        with h5py.File(folder/'beam_rcs.h5') as h:
            t=h['time_ns'][:]
            valid=np.array([x in kept for x in t]) & np.isfinite(h['rcs_dbsm'][:])
            valid &= (h['snr_db'][:]>=20)&(h['relative_gain_db'][:]>=-10)
            order=np.argsort(t)
            eligible,(score,_,_)=abrupt_drop((t[order]-t.min())/1e6,h['rcs_dbsm'][:][order],valid[order])
        cases.append((event,eligible,score))
    with h5py.File(root/'products/population_summary.h5','w') as out:
        out.attrs.update(trajectory_events=len(ids),orbital_events=len(orbit_ids),
                         classification='Mean of five criteria, equal weight per event with verified orbit posterior')
        for name,drop in [('drop',True),('no_drop',False)]:
            selected=[e for e,ok,s in cases if ok and (s>=5)==drop]
            g=out.create_group(name)
            g['event_id']=np.asarray(selected,dtype=h5py.string_dtype())
            orb=[e for e in selected if e in probability and np.isfinite(probability[e])]
            g.attrs.update(events=len(selected),orbital_events=len(orb),
                           mean_speed_km_s=np.mean([speed[e] for e in selected]),
                           cometary_fraction=np.mean([probability[e] for e in orb]))
            for kind,lookup in [('dynamic',dynamic),('rcs',rcs)]:
                selected_mass=np.array([lookup[e] for e in selected if e in lookup])
                g.attrs.update({kind+'_events':len(selected_mass),
                                kind+'_mean_kg':np.mean(selected_mass),
                                kind+'_median_kg':np.median(selected_mass)})
        masses=np.array(list(dynamic.values()))
        radii=(masses/(4*np.pi/3*3000))**(1/3)
        # Compact spheres, radiation-pressure efficiency one; SI constants.
        c=299792458.; luminosity=3.828e26; gm_sun=1.32712440018e20
        beta=3*luminosity/(16*np.pi*gm_sun*c*3000*radii)
        g=out.create_group('sizes')
        g['diameter_um_quantiles']=np.quantile(2*radii*1e6,[0,.25,.5,.75,1])
        g['beta_quantiles']=np.quantile(beta,[0,.25,.5,.75,1])
        lifetimes=[]
        for event,mass in dynamic.items():
            if event not in ae:
                continue
            aa,ee=ae[event]
            with h5py.File(root/'orbits/inputs'/(event+'.h5')) as h:
                radius=10**h['sample_params'][:,6]
            values=[]
            for semimajor,ecc,r in zip(aa,ee,radius):
                if not 0<=ecc<1 or semimajor<=0:
                    continue
                factor=(5/8 if ecc<1e-6 else
                        ecc**(-1.6)*(1-ecc*ecc)**2*
                        quad(lambda y:y**.6/(1-y*y)**1.5,0,ecc,epsabs=1e-9)[0])
                ballistic=4/3*3000*r
                tau=8*np.pi*ballistic*c*c*(semimajor*149597870700.)**2/(5*luminosity)*factor
                values.append(tau/(365.25*86400))
            if values:
                lifetimes.append((event,np.quantile(values,[.025,.5,.975])))
        g=out.create_group('pr_lifetime')
        g['event_id']=np.asarray([x[0] for x in lifetimes],dtype=h5py.string_dtype())
        g['interval95_yr']=np.array([x[1] for x in lifetimes])
        g['event_median_quantiles_yr']=np.quantile([x[1][1] for x in lifetimes],[0,.25,.5,.75,1])
        g.attrs['interpretation']='Joint sampled first-detection radius and orbit; fixed spherical density and Qpr=1; not pre-atmospheric residence times'
    print(root/'products/population_summary.h5',flush=True)


if __name__=='__main__':
    main()
