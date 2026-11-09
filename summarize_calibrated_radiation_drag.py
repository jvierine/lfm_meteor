"""Compact-sphere radiation scales for the current selected dynamic masses."""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from scipy.special import hyp2f1


def main():
    root=Path(__file__).resolve().parent
    source=root/'results/tristatic_calibrated_likelihood_20261109/products/mass_distribution.h5'
    orbit=root/'results/tristatic_seven_parameter_catalogue_20261108/orbits/orbits.h5'
    output=source.parent/'radiation_drag.h5'
    with h5py.File(source) as h:
        ids=h['event_id'].asstr()[:]; mass=h['initial_mass_kg'][:]
    radius=(3*mass/(4*np.pi*3000))**(1/3)
    c=299792458.; luminosity=3.828e26; gm=1.32712440018e20
    beta=3*luminosity/(16*np.pi*gm*c*3000*radius)
    values=[]; matched=[]
    with h5py.File(orbit) as h:
        index={e:i for i,e in enumerate(h['event_id'].asstr()[:])}
        for event,r in zip(ids,radius):
            if event not in index:continue
            i=index[event]; a=h['a_au'][i]; e=h['e'][i]
            valid=h['valid'][i] & np.isfinite(a) & np.isfinite(e) & (a>0) & (e>=0) & (e<1)
            a=a[valid]; e=e[valid]
            if not len(a):continue
            circular=np.pi*(4/3*3000*r)*c*c*(a*149597870700.)**2/luminosity
            tau=circular*(1-e*e)**2*hyp2f1(.8,1.5,1.8,e*e)/(365.25*86400)
            values.append(np.median(tau));matched.append(event)
    with h5py.File(output,'w') as h:
        h.attrs.update(mass_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            orbit_source_sha256=hashlib.sha256(orbit.read_bytes()).hexdigest(),density_kg_m3=3000.,Q_pr=1.,
            interpretation='Fixed best-fit first-detection radius; event-median secular timescale over saved gravity-only orbit samples, not residence time or joint mass-orbit uncertainty')
        h['event_id']=np.asarray(ids,dtype=h5py.string_dtype()); h['radius_m']=radius;h['beta']=beta
        h['diameter_um_quantiles']=np.quantile(2e6*radius,[0,.25,.5,.75,1])
        h['pr_event_id']=np.asarray(matched,dtype=h5py.string_dtype());h['pr_event_median_years']=values
        h['pr_years_quantiles']=np.quantile(values,[0,.25,.5,.75,1])
        print('selected',len(ids),'PR matched',len(matched))
        print('diameters',h['diameter_um_quantiles'][:],'median beta',np.median(beta))
        print('PR years',h['pr_years_quantiles'][:])


if __name__=='__main__':main()
