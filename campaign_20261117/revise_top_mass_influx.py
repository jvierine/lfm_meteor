"""Separate sensitivity product replacing the two audited dynamic masses only."""
import argparse
from pathlib import Path
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/gpfs/pub/data_other/juha/20261117_tristatic'))
    a=p.parse_args()
    with h5py.File(a.root/'november_mass_influx.h5') as h:
        names=h['dynamic/event_id'][:]
        mass=h['dynamic/mass_kg'][:]
        w=h['dynamic/weight_tonne_day_per_kg'][:]
        grid=h['dynamic/p'][:]
        all_names=h['event_id'][:]
        speed=h['speed_km_s'][:]
        lookup={name:i for i,name in enumerate(all_names)}
        v=np.array([speed[lookup[name]] for name in names])
        missing=int(h['dynamic'].attrs['missing_events'])
    old=mass.copy();revised=[];bounds=[];largest_residual=[]
    for event in (411,563):
        name=('tri_%06d'%event).encode()
        i=int(np.flatnonzero(names==name)[0])
        result=a.root/'mass_uncertainty_audit_20261122'/name.decode()/'mainlobe_corrected_000/result.h5'
        with h5py.File(result) as h:
            if not h.attrs['optimizer_success']:raise ValueError('Refit did not converge')
            mass[i]=h['mass_kg'][()]
            interval=h['mass_interval_kg'][:]
            if not np.all(np.isfinite(interval)):raise ValueError('Unresolved main-lobe mass interval')
            residual=float(np.max(np.abs(h['path_residual_m'][:])))
            if residual>=40:raise ValueError('Main-lobe residual fails 40 m criterion')
            revised.append(mass[i]);bounds.append(interval);largest_residual.append(residual)
    weights=w[None,:]*(72/v[None,:])**(grid[:,None]-3.25)
    contributions=weights*mass[None,:]
    rates=contributions.sum(axis=1)
    adjusted=rates+missing*np.median(contributions,axis=1)
    out=a.root/'november_mass_influx_top_two_mainlobe.h5'
    with h5py.File(out,'w') as h:
        h.attrs.update(interpretation='two-event main-lobe refit sensitivity; not a fully reprocessed catalogue',
            uncertainty='mass intervals conditional on receiver noise and fixed physical model; arc-selection systematic remains unresolved',
            held_fixed='all other masses, velocities, radiant exposures, collecting area and altitude correction',
            production_catalogue_changed=False)
        h['event_id']=names;h['original_mass_kg']=old;h['revised_mass_kg']=mass
        h['p']=grid;h['measured_rate_tonne_day']=rates;h['median_imputed_rate_tonne_day']=adjusted
        h['revised_event_id']=np.array([b'tri_000411',b'tri_000563'])
        h['revised_event_mass_kg']=revised;h['conditional_interval95_kg']=bounds
        h['maximum_used_path_residual_m']=largest_residual
    print('REVISED masses',revised,'conditional intervals',bounds,'max residual',largest_residual,flush=True)
    print('INFLUX p=3..3.5 measured',rates[[0,-1]],'median-adjusted',adjusted[[0,-1]],flush=True)
    print(out,flush=True)


if __name__=='__main__':main()
