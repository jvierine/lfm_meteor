"""Reproducible numerical cross-check for the current manuscript products."""
from pathlib import Path
import hashlib
import h5py
import numpy as np


def main():
    code=Path(__file__).resolve().parent
    root=code/'results/tristatic_calibrated_likelihood_20261109'
    old=code/'results/tristatic_seven_parameter_catalogue_20261108'
    output=root/'products/paper_number_audit.h5'
    paths=[root/'products/mass_distribution.h5',root/'products/mass_influx.h5',
           root/'current_mass_influx.h5',old/'products/population_summary.h5',
           old/'orbits/orbits.h5',old/'products/altitude_correction.h5',
           old/'products/area_uncertainty.h5']
    with (h5py.File(paths[0]) as mass,h5py.File(paths[1]) as influx,h5py.File(paths[2]) as previous,
          h5py.File(paths[3]) as population,h5py.File(paths[4]) as orbit,h5py.File(output,'w') as out):
        audit=out.create_group('provenance')
        for p in paths:audit.attrs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        ids=mass['event_id'].asstr()[:]; values=mass['initial_mass_kg'][:]
        dynamic=dict(zip(ids,values));rcs=dict(zip(previous['rcs/event_id'].asstr()[:],previous['rcs/mass_kg'][:]))
        paired=mass['rcs_event_id'].asstr()[:]
        ratio=np.array([dynamic[e]/rcs[e] for e in paired])
        g=out.create_group('mass')
        g.attrs.update(dynamic_events=len(ids),rcs_events=len(rcs),paired_events=len(paired),
            dynamic_total_kg=sum(values),dynamic_min_kg=min(values),dynamic_median_kg=np.median(values),
            dynamic_max_kg=max(values),rcs_total_kg=sum(rcs.values()),rcs_min_kg=min(rcs.values()),
            rcs_max_kg=max(rcs.values()),paired_rcs_median_kg=np.median([rcs[e] for e in paired]),
            dynamic_greater_count=sum(ratio>1),dynamic_greater_percent=np.mean(ratio>1)*100,
            median_dynamic_rcs_ratio=np.median(ratio))
        print('MASS',dict(g.attrs))
        for kind in ['drop','no_drop']:
            names=population[kind+'/event_id'].asstr()[:]
            d=np.array([dynamic[e] for e in names if e in dynamic]);r=np.array([rcs[e] for e in names if e in rcs])
            g=out.create_group(kind)
            g.attrs.update(**dict(population[kind].attrs))
            g.attrs.update(dynamic_events=len(d),dynamic_mean_kg=np.mean(d),dynamic_median_kg=np.median(d),
                rcs_events=len(r),rcs_mean_kg=np.mean(r))
            print(kind,dict(g.attrs))
        orbit.copy('classification_summary',out)
        out.attrs.update(orbit_events=len(orbit['event_id']),orbital_samples=int(np.sum(orbit['valid'][:])),
            hyperbolic_samples=int(np.sum(orbit['valid'][:] & (orbit['e'][:]>=1))),
            plotted_orbits=int(np.sum(orbit['plot_keep'][:])))
        for kind,names,m,source in [('dynamic',ids,values,influx),
                                    ('rcs',previous['rcs/event_id'].asstr()[:],previous['rcs/mass_kg'][:],previous['rcs'])]:
            stage='beam_sampling' if kind=='dynamic' else 'plus_altitude_heuristic'
            w=source[stage+('/weight_ton_day_per_kg' if kind=='dynamic' else '/weight_per_day')][:]
            if kind=='rcs':w=w/1000
            contribution=m*w;top=np.argsort(contribution)[-2:]
            g=out.create_group('influx_'+kind)
            g.attrs.update(top_two_fraction=contribution[top].sum()/contribution.sum(),
                omit_two_ton_day=contribution.sum()-contribution[top].sum(),
                matched_ton_day=sum(c for e,c in zip(names,contribution) if e in set(paired)))
            print('INFLUX',kind,dict(g.attrs))
        assert len(ids)==135 and len(paired)==115 and len(rcs)==197
    print(output)


if __name__=='__main__':main()
