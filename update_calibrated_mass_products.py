"""Refresh mass figures and fixed-weight influx endpoint envelopes.

Confidence projections are not posterior draws. Endpoint sums are sensitivity
envelopes, not a calibrated confidence interval for the global influx.
"""
from pathlib import Path
import hashlib
import subprocess
import h5py
import numpy as np


def main():
    root = Path(__file__).resolve().parent/'results/tristatic_calibrated_likelihood_20261109'
    out = root/'products'
    out.mkdir(exist_ok=True)
    with h5py.File(root/'mass_selection.h5') as h:
        names = h['events'].asstr()[:][h['dynamic_mass_included'][:]].tolist()
    with h5py.File(root/'radiants.h5') as h:
        ids = h['event_id'].asstr()[:]
        speeds = dict(zip(ids, h['speed_km_s'][:]))
        exposure = dict(zip(ids, h['family_effective_exposure_sum_h'][:]))
    mass, low, high = [], [], []
    for event in names:
        with h5py.File(root/'confidence'/event/'confidence.h5') as h:
            mass.append(4*np.pi/3*3000*10**(3*h['best_params'][-1]))
            lo, hi = h['mass_confidence_interval_kg'][:]
            low.append(lo); high.append(hi)
    mass, low, high = map(np.asarray, (mass, low, high))
    assert len(names)==len(set(names))==135
    assert np.all((low>0)&(low<=mass)&(mass<=high)&(high<10*low))
    provenance=dict(source_git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=root.parents[1],text=True).strip(),
        mass_selection_sha256=hashlib.sha256((root/'mass_selection.h5').read_bytes()).hexdigest(),
        weight_source_sha256=hashlib.sha256((root/'radiants.h5').read_bytes()).hexdigest())
    with h5py.File(root/'current_mass_influx.h5') as old:
        area = old.attrs['collecting_area_km2']
        factor = old.attrs['altitude_sampling_factor']
        r = old['rcs']
        ri = {e:i for i,e in enumerate(r['event_id'].asstr()[:])}
        paired = [e for e in names if e in ri]
        rm = r['mass_kg'][:]
        rlo, rhi = [], []
        for event in paired:
            rlo.append(np.quantile(r['input_mass_draws/'+event][:], .025))
            rhi.append(np.quantile(r['input_mass_draws/'+event][:], .975))
        with h5py.File(out/'mass_distribution.h5', 'w') as h:
            for key, value in dict(event_id=np.asarray(names,dtype=h5py.string_dtype()),
                initial_mass_kg=mass, mass_95_lo_kg=low, mass_95_hi_kg=high,
                initial_speed_km_s=[speeds[e] for e in names], selected=np.ones(len(names),bool),
                rcs_event_id=np.asarray(paired,dtype=h5py.string_dtype()),
                rcs_mass_kg=[rm[ri[e]] for e in paired], rcs_lo95_kg=rlo, rcs_hi95_kg=rhi,
                rcs_speed_km_s=[speeds[e] for e in paired]).items(): h[key]=value
            h.attrs.update(density_kg_m3=3000.,bin_width_dex=.48,
                uncertainty_method='Calibrated joint likelihood confidence projection; not posterior quantiles')
            h.attrs.update(**provenance)
        with h5py.File(out/'mass_influx.h5','w') as h:
            h.attrs.update(no_mass_imputation=True, fixed_historical_survey_weights=True,
                range_meaning='Sum of event confidence endpoints; not a calibrated global 95% interval',
                physical_systematic_uncertainty_included=False)
            h.attrs.update(**provenance)
            base = np.array([4*np.pi*(6371+95)**2/area*4*24/exposure[e]/1000 for e in names])
            velocity = np.array([(72/speeds[e])**3 for e in names])
            h['event_id']=np.asarray(names,dtype=h5py.string_dtype())
            h['mass_kg']=mass; h['mass_lo_kg']=low; h['mass_hi_kg']=high
            for name,w in [('area_exposure',base),('velocity',base*velocity),('beam_sampling',base*velocity*factor)]:
                g=h.create_group(name)
                g['weight_ton_day_per_kg']=w
                g['estimate_and_endpoint_envelope_ton_day']=[sum(mass*w),sum(low*w),sum(high*w)]
                print(name,g['estimate_and_endpoint_envelope_ton_day'][:])
            print('SELECTED',len(names),'PAIRED',len(paired),'TOTAL KG',mass.sum())
    with h5py.File(root/'rcs_figure.h5') as old,h5py.File(out/'rcs_figure.h5','w') as h:
        old.copy('pulses',h)
        g=h.create_group('mass_comparison')
        idx={e:i for i,e in enumerate(names)}
        for key,value in dict(event_id=np.asarray(paired,dtype=h5py.string_dtype()),
            dynamic_mass_kg=[mass[idx[e]] for e in paired],dynamic_lo95_kg=[low[idx[e]] for e in paired],
            dynamic_hi95_kg=[high[idx[e]] for e in paired],rcs_mass_kg=[rm[ri[e]] for e in paired],
            delta_mass_lo95_noise_only_kg=rlo,delta_mass_hi95_noise_only_kg=rhi).items():g[key]=value
        g.attrs['sample_count']=len(paired)


if __name__=='__main__':main()
