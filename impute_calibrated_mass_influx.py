"""Influx with explicit median-contribution imputation.

Missing events receive the median weighted contribution, not a claimed measured
mass. Endpoint envelopes retain their sensitivity interpretation.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np


def main():
    root=Path(__file__).resolve().parent/'results/tristatic_calibrated_likelihood_20261109'
    source=root/'products/mass_influx.h5'
    output=root/'products/imputed_mass_influx.h5'
    with h5py.File(root/'mass_selection.h5') as h:
        catalogue=h['events'].asstr()[:]
        statuses=h['status'].asstr()[:]
    with h5py.File(source) as h,h5py.File(output,'w') as out:
        measured=h['event_id'].asstr()[:]
        missing=sorted(set(catalogue)-set(measured))
        assert len(set(catalogue))==len(catalogue)==234
        assert len(measured)==135 and len(missing)==99
        out.attrs.update(n_catalogue=len(catalogue),n_measured=len(measured),n_imputed=len(missing),
            imputation_statistic='median weighted contribution',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            assumption='Each unusable-mass event contributes the median weighted mass contribution of selected events',
            limitation='Missingness may depend on mass and deceleration; imputation is not a validated selection correction',
            uncertainty='Endpoint sensitivity envelope with median imputation, not a calibrated global confidence interval; imputation/model uncertainty excluded')
        for key,values in [('measured_event_id',measured),('imputed_event_id',missing),
                           ('catalogue_event_id',catalogue),('catalogue_mass_status',statuses)]:
            out[key]=np.asarray(values,dtype=h5py.string_dtype())
        for name in ['area_exposure','velocity','beam_sampling']:
            original=h[name+'/estimate_and_endpoint_envelope_ton_day'][:]
            weight=h[name+'/weight_ton_day_per_kg'][:]
            medians=np.array([np.median(h[key][:]*weight) for key in ['mass_kg','mass_lo_kg','mass_hi_kg']])
            imputed=len(missing)*medians
            total=original+imputed
            g=out.create_group(name)
            g['measured_estimate_and_envelope_ton_day']=original
            g['imputed_estimate_and_envelope_ton_day']=imputed
            g['total_estimate_and_envelope_ton_day']=total
            g['imputed_event_contribution_ton_day']=np.full(len(missing),medians[0])
            np.testing.assert_allclose(original[0]+g['imputed_event_contribution_ton_day'][:].sum(),total[0])
            assert total[1]<=total[0]<=total[2]
            print(name,'total and endpoint envelope',total)
        print('events',len(measured),'measured +',len(missing),'median-imputed')
        with h5py.File(root/'current_mass_influx.h5') as previous:
            rcs=previous['rcs']; ids=rcs['event_id'].asstr()[:]
            assert set(ids).issubset(catalogue)
            bounds=np.array([np.quantile(rcs['input_mass_draws/'+event][:],[.025,.975]) for event in ids])
            g=out.create_group('rcs')
            g.attrs.update(n_measured=len(ids),n_imputed=len(catalogue)-len(ids),imputation_statistic='median weighted contribution',
                source_sha256=hashlib.sha256((root/'current_mass_influx.h5').read_bytes()).hexdigest())
            for name,stage in [('area_exposure','area_exposure'),('velocity','plus_velocity_heuristic'),
                               ('beam_sampling','plus_altitude_heuristic')]:
                target=g.create_group(name)
                rate=rcs[stage].attrs['rate_ton_day']
                median=np.median(rcs[stage+'/contribution_ton_day'][:])
                weights=rcs[stage+'/weight_per_day'][:]/1000
                weighted_bounds=bounds*weights[:,None]
                envelope=weighted_bounds.sum(axis=0)+(len(catalogue)-len(ids))*np.median(weighted_bounds,axis=0)
                target['total_endpoint_envelope_ton_day']=envelope
                target.attrs.update(measured_rate_ton_day=rate,imputed_per_event_ton_day=median,
                    total_rate_ton_day=rate+(len(catalogue)-len(ids))*median)
                print('RCS',name,'endpoint envelope',envelope)
            print('RCS imputed total',g['beam_sampling'].attrs['total_rate_ton_day'])


if __name__=='__main__':main()
