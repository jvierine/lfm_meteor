"""Count-scaled influx with explicit mean-contribution imputation.

Missing events receive the mean weighted contribution, not a claimed measured
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
        factor=len(catalogue)/len(measured)
        out.attrs.update(n_catalogue=len(catalogue),n_measured=len(measured),n_imputed=len(missing),
            factor=factor,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            assumption='Unusable-mass events have the same mean weighted mass contribution as selected events',
            limitation='Missingness may depend on mass and deceleration; imputation is not a validated selection correction',
            uncertainty='Scaled endpoint sensitivity envelope, not a calibrated global confidence interval; imputation/model uncertainty excluded')
        for key,values in [('measured_event_id',measured),('imputed_event_id',missing),
                           ('catalogue_event_id',catalogue),('catalogue_mass_status',statuses)]:
            out[key]=np.asarray(values,dtype=h5py.string_dtype())
        for name in ['area_exposure','velocity','beam_sampling']:
            original=h[name+'/estimate_and_endpoint_envelope_ton_day'][:]
            g=out.create_group(name)
            g['measured_estimate_and_envelope_ton_day']=original
            g['imputed_estimate_and_envelope_ton_day']=original*(factor-1)
            g['total_estimate_and_envelope_ton_day']=original*factor
            g['imputed_event_contribution_ton_day']=np.full(len(missing),original[0]/len(measured))
            np.testing.assert_allclose(original[0]+g['imputed_event_contribution_ton_day'][:].sum(),original[0]*factor)
            print(name,'total and endpoint envelope',original*factor)
        print('events',len(measured),'measured +',len(missing),'imputed; factor',factor)
        with h5py.File(root/'current_mass_influx.h5') as previous:
            rcs=previous['rcs']; ids=rcs['event_id'].asstr()[:]
            assert set(ids).issubset(catalogue)
            g=out.create_group('rcs')
            scale=len(catalogue)/len(ids)
            g.attrs.update(n_measured=len(ids),n_imputed=len(catalogue)-len(ids),factor=scale,
                source_sha256=hashlib.sha256((root/'current_mass_influx.h5').read_bytes()).hexdigest())
            for name,stage in [('area_exposure','area_exposure'),('velocity','plus_velocity_heuristic'),
                               ('beam_sampling','plus_altitude_heuristic')]:
                target=g.create_group(name)
                rate=rcs[stage].attrs['rate_ton_day']
                target.attrs.update(measured_rate_ton_day=rate,total_rate_ton_day=rate*scale)
            print('RCS imputed total',g['beam_sampling'].attrs['total_rate_ton_day'])


if __name__=='__main__':main()
