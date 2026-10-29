"""Publish audited block-bootstrap intervals without changing the best fit or flags."""
import argparse
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
from fit_cached_shrinking_voltage import native
from fit_shrinking_voltage import Model, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--event-folder',type=Path,required=True)
    p.add_argument('--bootstrap',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    a=p.parse_args(); folder=a.event_folder
    with h5py.File(a.bootstrap/'bootstrap.h5') as h:
        assert np.all(h['converged'][:]), 'Failed bootstrap optimization'
        selected=h['block_seconds'][:]==.04
        draws=h['params'][:][selected]
        revision=int(h.attrs['flag_revision'])
        assert len(draws)>=500
    with h5py.File(a.bootstrap/'central_refit.h5') as h:
        assert h.attrs['converged'] and abs(h.attrs['logp_gain'])<.01, 'Central optimum changed; regenerate fit products first'
    with h5py.File(folder/'input.h5') as h:
        d={k:h[k][()] for k in h}
        assert int(h.attrs['manual_flag_revision'])==revision
    with h5py.File(folder/'plot.h5') as h:
        fields={k:h[k][()] for k in h}; attrs=dict(h.attrs)
        assert attrs['fit_flag_revision']==revision
    assert np.all((draws[:,6]>-6.99)&(draws[:,6]<-2.01)), 'Boundary-limited bootstrap requires separate reporting'
    lib=native(a.library);model=Model(lib,folder,d)
    order=np.argsort(d['time_s']); paths=[]; speeds=[]
    for q in draws:
        state=model.states(q);paths.append(model.paths(q,state));speeds.append(state[order,1]/1000)
    paths=np.asarray(paths);speeds=np.asarray(speeds)
    assert np.isfinite(paths).all() and np.isfinite(speeds).all()
    radius=10**draws[:,6];mass=4*np.pi/3*3000*radius**3
    fields['initial_radius_m_quantiles']=np.quantile(radius,[.025,.5,.975])
    fields['initial_mass_kg_quantiles']=np.quantile(mass,[.025,.5,.975])
    fields['path_lower95_m'],fields['path_upper95_m']=np.quantile(paths,[.025,.975],axis=0)
    fields['speed_lower95_km_s'],fields['speed_upper95_km_s']=np.quantile(speeds,[.025,.975],axis=0)
    method='time-block exponential multiplier likelihood bootstrap'
    updates=dict(uncertainty_status='95% empirical time-block bootstrap interval',uncertainty_method=method,
                 uncertainty_updated_unix=time.time(),uncertainty_bootstrap_block_s=.04,
                 uncertainty_bootstrap_replicates=len(draws),uncertainty_converged=True,
                 mass_status='Direct shrinking-radius mass; 40 ms block-bootstrap uncertainty; density 3000 kg/m3, Gamma=1, sigma=1e-8 fixed; physical systematic errors excluded')
    attrs.update(updates)
    for k in list(attrs):
        if k.startswith('posterior_'): del attrs[k]
    backup=folder/'thermal_posterior_before_block_bootstrap'
    backup.mkdir(exist_ok=False)
    for name in ('plot.h5','complete.h5','dynamic_mass_posterior.h5'):
        shutil.copy2(folder/name,backup/name)
    save(folder/'dynamic_mass_posterior.h5',dict(radius_m=radius,mass_kg=mass),
         dict(model='direct shrinking radius',uncertainty_method=method,bootstrap_optimizer_converged=True,
              is_mcmc_posterior=False,confidence=.95,block_seconds=.04))
    save(folder/'block_bootstrap_states.h5',dict(params=draws,paths_m=paths,speeds_km_s=speeds),updates)
    save(folder/'plot.h5',fields,attrs)
    with h5py.File(folder/'complete.h5') as h: complete=dict(h.attrs)
    complete.update(updates)
    for k in list(complete):
        if k.startswith('posterior_'):del complete[k]
    save(folder/'complete.h5',{},complete)
    lib.voltage_free(model.handle)
    print('PUBLISHED',fields['initial_mass_kg_quantiles'],fields['initial_radius_m_quantiles'],flush=True)


if __name__=='__main__':main()
