"""Export compact HDF5 plot inputs; all rendering is native rustplotlib."""
import argparse
import os
from pathlib import Path
import h5py
import numpy as np
from run_native_voltage_catalog import apparent_paths


def posterior_bands(folder, data, t):
    paths = [folder / f'chain_{i}.h5' for i in range(4)]
    if not all(p.exists() for p in paths):
        return {}, dict(uncertainty_status='unavailable: four completed chains required')
    with_samples = []
    for p in paths:
        with h5py.File(p) as h:
            with_samples.append(h['samples'][:])
    samples = np.stack(with_samples)
    if samples.ndim != 3 or samples.shape[2] != 8 or not np.isfinite(samples).all():
        raise ValueError('Invalid posterior samples')
    import arviz as az
    inference = az.from_dict(posterior={'q': samples})
    rhat = float(np.max(az.rhat(inference, method='rank')['q'].values))
    ess = float(np.min(az.ess(inference, method='bulk')['q'].values))
    tail = float(np.min(az.ess(inference, method='tail')['q'].values))
    converged = bool(np.isfinite(rhat) and rhat < 1.01 and ess >= 400 and tail >= 400)
    # Use every production draw; no observation noise is added to model intervals.
    draws = samples.reshape(-1, 8)
    path_draws = np.stack([apparent_paths(q, data) for q in draws])
    speeds = (np.linalg.norm(data['center'][None, 3:] + 1000*draws[:, 3:6], axis=1)[:, None]
              - 10**draws[:, 6, None] * np.exp(10**draws[:, 7, None]*t[None, :]))/1000
    pl, pu = np.quantile(path_draws, [.025, .975], axis=0)
    vl, vu = np.quantile(speeds, [.025, .975], axis=0)
    return dict(path_lower95_m=pl, path_upper95_m=pu, speed_lower95_km_s=vl,
                speed_upper95_km_s=vu), dict(
        uncertainty_status='95% pointwise model interval' if converged else '95% sampled model interval; chains not converged',
        uncertainty_converged=converged, posterior_max_rhat=rhat,
        posterior_min_bulk_ess=ess, posterior_min_tail_ess=tail,
        uncertainty_draws=len(draws), uncertainty_alpha=.5,
        uncertainty_kind='Conditional equal-tail pointwise posterior; excludes measurement and physical-model errors')


def export(folder):
    folder=Path(folder)
    with h5py.File(folder/'input.h5') as h:
        data={k:h[k][()] for k in h}
    with h5py.File(folder/'fit.h5') as h:
        q=h['params'][:]
        fields={k:h[k][()] for k in ('time_s','station','seed_apparent_path_m',
                'residual_beat_hz','conditional_beat_std_hz')}
        passed=bool(h.attrs['beat_guard_pass'])
    fields['modeled_path_m']=apparent_paths(q,data)
    fields['time_ns']=data['time_ns']
    t=np.linspace(data['time_s'].min(),data['time_s'].max(),200)
    fields['speed_time_s']=t
    fields['speed_km_s']=(np.linalg.norm(data['center'][3:]+1000*q[3:6])-10**q[6]*np.exp(10**q[7]*t))/1000
    bands, diagnostics = posterior_bands(folder, data, t)
    fields.update(bands)
    with h5py.File(folder/'plot.h5.partial','w') as h:
        for k,v in fields.items(): h[k]=v
        h.attrs['beat_guard_pass']=passed
        h.attrs['event_id']=folder.name
        h.attrs['source_program']=__file__
        h.attrs.update(diagnostics)
    os.replace(folder/'plot.h5.partial', folder/'plot.h5')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args()
    root=Path(args.root)
    folders=[root] if (root/'fit.h5').exists() else sorted(x.parent for x in root.glob('*/fit.h5'))
    for folder in folders:
        export(folder)
    print(f'Exported {len(folders)} plot bundles')
