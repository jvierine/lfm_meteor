"""Export compact HDF5 plot inputs; all rendering is native rustplotlib."""
import argparse
import os
from pathlib import Path
import h5py
import numpy as np
from run_native_voltage_catalog import apparent_paths


def initial_dynamic_mass(draws, data):
    """Push the joint trajectory posterior through the free-molecular drag law."""
    import fit_all_ballistic_snr_weighted as base
    import jcoord
    i = int(np.argmin(data['time_s']))
    t = float(data.get('initial_detection_time_s', data['time_s'][i]))
    frame = data.get('initial_detection_frame', data['frames'][i])
    time_ns = int(data.get('initial_detection_time_ns', data['time_ns'][i]))
    x0 = data['center'][:3] + 1000*draws[:, :3]
    vinf = data['center'][3:] + 1000*draws[:, 3:6]
    speed0 = np.linalg.norm(vinf, axis=1)
    u = vinf/speed0[:, None]
    a, b = 10**draws[:, 6], 10**draws[:, 7]
    speed = speed0-a*np.exp(b*t)
    decel = a*b*np.exp(b*t)
    x = x0+(speed0*t-a*np.expm1(b*t)/b)[:, None]*u
    v = speed[:, None]*u
    rp, vp = frame[6:15].reshape(3,3), frame[15:24].reshape(3,3)
    rv, vv = frame[24:33].reshape(3,3), frame[33:42].reshape(3,3)
    xe = frame[:3]+x@rp.T+v@rv.T
    ve = frame[3:6]+x@vp.T+v@vv.T
    altitude = np.array([jcoord.ecef2geodetic(*p)[2] for p in xe])
    rho, meta = base.density_interpolator(np.array([time_ns]), np.median(xe,axis=0)[None,:])
    if np.any((altitude < base.MSIS_ALT_GRID_KM[0]*1000) | (altitude > base.MSIS_ALT_GRID_KM[-1]*1000)):
        raise ValueError('Mass posterior extends outside atmospheric model altitude grid')
    # Along-track drag acceleration includes gravity; air is assumed corotating, no wind.
    gravity = -base.EARTH_MU_M3_S2*x/np.linalg.norm(x,axis=1)[:,None]**3
    drag = decel+np.sum(gravity*u,axis=1)
    ue = u@rp.T
    numerator = rho(altitude)*np.linalg.norm(ve,axis=1)*np.sum(ve*ue,axis=1)
    if np.any(drag <= 0) or np.any(numerator <= 0):
        raise ValueError('Posterior includes states without positive along-track drag; mass is not bounded')
    radius = 3*numerator/(4*3000.*drag)
    mass = 4*np.pi/3*3000.*radius**3
    return radius, mass, meta


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
    from arviz_stats.base import array_stats as az
    diagnostic_samples = samples.transpose(2, 0, 1)
    rhat = float(np.max(az.rhat(diagnostic_samples, method='rank', chain_axis=1, draw_axis=2)))
    ess = float(np.min(az.ess(diagnostic_samples, method='bulk', chain_axis=1, draw_axis=2)))
    tail = float(np.min(az.ess(diagnostic_samples, method='tail', prob=.025, chain_axis=1, draw_axis=2)))
    converged = bool(np.isfinite(rhat) and rhat < 1.01 and ess >= 400 and tail >= 400)
    # Use every production draw; no observation noise is added to model intervals.
    draws = samples.reshape(-1, 8)
    path_draws = np.stack([apparent_paths(q, data) for q in draws])
    speeds = (np.linalg.norm(data['center'][None, 3:] + 1000*draws[:, 3:6], axis=1)[:, None]
              - 10**draws[:, 6, None] * np.exp(10**draws[:, 7, None]*t[None, :]))/1000
    pl, pu = np.quantile(path_draws, [.025, .975], axis=0)
    vl, vu = np.quantile(speeds, [.025, .975], axis=0)
    mass_fields = {}
    mass_status = 'Initial dynamic mass unavailable'
    try:
        radius, mass, atmosphere = initial_dynamic_mass(draws, data)
        mass_fields = dict(initial_radius_m_quantiles=np.quantile(radius,[.025,.5,.975]),
                           initial_mass_kg_quantiles=np.quantile(mass,[.025,.5,.975]))
        with h5py.File(folder/'dynamic_mass_posterior.h5.partial','w') as h:
            h['radius_m']=radius; h['mass_kg']=mass
            h.attrs.update(material_density_kg_m3=3000., Gamma=1., Cd=2.,
                           method='Pushforward of all joint WJ posterior draws at first detection; gravity-corrected along-track free-molecular drag, corotating atmosphere',
                           conditional=True, convergence_verified=converged, **atmosphere)
        os.replace(folder/'dynamic_mass_posterior.h5.partial',folder/'dynamic_mass_posterior.h5')
        mass_status = 'Conditional dynamic mass; density 3000 kg/m3, Gamma=1' + ('' if converged else '; chains not converged')
    except (ValueError, RuntimeError) as e:
        mass_status = 'Initial dynamic mass unavailable: '+str(e)
    return dict(path_lower95_m=pl, path_upper95_m=pu, speed_lower95_km_s=vl,
                speed_upper95_km_s=vu, **mass_fields), dict(
        mass_status=mass_status,
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
        fit_time=float(h.attrs.get('fit_unix_time',(folder/'fit.h5').stat().st_mtime))
        timestamp_source='recorded fit completion' if 'fit_unix_time' in h.attrs else 'fit file modification time (legacy)'
    with h5py.File(folder/'input.h5') as h:
        flag_revision=int(h.attrs.get('manual_flag_revision',0))
        excluded_count=int(h.attrs.get('manually_excluded_count',0))
    fields['modeled_path_m']=apparent_paths(q,data)
    fields['time_ns']=data['time_ns']
    t=np.linspace(data['time_s'].min(),data['time_s'].max(),200)
    fields['speed_time_s']=t
    fields['speed_km_s']=(np.linalg.norm(data['center'][3:]+1000*q[3:6])-10**q[6]*np.exp(10**q[7]*t))/1000
    bands, diagnostics = posterior_bands(folder, data, t)
    fields.update(bands)
    if 'initial_mass_kg_quantiles' in bands:
        try:
            r,m,_=initial_dynamic_mass(q[None,:],data)
            diagnostics.update(best_initial_mass_kg=float(m[0]),best_initial_radius_m=float(r[0]))
        except ValueError:
            pass
    with h5py.File(folder/'plot.h5.partial','w') as h:
        for k,v in fields.items(): h[k]=v
        h.attrs['beat_guard_pass']=passed
        h.attrs['event_id']=folder.name
        h.attrs['source_program']=__file__
        h.attrs.update(diagnostics)
        h.attrs.update(fit_unix_time=fit_time, fit_timestamp_source=timestamp_source,
                       fit_flag_revision=flag_revision, fit_excluded_count=excluded_count)
    os.replace(folder/'plot.h5.partial', folder/'plot.h5')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args()
    root=Path(args.root)
    folders=[root] if (root/'fit.h5').exists() else sorted(x.parent for x in root.glob('*/fit.h5'))
    for folder in folders:
        export(folder)
    print(f'Exported {len(folders)} plot bundles')
