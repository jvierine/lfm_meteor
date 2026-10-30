"""Direct Rust fits with station-stratified random measurement deletion.

Percentile ranges are subset robustness ranges, not calibrated confidence
intervals or posterior draws. Flagged measurements never enter any fit.
"""
import argparse
import ctypes as ct
import os
from pathlib import Path
import shutil
import time
import h5py
import numpy as np
from mpi4py import MPI


def deletion_weights(station, fraction, seed):
    rng=np.random.default_rng(seed)
    weights=np.ones(len(station))
    for s in np.unique(station):
        indices=np.flatnonzero(station==s)
        count=min(len(indices)-3, int(round(fraction*len(indices))))
        if count<1:raise ValueError('Insufficient station measurements for deletion')
        weights[rng.choice(indices,count,replace=False)]=0.
    return weights


def main():
    from fit_cached_shrinking_voltage import native, Cached, covariance, validate
    from fit_shrinking_voltage import Model, prepare, save, export, full_products, PTR
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','initial','flags','seed','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--library',required=True);p.add_argument('--event',required=True)
    p.add_argument('--replicates',type=int,default=256)
    p.add_argument('--budget',type=float,default=300.)
    args=p.parse_args();comm=MPI.COMM_WORLD;rank=comm.rank
    folder=args.output/args.event;folder.mkdir(parents=True,exist_ok=True)
    if rank==0:
        d,initial=prepare(args,args.event,folder)
        with h5py.File(args.seed/args.event/'fit.h5') as h:q=h['params'][:]
        with h5py.File(args.seed/args.event/'input.h5') as h:old_center=h['center'][:]
        q[:6]+=(old_center-d['center'])/1000
        with h5py.File(folder/'input.h5') as h:
            assert not set(h.attrs.get('manually_excluded_ids',[])) & {f'{s}:{t}' for s,t in zip(d['station'],d['time_ns'])}
        print('FLAGS',dict(revision=int(h5py.File(folder/'input.h5').attrs['manual_flag_revision']),retained=len(d['station'])),flush=True)
    else:q=None
    q=comm.bcast(q,root=0);comm.Barrier()
    with h5py.File(folder/'input.h5') as h:d={k:h[k][()] for k in h}
    lib=native(args.library);model=Model(lib,folder,d)
    work=folder/f'worker_{rank:03d}';work.mkdir(exist_ok=True)
    cache=Cached(model,q,0,1.,work/'likelihood_cache.h5')
    lib.voltage_cache_weighted_fit.argtypes=[ct.c_void_p,ct.c_void_p,PTR,PTR,PTR,PTR,ct.c_double]
    lib.voltage_cache_weighted_fit.restype=ct.c_int
    matrix=np.ascontiguousarray(np.linalg.cholesky(covariance(cache,q)))
    def fit(start,weights,seconds=45.):
        start=np.ascontiguousarray(start);weights=np.ascontiguousarray(weights);result=np.empty(7)
        ok=lib.voltage_cache_weighted_fit(cache.handle,model.handle,start.ctypes.data_as(PTR),matrix.ctypes.data_as(PTR),weights.ctypes.data_as(PTR),result.ctypes.data_as(PTR),seconds)
        return result,bool(ok)
    if rank==0:
        q,ok=fit(q,np.ones(len(d['station'])),90.)
        if not ok:raise RuntimeError('Central Rust fit did not converge')
        refined,ok=fit(q,np.ones(len(d['station'])),45.)
        gain=cache.logp(refined)-cache.logp(q)
        if not ok or gain>.01:raise RuntimeError('Central fit not stable on repeat optimization')
        q=refined
        if not validate(cache,q,covariance(cache,q),folder):raise RuntimeError('Raw likelihood parity check failed')
        meta=dict(fit_unix_time=time.time(),optimizer_success=True,model='direct shrinking radius; Rust joint voltage sufficient statistics',
                  cost=-cache.logp(q),material_density_kg_m3=3000.,Gamma=1.,Cd=2.,sigma_s2_m2=1e-8,
                  prior='Uniform x0,v0,log10 radius; radius 1e-7 to 1e-2 m')
        save(folder/'fit.h5',dict(params=q),meta)
        print('CENTRAL',q,'repeat_gain',gain,flush=True)
    q=comm.bcast(q,root=0);comm.Barrier()
    started=time.monotonic();jobs=[(f,j) for f in (.2,.35,.5) for j in range(args.replicates)];rows=[]
    for index in range(rank,len(jobs),comm.size):
        fraction,j=jobs[index];weights=deletion_weights(d['station'],fraction,310000+index)
        remaining=args.budget-(time.monotonic()-started)
        if remaining<=5:raise RuntimeError('Random deletion uncertainty budget exhausted')
        draw,ok=fit(q,weights,min(45.,remaining-2))
        if not ok and args.budget-(time.monotonic()-started)>10:
            draw,ok=fit(draw,weights,min(30.,args.budget-(time.monotonic()-started)-2))
        rows.append((index,fraction,ok,draw,weights))
        print('DRAW',index,ok,10**draw[6],flush=True)
    gathered=comm.gather(rows,root=0)
    if rank==0:
        rows=sorted([r for group in gathered for r in group],key=lambda r:r[0])
        draws=np.array([r[3] for r in rows]);fractions=np.array([r[1] for r in rows]);success=np.array([r[2] for r in rows])
        boundary=(draws[:,6]<=-6.99)|(draws[:,6]>=-2.01)
        save(folder/'random_deletion.h5',dict(params=draws,drop_fraction=fractions,optimizer_converged=success,
            retained_mask=np.array([r[4]>0 for r in rows]),prior_boundary=boundary),
            dict(method='station-stratified random deletion without replacement',replicates_per_fraction=args.replicates,
                 interval_interpretation='Unscaled subset percentile robustness range; not calibrated population confidence',elapsed_seconds=time.monotonic()-started))
        if not success.all():raise RuntimeError('Failed subset fits retained for audit; no silently filtered ranges published')
        export(folder,model,q,meta);full_products(args,args.event,folder,model,q)
        from voltage_path_uncertainty import path_intervals
        with h5py.File(folder/'plot.h5') as h:fields={k:h[k][:] for k in h};attrs=dict(h.attrs)
        intervals=path_intervals(d,model.prediction(q),fields['seed_apparent_path_m'])
        save(folder/'path_measurement_uncertainty.h5',dict(time_ns=d['time_ns'],station=d['station'],path_measurement_lower95_m=intervals[:,0],path_measurement_upper95_m=intervals[:,1]),dict(confidence=.95,conditional=True))
        # The predeclared primary fraction is 35%; the other fractions test sensitivity.
        chosen=draws[np.isclose(fractions,.35)];radius=10**chosen[:,6];mass=4*np.pi/3*3000*radius**3
        paths=[];speeds=[];order=np.argsort(d['time_s'])
        for draw in chosen:
            state=model.states(draw);paths.append(model.paths(draw,state));speeds.append(state[order,1]/1000)
        fields['initial_radius_m_quantiles']=np.quantile(radius,[.025,.5,.975])
        fields['initial_mass_kg_quantiles']=np.quantile(mass,[.025,.5,.975])
        fields['path_lower95_m'],fields['path_upper95_m']=np.quantile(paths,[.025,.975],axis=0)
        fields['speed_lower95_km_s'],fields['speed_upper95_km_s']=np.quantile(speeds,[.025,.975],axis=0)
        method='35% random-deletion subset robustness range (not calibrated confidence interval)'
        attrs.update(uncertainty_method=method,uncertainty_status='95% subset range; all subset optimizers converged',
            uncertainty_converged=True,uncertainty_alpha=.5,uncertainty_updated_unix=time.time(),
            mass_status='Direct shrinking-radius mass; 95% random-deletion robustness range; density 3000 kg/m3, Gamma=1, sigma=1e-8 fixed',
            uncertainty_prior_boundary=bool(boundary.any()),uncertainty_calibrated=False)
        save(folder/'dynamic_mass_posterior.h5',dict(radius_m=radius,mass_kg=mass),dict(uncertainty_method=method,is_mcmc_posterior=False,uncertainty_calibrated=False))
        save(folder/'plot.h5',fields,attrs);save(folder/'complete.h5',{},attrs)
        for f in (.2,.35,.5):
            m=4*np.pi/3*3000*10**(3*draws[np.isclose(fractions,f),6])
            print('INTERVAL',f,np.quantile(m,[.025,.5,.975]),flush=True)
        print('COMPLETE',folder,flush=True)
    cache.close();lib.voltage_free(model.handle)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc();MPI.COMM_WORLD.Abort(1)
