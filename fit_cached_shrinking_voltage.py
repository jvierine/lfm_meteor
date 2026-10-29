"""Rust cached joint delay/beat likelihood with bounded MH uncertainty, MPI events."""
import argparse
import ctypes as ct
import os
import time
import traceback
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from mpi4py import MPI
from fit_shrinking_voltage import Model, prepare, save, export, full_products, PTR
from run_native_voltage_catalog import load_library


def native(path):
    lib=load_library(path)
    lib.voltage_states.argtypes=[ct.c_void_p,PTR,PTR];lib.voltage_states.restype=ct.c_int
    lib.voltage_logp.argtypes=[ct.c_void_p,PTR];lib.voltage_logp.restype=ct.c_double
    lib.voltage_cache_build.argtypes=[ct.c_void_p,PTR,ct.c_size_t,ct.c_double,ct.c_char_p];lib.voltage_cache_build.restype=ct.c_void_p
    lib.voltage_cache_free.argtypes=[ct.c_void_p]
    lib.voltage_cache_logp.argtypes=[ct.c_void_p,ct.c_void_p,PTR,ct.c_int];lib.voltage_cache_logp.restype=ct.c_double
    lib.voltage_cache_margin.argtypes=[ct.c_void_p,ct.c_void_p,PTR];lib.voltage_cache_margin.restype=ct.c_double
    lib.voltage_cache_sample.argtypes=[ct.c_void_p,ct.c_void_p,PTR,PTR,ct.c_size_t,ct.c_size_t,ct.c_double,ct.c_uint64,ct.c_char_p]
    lib.voltage_cache_sample.restype=ct.c_int
    return lib


class Cached:
    def __init__(self,model,q,n,expand,path):
        self.model=model
        q=np.ascontiguousarray(q,dtype=float)
        self.handle=model.lib.voltage_cache_build(model.handle,q.ctypes.data_as(PTR),n,expand,os.fsencode(path))
        if not self.handle:raise RuntimeError('Cache construction failed; no posterior published')
    def logp(self,q,exact=False):
        q=np.ascontiguousarray(q,dtype=float)
        return self.model.lib.voltage_cache_logp(self.handle,self.model.handle,q.ctypes.data_as(PTR),int(exact))
    def close(self):
        self.model.lib.voltage_cache_free(self.handle)
    def margin(self,q):
        q=np.ascontiguousarray(q,dtype=float)
        return self.model.lib.voltage_cache_margin(self.handle,self.model.handle,q.ctypes.data_as(PTR))


def covariance(cache,q):
    # Profile-residual Fisher metric avoids subtracting large log likelihoods.
    # Raw voltages are evaluated only here, never inside the MCMC loop.
    step=np.r_[np.full(3,1e-5),np.full(3,1e-4),1e-4]
    upper=np.r_[np.full(3,20.),np.full(3,40.),-2.]
    lower=np.r_[np.full(3,-20.),np.full(3,-40.),-7.]
    step=np.where(q+2*step>upper,-step,step)
    baseline=cache.model.residual(q)
    for attempt in range(5):
        eye=np.diag(step);columns=[]
        for i in range(7):
            plus=cache.model.residual(q+eye[i])
            minus=cache.model.residual(q-eye[i]) if np.all(q-eye[i]>=lower) and np.all(q-eye[i]<=upper) else None
            if np.max(abs(plus))>=1e9:break
            columns.append((plus-minus)/2 if minus is not None and np.max(abs(minus))<1e9 else plus-baseline)
        if len(columns)==7:
            jac=np.asarray(columns);h=jac@jac.T
            if np.all(np.isfinite(h)):break
        step*=.2
    else:raise RuntimeError('Invalid physical derivative stencil')
    w,u=np.linalg.eigh(h)
    # Regularization affects proposal only, never likelihood or prior.
    inv=(u/np.maximum(w,max(np.max(w)*1e-12,1e-12)))@u.T
    cov=step[:,None]*inv*step[None,:]
    limit=np.r_[np.full(3,1.),np.full(3,10.),1.]
    scale=min(1.,float(np.min(limit/np.sqrt(np.diag(cov)))))
    return cov*scale**2


def optimize(cache,q):
    transform=np.linalg.cholesky(covariance(cache,q))
    def objective(z):
        lp=cache.logp(q+transform@z)
        return -lp if np.isfinite(lp) else 1e30
    fit=minimize(objective,np.zeros(7),method='Powell',options=dict(maxiter=100,xtol=1e-5,ftol=1e-9))
    return q+transform@fit.x,fit


def validate(cache,q,cov,folder):
    rng=np.random.default_rng(7021);points=[q]
    for _ in range(500):
        p=q+rng.multivariate_normal(np.zeros(7),cov)
        if np.isfinite(cache.logp(p)):points.append(p)
        if len(points)>=49:break
    if len(points)<16:raise RuntimeError('Insufficient interior validation probes; cache too narrow')
    exact=np.array([cache.logp(p,True) for p in points]);approx=np.array([cache.logp(p) for p in points])
    error=(approx-approx[0])-(exact-exact[0])
    # Independent comparison with the original residual-vector likelihood,
    # including the same complex-amplitude marginalization normalization.
    raw=[]
    for p in points[:3]:
        p=np.ascontiguousarray(p)
        raw.append(cache.model.lib.voltage_logp(cache.model.handle,p.ctypes.data_as(PTR)))
    parity=np.asarray(raw)-raw[0]-(exact[:3]-exact[0])
    passed=np.max(abs(error))<=.2 and np.max(abs(parity))<1e-5
    save(folder/'cache_validation.h5',dict(params=points,exact_logp=exact,cached_logp=approx,delta_error=error,raw_profile_logp=raw),
         dict(passed=passed,max_delta_logp_error=float(np.max(abs(error))),raw_parity_error=float(np.max(abs(parity))),tolerance=.2))
    return passed


def run(args,event):
    folder=args.output/event;folder.mkdir(parents=True,exist_ok=True)
    previous_seconds=0.
    if (folder/'complete.h5').exists():
        with h5py.File(folder/'complete.h5') as h:
            if bool(h.attrs.get('uncertainty_converged',False)):return
            previous_seconds=float(h.attrs.get('uncertainty_total_seconds',h.attrs.get('uncertainty_elapsed_seconds',0.)))
        completed_at=(folder/'complete.h5').stat().st_mtime
        for path in folder.glob('chain_*.h5'):
            if path.stat().st_mtime>completed_at:
                with h5py.File(path) as h:previous_seconds+=float(h['elapsed_seconds'][()]) if 'elapsed_seconds' in h else 0.
    elif list(folder.glob('chain_*.h5')):
        for path in folder.glob('chain_*.h5'):
            with h5py.File(path) as h:previous_seconds+=float(h['elapsed_seconds'][()]) if 'elapsed_seconds' in h else 0.
    if (folder/'uncertainty_budget.h5').exists():
        with h5py.File(folder/'uncertainty_budget.h5') as h:previous_seconds=max(previous_seconds,float(h.attrs['used_seconds']))
    historical_seconds=previous_seconds
    if args.budget_generation:
        same=False
        if (folder/'uncertainty_budget.h5').exists():
            with h5py.File(folder/'uncertainty_budget.h5') as h:same=h.attrs.get('generation','')==args.budget_generation
        if not same:previous_seconds=0.
    if previous_seconds>=args.budget-12:raise RuntimeError('Five-minute cumulative uncertainty budget exhausted; preserve existing draws')
    if (folder/'input.h5').exists():
        archive=folder/'attempts'/str(time.time_ns());archive.mkdir(parents=True)
        for path in folder.glob('*.h5'):shutil.copy2(path,archive/path.name)
        if (folder/'error.log').exists():shutil.move(folder/'error.log',archive/'error.log')
    seed=args.seed/event/'fit.h5'
    # Prefer the last physical fit; WJ is used only by prepare() as a seed.
    if (folder/'fit.h5').exists():seed=folder/'fit.h5'
    seeded=None
    if seed.exists():
        with h5py.File(seed) as h:seeded=h['params'][:]
        with h5py.File(seed.parent/'input.h5') as h:old_center=h['center'][:]
    began=time.monotonic();d,initial=prepare(args,event,folder);lib=native(args.library);model=Model(lib,folder,d)
    q=initial.copy()
    if seeded is not None:
        q=seeded
        q[:6]+=(old_center-d['center'])/1000
    def raw_logp(p):
        p=np.ascontiguousarray(p)
        return lib.voltage_logp(model.handle,p.ctypes.data_as(PTR))
    if not np.isfinite(raw_logp(q)):
        candidates=[]
        for center in (q,initial):
            for logr in (initial[6],-5.,-4.,-3.,-2.01):
                candidate=center.copy();candidate[6]=logr
                score=raw_logp(candidate)
                if np.isfinite(score):candidates.append((score,candidate))
        if not candidates:raise RuntimeError('No feasible physical seed for recorded pulse windows')
        q=max(candidates,key=lambda item:item[0])[1]
    cache=None
    for n in (0,):
        if cache is not None:cache.close()
        cache=Cached(model,q,n,args.expand,folder/'likelihood_cache.h5')
        try:
            q,fit=optimize(cache,q)
            cache.close()
            cache=Cached(model,q,0,args.expand,folder/'likelihood_cache.h5')
            cov=covariance(cache,q)
        except RuntimeError as exc:
            if 'Curvature stencil' not in str(exc):raise
            print('RECENTER',event,str(exc),flush=True)
            continue
        print('CACHE',event,n,'elapsed',time.monotonic()-began,flush=True)
        try:
            if validate(cache,q,cov,folder):break
        except RuntimeError as exc:
            if 'validation probes' not in str(exc):raise
            print('RECENTER',event,str(exc),flush=True)
    else:raise RuntimeError('Reduced sufficient-statistic likelihood failed raw parity gate')
    meta=dict(fit_unix_time=time.time(),cost=-cache.logp(q),optimizer_success=bool(fit.success),model='direct shrinking radius; analytic joint voltage sufficient statistics',
              Gamma=1.,Cd=2.,sigma_s2_m2=1e-8,material_density_kg_m3=3000.,prior='Uniform x0,v0,log10 r0; radius 1e-7 to 1e-2 m',
              cache_grid_size=n,uncertainty_budget_seconds=args.budget)
    save(folder/'fit.h5',dict(params=q,initial=initial),meta)
    export(folder,model,q,meta);full_products(args,event,folder,model,q)
    from voltage_path_uncertainty import path_intervals
    with h5py.File(folder/'plot.h5') as h:observed=h['seed_apparent_path_m'][:]
    intervals=path_intervals(d,model.prediction(q),observed)
    save(folder/'path_measurement_uncertainty.h5',dict(time_ns=d['time_ns'],station=d['station'],
         path_measurement_lower95_m=intervals[:,0],path_measurement_upper95_m=intervals[:,1]),dict(confidence=.95,conditional=True))
    print('FIT',event,'radius_m',10**q[6],'elapsed',time.monotonic()-began,flush=True)
    if args.validate_only:return
    # Start in the posterior, not merely somewhere with finite likelihood.
    # The centered Hessian proposal keeps the measured path/rate correlations.
    started=time.monotonic();chains=[];rng=np.random.default_rng(82304)
    for i in range(4):
        remaining=args.budget-previous_seconds-(time.monotonic()-started)-8
        if remaining<=0:raise RuntimeError('Uncertainty budget exhausted before four chains')
        seconds=remaining/(4-i)
        start=q.copy()
        for _ in range(8):
            candidate=q+rng.multivariate_normal(np.zeros(7),cov)*.2
            if cache.logp(candidate)>cache.logp(q)-10.:start=candidate;break
        remaining=args.budget-previous_seconds-(time.monotonic()-started)-8
        if remaining<=0:raise RuntimeError('Uncertainty budget exhausted while initializing chains')
        seconds=remaining/(4-i)
        start=np.ascontiguousarray(start);matrix=np.ascontiguousarray(cov)
        target=folder/f'chain_{i}.h5'
        ok=lib.voltage_cache_sample(cache.handle,model.handle,start.ctypes.data_as(PTR),matrix.ctypes.data_as(PTR),
                                    args.warmup,args.draws,seconds,20262000+i,os.fsencode(str(target)+'.partial'))
        if not ok:raise RuntimeError('Native cached MH failed')
        os.replace(str(target)+'.partial',target)
        with h5py.File(target) as h:chains.append(h['samples'][:])
        save(folder/'uncertainty_budget.h5',{},dict(used_seconds=previous_seconds+time.monotonic()-started,limit_seconds=args.budget,generation=args.budget_generation,previous_attempt_seconds=historical_seconds))
        print('CHAIN',event,i,len(chains[-1]),'elapsed',time.monotonic()-started,flush=True)
    length=min(map(len,chains))
    if length<1000:raise RuntimeError('Budget-limited: fewer than 1000 retained draws per chain')
    chains=np.stack([x[:length] for x in chains]);flat=chains.reshape(-1,7)
    probes=flat[np.linspace(0,len(flat)-1,64,dtype=int)]
    err=np.array([cache.logp(p)-cache.logp(p,True) for p in probes]);err-=cache.logp(q)-cache.logp(q,True)
    margin=min(cache.margin(p) for p in flat[np.linspace(0,len(flat)-1,min(2048,len(flat)),dtype=int)])
    save(folder/'posterior_cache_validation.h5',dict(params=probes,delta_error=err),dict(passed=bool(np.max(abs(err))<=.2 and margin>.1),max_delta_logp_error=float(np.max(abs(err))),minimum_boundary_margin=margin))
    if np.max(abs(err))>.2:raise RuntimeError('Posterior left validated interpolation accuracy region; refine cache')
    if margin<=.1:raise RuntimeError('Posterior approaches cache boundary; expand cache before reporting intervals')
    attrs=export(folder,model,q,meta,chains)
    save(folder/'uncertainty_budget.h5',{},dict(used_seconds=previous_seconds+time.monotonic()-started,limit_seconds=args.budget,generation=args.budget_generation,previous_attempt_seconds=historical_seconds))
    attrs.update(uncertainty_elapsed_seconds=time.monotonic()-started,uncertainty_budget_seconds=args.budget,
                 uncertainty_total_seconds=previous_seconds+time.monotonic()-started,
                 total_elapsed_seconds=time.monotonic()-began,cache_validated=True)
    save(folder/'complete.h5',{},attrs)
    cache.close();lib.voltage_free(model.handle)
    print('DONE',event,attrs,flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ('source','initial','flags','output','seed'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--library',required=True);p.add_argument('--event');p.add_argument('--budget',type=float,default=300)
    p.add_argument('--warmup',type=int,default=4000);p.add_argument('--draws',type=int,default=96000)
    p.add_argument('--expand',type=float,default=1.);p.add_argument('--validate-only',action='store_true')
    p.add_argument('--budget-generation',default='',help='Explicitly authorized fresh bounded attempt; old timings retained')
    args=p.parse_args();comm=MPI.COMM_WORLD
    events=[args.event] if args.event else sorted(f.parent.name for f in args.source.glob('tri_*/input.h5'))
    for event in events[comm.rank::comm.size]:
        try:run(args,event)
        except Exception:
            folder=args.output/event;folder.mkdir(parents=True,exist_ok=True)
            error=traceback.format_exc();(folder/'error.log').write_text(error);print('ERROR',event,error,flush=True)


if __name__=='__main__':main()
