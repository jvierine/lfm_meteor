"""Rust cached joint delay/beat likelihood with bounded MH uncertainty, MPI events."""
import argparse
import ctypes as ct
import os
import time
import traceback
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


def covariance(cache,q):
    # Numerical curvature in scaled coordinates; retain cross-parameter terms.
    step=np.r_[np.full(3,1e-4),np.full(3,1e-3),.003]
    eye=np.diag(step);h=np.empty((7,7));f=cache.logp(q)
    for i in range(7):
        h[i,i]=-(cache.logp(q+eye[i])-2*f+cache.logp(q-eye[i]))
        for j in range(i):
            h[i,j]=h[j,i]=-(cache.logp(q+eye[i]+eye[j])-cache.logp(q+eye[i]-eye[j])-cache.logp(q-eye[i]+eye[j])+cache.logp(q-eye[i]-eye[j]))/4
    if not np.all(np.isfinite(h)):raise RuntimeError('Curvature stencil outside cache; need recentering')
    w,u=np.linalg.eigh(h)
    # Regularization affects proposal only, never likelihood or prior.
    inv=(u/np.maximum(w,max(np.max(w)*1e-8,1e-6)))@u.T
    return step[:,None]*inv*step[None,:]


def optimize(cache,q):
    scales=np.r_[np.full(3,.001),np.full(3,.01),.1]
    def objective(z):
        lp=cache.logp(q+scales*z)
        return -lp if np.isfinite(lp) else 1e30
    fit=minimize(objective,np.zeros(7),method='Powell',options=dict(maxiter=100,xtol=1e-5,ftol=1e-9))
    return q+scales*fit.x,fit


def validate(cache,q,cov,folder):
    rng=np.random.default_rng(7021);points=[q]
    for _ in range(48):
        p=q+rng.multivariate_normal(np.zeros(7),cov)
        if np.isfinite(cache.logp(p)):points.append(p)
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
    if (folder/'complete.h5').exists():return
    began=time.monotonic();d,initial=prepare(args,event,folder);lib=native(args.library);model=Model(lib,folder,d)
    q=initial.copy();seed=args.seed/event/'fit.h5'
    if seed.exists():
        with h5py.File(seed) as h:q=h['params'][:]
        with h5py.File(args.seed/event/'input.h5') as h:old_center=h['center'][:]
        q[:6]+=(old_center-d['center'])/1000
    cache=None
    for n in (65,129,257):
        if cache is not None:cache.close()
        cache=Cached(model,q,n,args.expand,folder/'likelihood_cache.h5')
        q,fit=optimize(cache,q);cov=covariance(cache,q)
        print('CACHE',event,n,'elapsed',time.monotonic()-began,flush=True)
        if validate(cache,q,cov,folder):break
    else:raise RuntimeError('Cached likelihood failed interpolation accuracy gate')
    meta=dict(fit_unix_time=time.time(),cost=-cache.logp(q),optimizer_success=bool(fit.success),model='direct shrinking radius; cached joint voltage likelihood',
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
    started=time.monotonic();chains=[];rng=np.random.default_rng(82304)
    for i in range(4):
        remaining=args.budget-(time.monotonic()-started)-40
        if remaining<=0:raise RuntimeError('Uncertainty budget exhausted before four chains')
        seconds=remaining/(4-i)
        start=q.copy()
        for _ in range(100):
            candidate=q+rng.multivariate_normal(np.zeros(7),cov)*2
            if np.isfinite(cache.logp(candidate)):start=candidate;break
        start=np.ascontiguousarray(start);matrix=np.ascontiguousarray(cov)
        target=folder/f'chain_{i}.h5'
        ok=lib.voltage_cache_sample(cache.handle,model.handle,start.ctypes.data_as(PTR),matrix.ctypes.data_as(PTR),
                                    args.warmup,args.draws,seconds,20262000+i,os.fsencode(str(target)+'.partial'))
        if not ok:raise RuntimeError('Native cached MH failed')
        os.replace(str(target)+'.partial',target)
        with h5py.File(target) as h:chains.append(h['samples'][:])
        print('CHAIN',event,i,len(chains[-1]),'elapsed',time.monotonic()-started,flush=True)
    length=min(map(len,chains))
    if length<1000:raise RuntimeError('Budget-limited: fewer than 1000 retained draws per chain')
    chains=np.stack([x[:length] for x in chains]);flat=chains.reshape(-1,7)
    probes=flat[np.linspace(0,len(flat)-1,64,dtype=int)]
    err=np.array([cache.logp(p)-cache.logp(p,True) for p in probes]);err-=cache.logp(q)-cache.logp(q,True)
    save(folder/'posterior_cache_validation.h5',dict(params=probes,delta_error=err),dict(passed=bool(np.max(abs(err))<=.2),max_delta_logp_error=float(np.max(abs(err)))))
    if np.max(abs(err))>.2:raise RuntimeError('Posterior left validated interpolation accuracy region; refine cache')
    attrs=export(folder,model,q,meta,chains)
    attrs.update(uncertainty_elapsed_seconds=time.monotonic()-started,uncertainty_budget_seconds=args.budget,
                 total_elapsed_seconds=time.monotonic()-began,cache_validated=True)
    save(folder/'complete.h5',{},attrs)
    cache.close();lib.voltage_free(model.handle)
    print('DONE',event,attrs,flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ('source','initial','flags','output','seed'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--library',required=True);p.add_argument('--event');p.add_argument('--budget',type=float,default=300)
    p.add_argument('--warmup',type=int,default=4000);p.add_argument('--draws',type=int,default=24000)
    p.add_argument('--expand',type=float,default=1.);p.add_argument('--validate-only',action='store_true')
    args=p.parse_args();comm=MPI.COMM_WORLD
    events=[args.event] if args.event else sorted(f.parent.name for f in args.source.glob('tri_*/input.h5'))
    for event in events[comm.rank::comm.size]:
        try:run(args,event)
        except Exception:
            folder=args.output/event;folder.mkdir(parents=True,exist_ok=True)
            error=traceback.format_exc();(folder/'error.log').write_text(error);print('ERROR',event,error,flush=True)


if __name__=='__main__':main()
