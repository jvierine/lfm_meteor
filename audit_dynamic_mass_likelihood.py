"""Audit physical-model adequacy without changing catalogue masses or intervals."""
import argparse
import ctypes as ct
import hashlib
from pathlib import Path
import shutil
import subprocess
import time

import h5py
from mpi4py import MPI
import numpy as np
from scipy.signal import lfilter
from scipy.stats import chi2

from fit_cached_shrinking_voltage import native, Cached, covariance
from fit_shrinking_voltage import Model, PTR, save
from refit_voltage_reviewed import ROW_KEYS


def load(folder):
    with h5py.File(folder/'input.h5') as h:
        data={k:h[k][()] for k in h};attrs=dict(h.attrs)
    with h5py.File(folder/'fit.h5') as h:q=np.ascontiguousarray(h['params'][:])
    return data,attrs,q


def setup(library,folder,data,q):
    lib=native(library);model=Model(lib,folder,data)
    cache=Cached(model,q,0,1.,folder/'audit_cache.h5')
    return lib,model,cache


def waveform_stats(data,observables,paths,beats):
    fs=data['fs'];rate=data['bandwidth']/data['duration'];out=[]
    for i,(path,beat) in enumerate(zip(paths,beats)):
        tau=data['sample_time_s'][i]-path/299792458.
        fd=beat-rate*path/299792458.
        envelope=np.clip((tau+.5/fs)*fs,0,1)*np.clip((data['duration']-tau+.5/fs)*fs,0,1)
        template=envelope*np.exp(2j*np.pi*(data['bandwidth']/2*tau-rate/2*tau*tau+fd*data['sample_time_s'][i]))
        filt=data['whitener_real'][i]+1j*data['whitener_imag'][i]
        white=lfilter(filt,[1.],data['real'][i]+1j*data['imag'][i])
        unit=lfilter(filt,[1.],template);valid=np.arange(len(unit))>=len(filt)-1
        amplitude=np.vdot(unit[valid],white[valid])/np.vdot(unit[valid],unit[valid])
        error=abs(white-amplitude*unit)**2/data['noise_power'][i]
        guard=(len(filt)+10)/fs
        off=valid&((tau < -guard)|(tau>data['duration']+guard))
        edge=valid&(((tau>=-guard)&(tau<=guard))|((tau>=data['duration']-guard)&(tau<=data['duration']+guard)))
        interior=valid&(tau>guard)&(tau<data['duration']-guard)
        power=abs(white)**2/data['noise_power'][i]
        def total(mask,values):return [float(values[mask].sum()),int(mask.sum())]
        out.append([*total(valid,error),*total(off,power),*total(edge,error),
                    *total(interior,error),float(np.median(power[off])/np.log(2)) if off.any() else np.nan])
    return np.asarray(out)


def pulse_audit(source,output,library):
    data,attrs,q=load(source);output.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'input.h5',output/'input.h5')
    lib,model,cache=setup(library,output,data,q)
    n=len(data['time_s'])
    lib.voltage_observables.argtypes=[ct.c_void_p,PTR,PTR];lib.voltage_observables.restype=ct.c_int
    obs=np.empty((n,2));assert lib.voltage_observables(model.handle,q.ctypes.data_as(PTR),obs.ctypes.data_as(PTR))
    lib.voltage_cache_pulse_audit.argtypes=[ct.c_void_p,ct.c_void_p,PTR,PTR];lib.voltage_cache_pulse_audit.restype=ct.c_int
    pulse=np.empty((n,8));assert lib.voltage_cache_pulse_audit(cache.handle,model.handle,q.ctypes.data_as(PTR),pulse.ctypes.data_as(PTR))
    original_beats=obs[:,1]+data['bandwidth']/data['duration']/299792458.*obs[:,0]
    original=waveform_stats(data,obs,obs[:,0],original_beats)
    independent=waveform_stats(data,obs,pulse[:,0],pulse[:,1])
    lib.voltage_cache_terms.argtypes=[ct.c_void_p,ct.c_void_p,PTR,PTR];lib.voltage_cache_terms.restype=ct.c_int
    def terms(q):
        q=np.ascontiguousarray(q);values=np.empty(n)
        if not lib.voltage_cache_terms(cache.handle,model.handle,q.ctypes.data_as(PTR),values.ctypes.data_as(PTR)):
            raise ValueError('Invalid diagnostic derivative')
        return values
    metric=covariance(cache,q);factor=np.linalg.cholesky(metric);base=terms(q)
    scores=[];hessians=[]
    for step in [.1,.05]:
        plus=[terms(q+step*factor[:,j]) for j in range(7)]
        minus=[terms(q-step*factor[:,j]) for j in range(7)]
        scores.append(np.array([(a-b)/(2*step) for a,b in zip(plus,minus)]).T)
        hess=np.empty((7,7))
        for j in range(7):
            hess[j,j]=-np.sum(plus[j]-2*base+minus[j])/step**2
            for k in range(j):
                pp=terms(q+step*(factor[:,j]+factor[:,k]))
                pm=terms(q+step*(factor[:,j]-factor[:,k]))
                mp=terms(q+step*(-factor[:,j]+factor[:,k]))
                mm=terms(q-step*(factor[:,j]+factor[:,k]))
                hess[j,k]=hess[k,j]=-np.sum(pp-pm-mp+mm)/(4*step**2)
        hessians.append(hess)
    hessian=hessians[-1];eigen=np.linalg.eigvalsh(hessian)
    if eigen.min()>0:
        bread=np.linalg.inv(hessian)
        score=scores[-1]-scores[-1].mean(axis=0)
        robust=factor@bread@(score.T@score)@bread@factor.T
        conditional=factor@bread@factor.T
    else:
        robust=conditional=np.full((7,7),np.nan)
    gains=2*pulse[:,2];lack_df=2*n-7
    full_chi2=2*original[:,0].sum();free_chi2=2*independent[:,0].sum()
    np.testing.assert_allclose(full_chi2,np.sum(model.residual(q)**2),rtol=1e-7)
    full_df=2*original[:,1].sum()-2*n-7;free_df=2*independent[:,1].sum()-4*n
    result=dict(time_s=data['time_s'],time_ns=data['time_ns'],station=data['station'],
                trajectory_observables=obs,independent_pulse_fit=pulse,
                trajectory_waveform_statistics=original,independent_waveform_statistics=independent,
                pulse_scores=scores[-1],observed_information=hessian,
                information_eigenvalues=eigen,conditional_covariance=conditional,
                independent_pulse_sandwich_covariance=robust,params=q,
                curvature_step_comparison=np.array(hessians))
    meta=dict(event=source.name,diagnostic_only=True,production_products_unchanged=True,
              full_reduced_chi2=full_chi2/full_df,independent_pulse_reduced_chi2=free_chi2/free_df,
              trajectory_lack_of_fit_deviance=float(gains.sum()),trajectory_lack_of_fit_df=lack_df,
              trajectory_lack_of_fit_p=float(chi2.sf(gains.sum(),lack_df)),
              waveform_chi2=full_chi2,waveform_df=full_df,
              pulse_optimizer_successes=int(np.sum(pulse[:,6])),pulse_count=n,
              standard_error_logr_conditional=float(np.sqrt(conditional[6,6])),
              standard_error_logr_sandwich=float(np.sqrt(robust[6,6])),
              score_sum_norm=float(np.linalg.norm(scores[-1].sum(axis=0))),
              source_input_sha256=hashlib.sha256((source/'input.h5').read_bytes()).hexdigest(),
              source_fit_sha256=hashlib.sha256((source/'fit.h5').read_bytes()).hexdigest(),
              source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
              waveform_statistics_columns='residual energy,valid count,off-echo energy,off count,edge residual energy,edge count,interior residual energy,interior count,off-echo median calibrated power')
    save(output/'pulse_audit.h5',result,meta)
    print('PULSE',source.name,meta,flush=True)
    cache.close();lib.voltage_free(model.handle)


def subset(source,output,library,kind):
    data,attrs,q=load(source)
    t=data['time_s'];station=data['station'];keep=np.ones(len(t),bool)
    if kind.startswith('omit_chunk_'):
        j=int(kind.rsplit('_',1)[1]);bins=np.minimum(5,np.floor((t-t.min())/(np.ptp(t)+1e-12)*6).astype(int));keep=bins!=j
    elif kind.startswith('omit_station_'):keep=station!=int(kind[-1])
    elif kind in ('early','late'):
        keep=t<=np.median(t) if kind=='early' else t>=np.median(t)
    elif kind in ('even','odd'):
        for s in np.unique(station):
            indices=np.flatnonzero(station==s);indices=indices[np.argsort(t[indices])]
            keep[indices[1 if kind=='even' else 0::2]]=False
    output.mkdir(parents=True,exist_ok=True)
    for key in ROW_KEYS:data[key]=data[key][keep]
    save(output/'input.h5',data,attrs)
    lib,model,cache=setup(library,output,data,q)
    metric=np.ascontiguousarray(covariance(cache,q))
    lib.voltage_importance_optimize.argtypes=[ct.c_void_p,ct.c_void_p,PTR,PTR,ct.c_double,PTR];lib.voltage_importance_optimize.restype=ct.c_int
    def optimize(start):
        start=np.ascontiguousarray(start);result=np.empty(7)
        ok=lib.voltage_importance_optimize(cache.handle,model.handle,start.ctypes.data_as(PTR),
                  metric.ctypes.data_as(PTR),4.,result.ctypes.data_as(PTR))
        if not ok:raise RuntimeError('Subset optimization failed')
        return result
    fit=optimize(q);repeat=optimize(fit);third=optimize(repeat)
    repeat_gain=abs(cache.logp(third)-cache.logp(repeat))
    fit=max([fit,repeat,third],key=cache.logp)
    # Local errors are diagnostic Fisher scales, not new published intervals.
    metric=covariance(cache,fit)
    save(output/'fit.h5',dict(params=fit,keep=keep,fisher_covariance=metric),
         dict(event=source.name,subset=kind,mass_kg=4*np.pi/3*3000*10**(3*fit[6]),
              log_likelihood_gain=cache.logp(fit)-cache.logp(q),repeat_gain=repeat_gain,
              diagnostic_only=True,production_products_unchanged=True))
    print('SUBSET',source.name,kind,'mass',4*np.pi/3*3000*10**(3*fit[6]),flush=True)
    cache.close();lib.voltage_free(model.handle)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--events',nargs='+',required=True)
    a=p.parse_args();comm=MPI.COMM_WORLD
    kinds=['pulse','full']+[f'omit_chunk_{j}' for j in range(6)]+[f'omit_station_{s}' for s in range(3)]+['early','late','even','odd']
    jobs=[(e,k) for e in a.events for k in kinds]
    failures=[]
    for event,kind in jobs[comm.rank::comm.size]:
        try:
            source=a.source/event;output=a.output/event/kind
            if kind=='pulse':pulse_audit(source,output,a.library)
            else:subset(source,output,a.library,kind)
        except Exception as exc:
            import traceback
            traceback.print_exc();failures.append((event,kind,repr(exc)))
    all_failures=comm.gather(failures,root=0)
    if comm.rank==0:
        failures=sum(all_failures,[])
        save(a.output/'status.h5',dict(failures=np.asarray([str(x) for x in failures],dtype='S')),
             dict(completed_unix=time.time(),jobs=len(jobs),failures_count=len(failures),diagnostic_only=True))


if __name__=='__main__':main()
