"""Project a multiplier-calibrated joint likelihood confidence region onto mass.

Not a marginal posterior or an equal-tail Bayesian credible interval. The joint
seven-parameter region is calibrated with whole-pulse weights; its mass projection
is conservative. Fixed-radius profiles still refit all six position/velocity terms.
"""
import argparse
import ctypes as ct
import hashlib
from pathlib import Path
import subprocess
import time

import h5py
from mpi4py import MPI
import numpy as np

from fit_cached_shrinking_voltage import native, Cached, covariance
from fit_shrinking_voltage import Model, PTR, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','posterior','output','library']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--events',nargs='+',required=True)
    p.add_argument('--bounds',action='store_true')
    a=p.parse_args()
    world=MPI.COMM_WORLD
    results=[]
    for event in a.events[world.rank::world.size]:
        began=time.monotonic();source=a.source/event;output=a.output/event
        output.mkdir(parents=True,exist_ok=True)
        with h5py.File(source/'input.h5') as h:data={k:h[k][()] for k in h}
        with h5py.File(source/'fit.h5') as h:seed=np.ascontiguousarray(h['params'][:])
        with h5py.File(a.posterior/event/'central_fit.h5') as h:
            center=np.ascontiguousarray(h['params'][:]);metric=np.ascontiguousarray(h['covariance'][:])
        with h5py.File(a.posterior/event/'posterior.h5') as h:
            draws=len(h['samples']);numerical=bool(h.attrs['numerical_checks_passed'])
        # Reconstruct the same cache offset used for the stored weighted maxima.
        # Real runs seed from the original parametric posterior, simulations from fit.h5.
        original=a.source.parent/'posterior'/event/'posterior.h5'
        if original.exists():
            with h5py.File(original) as h:seed=np.ascontiguousarray(h['proposal_center'][:])
        lib=native(a.library);model=Model(lib,source,data)
        cache=Cached(model,seed,0,1.,output/'cache.h5')
        lib.voltage_cache_terms.argtypes=[ct.c_void_p,ct.c_void_p,PTR,PTR]
        lib.voltage_cache_terms.restype=ct.c_int
        terms=np.empty(len(data['time_s']))
        assert lib.voltage_cache_terms(cache.handle,model.handle,center.ctypes.data_as(PTR),terms.ctypes.data_as(PTR))
        gains=[]
        for draw in range(draws):
            with h5py.File(a.posterior/event/f'draw_{draw:04d}.h5') as h:
                gains.append(2*(h['diagnostics'][0]-np.dot(h['weights'][:],terms)))
        gains=np.asarray(gains)
        if gains.min() < -.01:
            raise ValueError('Weighted optimum is worse than central fit; cannot calibrate')
        cutoff=float(np.quantile(np.maximum(gains,0),.95,method='weibull'))
        base=cache.logp(center)
        lib.voltage_profile_radius_fit.argtypes=[ct.c_void_p,ct.c_void_p,PTR,PTR,ct.c_double,ct.c_double,PTR,PTR]
        lib.voltage_profile_radius_fit.restype=ct.c_int
        profile=[]
        solutions=[(float(center[6]),center.copy())]
        def at(radius):
            # Continue from nearby solved profiles, with the original optimum
            # as an independent start. This avoids numerical branch hopping.
            nearby=sorted(solutions,key=lambda row:abs(row[0]-radius))[:2]
            candidates=[]
            for _,start in nearby+[(float(center[6]),center)]:
                local_metric=np.ascontiguousarray(covariance(cache,start))
                trial=np.empty(7);diagnostic=np.empty(2)
                ok=lib.voltage_profile_radius_fit(cache.handle,model.handle,
                    np.ascontiguousarray(start).ctypes.data_as(PTR),local_metric.ctypes.data_as(PTR),
                    float(radius),.5,trial.ctypes.data_as(PTR),diagnostic.ctypes.data_as(PTR))
                if ok:candidates.append((float(diagnostic[0]),trial,diagnostic.copy()))
            ok=bool(candidates)
            if ok:_,params,diagnostic=max(candidates,key=lambda row:row[0])
            if not ok:
                raise ValueError('Conditional profile failed; cannot treat optimizer failure as exclusion')
            else:
                for _ in range(12):
                    if diagnostic[1]<.01:break
                    next_params=np.empty(7)
                    local_metric=np.ascontiguousarray(covariance(cache,params))
                    lib.voltage_profile_radius_fit(cache.handle,model.handle,
                        np.ascontiguousarray(params).ctypes.data_as(PTR),local_metric.ctypes.data_as(PTR),
                        float(radius),.5,next_params.ctypes.data_as(PTR),diagnostic.ctypes.data_as(PTR))
                    params=next_params
                if diagnostic[1]>=.01:
                    save(output/'profile_failure.h5',dict(params=params,diagnostics=diagnostic,
                        previous_profile=np.asarray(profile)),dict(log_radius=radius,event=event))
                    raise ValueError('Unresolved conditional profile optimization')
                value=2*(base-diagnostic[0])
                if value < -.01:
                    raise ValueError('Profile improves central fit; central optimum must be refit')
            profile.append((radius,value))
            solutions.append((float(radius),params.copy()))
            return value-cutoff
        fields=dict(bootstrap_joint_likelihood_ratio=gains,best_params=center)
        attrs=dict(uncertainty_method='Whole-pulse multiplier-calibrated joint likelihood confidence region',
            Bayesian_credible_interval=False,joint_parameter_count=7,
            nominal_joint_confidence=.95,mass_projection_conservative=True,
            cutoff=cutoff,bootstrap_numerical_checks_passed=numerical,
            physical_systematic_errors_included=False,
            source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
            source_input_sha256=hashlib.sha256((source/'input.h5').read_bytes()).hexdigest())
        attrs.update(density_kg_m3=3000.,Gamma=1.,sigma_s2_m2=1e-8,
            confidence_region='2*(maximum log likelihood - trial log likelihood) <= calibrated cutoff')
        if (source/'truth.h5').exists():
            with h5py.File(source/'truth.h5') as h:truth=h['params'][:]
            attrs['mass_projection_covered']=bool(at(truth[6])<=0)
            attrs['joint_truth_likelihood_ratio']=2*(base-cache.logp(truth))
            attrs['joint_truth_covered']=bool(attrs['joint_truth_likelihood_ratio']<=cutoff)
        if a.bounds:
            endpoints=[];limited=[]
            for direction,boundary in [(-1.,-7.),(1.,-2.)]:
                # A feasible point at the largest tested radius already proves
                # that no upper bound has been resolved within this domain.
                if direction>0 and at(boundary)<=0:
                    endpoints.append(boundary);limited.append(True);continue
                inside=center[6];outside=None
                for step in [.01,.03,.1,.3,.6,1.,2.,5.]:
                    trial=float(np.clip(center[6]+direction*step,-7.,-2.))
                    if at(trial)>0:outside=trial;break
                    inside=trial
                    if trial==boundary:break
                limited.append(outside is None)
                if outside is None:endpoints.append(boundary);continue
                for _ in range(16):
                    mid=(inside+outside)/2
                    if at(mid)>0:outside=mid
                    else:inside=mid
                    if abs(outside-inside)<.0005:break
                endpoints.append((inside+outside)/2)
            fields['log_radius_profile_endpoints']=np.array(endpoints)
            fields['mass_profile_endpoints_kg']=4*np.pi/3*3000*10**(3*np.array(endpoints))
            reported=fields['mass_profile_endpoints_kg'].copy()
            if limited[0]:reported[0]=0.
            if limited[1]:reported[1]=np.inf
            fields['mass_confidence_interval_kg']=reported
            attrs.update(lower_endpoint_boundary_limited=limited[0],upper_endpoint_boundary_limited=limited[1])
        fields['profile_radius_and_likelihood_ratio']=np.asarray(profile)
        attrs['elapsed_seconds']=time.monotonic()-began
        save(output/'confidence.h5',fields,attrs)
        results.append((event,attrs))
        print('REGION',event,attrs,fields.get('mass_profile_endpoints_kg',''),flush=True)
        cache.close();lib.voltage_free(model.handle)
    rows=world.gather(results,root=0)
    if world.rank==0:
        rows=[x for group in rows for x in group]
        if rows and 'mass_projection_covered' in rows[0][1]:
            print('REGION COVERAGE mass',sum(x[1]['mass_projection_covered'] for x in rows),
                'joint',sum(x[1]['joint_truth_covered'] for x in rows),'of',len(rows),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
