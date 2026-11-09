"""Frozen full catalogue: native pulse bootstrap plus calibrated likelihood bounds.

All expensive fits use Rust. MPI runs bootstrap batches; independent native
profile subprocesses run concurrently. No old uncertainty or WJ fallback is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import h5py
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flag_state(path):
    if not path.exists():return 0,False,set()
    with h5py.File(path) as h:
        return (int(h.attrs.get('version',0)),bool(h.attrs.get('bad',False)),
                set(h['excluded'].asstr()[:]))


def write(path, fields, attrs):
    temporary=path.with_suffix('.h5.partial')
    with h5py.File(temporary,'w') as h:
        for key,value in fields.items():h[key]=value
        h.attrs.update(**attrs)
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','flags','output','library']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cores',type=int,default=128)
    p.add_argument('--draws',type=int,default=256)
    p.add_argument('--event-seconds',type=float,default=300.)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    code=Path(__file__).resolve().parent;a.output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    events=sorted(p.parent.name for p in (a.source/'inputs').glob('tri_*/input.h5'))
    retained=[];rejected=[]
    snapshot=a.output/'flags_snapshot';snapshot.mkdir(exist_ok=True)
    for event in events:
        live=a.flags/(event+'.h5');previous=a.source/'flags_snapshot'/(event+'.h5')
        if flag_state(live)!=flag_state(previous):
            raise ValueError(f'{event}: flags changed; rebuild masked input rather than reuse stale measurements')
        if live.exists():shutil.copy2(live,snapshot/live.name)
        if flag_state(live)[1]:rejected.append(event);continue
        retained.append(event)
        target=a.output/'inputs'/event;target.mkdir(parents=True,exist_ok=True)
        original=a.source/'inputs'/event/'input.h5'
        if a.resume and (target/'input.h5').exists():
            if digest(target/'input.h5')!=digest(original):raise ValueError('Frozen input changed')
        else:shutil.copy2(original,target/'input.h5')
        # Import a point initializer only, never its old uncertainty.
        with h5py.File(a.source/'posterior'/event/'posterior.h5') as h:q=h['proposal_center'][:]
        if q.shape!=(7,):raise ValueError('Direct shrinking-radius initializer required')
        write(target/'fit.h5',{'params':q},{'initializer_only':True,
            'old_uncertainty_inherited':False,'source_git_commit':commit})
    strings=lambda x:np.asarray(x,dtype=h5py.string_dtype())
    write(a.output/'launch.h5',{'input_events':strings(events),'retained_events':strings(retained),
        'review_rejected_events':strings(rejected)},dict(started_unix=time.time(),cores=a.cores,
        bootstrap_draws=a.draws,event_uncertainty_budget_seconds=a.event_seconds,
        source_git_commit=commit,library_sha256=digest(a.library),
        uncertainty_method='Whole-pulse multiplier-calibrated joint likelihood confidence region',
        Bayesian_credible_intervals=False,canonical_products_unchanged=True))
    def progress(stage, **extra):
        write(a.output/'run_status.h5',{},dict(stage=stage,updated_unix=time.time(),
            retained_events=len(retained),review_rejected_events=len(rejected),
            source_git_commit=commit,**extra))
    def run(command, log, timeout):
        with open(log,'ab') as stream:
            process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                return 124
    batch_size=max(1,a.cores//8)
    for first in range(0,len(retained),batch_size):
        batch=retained[first:first+batch_size]
        progress('bootstrap',batch_first=first,processed_bootstrap_events=first)
        command=['/usr/bin/mpirun','-np',str(min(a.cores,8*len(batch))),'--bind-to','core',
            sys.executable,'-u',str(code/'posterior_pulse_bootstrap.py'),
            '--source',str(a.output/'inputs'),'--output',str(a.output/'bootstrap'),
            '--library',str(a.library),'--draws',str(a.draws),'--optimize-seconds','1.5',
            '--events',*batch]
        if a.resume:command.append('--resume')
        rc=run(command,a.output/f'bootstrap_{first:04d}.log',a.event_seconds)
        print('BOOTSTRAP BATCH',first,len(batch),'return',rc,flush=True)
    def profile(event):
        posterior=a.output/'bootstrap'/event/'posterior.h5'
        if not posterior.exists():return event,'bootstrap did not finish'
        with h5py.File(posterior) as h:
            if not h.attrs['numerical_checks_passed']:return event,'bootstrap numerical checks unresolved'
            elapsed=float(h.attrs['elapsed_seconds'])
        remaining=a.event_seconds-elapsed
        if remaining<1:return event,'uncertainty budget exhausted before profile'
        target=a.output/'confidence'/event/'confidence.h5'
        command=[sys.executable,'-u',str(code/'calibrate_pulse_likelihood_region.py'),
            '--source',str(a.output/'inputs'),'--posterior',str(a.output/'bootstrap'),
            '--output',str(a.output/'confidence'),'--library',str(a.library),
            '--events',event,'--bounds']
        rc=run(command,a.output/(event+'_profile.log'),remaining)
        if rc or not target.exists():return event,f'profile did not finish, return {rc}'
        with h5py.File(target) as h:
            bounds=h['mass_confidence_interval_kg'][:]
            if not h.attrs['bootstrap_numerical_checks_passed']:return event,'uncalibrated profile'
            if h.attrs['source_input_sha256']!=digest(a.output/'inputs'/event/'input.h5'):
                return event,'input identity mismatch'
            finite=bounds[0]>0 and np.isfinite(bounds[1])
            included=finite and np.log10(bounds[1]/bounds[0])<1
            reason=('' if included else 'confidence interval spans at least one order of magnitude'
                if finite else 'upper or lower mass bound unresolved')
            q=h['best_params'][:]
            write(a.output/'confidence'/event/'complete.h5',
                {'params':q,'mass_confidence_interval_kg':bounds},dict(
                    uncertainty_method=h.attrs['uncertainty_method'],
                    Bayesian_credible_interval=False,dynamic_mass_included=included,
                    dynamic_mass_exclusion_reason=reason,finished_unix=time.time(),
                    source_git_commit=commit,source_input_sha256=h.attrs['source_input_sha256']))
        return event,''
    progress('profiles',processed_bootstrap_events=len(retained))
    results=[]
    with ThreadPoolExecutor(max_workers=a.cores) as pool:
        futures={pool.submit(profile,event):event for event in retained}
        for future in as_completed(futures):
            try:row=future.result()
            except Exception as exc:row=(futures[future],repr(exc))
            results.append(row)
            print('PROFILE',*row,flush=True)
            progress('profiles',processed_profile_events=len(results),
                unresolved_profile_events=sum(bool(x[1]) for x in results))
    done=[e for e,error in results if not error];failed=[(e,error) for e,error in results if error]
    write(a.output/'complete.h5',{'completed_events':strings(done),
        'unresolved_events':strings([x[0] for x in failed]),
        'unresolved_reasons':strings([x[1] for x in failed]),'review_rejected_events':strings(rejected)},
        dict(finished_unix=time.time(),expected_events=len(retained),completed_events=len(done),
            unresolved_events=len(failed),all_retained_events_completed=not failed,
            source_git_commit=commit,canonical_products_unchanged=True))
    progress('finished',completed_events=len(done),unresolved_events=len(failed))
    print('FINISHED',len(done),'completed;',len(failed),'unresolved',flush=True)


if __name__=='__main__':main()
