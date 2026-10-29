"""Audit narrow mass intervals, then run bounded MPI block-bootstrap refits."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import h5py
import numpy as np


def audit(root, flags, output):
    records=[]
    for folder in sorted(root.glob('tri_*')):
        if not (folder/'plot.h5').exists():continue
        try:
            with h5py.File(folder/'plot.h5') as h:
                attrs=dict(h.attrs); q=h['initial_mass_kg_quantiles'][:]
                stations=h['station'][:];times=h['time_s'][:]
                beat=h['residual_beat_hz'][:];std=h['conditional_beat_std_hz'][:]
            width=float(np.log10(q[2]/q[0])) if np.all(np.isfinite(q)) and q[0]>0 else np.inf
            excess=[];correlations=[];cadences=[]
            for s in np.unique(stations):
                idx=np.flatnonzero(stations==s);idx=idx[np.argsort(times[idx])]
                z=beat[idx]/np.maximum(std[idx],1e-8)
                excess.append(np.sqrt(np.mean(z*z)))
                residual=beat[idx]-np.mean(beat[idx])
                denom=np.dot(residual,residual)
                correlations.append(np.dot(residual[:-1],residual[1:])/denom if denom>0 else 0.)
                cadences.extend(np.diff(times[idx]).tolist())
            duration=float(np.ptp(times)); cadence=float(np.median(cadences))
            primary=max(cadence,min(.04,duration/8))
            eligible=duration/primary>=6 and len(times)>=21
            flagfile=flags/(folder.name+'.h5'); revision=0
            if flagfile.exists():
                with h5py.File(flagfile) as h:revision=int(h.attrs.get('version',0))
            stale=revision!=int(attrs.get('fit_flag_revision',-1))
            done='bootstrap' in attrs.get('uncertainty_method','')
            suspicious=width<.5 and (max(excess)>1.5 or max(correlations)>.3)
            selected=(suspicious or folder.name.startswith('tri_0112_')) and not done and eligible and not stale
            records.append((folder.name,width,max(excess),max(correlations),primary,selected,stale,eligible,done))
        except Exception as exc:
            print('AUDIT_ERROR',folder.name,repr(exc),flush=True)
            records.append((folder.name,np.nan,np.nan,np.nan,np.nan,False,False,False,False))
    with h5py.File(output/'audit.h5','w') as h:
        names=['event','mass_interval_dex','max_standardized_beat_rms','max_lag1_correlation','block_seconds','selected','stale_flags','enough_time_blocks','already_bootstrapped']
        for i,name in enumerate(names):
            h[name]=np.asarray([r[i] for r in records],dtype=h5py.string_dtype() if i==0 else None)
        h.attrs.update(max_interval_dex=.5,min_excess_rms=1.5,min_lag1_correlation=.3,
                       criterion='screening only; not a scientific rejection',created_unix=time.time())
    return sorted([r for r in records if r[5]],key=lambda r:(not r[0].startswith('tri_0112_'),r[0]))


def command(cmd,log,seconds):
    with open(log,'a') as stream:
        child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
             env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        try:return child.wait(timeout=max(1,seconds))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            return 124


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','flags','output','library'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--workers',type=int,default=128)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    selected=audit(a.root,a.flags,a.output)
    print('SELECTED',len(selected),[r[0] for r in selected],flush=True)
    if a.audit_only:return
    code=Path(__file__).resolve().parent
    for record in selected:
        event=record[0];out=a.output/event;out.mkdir(exist_ok=True)
        began=time.monotonic()
        cmd=['mpirun','-np',str(a.workers),'--bind-to','core',sys.executable,str(code/'bootstrap_voltage_blocks.py'),
             '--source',str(a.root/event),'--output',str(out),'--library',str(a.library),'--replicates','512','--block-seconds',str(record[4])]
        print('START',event,'block_s',record[4],flush=True)
        rc=command(cmd,out/'run.log',240)
        if rc and time.monotonic()-began<280:
            rc=command(cmd+['--resume'],out/'run.log',300-(time.monotonic()-began))
        published=False;reason='bootstrap failed or budget exhausted'
        if rc==0:
            with h5py.File(out/'bootstrap.h5') as h:
                masses=h['mass_kg'][:][np.isclose(h['block_seconds'][:],record[4],rtol=1e-10,atol=0)]
                quant=np.quantile(masses,[.025,.975]);new_width=np.log10(quant[1]/quant[0])
            if new_width<=record[1]:reason='empirical interval not wider; retained original interval'
            else:
                pub=[sys.executable,str(code/'publish_voltage_block_uncertainty.py'),'--event-folder',str(a.root/event),
                     '--bootstrap',str(out),'--library',str(a.library),'--flags',str(a.flags)]
                published=command(pub,out/'publish.log',120)==0
                reason='published' if published else 'publication guard failed; retained original interval'
        with h5py.File(out/'status.h5','w') as h:
            h.attrs.update(event=event,published=published,reason=reason,finished_unix=time.time(),elapsed_seconds=time.monotonic()-began)
        print('FINISHED',event,reason,flush=True)
    with h5py.File(a.output/'finished.h5','w') as h:h.attrs['finished_unix']=time.time()


if __name__=='__main__':main()
