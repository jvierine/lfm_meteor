"""Cross-validate observed echo shapes and a noise-filter response hypothesis."""
import argparse
from pathlib import Path

import h5py
from mpi4py import MPI
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from fit_shrinking_voltage import save


def read(source,audit,event):
    with h5py.File(source/event/'input.h5') as h:d={k:h[k][()] for k in h}
    with h5py.File(audit/event/'pulse/pulse_audit.h5') as h:
        pulse=h['independent_pulse_fit'][:]
    return d,pulse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    a=p.parse_args();comm=MPI.COMM_WORLD
    grid=np.linspace(-10e-6,210e-6,881)
    profiles={};weights={};datasets={};jobs=[]
    for folder in sorted(a.audit.glob('tri_*')):
        event=folder.name;d,pulse=read(a.source,a.audit,event);datasets[event]=(d,pulse)
        rows=[];w=[];indices=np.flatnonzero(d['station']==0)
        for i in indices:
            path,beat=pulse[i,:2];t=d['sample_time_s'][i];rate=d['bandwidth']/d['duration']
            tau=t-path/299792458.;fd=beat-rate*path/299792458.
            phase=np.exp(2j*np.pi*(d['bandwidth']/2*tau-rate/2*tau*tau+fd*t))
            env=np.clip((tau+.5/d['fs'])*d['fs'],0,1)*np.clip((d['duration']-tau+.5/d['fs'])*d['fs'],0,1)
            filt=d['whitener_real'][i]+1j*d['whitener_imag'][i]
            raw=d['real'][i]+1j*d['imag'][i];white=lfilter(filt,[1.],raw);unit=lfilter(filt,[1.],env*phase)
            valid=slice(len(filt)-1,None);amp=np.vdot(unit[valid],white[valid])/np.vdot(unit[valid],unit[valid])
            z=raw*np.conj(phase)/amp
            rows.append(np.interp(grid,tau,z.real)+1j*np.interp(grid,tau,z.imag))
            w.append(abs(amp)**2/d['noise_power'][i])
        profiles[event]=np.asarray(rows);weights[event]=np.asarray(w)
        strongest=indices[np.argsort(weights[event])[-3:]]
        jobs.extend((event,int(i)) for i in strongest)
    results=[]
    for event,index in jobs[comm.rank::comm.size]:
        d,pulse=datasets[event];indices=np.flatnonzero(d['station']==0)
        other=np.concatenate([profiles[e] for e in profiles if e!=event])
        other_weights=np.concatenate([weights[e] for e in profiles if e!=event])
        own=indices!=index
        alternatives={'heldout_event':np.average(other,axis=0,weights=other_weights),
                      'heldout_pulse':np.average(profiles[event][own],axis=0,weights=weights[event][own])}
        raw=d['real'][index]+1j*d['imag'][index];t=d['sample_time_s'][index]
        filt=d['whitener_real'][index]+1j*d['whitener_imag'][index];white=lfilter(filt,[1.],raw)
        valid=slice(len(filt)-1,None);rate=d['bandwidth']/d['duration'];fs=d['fs'];noise=d['noise_power'][index]
        origin=pulse[index,:2]
        for kind in ['original','whitened_ideal','heldout_event','heldout_pulse']:
            def objective(q):
                path=origin[0]+q[0];beat=origin[1]+100*q[1]
                tau=t-path/299792458.;fd=beat-rate*path/299792458.
                phase=np.exp(2j*np.pi*(d['bandwidth']/2*tau-rate/2*tau*tau+fd*t))
                if kind in alternatives:
                    z=alternatives[kind]
                    env=np.interp(tau,grid,z.real,left=0,right=0)+1j*np.interp(tau,grid,z.imag,left=0,right=0)
                else:
                    env=np.clip((tau+.5/fs)*fs,0,1)*np.clip((d['duration']-tau+.5/fs)*fs,0,1)
                unit=env*phase
                if kind!='whitened_ideal':unit=lfilter(filt,[1.],unit)
                amp=np.vdot(unit[valid],white[valid])/np.vdot(unit[valid],unit[valid])
                return float(np.sum(abs(white[valid]-amp*unit[valid])**2)/noise)
            fit=minimize(objective,np.zeros(2),method='Powell',options=dict(xtol=1e-7,ftol=1e-9,maxiter=100))
            result=dict(event=event,index=index,model=kind,noise_normalized_energy=float(fit.fun),
                        expected_noise_energy=len(t)-len(filt)-1,optimizer_success=bool(fit.success))
            print('TEMPLATE',result,flush=True);results.append(result)
    gathered=comm.gather(results,root=0)
    if comm.rank==0:
        rows=sum(gathered,[])
        with h5py.File(a.audit/'template_summary.h5','w') as h:
            for k in rows[0]:
                h[k]=np.asarray([r[k] for r in rows],dtype=h5py.string_dtype() if k in ('event','model') else None)
            for event in profiles:
                g=h.create_group('profiles/'+event);mean=np.average(profiles[event],axis=0,weights=weights[event])
                g['time_us']=grid*1e6;g['mean_real']=mean.real;g['mean_imag']=mean.imag
            h.attrs.update(diagnostic_only=True,production_products_unchanged=True,
                heldout='Event/pulse being tested is excluded from its empirical template')


if __name__=='__main__':main()
