"""Non-destructive uncertainty audit of the two dominant November masses."""
import argparse
import concurrent.futures
import hashlib
from pathlib import Path
import traceback
import h5py
import numpy as np
from scipy.optimize import least_squares
from analyze import Native,C,GAMMA,save
from physical import ROWS
from derived import rcs_beam
from fit_numerics import STEPS,jacobian,profile_interval

LO=np.r_[[-30.]*3,[-50.]*3,-7.]
HI=np.r_[[30.]*3,[50.]*3,-2.]


def optimize(fun,start,factor=1.,nfev=300):
    n=len(start)
    return least_squares(fun,np.clip(start,LO[:n]+1e-9,HI[:n]-1e-9),
        jac=lambda q:jacobian(fun,q,STEPS[:n]*factor,(LO[:n],HI[:n])),
        bounds=(LO[:n],HI[:n]),x_scale=np.r_[np.ones(6),.3][:n],
        max_nfev=nfev,ftol=1e-11,xtol=1e-11,gtol=1e-6)


def pulse_groups(times):
    return np.rint((times-times.min())/.005).astype(int)


def variant_mask(kind,mask,times,station,position,snr,index=0):
    keep=mask.copy();groups=pulse_groups(times);unique=np.unique(groups[mask])
    if kind.startswith('trim_'):
        fraction=.1 if '10' in kind else .2
        count=max(1,int(np.ceil(len(unique)*fraction)))
        if 'start' in kind:keep &= groups>=unique[count]
        else:keep &= groups<=unique[-count-1]
    elif kind=='strict_simultaneous':
        shared=[g for g in unique if len(np.unique(station[mask&(groups==g)]))==3]
        keep &= np.isin(groups,shared)
    elif kind=='mainlobe_corrected':keep &= rcs_beam(position)[0]>=-10.
    elif kind=='even_pulses':keep &= groups%2==0
    elif kind=='odd_pulses':keep &= groups%2==1
    elif kind=='drop_weakest20':
        for s in range(3):
            ids=np.flatnonzero(mask&(station==s));ids=ids[np.argsort(snr[ids])]
            keep[ids[:int(.2*len(ids))]]=False
    elif kind=='delete_pulse':keep &= groups!=index
    return keep


def one(task):
    root,library,event,kind,index=task
    event_dir=root/'events'/('tri_%06d'%event)
    out=root/'mass_uncertainty_audit_20261122'/event_dir.name/(kind+'_%03d'%index)
    out.mkdir(parents=True,exist_ok=True)
    try:
        with h5py.File(event_dir/'physical.h5') as h:
            q0=h['params'][:];mask=h['dynamic_mass_fit_mask'][:]
            pos=h['position_ecef_m'][:];old_cov=h['covariance'][:]
            published=h['mass_local95_kg'][:];old_cost=float(h.attrs['voltage_cost'])
        with h5py.File(event_dir/'drag_model/input.h5') as h:d={k:h[k][()] for k in h}
        with h5py.File(event_dir/'fit.h5') as h:
            observed=h['apparent_path_m'][:];snr=h['snr_db'][:];seed=h['params'][:6]
        keep=variant_mask(kind,mask,d['time_s'],d['station'],pos,snr,index)
        if any(sum(keep&(d['station']==s))<6 for s in range(3)):
            raise ValueError('Insufficient station measurements in diagnostic subset')
        selected={k:(v[keep] if k in ROWS else v) for k,v in d.items()}
        save(out/'input.h5',selected,{'diagnostic_only':True,'original_epoch_preserved':True})
        model=Native(library,out/'input.h5',int(keep.sum()))
        factor={0:.3,1:1.,2:3.}.get(index,1.) if kind=='derivative_step' else 1.
        truth=4*np.pi/3*3000*10**(3*q0[6])
        if kind=='noise_recovery':
            prediction=model.call('voltage_prediction',q0,(*selected['real'].shape,2))
            model.close();rng=np.random.default_rng(20261122000+event*100+index)
            sigma=np.sqrt(selected['noise_power'][:,None]/2)
            selected['real']=prediction[:,:,0]+rng.normal(size=selected['real'].shape)*sigma
            selected['imag']=prediction[:,:,1]+rng.normal(size=selected['imag'].shape)*sigma
            save(out/'input.h5',selected,{'diagnostic_only':True,'simulation':'fitted finite-pulse model plus independent circular complex Gaussian receiver noise'})
            model=Native(library,out/'input.h5',int(keep.sum()))
        if kind=='diagnostics':
            prediction=model.call('voltage_prediction',q0,(*selected['real'].shape,2))
            r=model.residual(q0).reshape(*selected['real'].shape,2)
            signal=np.linalg.norm(prediction,axis=2)>.2*np.max(np.linalg.norm(prediction,axis=2),axis=1)[:,None]
            fields=dict(used_mask=keep,time_s=selected['time_s'],station=selected['station'],
                signal_mean_squared_normalized_residual=np.array([np.mean(r[i,signal[i]]**2) for i in range(len(r))]),
                offpulse_normalized_noise_ratio=np.array([np.median(np.sum(r[i,~signal[i]]**2,axis=1))/(2*np.log(2)) for i in range(len(r))]))
            obs=model.call('voltage_observables',q0,(int(keep.sum()),2))
            fields['path_residual_m']=observed[keep]-obs[:,0]-C*obs[:,1]/GAMMA
            acf=[]
            for s in range(3):
                x=fields['path_residual_m'][selected['station']==s]
                acf.append(np.corrcoef(x[:-1],x[1:])[0,1])
            fields['site_path_lag1_correlation']=acf
            jacs=[jacobian(model.residual,q0,STEPS*f,(LO,HI)) for f in (.1,1.,10.)]
            fields['derivative_relative_changes']=np.array([np.linalg.norm(j-jacs[1],axis=0)/np.maximum(np.linalg.norm(jacs[1],axis=0),1e-30) for j in jacs])
            save(out/'result.h5',fields,{'diagnostic_only':True,'mass_kg':truth,'baseline_cost':old_cost})
            model.close();return event,kind,index,'done'
        if kind=='radius_scan':
            radius=float(index)/1000000
            transport=q0[:6]+old_cov[:6,6]/max(old_cov[6,6],1e-30)*(radius-q0[6])
            fun=lambda x:model.residual(np.r_[x,radius])
            fits=[optimize(fun,x,nfev=400) for x in (q0[:6],transport,seed)]
            fit=min(fits,key=lambda f:f.cost);q=np.r_[fit.x,radius]
            interval=np.full(2,np.nan)
        else:
            starts=[q0.copy()]
            if kind!='delete_pulse':
                for radius in (q0[6]-.25,q0[6]+.25):
                    starts.append(np.r_[q0[:6],np.clip(radius,-6.9,-2.01)])
            fits=[optimize(model.residual,x,factor) for x in starts]
            fit=min(fits,key=lambda f:f.cost);q=fit.x
            interval=np.full(2,np.nan)
            if kind in ('baseline','noise_recovery','mainlobe_corrected'):
                cache={float(q[6]):(fit.cost,q[:6].copy())}
                def at(radius):
                    if radius not in cache:
                        near=min(cache,key=lambda x:abs(x-radius))
                        f=optimize(lambda x:model.residual(np.r_[x,radius]),cache[near][1],factor,240)
                        cache[radius]=(f.cost,f.x)
                    return cache[radius][0]
                interval=profile_interval(at,q[6],fit.cost)
        obs=model.call('voltage_observables',q,(int(keep.sum()),2))
        residual=observed[keep]-obs[:,0]-C*obs[:,1]/GAMMA
        if kind=='noise_recovery':
            residual=np.full(int(keep.sum()),np.nan)
        mass=4*np.pi/3*3000*10**(3*q[6]);mass_interval=4*np.pi/3*3000*10**(3*interval)
        fields=dict(params=q,mass_kg=mass,mass_interval_kg=mass_interval,published_interval_kg=published,
            used_mask=keep,time_s=selected['time_s'],station=selected['station'],path_residual_m=residual,
            candidate_cost=[f.cost for f in fits],candidate_success=[f.success for f in fits],
            candidate_optimality=[f.optimality for f in fits])
        save(out/'result.h5',fields,{'diagnostic_only':True,'cost':fit.cost,'baseline_cost':old_cost,
            'optimizer_success':fit.success,'optimality':fit.optimality,'nfev':fit.nfev,
            'noise_only_truth_mass_kg':truth,'simulation_covered':bool(mass_interval[0]<=truth<=mass_interval[1]),
            'mass_at_radius_boundary':bool(q[6]<-6.99 or q[6]>-2.01),
            'input_physical_sha256':hashlib.sha256((event_dir/'physical.h5').read_bytes()).hexdigest(),
            'interval_meaning':'conditional receiver-noise profile only; subset range is not a confidence interval'})
        model.close()
        print(event,kind,index,'mass',mass,'cost',fit.cost,'success',fit.success,flush=True)
        return event,kind,index,'done'
    except Exception:
        (out/'error.log').write_text(traceback.format_exc())
        return event,kind,index,'failed'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/gpfs/pub/data_other/juha/20261117_tristatic'))
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--simulations',type=int,default=32)
    p.add_argument('--kinds',nargs='+',help='Run only these diagnostic task kinds')
    a=p.parse_args();tasks=[]
    for event in (411,563):
        folder=a.root/'events'/('tri_%06d'%event)
        with h5py.File(folder/'physical.h5') as h:
            q=h['params'][:];mask=h['dynamic_mass_fit_mask'][:];times=h['time_s'][:]
        for kind in ('diagnostics','baseline','strict_simultaneous','mainlobe_corrected',
                     'trim_start10','trim_end10','trim_start20','trim_end20','even_pulses','odd_pulses','drop_weakest20'):
            tasks.append((a.root,a.library.resolve(),event,kind,0))
        for index in range(3):tasks.append((a.root,a.library.resolve(),event,'derivative_step',index))
        for group in np.unique(pulse_groups(times)[mask]):
            tasks.append((a.root,a.library.resolve(),event,'delete_pulse',int(group)))
        radii=np.unique(np.r_[np.linspace(max(-6.9,q[6]-.8),-2.001,25),q[6],q[6]-.04,q[6]+.04])
        for radius in radii:tasks.append((a.root,a.library.resolve(),event,'radius_scan',int(round(radius*1000000))))
        for index in range(a.simulations):tasks.append((a.root,a.library.resolve(),event,'noise_recovery',index))
    out=a.root/'mass_uncertainty_audit_20261122'
    if a.kinds:
        tasks=[task for task in tasks if task[3] in a.kinds]
        if not tasks:raise ValueError('No tasks match --kinds')
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
        result=list(pool.map(one,tasks,chunksize=1))
    status_name='status.h5' if not a.kinds else 'status_selected.h5'
    save(out/status_name,dict(event=np.array([r[0] for r in result]),
        kind=np.array([r[1] for r in result],dtype='S32'),index=np.array([r[2] for r in result]),
        status=np.array([r[3] for r in result],dtype='S16')),
        {'diagnostic_only':True,'production_files_changed':False,'noise_simulations_per_event':a.simulations})
    print('AUDIT tasks',len(result),'failed',sum(r[3]=='failed' for r in result),flush=True)


if __name__=='__main__':main()
