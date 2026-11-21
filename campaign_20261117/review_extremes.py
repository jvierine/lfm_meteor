"""Non-destructive full/core voltage refits of the November extreme events."""
import argparse
import concurrent.futures
from pathlib import Path
import hashlib
import traceback
import h5py
import numpy as np
from analyze import Native, save
from fit_numerics import optimize, profile_interval

EVENTS = (46,644,879,35,179,56,70,581,772,773,211,585,515,586,837)
ROWS = ('time_s','time_ns','station','frames','rx','real','imag','noise_power',
        'sample_time_s','whitener_real','whitener_imag')


def core_mask(times, station):
    lower=max(times[station==s].min() for s in range(3))
    upper=min(times[station==s].max() for s in range(3))
    return (times>=lower)&(times<=upper)


def one(task):
    root,library,event,variant=task
    folder=root/'events'/('tri_%06d'%event)
    output=root/'extreme_review'/folder.name/variant
    output.mkdir(parents=True,exist_ok=True)
    try:
        with h5py.File(folder/'drag_model/input.h5') as h:
            data={k:h[k][()] for k in h}
        with h5py.File(folder/'physical.h5') as h:
            original=h['params'][:]
        with h5py.File(folder/'fit.h5') as h:
            measured=h['apparent_path_m'][:]
        mask=np.ones(len(data['time_s']),bool)
        if variant=='core': mask=core_mask(data['time_s'],data['station'])
        # Retain the original reference epoch, so initial masses remain comparable.
        selected={k:(v[mask] if k in ROWS else v) for k,v in data.items()}
        save(output/'input.h5',selected,{'selection':variant,'original_epoch_preserved':True})
        model=Native(library,output/'input.h5',sum(mask))
        lo=np.r_[[-30.]*3,[-50.]*3,-7.];hi=np.r_[[30.]*3,[50.]*3,-2.]
        candidates=[]
        for radius in (original[6],-5.,-4.,-3.,-2.1):
            q=original.copy();q[6]=radius
            candidates.append(optimize(model.residual,q,(lo,hi),max_nfev=400))
        best=min(candidates,key=lambda f:f.cost)
        q=best.x
        profiles={}
        def cost_at(r):
            if r not in profiles:
                fun=lambda x:model.residual(np.r_[x,r])
                fit=optimize(fun,q[:6],(lo[:6],hi[:6]),max_nfev=300)
                profiles[r]=(fit.cost,fit.x,fit.success,fit.optimality)
            return profiles[r][0]
        interval=profile_interval(cost_at,q[6],best.cost)
        # A broad scan reveals flat tails and failed local optimization hidden by narrow intervals.
        for r in np.unique(np.r_[np.linspace(max(-6.99,q[6]-.5),min(-2.01,q[6]+.5),9),-2.01]):
            cost_at(float(r))
        obs=model.call('voltage_observables',q,(sum(mask),2))
        states=model.call('voltage_states',q,(sum(mask),4))
        residual=measured[mask]-obs[:,0]-299792458*obs[:,1]/2e10
        grid=np.array(sorted(profiles))
        save(output/'review.h5',dict(retained_mask=mask,original_params=original,
            params=q,cost=best.cost,success=best.success,optimality=best.optimality,
            candidate_params=np.array([f.x for f in candidates]),
            candidate_cost=np.array([f.cost for f in candidates]),
            candidate_optimality=np.array([f.optimality for f in candidates]),
            time_s=selected['time_s'],station=selected['station'],path_residual_m=residual,
            speed_mps=states[:,1],profile_log10_radius=grid,
            profile_cost=[profiles[r][0] for r in grid],
            profile_success=[profiles[r][2] for r in grid],
            profile_optimality=[profiles[r][3] for r in grid],
            radius_interval_m=10**interval,initial_mass_kg=4*np.pi/3*3000*10**(3*q[6])),
            {'diagnostic_only':True,'source_input_sha256':hashlib.sha256((folder/'drag_model/input.h5').read_bytes()).hexdigest(),
             'interval_method':'conditional profile; not approved for catalogue replacement'})
        model.close()
        print(folder.name,variant,'mass',4*np.pi/3*3000*10**(3*q[6]),'rms',np.sqrt(np.mean(residual**2)),'optimality',best.optimality,flush=True)
        return folder.name,variant,'done'
    except Exception:
        (output/'error.log').write_text(traceback.format_exc())
        return folder.name,variant,'failed'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/gpfs/pub/data_other/juha/20261117_tristatic'))
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    tasks=[(a.root,a.library.resolve(),n,v) for n in EVENTS for v in ('full','core')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(one,tasks))
    save(a.root/'extreme_review/status.h5',{'event':np.array([r[0] for r in results],dtype='S32'),
        'variant':np.array([r[1] for r in results],dtype='S16'),'status':np.array([r[2] for r in results],dtype='S16')})
    if any(r[2]=='failed' for r in results):raise SystemExit('Review failures: inspect error.log')


if __name__=='__main__':main()
