"""Resample failed convergence checks without changing the likelihood or flags."""
import argparse
import ctypes as ct
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI
from arviz_stats.base import array_stats
from run_native_voltage_catalog import load_library


def diagnostics(samples):
    a=samples.transpose(2,0,1)
    return dict(rhat=float(np.max(array_stats.rhat(a,chain_axis=1,draw_axis=2))),
                bulk=float(np.min(array_stats.ess(a,chain_axis=1,draw_axis=2))),
                tail=float(np.min(array_stats.ess(a,method='tail',prob=.025,chain_axis=1,draw_axis=2))))


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--library',required=True);p.add_argument('--warmup',type=int,default=32000)
    p.add_argument('--draws',type=int,default=64000);args=p.parse_args()
    world=MPI.COMM_WORLD
    if world.size%4:raise ValueError('Four ranks per event required')
    group=world.Split(world.rank//4,world.rank%4)
    events=[]
    if world.rank==0:
        for f in sorted(args.root.glob('tri_*/plot.h5')):
            with h5py.File(f) as h:
                if not h.attrs.get('uncertainty_converged',False) and (f.parent/'complete.h5').exists():events.append(f.parent.name)
        events.sort(key=lambda e:(not e.startswith('tri_0060'),e))
        print('RETRY',len(events),'events',flush=True)
    events=world.bcast(events)
    lib=load_library(args.library);ptr=ct.POINTER(ct.c_double)
    for event in events[world.rank//4::world.size//4]:
        folder=args.root/event;stage=folder/'posterior_retry'
        if group.rank==0:
            stage.mkdir(exist_ok=True)
            for name in ('input.h5','fit.h5'):shutil.copy2(folder/name,stage/name)
            print('START',event,flush=True)
        group.Barrier()
        with h5py.File(folder/f'chain_{group.rank}.h5') as h:q=np.ascontiguousarray(h['samples'][-1])
        widths=np.r_[np.full(3,1e-4),np.full(3,1e-3),.01,.01].astype(float)
        handle=lib.voltage_load(os.fsencode(stage/'input.h5'))
        target=stage/f'chain_{group.rank}.h5'
        ok=bool(handle) and bool(lib.voltage_sample(handle,q.ctypes.data_as(ptr),widths.ctypes.data_as(ptr),args.warmup,args.draws,20261030+world.rank,os.fsencode(target)))
        if handle:lib.voltage_free(handle)
        success=group.allreduce(ok,op=MPI.LAND)
        if group.rank==0 and success:
            try:
                arrays=[]
                for i in range(4):
                    with h5py.File(stage/f'chain_{i}.h5') as h:arrays.append(h['samples'][:])
                d=diagnostics(np.stack(arrays));passed=d['rhat']<1.01 and min(d['bulk'],d['tail'])>=400
                with h5py.File(stage/'diagnostics.h5','w') as h:h.attrs.update(**d,converged=passed,warmup=args.warmup,draws=args.draws)
                print('DIAGNOSTICS',event,d,'converged',passed,flush=True)
                if passed:
                    from export_voltage_plot import export
                    export(stage)
                    archive=folder/'posterior_before_retry';archive.mkdir(exist_ok=True)
                    for name in [*(f'chain_{i}.h5' for i in range(4)),'dynamic_mass_posterior.h5','plot.h5']:
                        if (stage/name).exists():
                            if (folder/name).exists():shutil.copy2(folder/name,archive/name)
                            shutil.copy2(stage/name,folder/(name+'.retry'));os.replace(folder/(name+'.retry'),folder/name)
                    print('PUBLISHED',event,flush=True)
            except Exception as e:print('ERROR',event,repr(e),flush=True)
        elif group.rank==0:print('ERROR',event,'sampling failed',flush=True)
        group.Barrier()


if __name__=='__main__':main()
