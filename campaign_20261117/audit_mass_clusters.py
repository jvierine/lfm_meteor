"""Test finite-difference stability and radius-start dependence on real echoes."""
import argparse
import concurrent.futures
from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import least_squares
from analyze import Native, save


def central_jacobian(fun, x, steps):
    columns=[]
    for j,step in enumerate(steps):
        a=x.copy(); b=x.copy(); a[j]+=step; b[j]-=step
        columns.append((fun(a)-fun(b))/(2*step))
    return np.column_stack(columns)


def one(task):
    folder, library, output = task
    with h5py.File(folder/"physical.h5") as h:
        q=h["params"][:]; original_cost=float(h.attrs["voltage_cost"])
    with h5py.File(folder/"drag_model/input.h5") as h: n=len(h["time_s"])
    model=Native(library,folder/"drag_model/input.h5",n)
    lo=np.r_[[-30.]*3,[-50.]*3,-7.]; hi=np.r_[[30.]*3,[50.]*3,-2.]
    scales=np.array([1e-4]*6+[1e-3])
    jacobians=[central_jacobian(model.residual,q,scales*factor) for factor in [1e-3,.1,1.,10.]]
    norms=np.array([np.linalg.norm(j,axis=0) for j in jacobians])
    relative=np.array([np.linalg.norm(j-jacobians[2],axis=0)/np.maximum(norms[2],1e-30) for j in jacobians])
    candidates=[]
    for initial in [q[6],q[6]-.23,q[6]+.23]:
        start=q.copy(); start[6]=np.clip(initial,-6.9,-2.1)
        fit=least_squares(model.residual,start,jac=lambda x:central_jacobian(model.residual,x,scales),
            bounds=(lo,hi),x_scale=np.array([1.]*6+[.3]),max_nfev=160,
            ftol=1e-10,xtol=1e-10,gtol=1e-6)
        candidates.append(fit)
    best=min(candidates,key=lambda f:f.cost)
    radius=np.linspace(max(-6.9,best.x[6]-.6),min(-2.1,best.x[6]+.6),13)
    profile=[]; states=[]
    seed=best.x[:6].copy()
    for r in radius:
        fun=lambda x:model.residual(np.r_[x,r])
        fit=least_squares(fun,seed,jac=lambda x:central_jacobian(fun,x,scales[:6]),
            bounds=(lo[:6],hi[:6]),x_scale=np.ones(6),max_nfev=100,
            ftol=1e-9,xtol=1e-9,gtol=1e-6)
        profile.append(fit.cost);states.append(fit.x)
    save(output/(folder.name+".h5"),dict(original_params=q,original_cost=original_cost,
        derivative_steps=scales,derivative_norms=norms,derivative_relative_changes=relative,
        refined_params=np.array([f.x for f in candidates]),refined_cost=np.array([f.cost for f in candidates]),
        refined_success=np.array([f.success for f in candidates]),
        refined_optimality=np.array([f.optimality for f in candidates]),
        profile_log10_radius=radius,profile_cost=profile,profile_state=states),
        {"diagnostic_only":True,"interval_claim":"profile scan is not a posterior interval"})
    model.close()
    print(folder.name,"old",q[6],original_cost,"refined",best.x[6],best.cost,flush=True)
    return folder.name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("/gpfs/pub/data_other/juha/20261117_tristatic"))
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--workers",type=int,default=1)
    args=p.parse_args()
    with h5py.File(args.root/"association_selection.h5") as h:
        accepted={n.decode() for n,k in zip(h["event_id"][:],h["accepted"][:]) if k}
    groups=[[],[],[]]
    for folder in sorted((args.root/"events").glob("tri_*")):
        if folder.name not in accepted:continue
        with h5py.File(folder/"physical.h5") as h:
            if not h.attrs["mass_resolved_local_curvature"]:continue
            radius=float(h["params"][6])
        group=0 if abs(radius+4)<.005 else 1 if abs(radius+3)<.005 else 2
        if len(groups[group])<3:groups[group].append(folder)
    output=args.root/"mass_cluster_audit";output.mkdir(exist_ok=True)
    tasks=[(f,args.library.resolve(),output) for g in groups for f in g]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(one,tasks))


if __name__=="__main__":main()
