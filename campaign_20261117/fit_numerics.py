"""Explicit physical step sizes and conditional radius-profile checks."""
import numpy as np
from scipy.optimize import least_squares, brentq

STEPS = np.array([1e-4]*6+[1e-3])  # 0.1 m, 0.1 m/s, 0.001 dex.


def jacobian(fun, x, steps=STEPS, bounds=None):
    columns=[]
    for j,step in enumerate(steps):
        a=x.copy(); b=x.copy(); a[j]+=step; b[j]-=step
        if bounds is not None:
            a[j]=min(a[j],bounds[1][j]); b[j]=max(b[j],bounds[0][j])
        columns.append((fun(a)-fun(b))/(a[j]-b[j]))
    return np.column_stack(columns)


def optimize(fun, start, bounds, max_nfev=240):
    n=len(start)
    return least_squares(fun,start,jac=lambda x:jacobian(fun,x,STEPS[:n],bounds),
        bounds=bounds,x_scale=np.r_[np.ones(6),.3][:n],max_nfev=max_nfev,
        ftol=1e-10,xtol=1e-10,gtol=1e-6)


def profile_interval(profile_cost, center, minimum_cost, bounds=(-6.999,-2.001), threshold=1.920729410347062):
    """Conditional one-parameter LR crossings, not a posterior interval."""
    target=minimum_cost+threshold
    if not bounds[0]<center<bounds[1]:return np.full(2,np.nan)
    endpoints=[]
    for direction,bound in zip([-1,1],bounds):
        inside=center
        crossing=np.nan
        for k in range(16):
            outside=float(np.clip(center+direction*.025*1.7**k,*bounds))
            if outside==inside: break
            value=profile_cost(outside)-target
            if np.isfinite(value) and value>0:
                candidate=brentq(lambda x:profile_cost(x)-target,min(inside,outside),max(inside,outside),xtol=1e-7)
                # A sign change across a discontinuity is not an LR crossing.
                if abs(profile_cost(candidate)-target) <= .02:
                    crossing=candidate
                break
            if outside==bound: break
            inside=outside
        endpoints.append(crossing)
    return np.array(endpoints)
