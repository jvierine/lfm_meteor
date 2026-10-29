"""Independent raw-voltage parity and timing on prepared physical event inputs."""
import argparse
import ctypes as ct
import time
from pathlib import Path
import h5py
import numpy as np
from fit_cached_shrinking_voltage import native, Cached, PTR
from fit_shrinking_voltage import Model, save

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--library',required=True)
    p.add_argument('--event',required=True)
    a=p.parse_args();folder=a.root/a.event
    with h5py.File(folder/'input.h5') as h:d={k:h[k][()] for k in h}
    with h5py.File(folder/'fit.h5') as h:q=h['params'][:]
    lib=native(a.library);model=Model(lib,folder,d)
    t=time.monotonic();cache=Cached(model,q,0,1.,folder/'reduced_statistics_validation.h5')
    build=time.monotonic()-t;rng=np.random.default_rng(377)
    points=[q]
    for scale in [0.001,0.01,0.1,1.]:
        for _ in range(16):
            probe=q+rng.normal(size=7)*np.r_[np.full(3,scale),np.full(3,scale),scale]
            if np.isfinite(cache.logp(probe)):points.append(probe)
    fast=np.array([cache.logp(x) for x in points]);raw=[]
    for x in points:
        x=np.ascontiguousarray(x);raw.append(lib.voltage_logp(model.handle,x.ctypes.data_as(PTR)))
    error=(fast-fast[0])-(np.array(raw)-raw[0])
    t=time.monotonic()
    for _ in range(1000):cache.logp(q)
    elapsed=(time.monotonic()-t)/1000
    attrs=dict(build_seconds=build,evaluation_seconds=elapsed,max_delta_logp_error=float(np.max(abs(error))),passed=bool(np.max(abs(error))<1e-4))
    save(folder/'reduced_raw_validation.h5',dict(params=points,delta_error=error),attrs)
    print(a.event,attrs,flush=True)
    cache.close();lib.voltage_free(model.handle)
    if not attrs['passed']:raise RuntimeError('Raw likelihood parity failed')

if __name__=='__main__':main()
