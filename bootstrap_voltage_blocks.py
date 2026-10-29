"""Time-block multiplier bootstrap of complete joint voltage likelihoods.

Rust performs every physical-model optimization. Python coordinates MPI and HDF5.
These empirical sampling intervals are not thermal-noise posterior intervals,
nor do they cover density, drag, ablation, or unobserved systematic bias.
"""
import argparse
import ctypes as ct
import os
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
from mpi4py import MPI
from fit_cached_shrinking_voltage import native, Cached, covariance
from fit_shrinking_voltage import Model, PTR, save


def block_weights(times, width, seed):
    rng = np.random.default_rng(seed)
    # Random origin avoids a privileged partition. All stations share blocks.
    blocks = np.floor((times - times.min() + rng.uniform(0, width)) / width).astype(int)
    weights = rng.exponential(size=blocks.max()+1)[blocks]
    return np.ascontiguousarray(weights / weights.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--replicates', type=int, default=128)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--block-seconds', type=float, default=.04)
    a = p.parse_args()
    comm = MPI.COMM_WORLD; rank = comm.rank
    started = time.monotonic()
    work = a.output / f'worker_{rank:03d}'
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.source/'input.h5', work/'input.h5')
    with h5py.File(work/'input.h5') as h:
        d = {k: h[k][()] for k in h}
        flag_revision = int(h.attrs['manual_flag_revision'])
    with h5py.File(a.source/'fit.h5') as h: q = h['params'][:]
    lib = native(a.library); model = Model(lib, work, d)
    cache = Cached(model, q, 0, 1., work/'likelihood_cache.h5')
    lib.voltage_cache_weighted_fit.argtypes = [ct.c_void_p, ct.c_void_p, PTR, PTR, PTR, PTR, ct.c_double]
    lib.voltage_cache_weighted_fit.restype = ct.c_int
    matrix = np.ascontiguousarray(np.linalg.cholesky(covariance(cache, q)))
    def fit(weights, start=q):
        start = np.ascontiguousarray(start); result = np.empty(7)
        ok = lib.voltage_cache_weighted_fit(cache.handle, model.handle,
             start.ctypes.data_as(PTR), matrix.ctypes.data_as(PTR),
             weights.ctypes.data_as(PTR), result.ctypes.data_as(PTR), 65.)
        return result, bool(ok)
    if rank == 0:
        center, ok = fit(np.ones(len(d['time_s'])))
        save(a.output/'central_refit.h5', dict(params=center, previous=q),
             dict(converged=ok, logp_gain=cache.logp(center)-cache.logp(q)))
        print('CENTRAL', ok, cache.logp(center)-cache.logp(q), flush=True)
    else: center = None
    center = comm.bcast(center, root=0)
    block_widths=np.array([.5,1.,2.])*a.block_seconds
    jobs = [(width, j) for width in block_widths for j in range(a.replicates)]
    rows = []
    for index in range(rank, len(jobs), comm.size):
        width, j = jobs[index]
        target=a.output/f'draw_{index:04d}.h5'
        start=center
        if a.resume and target.exists():
            with h5py.File(target) as h:
                previous=h['params'][:]
                if h.attrs['optimizer_converged']:
                    rows.append((width,j,True,previous));continue
                start=previous
        weights = block_weights(d['time_s'], width, 441100+index)
        result, ok = fit(weights, start)
        rows.append((width, j, ok, result))
        save(a.output/f'draw_{index:04d}.h5', dict(params=result, weights=weights),
             dict(block_seconds=width, replicate=j, optimizer_converged=ok))
        print('DRAW', index, width, ok, 10**result[6], flush=True)
    rows = comm.gather(rows, root=0)
    if rank == 0:
        rows = [row for chunk in rows for row in chunk]
        widths = np.array([r[0] for r in rows]); ok = np.array([r[2] for r in rows]); params = np.array([r[3] for r in rows])
        masses = 4*np.pi/3*3000*10**(3*params[:,6])
        save(a.output/'bootstrap.h5', dict(block_seconds=widths, converged=ok, params=params, mass_kg=masses),
             dict(method='time-block exponential multiplier likelihood bootstrap', primary_block_seconds=a.block_seconds,
                  flag_revision=flag_revision, source=str(a.source), elapsed_seconds=time.monotonic()-started,
                  replicates_per_width=a.replicates, physical_parameters_fixed=True))
        for width in block_widths:
            select=(widths==width)&ok
            print('SUMMARY', width, int(select.sum()), np.quantile(masses[select],[.025,.5,.975]), flush=True)
        if not np.all(ok):
            raise RuntimeError('Some bootstrap optimizations failed: do not publish a silently filtered interval')
    cache.close(); lib.voltage_free(model.handle)


if __name__ == '__main__': main()
