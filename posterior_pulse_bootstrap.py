"""Native MPI posterior bootstrap of complete per-pulse voltage likelihoods.

This is a nonparametric posterior over the fitted-trajectory functional, not
the receiver-noise-only parametric posterior. No correlation between pulses
is imposed. Fixed-station sample counts are respected by stratified weights.
"""
import argparse
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import time

import h5py
from mpi4py import MPI
import numpy as np

from fit_cached_shrinking_voltage import native, Cached, covariance
from fit_shrinking_voltage import Model, PTR, save


def weights_for(stations, seed):
    rng = np.random.default_rng(seed)
    weights = rng.exponential(size=len(stations))
    for site in np.unique(stations):
        selected = stations == site
        weights[selected] *= np.sum(selected)/weights[selected].sum()
    return np.ascontiguousarray(weights)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['source', 'output', 'library']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--posterior', type=Path)
    p.add_argument('--events', nargs='+', required=True)
    p.add_argument('--draws', type=int, default=256)
    p.add_argument('--optimize-seconds', type=float, default=2.)
    p.add_argument('--seed', type=int, default=202611090000)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    world = MPI.COMM_WORLD
    index = world.rank % len(a.events)
    comm = world.Split(index, world.rank)
    event = a.events[index]
    source, output = a.source/event, a.output/event
    output.mkdir(parents=True, exist_ok=True)
    input_hash = hashlib.sha256((source/'input.h5').read_bytes()).hexdigest()
    if a.resume and (output/'posterior.h5').exists():
        with h5py.File(output/'posterior.h5') as h:
            if h.attrs['source_input_sha256'] != input_hash:
                raise ValueError('Cannot resume with changed measurements or flags')
    began = time.monotonic()
    with h5py.File(source/'input.h5') as h:
        data = {k: h[k][()] for k in h}
        revision = int(h.attrs.get('manual_flag_revision', 0))
    with h5py.File(source/'fit.h5') as h:
        q = np.ascontiguousarray(h['params'][:])
    if a.posterior:
        with h5py.File(a.posterior/event/'posterior.h5') as h:
            q = np.ascontiguousarray(h['proposal_center'][:])
    if q.shape != (7,) or 'height_m' not in data:
        raise ValueError('A direct seven-parameter shrinking-radius model with atmosphere is required')
    lib = native(a.library)
    model = Model(lib, source, data)
    cache = Cached(model, q, 0, 1., output/f'cache_{comm.rank:03d}.h5')
    lib.voltage_posterior_bootstrap_fit.argtypes = [ct.c_void_p,ct.c_void_p,PTR,PTR,PTR,ct.c_double,PTR,PTR]
    lib.voltage_posterior_bootstrap_fit.restype = ct.c_int
    lib.voltage_cache_weighted_logp.argtypes = [ct.c_void_p,ct.c_void_p,PTR,PTR,ct.c_int]
    lib.voltage_cache_weighted_logp.restype = ct.c_double
    def fit(seed, metric, weights):
        qnew, diagnostics = np.empty(7), np.empty(4)
        ok = lib.voltage_posterior_bootstrap_fit(cache.handle, model.handle,
            seed.ctypes.data_as(PTR), metric.ctypes.data_as(PTR), weights.ctypes.data_as(PTR),
            a.optimize_seconds, qnew.ctypes.data_as(PTR), diagnostics.ctypes.data_as(PTR))
        if not ok:
            raise RuntimeError('Invalid weighted trajectory optimization')
        return qnew, diagnostics
    if comm.rank == 0:
        metric = np.ascontiguousarray(covariance(cache, q))
        center, diagnostics = fit(q, metric, np.ones(len(data['station'])))
        metric = np.ascontiguousarray(covariance(cache, center))
        save(output/'central_fit.h5', dict(params=center, previous_params=q,
            covariance=metric, diagnostics=diagnostics),
            dict(repeat_gain=float(diagnostics[2]), unchanged_likelihood=True))
        print('CENTER', event, 'gain', diagnostics[1], 'repeat',diagnostics[2],flush=True)
    else:
        metric = center = None
    center = np.ascontiguousarray(comm.bcast(center))
    metric = np.ascontiguousarray(comm.bcast(metric))
    rows = []
    for draw in range(comm.rank, a.draws, comm.size):
        seed = (a.seed ^ int.from_bytes(hashlib.sha256(event.encode()).digest()[:8], 'little') ^ draw) % 2**64
        weights = weights_for(data['station'], seed)
        target = output/f'draw_{draw:04d}.h5'
        start, current_metric = center, metric
        if a.resume and target.exists():
            with h5py.File(target) as h:
                np.testing.assert_array_equal(h['weights'][:],weights)
                old_params, old_diagnostics = h['params'][:], h['diagnostics'][:]
                parity = float(h.attrs['cache_raw_difference'])
            if old_diagnostics[2] < .01:
                rows.append((draw,old_params,old_diagnostics,parity))
                continue
            start = np.ascontiguousarray(old_params)
            current_metric = np.ascontiguousarray(covariance(cache,start))
            save(output/f'draw_{draw:04d}_resume_input.h5',
                 dict(params=old_params,diagnostics=old_diagnostics),dict(preserved_previous_attempt=True))
        params, diagnostic = fit(start, current_metric, weights)
        for attempt in range(6):
            if diagnostic[2] <= .01:
                break
            save(output/f'draw_{draw:04d}_attempt_{attempt}.h5',
                 dict(params=params, diagnostics=diagnostic), dict(unresolved_optimization=True))
            previous = diagnostic[0]
            current_metric = np.ascontiguousarray(covariance(cache,params))
            params, diagnostic = fit(np.ascontiguousarray(params), current_metric, weights)
            diagnostic[2] = max(diagnostic[2], diagnostic[0]-previous)
        parity = np.nan
        if draw < comm.size:
            values = [lib.voltage_cache_weighted_logp(cache.handle,model.handle,
                params.ctypes.data_as(PTR),weights.ctypes.data_as(PTR),exact) for exact in [0,1]]
            parity = abs(values[0]-values[1])
        save(output/f'draw_{draw:04d}.h5', dict(params=params, weights=weights, diagnostics=diagnostic),
            dict(draw=draw, seed=seed, cache_raw_difference=parity,
                 optimizer_repeat_gain=float(diagnostic[2])))
        rows.append((draw, params, diagnostic, parity))
    gathered = comm.gather(rows, root=0)
    if comm.rank == 0:
        rows = sorted([r for group in gathered for r in group], key=lambda r:r[0])
        params = np.array([r[1] for r in rows])
        diagnostics = np.array([r[2] for r in rows])
        parity = np.array([r[3] for r in rows])
        radii = 10**params[:,6]
        masses = 4*np.pi/3*3000*radii**3
        boundary = (params[:,6] < -6.99) | (params[:,6] > -2.01)
        rng = np.random.default_rng(19)
        endpoints = np.array([np.quantile(3*params[rng.integers(0,len(params),len(params)),6],
            [.025,.5,.975]) for _ in range(128)])
        checks = (len(rows)==a.draws and np.isfinite(params).all()
            and np.max(diagnostics[:,2]) < .01 and np.nanmax(parity) < .01)
        save(output/'posterior.h5', dict(samples=params, mass_kg=masses, radius_m=radii,
            mass_kg_quantiles=np.quantile(masses,[.025,.5,.975]),
            radius_m_quantiles=np.quantile(radii,[.025,.5,.975]),
            parameter_quantiles=np.quantile(params,[.025,.5,.975],axis=0),
            best_params=center, diagnostics=diagnostics, cache_raw_difference=parity,
            radius_prior_boundary=boundary,
            log_mass_quantile_monte_carlo_se=np.std(endpoints,axis=0,ddof=1)),
            dict(method='Stratified independent pulse weighted-likelihood posterior bootstrap',
                parameter_count=7, nuisance_scatter_parameters=0,
                original_parametric_posterior=False, receiver_noise_unchanged=True,
                inter_pulse_correlation_assumed=False, manual_flag_revision=revision,
                numerical_checks_passed=bool(checks), model_discrepancy_validated=False,
                interval_usable=False, prior_boundary_fraction=float(np.mean(boundary)),
                density_kg_m3=3000.,Gamma=1.,sigma_s2_m2=1e-8,
                source_input_sha256=input_hash,
                library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
                source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                elapsed_seconds=time.monotonic()-began,
                uncertainty_scope='Independent pulse variation of physical fit; not shared systematic density, drag, ablation or calibration errors'))
        print('POSTERIOR',event,'checks',checks,'mass95',np.quantile(masses,[.025,.5,.975]),
              'boundary_fraction',np.mean(boundary),'repeat_max',np.max(diagnostics[:,2]),
              'seconds',time.monotonic()-began,flush=True)
    cache.close()
    lib.voltage_free(model.handle)


if __name__ == '__main__':
    main()
