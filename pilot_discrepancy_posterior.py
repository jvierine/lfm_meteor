"""MPI pilot: seven physical parameters, marginalized joint-measurement scatter.

The Gaussian compression of each pulse likelihood is experimental. These outputs
are not accepted mass estimates until quadrature, subset and coverage tests pass.
Python only orchestrates native Rust likelihood, optimization and posterior draws.
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
from scipy.special import logsumexp

from fit_cached_shrinking_voltage import native
from fit_shrinking_voltage import Model, PTR, save
from sample_seven_parameter_importance import quantiles


def metric(logp, q, base, step=.25):
    factor = np.linalg.cholesky(base)
    origin = logp(q)
    information = np.empty((7, 7))
    for j in range(7):
        a = step*factor[:, j]
        information[j, j] = -(logp(q+a)-2*origin+logp(q-a))/step**2
        for k in range(j):
            b = step*factor[:, k]
            information[j, k] = information[k, j] = -(logp(q+a+b)-logp(q+a-b)
                -logp(q-a+b)+logp(q-a-b))/(4*step**2)
    if not np.isfinite(information).all():
        if step > 1e-5:
            return metric(logp, q, base, step/2)
        raise ValueError('Invalid pilot curvature; no posterior precision claim')
    eigenvalues, vectors = np.linalg.eigh(information)
    # Proposal stabilization only. Importance weights use the unmodified target.
    proposal = factor@vectors@np.diag(1/np.maximum(eigenvalues, max(eigenvalues.max()*1e-5, 1e-6)))@vectors.T@factor.T
    return np.ascontiguousarray((proposal+proposal.T)/2), eigenvalues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['source', 'audit', 'posterior', 'output', 'library']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--events', nargs='+', required=True)
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--draws', type=int, default=32768)
    a = parser.parse_args()
    world = MPI.COMM_WORLD
    group = world.rank % len(a.events)
    comm = world.Split(group, world.rank)
    event = a.events[group]
    output = a.output/event
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    source = a.source/event
    with h5py.File(source/'input.h5') as h:
        data = {k: h[k][()] for k in h}
    with h5py.File(a.audit/event/'pulse'/'pulse_audit.h5') as h:
        pulse = np.ascontiguousarray(h['independent_pulse_fit'][:])
        np.testing.assert_array_equal(h['time_ns'][:], data['time_ns'])
    with h5py.File(a.posterior/event/'posterior.h5') as h:
        q = np.ascontiguousarray(h['proposal_center'][:])
        base = np.ascontiguousarray(h['posterior_covariance'][:])
    lib = native(a.library)
    model = Model(lib, source, data)
    lib.voltage_discrepancy_new.argtypes = [ct.c_void_p, PTR, ct.POINTER(ct.c_uint64), ct.c_size_t, ct.c_size_t]
    lib.voltage_discrepancy_new.restype = ct.c_void_p
    lib.voltage_discrepancy_logp.argtypes = [ct.c_void_p, ct.c_void_p, PTR]
    lib.voltage_discrepancy_logp.restype = ct.c_double
    lib.voltage_discrepancy_free.argtypes = [ct.c_void_p]
    station = np.ascontiguousarray(data['station'], dtype=np.uint64)
    def make(nodes):
        handle = lib.voltage_discrepancy_new(model.handle, pulse.ctypes.data_as(PTR),
            station.ctypes.data_as(ct.POINTER(ct.c_uint64)), len(station), nodes)
        if not handle:
            raise ValueError('Invalid independent pulse maximum/covariance')
        return handle
    handle = make(a.nodes)
    def logp(q):
        q = np.ascontiguousarray(q)
        return lib.voltage_discrepancy_logp(handle, model.handle, q.ctypes.data_as(PTR))
    if comm.rank == 0:
        lib.voltage_discrepancy_optimize.argtypes = [ct.c_void_p,ct.c_void_p,PTR,PTR,ct.c_double,PTR]
        for _ in range(3):
            proposal, eig = metric(logp, q, base*100)
            trial = np.empty(7)
            ok = lib.voltage_discrepancy_optimize(handle, model.handle, q.ctypes.data_as(PTR),
                proposal.ctypes.data_as(PTR), 8., trial.ctypes.data_as(PTR))
            if not ok:
                raise ValueError('Pilot optimization failed')
            q = trial
        proposal, eig = metric(logp, q, proposal)
        save(output/'proposal.h5', dict(params=q, covariance=proposal, curvature_eigenvalues=eig),
             dict(pilot_only=True, model_discrepancy_validated=False))
        print('PILOT MODE', event, q[6], logp(q), flush=True)
    else:
        proposal = None
    q = np.ascontiguousarray(comm.bcast(q))
    proposal = np.ascontiguousarray(comm.bcast(proposal))
    lib.voltage_discrepancy_sample.argtypes = [ct.c_void_p,ct.c_void_p,PTR,PTR,ct.c_size_t,ct.c_uint64,ct.c_char_p]
    file = output/f'particles_{comm.rank:03d}.h5'
    seed = int.from_bytes(hashlib.sha256(f'{event}:{comm.rank}'.encode()).digest()[:8], 'little')
    if not lib.voltage_discrepancy_sample(handle, model.handle, q.ctypes.data_as(PTR),
            proposal.ctypes.data_as(PTR), a.draws, seed, os.fsencode(str(file)+'.partial')):
        raise RuntimeError('Native pilot sampling failed')
    os.replace(str(file)+'.partial', file)
    comm.Barrier()
    if comm.rank == 0:
        arrays, logs = [], []
        for rank in range(comm.size):
            with h5py.File(output/f'particles_{rank:03d}.h5') as h:
                arrays.append(h['samples'][:]); logs.append(h['log_weights'][:])
        x, logw = np.concatenate(arrays), np.concatenate(logs)
        w = np.exp(logw-logsumexp(logw))
        radius = quantiles(x[:, 6], w)
        fine = make(2*a.nodes)
        indices = np.random.default_rng(17).choice(len(x), min(64,len(x)), p=w)
        differences = np.array([lib.voltage_discrepancy_logp(fine, model.handle,
            np.ascontiguousarray(x[i]).ctypes.data_as(PTR))-logp(x[i]) for i in indices])
        lib.voltage_discrepancy_free(fine)
        mass = 4*np.pi/3*3000*10**(3*radius)
        save(output/'posterior.h5', dict(samples=x, log_weights=logw, weights=w,
            log_radius_quantiles=radius, mass_kg_quantiles=mass,
            quadrature_log_difference=differences, params=q, proposal_covariance=proposal),
            dict(pilot_only=True, model_discrepancy_validated=False,
                parameter_count=7, sampled_scatter_parameters=0,
                marginalized_scatter_scales=6, thermal_covariance_unchanged=True,
                inter_pulse_correlation_assumed=False,
                likelihood='Joint Gaussian pulse compression plus independent additive path/Doppler discrepancy; station scales marginalized',
                scatter_prior='Independent half Student t(3); path scale c/B, Doppler scale 1/T; numerical integration over 8e-7 to 100 resolution units',
                importance_ess=1/np.sum(w*w), maximum_weight=w.max(),
                quadrature_relative_log_range=np.ptp(differences),
                elapsed_seconds=time.monotonic()-began,
                source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        print('PILOT POSTERIOR', event, 'mass95', mass, 'ESS', 1/np.sum(w*w),
              'quadrature_log_range',np.ptp(differences),flush=True)
    lib.voltage_discrepancy_free(handle)
    lib.voltage_free(model.handle)


if __name__ == '__main__':
    main()
