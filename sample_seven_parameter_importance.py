"""Independent native importance draws of the seven-parameter voltage posterior."""
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

from fit_cached_shrinking_voltage import native, Cached, covariance
from fit_shrinking_voltage import Model, PTR, save


def quantiles(values, weights):
    valid = weights > 0
    values, weights = values[valid], weights[valid]
    indices = np.argsort(values)
    w = weights[indices]
    cdf = (np.cumsum(w)-.5*w)/np.sum(w)
    return np.interp([.025, .5, .975], cdf, values[indices])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "flags", "output", "library"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--particles", type=int, default=131072)
    a = parser.parse_args()
    world = MPI.COMM_WORLD
    event_index = world.rank % len(a.events)
    comm = world.Split(event_index, world.rank)
    event = a.events[event_index]
    source, output = a.source/event, a.output/event
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    with h5py.File(source/"input.h5") as h:
        data = {k: h[k][()] for k in h}
        revision = int(h.attrs.get("manual_flag_revision", 0))
    flag = a.flags/(event+".h5")
    if flag.exists():
        with h5py.File(flag) as h:
            if revision != int(h.attrs.get("version", 0)):
                raise ValueError("Measurement flags differ from fitted input")
            excluded = set(h["excluded"].asstr()[:])
        if excluded & {f"{int(s)}:{int(t)}" for s, t in zip(data["station"], data["time_ns"])}:
            raise ValueError("Flagged measurements remain in input")
    with h5py.File(source/"fit.h5") as h:
        seed = np.ascontiguousarray(h["params"][:])
    if seed.shape != (7,):
        raise ValueError("Exactly seven physical parameters required")
    lib = native(a.library)
    model = Model(lib, source, data)
    cache = Cached(model, seed, 0, 1., output/f"cache_{comm.rank}.h5")
    metric = np.ascontiguousarray(comm.bcast(covariance(cache, seed) if comm.rank == 0 else None))
    lib.voltage_importance_sample.argtypes = [ct.c_void_p, ct.c_void_p, PTR, PTR,
        ct.c_size_t, ct.c_uint64, ct.c_char_p]
    lib.voltage_importance_sample.restype = ct.c_int
    draws = (a.particles+comm.size-1)//comm.size
    path = output/f"particles_{comm.rank:03d}.h5"
    ok = lib.voltage_importance_sample(cache.handle, model.handle,
        seed.ctypes.data_as(PTR), metric.ctypes.data_as(PTR), draws,
        202611080000+event_index*10000+comm.rank, os.fsencode(str(path)+".partial"))
    if not ok:
        raise RuntimeError("Native independent sampling failed")
    os.replace(str(path)+".partial", path)
    with h5py.File(path) as h:
        x = h["samples"][:]
        logw = h["log_weights"][:]
    probes = np.flatnonzero(np.isfinite(logw))
    probes = probes[np.linspace(0, len(probes)-1, min(4, len(probes)), dtype=int)]
    parity = max([abs(cache.logp(x[i])-cache.logp(x[i], True)) for i in probes] or [np.inf])
    checks = comm.gather(parity, root=0)
    comm.Barrier()
    if comm.rank == 0:
        arrays, log_weights, labels = [], [], []
        for rank in range(comm.size):
            with h5py.File(output/f"particles_{rank:03d}.h5") as h:
                arrays.append(h["samples"][:])
                log_weights.append(h["log_weights"][:])
                labels.append(np.full(len(h["log_weights"]), rank % 4))
        x, logw, labels = np.concatenate(arrays), np.concatenate(log_weights), np.concatenate(labels)
        total = len(logw)
        normalizer = logsumexp(logw)
        weights = np.exp(logw-normalizer)
        effective = 1./np.sum(weights**2)
        batch_z, batch_radius = [], []
        for batch in range(4):
            selected = labels == batch
            batch_z.append(logsumexp(logw[selected])-np.log(np.sum(selected)))
            batch_radius.append(quantiles(x[selected, 6], weights[selected]))
        batch_radius = np.asarray(batch_radius)
        radius = quantiles(x[:, 6], weights)
        mc_error = np.std(batch_radius, axis=0, ddof=1)/2.
        verified = (effective >= 10000 and np.max(weights) < .005 and max(checks) < .01
                    and np.std(batch_z, ddof=1) < .02
                    and np.max(mc_error)/(radius[2]-radius[0]) < .01)
        valid = weights > 0
        draws_valid, w_valid = x[valid], weights[valid]
        mean = np.sum(draws_valid*w_valid[:, None], axis=0)
        centered = draws_valid-mean
        posterior_covariance = centered.T@(centered*w_valid[:, None])
        attrs = dict(parameter_count=7, sampled_scatter_parameters=0,
            method="Independent defensive importance sampling of the voltage posterior",
            prior="Uniform position/velocity and log10 radius, log10 radius in [-7,-2]",
            receiver_noise_unchanged=True, inter_pulse_correlation_assumed=False,
            material_density_kg_m3=3000.,Gamma=1.,sigma_s2_m2=1e-8,
            convergence_verified=bool(verified), importance_ess=effective,
            maximum_normalized_weight=float(np.max(weights)),
            maximum_exact_likelihood_difference=max(checks),
            log_normalizer=normalizer-np.log(total),
            batch_log_normalizer_std=float(np.std(batch_z, ddof=1)),
            elapsed_seconds=time.monotonic()-began,
            source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
            production_products_unchanged=True,
            uncertainty_scope="Conditional on fixed drag, density, ablation and receiver calibration; not total physical uncertainty")
        save(output/"posterior.h5", dict(samples=x, log_weights=logw, weights=weights,
            posterior_mean=mean, posterior_covariance=posterior_covariance,
            log_radius_quantiles=radius, log_radius_quantile_mc_se=mc_error,
            radius_m_quantiles=10**radius,
            mass_kg_quantiles=4*np.pi/3*3000*10**(3*radius),
            batch_log_radius_quantiles=batch_radius,batch_log_normalizer=batch_z), attrs)
        print("IMPORTANCE", event, "verified", verified, "ESS", effective,
              "mass95", 4*np.pi/3*3000*10**(3*radius),
              "elapsed", time.monotonic()-began, flush=True)
    cache.close()
    lib.voltage_free(model.handle)
    world.Barrier()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
