"""MPI orchestration of native seven-parameter posterior quadrature, no scatter model."""
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
from scipy.integrate import cumulative_trapezoid

from fit_cached_shrinking_voltage import native, Cached, covariance
from fit_shrinking_voltage import Model, PTR, save


def summary(radius, log_density):
    density = np.exp(log_density - np.max(log_density))
    cdf = np.r_[0., cumulative_trapezoid(density, radius)]
    norm = cdf[-1]
    density /= norm
    cdf /= norm
    return dict(density=density, cdf=cdf,
                log_normalizer=np.max(log_density)+np.log(norm),
                log_radius_quantiles=np.interp([.025, .5, .975], cdf, radius))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "flags", "output", "library"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--events", nargs="+", required=True)
    p.add_argument("--orders", type=int, nargs="+", default=[3, 5])
    p.add_argument("--radius-nodes", type=int, default=81)
    p.add_argument("--optimize-seconds", type=float, default=1.)
    a = p.parse_args()
    world = MPI.COMM_WORLD
    index = world.rank % len(a.events)
    comm = world.Split(index, world.rank)
    event = a.events[index]
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
                raise ValueError("Flags changed: refit the current measurement selection first")
            excluded = set(h["excluded"].asstr()[:])
        retained = {f"{int(s)}:{int(t)}" for s, t in zip(data["station"], data["time_ns"])}
        if retained & excluded:
            raise ValueError("Excluded measurements remain in the input")
    with h5py.File(source/"fit.h5") as h:
        seed = np.ascontiguousarray(h["params"][:])
    if seed.shape != (7,):
        raise ValueError("Exactly seven physical parameters required")
    worker = output/f"worker_{comm.rank}"
    worker.mkdir(exist_ok=True)
    lib = native(a.library)
    model = Model(lib, worker, data)
    cache = Cached(model, seed, 0, 1., worker/"cache.h5")
    metric = comm.bcast(covariance(cache, seed) if comm.rank == 0 else None)
    metric = np.ascontiguousarray(metric)
    lib.voltage_radius_grid_point.argtypes = [ct.c_void_p, ct.c_void_p, PTR, PTR,
        ct.c_double, ct.c_size_t, ct.c_double, ct.c_double, ct.c_char_p]
    lib.voltage_radius_grid_point.restype = ct.c_int
    lib.voltage_radius_grid_integrate.argtypes = [ct.c_void_p, ct.c_void_p,
        ct.c_char_p, ct.c_size_t, ct.c_double, ct.c_char_p]
    lib.voltage_radius_grid_integrate.restype = ct.c_int
    profiles = output/"profiles"
    profiles.mkdir(exist_ok=True)

    def profile_path(r):
        return profiles/f"r_{r:+.12f}.h5"

    def profile(points):
        for r in points[comm.rank::comm.size]:
            path = profile_path(r)
            if path.exists():
                continue
            partial = str(path)+".partial"
            ok = lib.voltage_radius_grid_point(cache.handle, model.handle,
                seed.ctypes.data_as(PTR), metric.ctypes.data_as(PTR), float(r), 1,
                1., a.optimize_seconds, os.fsencode(partial))
            if not ok:
                raise RuntimeError(f"Conditional grid calculation failed: {event} r={r}")
            os.replace(partial, path)
        comm.Barrier()
        rows = []
        for r in points:
            with h5py.File(profile_path(r)) as h:
                rows.append(float(h["log_marginal"][()]))
        return np.asarray(rows)

    sigma = float(np.sqrt(metric[6, 6]))
    initial = np.unique(np.round(np.r_[np.linspace(-7, -2, 41),
        np.clip(seed[6]+np.linspace(-8, 8, 17)*sigma, -7, -2), seed[6]], 12))
    screen = profile(initial)
    finite = np.isfinite(screen)
    if not np.any(finite):
        raise RuntimeError("No finite conditional likelihoods")
    active = np.flatnonzero(screen > np.max(screen[finite])-30.)
    lower = initial[max(0, active[0]-1)]
    upper = initial[min(len(initial)-1, active[-1]+1)]
    radius = np.unique(np.round(np.r_[initial, np.linspace(lower, upper, a.radius_nodes)], 12))
    profile_density = profile(radius)
    if comm.rank == 0:
        print("PROFILE", event, len(radius), "radius nodes", flush=True)
    orders = {}
    for order in a.orders:
        folder = output/f"order_{order}"
        folder.mkdir(exist_ok=True)
        for i in range(comm.rank, len(radius), comm.size):
            if not np.isfinite(profile_density[i]):
                continue
            r = radius[i]
            path = folder/f"r_{r:+.12f}.h5"
            if path.exists():
                continue
            partial = str(path)+".partial"
            ok = lib.voltage_radius_grid_integrate(cache.handle, model.handle,
                os.fsencode(profile_path(r)), order, 1., os.fsencode(partial))
            if not ok:
                raise RuntimeError(f"Quadrature failed: {event} r={r}, order={order}")
            os.replace(partial, path)
        comm.Barrier()
        if comm.rank == 0:
            log_density = np.full(len(radius), -np.inf)
            for i, r in enumerate(radius):
                path = folder/f"r_{r:+.12f}.h5"
                if path.exists():
                    with h5py.File(path) as h:
                        log_density[i] = h["log_marginal"][()]
            result = summary(radius, log_density)
            orders[order] = dict(log_density=log_density, **result)
            print("GRID", event, order, "mass95", 4*np.pi/3*3000*10**(3*result["log_radius_quantiles"]), flush=True)
            partial = output/"posterior.partial.h5"
            with h5py.File(partial, "w") as h:
                h.attrs.update(parameter_count=7, sampled_scatter_parameters=0,
                    method="Log-radius grid; native six-dimensional Gauss-Hermite integration of the voltage likelihood",
                    prior="Uniform position/velocity and log10 radius; log10 radius in [-7,-2]",
                    receiver_noise_unchanged=True, inter_pulse_correlation_assumed=False,
                    source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
                    elapsed_seconds=time.monotonic()-began,
                    status="Trial: quadrature, radius-grid and tail convergence must be checked before use",
                    production_products_unchanged=True)
                h["log10_radius_m"] = radius
                h["initial_seed"] = seed
                h["initial_covariance"] = metric
                for n, values in orders.items():
                    g = h.create_group(f"order_{n}")
                    for key, value in values.items():
                        g[key] = value
            partial.replace(output/"posterior.h5")
    cache.close()
    lib.voltage_free(model.handle)
    comm.Barrier()
    if comm.rank == 0:
        print("FINISHED", event, "elapsed", time.monotonic()-began, flush=True)
    world.Barrier()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
