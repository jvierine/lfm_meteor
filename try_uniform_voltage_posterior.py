"""Isolated four-chain posterior-only trial of existing physical trajectory fits."""
import argparse
import ctypes as ct
import hashlib
import os
from pathlib import Path
import shutil
import time
import traceback

import h5py
from mpi4py import MPI
import numpy as np

from fit_cached_shrinking_voltage import native, Cached, covariance
from fit_shrinking_voltage import Model, PTR, finish_export, save
from arviz_stats.base import array_stats


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "flags", "output", "library"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--events", nargs="+", required=True)
    p.add_argument("--draws", type=int, default=96000)
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--chain-seconds", type=float, default=65.)
    p.add_argument("--independent-scatter", action="store_true")
    a = p.parse_args()
    world = MPI.COMM_WORLD
    if world.size % 4:
        raise ValueError("Use four MPI ranks per event")
    group_index, chain_index = divmod(world.rank, 4)
    comm = world.Split(group_index, chain_index)
    groups = world.size // 4
    a.output.mkdir(parents=True, exist_ok=True)
    for event in a.events[group_index::groups]:
        source, folder = a.source / event, a.output / event
        folder.mkdir(parents=True, exist_ok=True)
        error, model, cache, lib = "", None, None, None
        began = time.monotonic()
        try:
            with h5py.File(source / "input.h5") as h:
                data = {k: h[k][()] for k in h}
                revision = int(h.attrs.get("manual_flag_revision", 0))
            flag = a.flags / (event + ".h5")
            if flag.exists():
                with h5py.File(flag) as h:
                    if revision != int(h.attrs.get("version", 0)):
                        raise ValueError("Current flags differ from fitted input; refit first")
                    excluded = set(h["excluded"].asstr()[:])
                retained = {f"{int(s)}:{int(t)}" for s, t in zip(data["station"], data["time_ns"])}
                if retained & excluded:
                    raise ValueError("Excluded measurements remain in fitted input")
            with h5py.File(source / "fit.h5") as h:
                q = h["params"][:]
            if q.shape != (7,):
                raise ValueError("Only direct shrinking-radius fits are permitted")
            provenance = {name: digest(source / name) for name in ("input.h5", "fit.h5", "plot.h5")}
            provenance["flags"] = digest(flag) if flag.exists() else "absent"
            work = folder / f"worker_{chain_index}"
            work.mkdir(exist_ok=True)
            shutil.copy2(source / "input.h5", work / "input.h5")
            lib = native(a.library)
            model = Model(lib, work, data)
            cache = Cached(model, q, 0, 1., work / "likelihood_cache.h5")
            cov = np.ascontiguousarray(covariance(cache, q))
            rng = np.random.default_rng(2026110800 + chain_index)
            start = q.copy()
            for _ in range(32):
                candidate = q + rng.multivariate_normal(np.zeros(7), cov) * .5
                if np.isfinite(cache.logp(candidate)) and cache.logp(candidate) > cache.logp(q) - 20.:
                    start = candidate
                    break
            start = np.ascontiguousarray(start)
            target = folder / f"chain_{chain_index}.h5"
            if a.independent_scatter:
                station = np.ascontiguousarray(data["station"], dtype=np.uint64)
                lib.voltage_scatter_sample.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64),
                    PTR, PTR, ct.c_size_t, ct.c_size_t, ct.c_double, ct.c_uint64, ct.c_char_p]
                lib.voltage_scatter_sample.restype = ct.c_int
                ok = lib.voltage_scatter_sample(model.handle,
                    station.ctypes.data_as(ct.POINTER(ct.c_uint64)), start.ctypes.data_as(PTR),
                    cov.ctypes.data_as(PTR), a.warmup, a.draws, a.chain_seconds,
                    2026110800 + chain_index, os.fsencode(str(target) + ".partial"))
            else:
                ok = lib.voltage_cache_sample(cache.handle, model.handle,
                    start.ctypes.data_as(PTR), cov.ctypes.data_as(PTR), a.warmup,
                    a.draws, a.chain_seconds, 2026110800 + chain_index,
                    os.fsencode(str(target) + ".partial"))
            if not ok:
                raise RuntimeError("Native posterior sampler failed")
            os.replace(str(target) + ".partial", target)
            with h5py.File(target) as h:
                draws = h["samples"][:]
            probes = draws[np.linspace(0, len(draws)-1, min(64, len(draws)), dtype=int)]
            differences = np.array([cache.logp(x)-cache.logp(x, True) for x in probes])
            differences -= cache.logp(q)-cache.logp(q, True)
            parity = float(np.max(np.abs(differences)))
            if len(draws) < 1000 or parity > .2:
                raise RuntimeError(f"Insufficient draws or likelihood parity failure: n={len(draws)}, error={parity}")
            save(folder / f"chain_validation_{chain_index}.h5", dict(delta_logp_error=differences),
                 dict(passed=True, draws=len(draws), elapsed_seconds=time.monotonic()-began))
            print("CHAIN", event, chain_index, len(draws), parity, flush=True)
        except Exception:
            error = traceback.format_exc()
            (folder / f"error_chain_{chain_index}.log").write_text(error)
            print("ERROR", event, chain_index, error, flush=True)
        errors = comm.allgather(error)
        if comm.rank == 0:
            try:
                if any(errors):
                    raise RuntimeError("At least one chain failed; no interval promoted")
                arrays = []
                for i in range(4):
                    with h5py.File(folder / f"chain_{i}.h5") as h:
                        arrays.append(h["samples"][:])
                n = min(map(len, arrays))
                chains = np.stack([x[:n] for x in arrays])
                with h5py.File(source / "plot.h5") as h:
                    fields = {k: h[k][()] for k in h}
                    old_attrs = dict(h.attrs)
                old_mass = fields["initial_mass_kg_quantiles"].copy()
                old_method = str(old_attrs.get("uncertainty_method", "thermal posterior"))
                attrs = {k: v for k, v in old_attrs.items()
                         if not k.startswith(("uncertainty_", "posterior_"))}
                attrs.update(uncertainty_method="Metropolis-Hastings posterior of joint complex-voltage likelihood",
                             prior="Uniform position, velocity, log10 radius; 1e-7 < radius < 1e-2 m",
                             likelihood="Unmodified cached voltage likelihood, checked against exact voltage calculation; receiver noise fixed",
                             source_hashes_json=__import__("json").dumps(provenance),
                             library_sha256=digest(a.library),
                             source_git_commit=__import__("subprocess").check_output(
                                 ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True).strip(),
                             uncertainty_updated_unix=time.time())
                finish_export(folder, model, fields, attrs, chains)
                if a.independent_scatter:
                    scatter = []
                    for i in range(4):
                        with h5py.File(folder / f"chain_{i}.h5") as h:
                            scatter.append(h["scatter_samples"][:n])
                    scatter = np.stack(scatter)
                    statistics = np.log(scatter).transpose(2, 0, 1)
                    rh = float(np.max(array_stats.rhat(statistics, chain_axis=1, draw_axis=2)))
                    bulk = float(np.min(array_stats.ess(statistics, chain_axis=1, draw_axis=2)))
                    tail = float(np.min(array_stats.ess(statistics, method="tail", prob=.025, chain_axis=1, draw_axis=2)))
                    nuisance_pass = rh < 1.01 and min(bulk, tail) >= 400
                    with h5py.File(folder / "plot.h5", "r+") as h:
                        h.attrs.update(uncertainty_method="Metropolis-Hastings posterior with independent per-pulse delay/Doppler scatter",
                            likelihood="Exact sampled-voltage likelihood plus independent Gaussian physical delay/Doppler deviations; receiver-noise variance unchanged",
                            scatter_prior="Independent half-Student-t(3) scales: c/bandwidth metres and 1/pulse_duration Hz",
                            pulse_correlation_assumed=False, scatter_max_rhat=rh,
                            scatter_min_bulk_ess=bulk, scatter_min_tail_ess=tail,
                            scatter_converged=nuisance_pass,
                            uncertainty_converged=bool(h.attrs["uncertainty_converged"]) and nuisance_pass)
                        h["scatter_sigma_quantiles"] = np.quantile(scatter.reshape(-1, 6), [.025, .5, .975], axis=0)
                    save(folder / "scatter_posterior.h5", dict(samples=scatter),
                         dict(columns="Sanya path_m,doppler_Hz,Danzhou path_m,doppler_Hz,Wenchang path_m,doppler_Hz",
                              converged=nuisance_pass, max_rhat=rh, min_bulk_ess=bulk, min_tail_ess=tail))
                    with h5py.File(folder / "dynamic_mass_posterior.h5", "r+") as h:
                        h.attrs["convergence_verified"] = bool(h.attrs["convergence_verified"]) and nuisance_pass
                with h5py.File(folder / "plot.h5") as h:
                    result = dict(h.attrs)
                    quantiles = h["initial_mass_kg_quantiles"][:]
                save(folder / "comparison.h5", dict(previous_mass_95=old_mass, posterior_mass_95=quantiles),
                     dict(previous_method=old_method, posterior_converged=result["uncertainty_converged"],
                          rhat=result["posterior_max_rhat"], bulk_ess=result["posterior_min_bulk_ess"],
                          tail_ess=result["posterior_min_tail_ess"],
                          elapsed_seconds=time.monotonic()-began, production_products_unchanged=True))
                print("RESULT", event, quantiles, result["uncertainty_converged"], result["posterior_max_rhat"], flush=True)
            except Exception:
                error = traceback.format_exc()
                (folder / "error.log").write_text(error)
                print("ERROR", event, error, flush=True)
        comm.Barrier()
        if cache is not None:
            cache.close()
        if model is not None:
            lib.voltage_free(model.handle)
    world.Barrier()
    if world.rank == 0:
        save(a.output / "finished.h5", {}, dict(finished_unix=time.time(),
             meaning="All tasks ended; inspect per-event comparison and error files before accepting intervals"))


if __name__ == "__main__":
    main()
