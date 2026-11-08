"""Synthetic voltage ensembles with independent receiver noise and pulse scatter."""
import argparse
import ctypes as ct
from pathlib import Path
import shutil

import h5py
import numpy as np

from fit_cached_shrinking_voltage import native
from fit_shrinking_voltage import Model, PTR, save, C
from voltage_noise_covariance import colored_noise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "library"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--event", required=True)
    p.add_argument("--replicates", type=int, default=32)
    p.add_argument("--sigma-path", type=float, nargs=3, default=[5.,3.,3.])
    p.add_argument("--sigma-doppler", type=float, nargs=3, default=[300.,150.,150.])
    a = p.parse_args()
    source = a.source / a.event
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "flags").mkdir(exist_ok=True)
    with h5py.File(source / "input.h5") as h:
        data = {k: h[k][()] for k in h}
        attrs = dict(h.attrs)
    with h5py.File(source / "fit.h5") as h:
        truth = np.ascontiguousarray(h["params"][:])
        fit_attrs = dict(h.attrs)
    lib = native(a.library)
    lib.voltage_observables.argtypes = [ct.c_void_p, PTR, PTR]
    lib.voltage_observables.restype = ct.c_int
    model = Model(lib, source, data)
    observables = np.empty((len(data["time_s"]), 2))
    if not lib.voltage_observables(model.handle, truth.ctypes.data_as(PTR), observables.ctypes.data_as(PTR)):
        raise RuntimeError("Cannot obtain synthetic truth observables")
    def waveform(path, doppler):
        tau = data["sample_time_s"] - path[:, None] / C
        env = np.clip((tau + .5/data["fs"])*data["fs"], 0, 1) * np.clip(
            (data["duration"]-tau+.5/data["fs"])*data["fs"], 0, 1)
        phase = 2*np.pi*(data["bandwidth"]/2*tau-data["bandwidth"]/(2*data["duration"])*tau*tau
                         + doppler[:, None]*data["sample_time_s"])
        return env*np.exp(1j*phase)
    unit = waveform(observables[:, 0], observables[:, 1])
    prediction = model.prediction(truth)
    amplitudes = np.sum(np.conj(unit)*prediction, axis=1)/np.sum(abs(unit)**2, axis=1)
    sigma_path = np.array(a.sigma_path)[data["station"].astype(int)]
    sigma_doppler = np.array(a.sigma_doppler)[data["station"].astype(int)]
    for index in range(a.replicates):
        target = a.output / f"sim_{index:03d}"
        target.mkdir(exist_ok=True)
        rng = np.random.default_rng(20261108000+index)
        path_error = rng.normal(size=len(sigma_path))*sigma_path
        doppler_error = rng.normal(size=len(sigma_path))*sigma_doppler
        signal = amplitudes[:, None]*waveform(observables[:, 0]+path_error, observables[:, 1]+doppler_error)
        noise = np.array([colored_noise(rng, data["real"].shape[1],
                         -(data["whitener_real"][i, 1:]+1j*data["whitener_imag"][i, 1:]), data["noise_power"][i])
                          for i in range(len(signal))])
        simulated = dict(data, real=(signal+noise).real, imag=(signal+noise).imag)
        save(target / "input.h5", simulated, dict(attrs, manual_flag_revision=0, synthetic=True))
        start = truth.copy()
        start[:6] += rng.normal(size=6)*np.array([.002, .002, .002, .02, .02, .02])
        start[6] += rng.normal()*.02
        save(target / "fit.h5", dict(params=start), dict(fit_attrs, synthetic_seed_not_truth=True))
        shutil.copy2(source / "plot.h5", target / "plot.h5")
        save(target / "truth.h5", dict(params=truth, sigma_path_m=sigma_path,
             sigma_doppler_hz=sigma_doppler, path_error_m=path_error, doppler_error_hz=doppler_error),
             dict(seed=20261108000+index, receiver_noise_unchanged=True,
                  inter_pulse_correlation=False, source_program=Path(__file__).read_text()))
    lib.voltage_free(model.handle)
    print("Prepared", a.replicates, "independent simulated events", flush=True)


if __name__ == "__main__":
    main()
