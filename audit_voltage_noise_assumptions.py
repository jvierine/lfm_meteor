"""Measure off-echo receiver noise separately from on-echo model residuals."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import lfilter

from fit_cached_shrinking_voltage import native
from fit_shrinking_voltage import Model, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "library"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--events", nargs="+", required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    lib = native(a.library)
    for event in a.events:
        folder = a.source / event
        with h5py.File(folder / "input.h5") as h:
            d = {k: h[k][()] for k in h}
        with h5py.File(folder / "fit.h5") as h:
            q = h["params"][:]
        model = Model(lib, folder, d)
        pred = model.prediction(q)
        rows = d["real"] + 1j*d["imag"]
        white, residual, off_masks, fractions = [], [], [], []
        for i in range(len(rows)):
            filt = d["whitener_real"][i] + 1j*d["whitener_imag"][i]
            z = lfilter(filt, [1.], rows[i]) / np.sqrt(d["noise_power"][i])
            r = lfilter(filt, [1.], rows[i]-pred[i]) / np.sqrt(d["noise_power"][i])
            # Discard FIR startup and every row whose FIR support meets the echo.
            support = abs(pred[i]) > max(abs(pred[i])) * 1e-8
            contaminated = np.convolve(support.astype(int), np.ones(len(filt), dtype=int))[:len(z)] > 0
            off = ~contaminated
            off[:len(filt)-1] = False
            white.append(z)
            residual.append(r)
            off_masks.append(off)
            fractions.append(np.mean(abs(z[off])**2))
        white, residual, off_masks = map(np.asarray, (white, residual, off_masks))
        result = []
        for station in range(3):
            keep = d["station"] == station
            z, r, mask = white[keep], residual[keep], off_masks[keep]
            off = z[mask]
            correlations, pairs = [], []
            for lag in (1, 2, 4, 8):
                both = mask[:, lag:] & mask[:, :-lag]
                x, y = z[:, :-lag][both], z[:, lag:][both]
                correlations.append(np.vdot(x, y)/np.sqrt(np.vdot(x, x).real*np.vdot(y, y).real))
                pairs.append(len(x))
            both = mask[:-1] & mask[1:]
            x, y = z[:-1][both], z[1:][both]
            temporal = np.vdot(x, y)/np.sqrt(np.vdot(x, x).real*np.vdot(y, y).real)
            valid = np.ones_like(mask, dtype=bool)
            valid[:, :d["whitener_real"].shape[1]-1] = False
            echo = valid & ~mask
            row = dict(station=station, off_echo_samples=len(off),
                       measured_to_assumed_noise_power=float(np.mean(abs(off)**2)),
                       residual_to_assumed_power=float(np.mean(abs(r[valid])**2)),
                       on_echo_residual_to_assumed_power=float(np.mean(abs(r[echo])**2)),
                       off_echo_neighboring_pulse_correlation_abs=float(abs(temporal)),
                       off_echo_neighboring_pulse_pairs=len(x),
                       off_echo_sample_correlation_abs=np.abs(correlations).tolist(),
                       off_echo_sample_pair_counts=pairs)
            result.append(row)
        save(a.output / (event + ".h5"),
             dict(off_echo_noise_power_ratio_per_pulse=fractions),
             dict(summary_json=json.dumps(result),
                  scope="Independent receiver-noise audit; on-echo excess can reflect waveform or trajectory mismatch, not necessarily correlated receiver noise",
                  source_program=Path(__file__).read_text()))
        print(event, json.dumps(result), flush=True)
        lib.voltage_free(model.handle)


if __name__ == "__main__":
    main()
