import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import fit_gcrs_trajectories_lfm_ambiguity as gfit
import test_rank02_range_interpolation as interp


SCRIPT_VERSION = "v20260618a"
DEFAULT_OUTPUT_BASE = os.path.join("results", f"lfm_sampling_convention_audit_{SCRIPT_VERSION}")


def synthesize_row(gate, fd_hz, n_samples, sr_mhz, bw_mhz, pulse_length_us, amp=1.0):
    code, t_s = interp.lfm(length_us=pulse_length_us, sr_mhz=sr_mhz, bandwidth_hz=bw_mhz * 1e6)
    row = np.zeros(n_samples, dtype=np.complex128)
    start = int(round(gate)) - len(code) // 2
    stop = start + len(code)
    if start < 0 or stop > n_samples:
        raise ValueError("synthetic echo outside row")
    row[start:stop] += amp * code.astype(np.complex128) * np.exp(1j * 2.0 * np.pi * fd_hz * t_s)
    return row, code


def matched_peak(row, code, method):
    if method == "np_convolve":
        corr = np.convolve(row, np.conj(code), mode="same")
    elif method == "fftconvolve":
        corr = sig.fftconvolve(row, np.conj(code), mode="same")
    else:
        raise ValueError(method)
    power = np.abs(corr) ** 2.0
    idx = int(np.argmax(power))
    delta = 0.0
    if 0 < idx < len(power) - 1:
        ym1, y0, yp1 = float(power[idx - 1]), float(power[idx]), float(power[idx + 1])
        denom = ym1 - 2.0 * y0 + yp1
        if denom < 0.0:
            delta = float(np.clip(0.5 * (ym1 - yp1) / denom, -0.5, 0.5))
    return float(idx) + delta


def fft_beat(row, gate, sr_mhz, bw_mhz, pulse_length_us, zero_pad_factor=64, gate_upsample_factor=1, center_offset=0, code_roll=0):
    if gate_upsample_factor > 1:
        row = sig.resample_poly(row, gate_upsample_factor, 1).astype(np.complex128)
        sr_mhz = float(sr_mhz) * gate_upsample_factor
        center = int(round(float(gate) * gate_upsample_factor)) + int(center_offset)
    else:
        row = np.asarray(row, dtype=np.complex128)
        center = int(round(float(gate))) + int(center_offset)
    code, _t_s = interp.lfm(length_us=pulse_length_us, sr_mhz=sr_mhz, bandwidth_hz=bw_mhz * 1e6)
    if code_roll:
        code = np.roll(code, int(code_roll))
    n_code = len(code)
    start = center - n_code // 2
    stop = start + n_code
    segment = row[start:stop]
    y = segment * np.conj(code.astype(np.complex128)) * np.hanning(n_code)
    n_fft = 1
    while n_fft < zero_pad_factor * n_code:
        n_fft *= 2
    sr_hz = sr_mhz * 1e6
    spec = np.fft.fftshift(np.fft.fft(y, n=n_fft))
    freq = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sr_hz))
    power = 10.0 * np.log10(np.maximum(np.abs(spec) ** 2.0, 1e-300))
    idx = int(np.argmax(power))
    delta = 0.0
    if 0 < idx < len(power) - 1:
        ym1, y0, yp1 = map(float, power[idx - 1 : idx + 2])
        denom = ym1 - 2.0 * y0 + yp1
        if abs(denom) > 1e-30:
            delta = float(np.clip(0.5 * (ym1 - yp1) / denom, -1.0, 1.0))
    return float(freq[idx] + delta * (freq[1] - freq[0]))


def run_audit(output_base):
    sr_mhz = 4.0
    bw_mhz = 4.0
    pulse_us = 199.0
    n_samples = 4096
    gate0 = 1800
    fd_values = np.linspace(-100e3, 100e3, 41)
    rows = []
    offset_rows = []
    for fd in fd_values:
        row, code = synthesize_row(gate0, fd, n_samples, sr_mhz, bw_mhz, pulse_us)
        np_gate = matched_peak(row, code, "np_convolve")
        fft_gate = matched_peak(row, code, "fftconvolve")
        beat_at_np = fft_beat(row, np_gate, sr_mhz, bw_mhz, pulse_us, gate_upsample_factor=16)
        beat_at_fft = fft_beat(row, fft_gate, sr_mhz, bw_mhz, pulse_us, gate_upsample_factor=16)
        rows.append((fd, np_gate, fft_gate, beat_at_np, beat_at_fft))
    test_fd = 50e3
    row, code = synthesize_row(gate0, test_fd, n_samples, sr_mhz, bw_mhz, pulse_us)
    gate = matched_peak(row, code, "np_convolve")
    for center_offset in range(-3, 4):
        for code_roll in range(-3, 4):
            beat = fft_beat(
                row,
                gate,
                sr_mhz,
                bw_mhz,
                pulse_us,
                gate_upsample_factor=16,
                center_offset=center_offset,
                code_roll=code_roll,
            )
            offset_rows.append((center_offset, code_roll, beat))
    rows = np.asarray(rows, dtype=np.float64)
    offset_rows = np.asarray(offset_rows, dtype=np.float64)
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    with h5py.File(f"{output_base}.h5", "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["chirp_rate_hz_per_s"] = gfit.CHIRP_RATE_HZ_PER_S
        h.attrs["sr_mhz"] = sr_mhz
        h.attrs["bw_mhz"] = bw_mhz
        h.attrs["pulse_length_us"] = pulse_us
        h.attrs["injected_gate"] = gate0
        h["fd_injected_hz"] = rows[:, 0]
        h["np_convolve_gate"] = rows[:, 1]
        h["fftconvolve_gate"] = rows[:, 2]
        h["beat_at_np_gate_hz"] = rows[:, 3]
        h["beat_at_fft_gate_hz"] = rows[:, 4]
        h["offset_code_roll_test"] = offset_rows

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(rows[:, 0] / 1e3, rows[:, 1] - gate0, label="np.convolve same")
    axes[0].plot(rows[:, 0] / 1e3, rows[:, 2] - gate0, label="fftconvolve same", ls="--")
    axes[0].set_ylabel("Gate bias (samples)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(rows[:, 0] / 1e3, rows[:, 3] / 1e3, label="beat at np gate")
    axes[1].plot(rows[:, 0] / 1e3, rows[:, 4] / 1e3, label="beat at fft gate", ls="--")
    axes[1].axhline(0, color="0.2", lw=1.0)
    axes[1].set_xlabel("Injected Doppler (kHz)")
    axes[1].set_ylabel("Residual beat (kHz)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle("LFM sampling convention audit")
    fig.savefig(f"{output_base}.png", dpi=220)
    fig.savefig(f"{output_base}.pdf")
    plt.close(fig)
    print(f"output_h5={output_base}.h5")
    print(f"output_png={output_base}.png")
    print(f"gate_bias_np_samples_range={np.min(rows[:,1]-gate0):.6f},{np.max(rows[:,1]-gate0):.6f}")
    print(f"gate_bias_fft_samples_range={np.min(rows[:,2]-gate0):.6f},{np.max(rows[:,2]-gate0):.6f}")
    print(f"beat_at_np_gate_hz_rms={np.sqrt(np.mean(rows[:,3]**2)):.3f}")
    print(f"beat_at_fft_gate_hz_rms={np.sqrt(np.mean(rows[:,4]**2)):.3f}")
    best = offset_rows[np.argmin(np.abs(offset_rows[:, 2]))]
    print(f"best_center_offset_code_roll_beat_hz={best[0]:.0f},{best[1]:.0f},{best[2]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Audit LFM matched-filter and FFT beat sampling conventions.")
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    args = parser.parse_args()
    run_audit(args.output_base)


if __name__ == "__main__":
    main()
