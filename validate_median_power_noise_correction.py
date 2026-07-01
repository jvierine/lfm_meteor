#!/usr/bin/env python3
"""Validate median-power noise correction with LFM matched-filter simulations."""

from __future__ import annotations

import numpy as np


def lfm_code(n_samples: int = 796, bandwidth_hz: float = 4.0e6, sample_rate_hz: float = 4.0e6) -> np.ndarray:
    """Return a unit-amplitude down-chirp similar to the Sanya meteor code."""

    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    duration_s = n_samples / sample_rate_hz
    chirp_rate_hz_s = bandwidth_hz / duration_s
    phase_cycles = 0.5 * bandwidth_hz * t - 0.5 * chirp_rate_hz_s * t**2
    return np.exp(1j * 2.0 * np.pi * phase_cycles).astype(np.complex64)


def matched_filter(raw: np.ndarray, code: np.ndarray) -> np.ndarray:
    """Apply the same convention as the detector: convolution with conj(code)."""

    return np.convolve(raw, np.conj(code), mode="same")


def simulate_lfm_pulses(
    rng: np.random.Generator,
    code: np.ndarray,
    n_pulses: int = 3000,
    n_range: int = 4096,
    input_noise_power: float = 1.0,
    output_snr_power: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Simulate raw-voltage pulses with one code-shaped echo per pulse."""

    noise_scale = np.sqrt(input_noise_power / 2.0)
    raw = noise_scale * (
        rng.normal(size=(n_pulses, n_range)) + 1j * rng.normal(size=(n_pulses, n_range))
    )
    raw = raw.astype(np.complex64)

    code_energy = float(np.sum(np.abs(code) ** 2))
    # Matched-filter output signal power is |a|^2 E_code^2, while output noise
    # power is input_noise_power E_code.  Choose a for the requested output SNR.
    echo_amplitude = np.sqrt(output_snr_power * input_noise_power / code_energy)
    first_gate = len(code)
    last_gate = n_range - len(code) - 1
    gates = rng.integers(first_gate, last_gate, size=n_pulses)
    template = np.zeros(n_range, dtype=np.complex64)
    template_gate = n_range // 2
    template_start = template_gate - len(code) // 2
    template[template_start : template_start + len(code)] = code
    peak_offset = int(np.argmax(np.abs(matched_filter(template, code)) ** 2)) - template_gate
    for pulse_idx, gate in enumerate(gates):
        start = gate - len(code) // 2
        raw[pulse_idx, start : start + len(code)] += echo_amplitude * code

    mf = np.empty_like(raw)
    for pulse_idx in range(n_pulses):
        mf[pulse_idx] = matched_filter(raw[pulse_idx], code)
    return mf, gates + peak_offset, peak_offset


def main() -> None:
    rng = np.random.default_rng(20260629)
    n = 5_000_000

    z = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2.0)
    complex_power = np.abs(z) ** 2
    complex_mean = float(np.mean(complex_power))
    complex_median = float(np.median(complex_power))
    complex_corrected = complex_median / np.log(2.0)

    x = rng.normal(size=n)
    real_power = x**2
    real_mean = float(np.mean(real_power))
    real_median = float(np.median(real_power))
    real_median_factor = 0.454936423119572
    real_corrected = real_median / real_median_factor

    print("Complex circular Gaussian noise distribution")
    print(f"  E[|z|^2]                         = {complex_mean:.6f}")
    print(f"  median(|z|^2)                    = {complex_median:.6f}")
    print(f"  median(|z|^2) / E[|z|^2]         = {complex_median / complex_mean:.6f}")
    print(f"  ln(2)                            = {np.log(2.0):.6f}")
    print(f"  median(|z|^2) / ln(2)            = {complex_corrected:.6f}")
    print(f"  corrected / E[|z|^2]             = {complex_corrected / complex_mean:.6f}")
    print()
    print("Real Gaussian noise, for comparison")
    print(f"  E[x^2]                           = {real_mean:.6f}")
    print(f"  median(x^2)                      = {real_median:.6f}")
    print(f"  median(x^2) / E[x^2]             = {real_median / real_mean:.6f}")
    print(f"  [Phi^-1(0.75)]^2                 = {real_median_factor:.6f}")
    print(f"  median(x^2) / [Phi^-1(0.75)]^2   = {real_corrected:.6f}")
    print(f"  corrected / E[x^2]               = {real_corrected / real_mean:.6f}")
    print()

    code = lfm_code()
    code_energy = float(np.sum(np.abs(code) ** 2))
    expected_noise_power = code_energy
    requested_snr = 100.0
    mf, gates, peak_offset = simulate_lfm_pulses(rng, code, output_snr_power=requested_snr)
    half_guard = len(code)
    valid = np.zeros(mf.shape[1], dtype=bool)
    valid[len(code) : mf.shape[1] - len(code)] = True

    noise_mean_by_pulse = []
    noise_median_by_pulse = []
    snr_est_by_pulse = []
    snr_at_true_peak_by_pulse = []
    peak_gate_error = []
    for pulse_idx, gate in enumerate(gates):
        power = np.abs(mf[pulse_idx]) ** 2
        mask = valid.copy()
        mask[max(0, gate - half_guard) : min(power.size, gate + half_guard + 1)] = False
        noise_power = power[mask]
        median_power = float(np.median(noise_power))
        noise_mean_by_pulse.append(float(np.mean(noise_power)))
        noise_median_by_pulse.append(median_power)
        corrected = median_power / np.log(2.0)
        snr_est_by_pulse.append(float(np.max(power[valid]) / corrected))
        snr_at_true_peak_by_pulse.append(float(power[int(gate)] / corrected))
        peak_gate_error.append(int(np.argmax(np.where(valid, power, -np.inf))) - int(gate))

    noise_mean = float(np.mean(noise_mean_by_pulse))
    noise_median = float(np.mean(noise_median_by_pulse))
    corrected_noise = noise_median / np.log(2.0)
    snr_est = np.asarray(snr_est_by_pulse, dtype=np.float64)
    snr_at_true_peak = np.asarray(snr_at_true_peak_by_pulse, dtype=np.float64)
    gate_error = np.asarray(peak_gate_error, dtype=np.int64)

    print("LFM raw-voltage and matched-filter simulation")
    print(f"  code samples                       = {len(code)}")
    print(f"  code energy                        = {code_energy:.3f}")
    print(f"  calibrated mode='same' peak offset = {peak_offset:d} samples")
    print(f"  expected MF noise power            = {expected_noise_power:.3f}")
    print(f"  measured MF noise mean power       = {noise_mean:.3f}")
    print(f"  measured MF noise median power     = {noise_median:.3f}")
    print(f"  median / mean                      = {noise_median / noise_mean:.6f}")
    print(f"  median / ln(2)                     = {corrected_noise:.3f}")
    print(f"  corrected / measured mean          = {corrected_noise / noise_mean:.6f}")
    print(f"  requested output SNR               = {requested_snr:.3f}")
    print(f"  true-peak output SNR median        = {np.median(snr_at_true_peak):.3f}")
    print(f"  true-peak output SNR p05/p95       = {np.percentile(snr_at_true_peak, 5):.3f} / {np.percentile(snr_at_true_peak, 95):.3f}")
    print(f"  estimated output SNR median        = {np.median(snr_est):.3f}")
    print(f"  estimated output SNR p05/p95       = {np.percentile(snr_est, 5):.3f} / {np.percentile(snr_est, 95):.3f}")
    print(f"  exact peak-gate recovery fraction  = {np.mean(gate_error == 0):.6f}")


if __name__ == "__main__":
    main()
