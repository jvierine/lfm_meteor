import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import scipy.signal as sig

C = 299792458.0
SCRIPT_VERSION = "v20260613a"
OUTPUT_BASE = os.path.join("results", f"range_sampling_interpolation_simulation_{SCRIPT_VERSION}")

FS_HZ = 4.0e6
B_HZ = 4.0e6
T_S = 199.0e-6
CODE_LEN = int(round(T_S * FS_HZ))
CENTER_SAMPLE = 180.0
N_SAMPLES = 1200
UPSAMPLE_FACTORS = (1, 2, 4, 8, 16, 32)
FRACTIONAL_PHASES = np.linspace(0.0, 1.0, 41, endpoint=False)
SNR_DB = 23.0
N_MONTE_CARLO = 64
RNG_SEED = 20260613


def lfm_continuous(t_s):
    t_s = np.asarray(t_s, dtype=np.float64)
    phase_cycles = 0.5 * B_HZ * t_s - 0.5 * (B_HZ / T_S) * t_s**2.0
    out = np.exp(1j * 2.0 * np.pi * phase_cycles)
    return np.where((t_s >= 0.0) & (t_s < T_S), out, 0.0).astype(np.complex64)


def lfm_discrete(fs_hz):
    t_s = np.arange(int(round(T_S * fs_hz)), dtype=np.float64) / fs_hz
    return lfm_continuous(t_s)


def simulate_row(delay_samples, rng=None, snr_db=None):
    n = np.arange(N_SAMPLES, dtype=np.float64)
    row = lfm_continuous((n - float(delay_samples)) / FS_HZ).astype(np.complex64)
    if rng is not None and snr_db is not None:
        code_energy = float(np.sum(np.abs(lfm_discrete(FS_HZ)) ** 2.0))
        noise_var = code_energy / (10.0 ** (float(snr_db) / 10.0))
        noise = np.sqrt(noise_var / 2.0) * (
            rng.standard_normal(N_SAMPLES) + 1j * rng.standard_normal(N_SAMPLES)
        )
        row = (row + noise.astype(np.complex64)).astype(np.complex64)
    return row


def matched_filter_peak(row, upsample_factor):
    if upsample_factor == 1:
        row_work = row
        fs_work = FS_HZ
    else:
        row_work = sig.resample_poly(row, upsample_factor, 1).astype(np.complex64)
        fs_work = FS_HZ * upsample_factor
    code = lfm_discrete(fs_work)
    corr = sig.fftconvolve(row_work, np.conj(code), mode="same")
    power = np.abs(corr) ** 2.0
    idx0 = int(np.argmax(power))
    delta = 0.0
    if 0 < idx0 < len(power) - 1:
        ym1, y0, yp1 = float(power[idx0 - 1]), float(power[idx0]), float(power[idx0 + 1])
        denom = ym1 - 2.0 * y0 + yp1
        if denom < 0.0:
            delta = float(np.clip(0.5 * (ym1 - yp1) / denom, -0.5, 0.5))
    return (float(idx0) + delta) / float(upsample_factor)


def continuous_template_delay(row, bracket_center):
    n = np.arange(N_SAMPLES, dtype=np.float64)
    row64 = row.astype(np.complex128)

    def residual_power(delay_samples):
        template = lfm_continuous((n - float(delay_samples)) / FS_HZ).astype(np.complex128)
        denom = np.vdot(template, template).real
        if denom <= 0.0:
            return np.inf
        amp = np.vdot(template, row64) / denom
        resid = row64 - amp * template
        return float(np.vdot(resid, resid).real)

    result = so.minimize_scalar(
        residual_power,
        bounds=(float(bracket_center) - 1.25, float(bracket_center) + 1.25),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(result.x)


def range_error_m(sample_error):
    return float(sample_error) * C / (2.0 * FS_HZ)


def noiseless_phase_sweep():
    zero_peaks = {
        q: matched_filter_peak(simulate_row(CENTER_SAMPLE), q)
        for q in UPSAMPLE_FACTORS
    }
    rows = []
    for frac in FRACTIONAL_PHASES:
        true_delay = CENTER_SAMPLE + float(frac)
        row = simulate_row(true_delay)
        entry = {"fractional_sample": float(frac)}
        for q in UPSAMPLE_FACTORS:
            measured_frac = matched_filter_peak(row, q) - zero_peaks[q]
            entry[f"current_q{q}_error_m"] = range_error_m(measured_frac - frac)
        entry["template_error_m"] = range_error_m(
            continuous_template_delay(row, true_delay) - true_delay
        )
        rows.append(entry)
    return rows


def noisy_monte_carlo():
    rng = np.random.default_rng(RNG_SEED)
    errors = {f"current_q{q}": [] for q in UPSAMPLE_FACTORS}
    errors["template"] = []
    zero_peaks = {
        q: matched_filter_peak(simulate_row(CENTER_SAMPLE), q)
        for q in UPSAMPLE_FACTORS
    }
    for frac in FRACTIONAL_PHASES:
        true_delay = CENTER_SAMPLE + float(frac)
        for _ in range(N_MONTE_CARLO):
            row = simulate_row(true_delay, rng=rng, snr_db=SNR_DB)
            for q in UPSAMPLE_FACTORS:
                measured_frac = matched_filter_peak(row, q) - zero_peaks[q]
                errors[f"current_q{q}"].append(range_error_m(measured_frac - frac))
            errors["template"].append(
                range_error_m(continuous_template_delay(row, true_delay) - true_delay)
            )
    summary = {}
    for key, values in errors.items():
        values = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "rms_error_m": float(np.sqrt(np.mean(values**2.0))),
            "median_abs_error_m": float(np.median(np.abs(values))),
            "p95_abs_error_m": float(np.percentile(np.abs(values), 95.0)),
            "bias_m": float(np.mean(values)),
        }
    return summary


def plot_results(phase_rows, noise_summary):
    fig, axes = plt.subplots(2, 1, figsize=(7.3, 6.6), constrained_layout=True)
    frac = np.asarray([row["fractional_sample"] for row in phase_rows])
    for q in (1, 2, 4, 8, 16):
        axes[0].plot(frac, [row[f"current_q{q}_error_m"] for row in phase_rows], label=f"{q}x")
    axes[0].plot(frac, [row["template_error_m"] for row in phase_rows], "k--", label="sampling-aware fit")
    axes[0].axhline(0.0, color="0.4", lw=0.8)
    axes[0].set_xlabel("True delay within native 4 MHz sample")
    axes[0].set_ylabel("Noiseless range error (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncols=3, fontsize=8)

    q_values = np.asarray(UPSAMPLE_FACTORS, dtype=np.float64)
    current_rms = np.asarray([noise_summary[f"current_q{q}"]["rms_error_m"] for q in UPSAMPLE_FACTORS])
    current_p95 = np.asarray([noise_summary[f"current_q{q}"]["p95_abs_error_m"] for q in UPSAMPLE_FACTORS])
    template_rms = noise_summary["template"]["rms_error_m"]
    template_p95 = noise_summary["template"]["p95_abs_error_m"]
    axes[1].plot(q_values, current_rms, "o-", label="current peak picker RMS")
    axes[1].plot(q_values, current_p95, "s-", label="current peak picker 95% abs.")
    axes[1].axhline(template_rms, color="k", ls="--", label="sampling-aware RMS")
    axes[1].axhline(template_p95, color="0.35", ls=":", label="sampling-aware 95% abs.")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Raw-voltage interpolation factor")
    axes[1].set_ylabel(f"Range error at SNR={SNR_DB:.0f} dB (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Synthetic sampling-discretization test for Sanya LFM range interpolation")
    fig.savefig(f"{OUTPUT_BASE}.png", dpi=220)
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)
    phase_rows = noiseless_phase_sweep()
    noise_summary = noisy_monte_carlo()
    with open(f"{OUTPUT_BASE}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "script": os.path.basename(__file__),
                "script_version": SCRIPT_VERSION,
                "fs_hz": FS_HZ,
                "bandwidth_hz": B_HZ,
                "pulse_duration_s": T_S,
                "code_length_samples": CODE_LEN,
                "snr_db": SNR_DB,
                "n_monte_carlo": N_MONTE_CARLO,
                "rng_seed": RNG_SEED,
                "upsample_factors": list(UPSAMPLE_FACTORS),
                "phase_rows": phase_rows,
                "noise_summary": noise_summary,
            },
            f,
            indent=2,
        )
    plot_results(phase_rows, noise_summary)
    print(f"wrote {OUTPUT_BASE}.json")
    print(f"wrote {OUTPUT_BASE}.png")
    for q in UPSAMPLE_FACTORS:
        stats = noise_summary[f"current_q{q}"]
        print(f"current {q:2d}x rms={stats['rms_error_m']:.2f} m p95={stats['p95_abs_error_m']:.2f} m")
    stats = noise_summary["template"]
    print(f"template rms={stats['rms_error_m']:.2f} m p95={stats['p95_abs_error_m']:.2f} m")


if __name__ == "__main__":
    main()
