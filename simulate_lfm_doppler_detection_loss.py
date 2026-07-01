"""Simulate LFM matched-filter detection loss from Doppler-induced peak shift."""

import argparse
import os
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


C_M_S = 299_792_458.0
DEFAULT_OUTPUT_BASE = "results/lfm_doppler_detection_loss_v20260629a"
PAPER_MEMO_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/memos/figures"


def make_lfm(fs_hz, duration_s, bandwidth_hz):
    n = np.arange(int(round(fs_hz * duration_s)), dtype=np.float64)
    t = n / fs_hz
    gamma_hz_s = bandwidth_hz / duration_s
    phase_cycles = 0.5 * bandwidth_hz * t - 0.5 * gamma_hz_s * t**2
    return t, np.exp(2.0j * np.pi * phase_cycles)


def ambiguity_peak(s, t, fs_hz, doppler_hz):
    echo = s * np.exp(2.0j * np.pi * doppler_hz * t)
    corr = signal.correlate(echo, s, mode="full", method="fft")
    lags = signal.correlation_lags(echo.size, s.size, mode="full")
    power = np.abs(corr) ** 2
    idx = int(np.argmax(power))
    return float(power[idx]), float(lags[idx] / fs_hz)


def matched_filter_record(record, s):
    corr = signal.correlate(record, s, mode="full", method="fft")
    lags = signal.correlation_lags(record.size, s.size, mode="full")
    return lags, corr


def noisy_detection_trials(s, t, args, doppler_hz_values):
    rng = np.random.default_rng(args.random_seed)
    n_pulse = s.size
    true_delay_samples = int(round(args.true_delay_s * args.sample_rate_hz))
    guard_samples = int(round(args.noise_guard_s * args.sample_rate_hz))
    search_half_samples = int(round(args.search_half_width_s * args.sample_rate_hz))
    n_record = true_delay_samples + n_pulse + int(round(args.post_echo_s * args.sample_rate_hz))
    echo_snr_linear = 10.0 ** (args.output_snr_db / 10.0)
    # Unit-variance complex noise gives matched-filter noise-power mean equal to
    # pulse energy.  Choose the echo amplitude so the zero-Doppler peak power
    # SNR is args.output_snr_db before finite-pulse Doppler loss.
    pulse_energy = float(np.vdot(s, s).real)
    echo_amplitude = np.sqrt(echo_snr_linear / pulse_energy)

    observed_offset_s = np.empty((doppler_hz_values.size, args.n_noise_trials), dtype=np.float64)
    detected_snr = np.empty_like(observed_offset_s)
    for i, fd in enumerate(doppler_hz_values):
        echo = echo_amplitude * s * np.exp(2.0j * np.pi * fd * t)
        for j in range(args.n_noise_trials):
            noise = (rng.standard_normal(n_record) + 1.0j * rng.standard_normal(n_record)) / np.sqrt(2.0)
            record = noise.astype(np.complex128, copy=False)
            record = record.copy()
            record[true_delay_samples:true_delay_samples + n_pulse] += echo
            lags, corr = matched_filter_record(record, s)
            power = np.abs(corr) ** 2
            search = np.abs(lags - true_delay_samples) <= search_half_samples
            idx_search = np.flatnonzero(search)
            idx_peak = idx_search[int(np.argmax(power[idx_search]))]
            full_overlap = (lags >= 0) & (lags <= n_record - n_pulse)
            noise_mask = full_overlap & (np.abs(lags - true_delay_samples) > guard_samples)
            noise_power = np.median(power[noise_mask]) / np.log(2.0)
            observed_offset_s[i, j] = (lags[idx_peak] - true_delay_samples) / args.sample_rate_hz
            detected_snr[i, j] = power[idx_peak] / noise_power

    return {
        "doppler_hz": doppler_hz_values,
        "true_delay_samples": true_delay_samples,
        "pulse_energy": pulse_energy,
        "echo_amplitude": echo_amplitude,
        "observed_offset_s": observed_offset_s,
        "detected_snr": detected_snr,
    }


def analytic_power_ratio(doppler_hz, gamma_hz_s, duration_s):
    shift_s = np.abs(doppler_hz) / gamma_hz_s
    overlap = np.maximum(0.0, 1.0 - shift_s / duration_s)
    return overlap**2


def run(args):
    t, s = make_lfm(args.sample_rate_hz, args.duration_s, args.bandwidth_hz)
    gamma_hz_s = args.bandwidth_hz / args.duration_s

    velocities_km_s = np.linspace(0.0, args.max_velocity_km_s, args.n_velocity)
    doppler_hz = 2.0 * (velocities_km_s * 1e3) / (C_M_S / args.carrier_hz)
    peak_power = np.empty_like(doppler_hz)
    peak_delay_s = np.empty_like(doppler_hz)
    for i, fd in enumerate(doppler_hz):
        peak_power[i], peak_delay_s[i] = ambiguity_peak(s, t, args.sample_rate_hz, fd)

    peak_power_ratio = peak_power / peak_power[0]
    analytic_ratio = analytic_power_ratio(doppler_hz, gamma_hz_s, args.duration_s)
    analytic_shift_s = doppler_hz / gamma_hz_s

    idx_72 = int(np.argmin(np.abs(velocities_km_s - args.reference_velocity_km_s)))
    trial_velocities_km_s = np.asarray([0.0, args.reference_velocity_km_s], dtype=np.float64)
    trial_doppler_hz = 2.0 * (trial_velocities_km_s * 1e3) / (C_M_S / args.carrier_hz)
    noisy_trials = noisy_detection_trials(s, t, args, trial_doppler_hz)
    detected_snr_median = np.median(noisy_trials["detected_snr"], axis=1)
    observed_offset_median_s = np.median(noisy_trials["observed_offset_s"], axis=1)
    noisy_power_loss_percent = 100.0 * (1.0 - detected_snr_median[1] / detected_snr_median[0])

    os.makedirs(os.path.dirname(args.output_base), exist_ok=True)
    with h5py.File(args.output_base + ".h5", "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["sample_rate_hz"] = float(args.sample_rate_hz)
        h.attrs["duration_s"] = float(args.duration_s)
        h.attrs["bandwidth_hz"] = float(args.bandwidth_hz)
        h.attrs["carrier_hz"] = float(args.carrier_hz)
        h.attrs["chirp_rate_hz_s"] = float(gamma_hz_s)
        h.attrs["reference_velocity_km_s"] = float(velocities_km_s[idx_72])
        h.attrs["reference_doppler_hz"] = float(doppler_hz[idx_72])
        h.attrs["reference_peak_delay_us"] = float(1e6 * peak_delay_s[idx_72])
        h.attrs["reference_analytic_shift_us"] = float(1e6 * analytic_shift_s[idx_72])
        h.attrs["reference_simulated_power_loss_percent"] = float(100.0 * (1.0 - peak_power_ratio[idx_72]))
        h.attrs["reference_analytic_power_loss_percent"] = float(100.0 * (1.0 - analytic_ratio[idx_72]))
        h.attrs["noise_trials"] = int(args.n_noise_trials)
        h.attrs["output_snr_db"] = float(args.output_snr_db)
        h.attrs["true_delay_s"] = float(args.true_delay_s)
        h.attrs["search_half_width_s"] = float(args.search_half_width_s)
        h.attrs["reference_noisy_median_offset_us"] = float(1e6 * observed_offset_median_s[1])
        h.attrs["reference_noisy_median_snr"] = float(detected_snr_median[1])
        h.attrs["zero_doppler_noisy_median_snr"] = float(detected_snr_median[0])
        h.attrs["reference_noisy_power_loss_percent"] = float(noisy_power_loss_percent)
        h["velocity_km_s"] = velocities_km_s
        h["doppler_hz"] = doppler_hz
        h["peak_delay_s"] = peak_delay_s
        h["analytic_shift_s"] = analytic_shift_s
        h["peak_power_ratio"] = peak_power_ratio
        h["analytic_power_ratio"] = analytic_ratio
        g = h.create_group("noisy_detection_trials")
        g["velocity_km_s"] = trial_velocities_km_s
        g["doppler_hz"] = trial_doppler_hz
        g["observed_offset_s"] = noisy_trials["observed_offset_s"]
        g["detected_snr"] = noisy_trials["detected_snr"]

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=False, constrained_layout=True)
    ax = axes[0]
    ax.plot(velocities_km_s, 100.0 * (1.0 - peak_power_ratio), "o", ms=3.0, color="#2563a7", label="Sampled LFM simulation")
    ax.plot(velocities_km_s, 100.0 * (1.0 - analytic_ratio), "-", color="#111827", lw=1.6, label="Rectangular-overlap model")
    ax.axvline(velocities_km_s[idx_72], color="#a72222", ls="--", lw=1.2)
    ax.set_ylabel("Peak power-SNR loss (%)")
    ax.set_title("Doppler loss after matched-filter range search")
    ax.grid(True, color="0.88", lw=0.8)
    ax.legend(loc="upper left", frameon=True, framealpha=0.94)

    ax = axes[1]
    ax.plot(velocities_km_s, 1e6 * peak_delay_s, "o", ms=3.0, color="#2f7f6f", label="Sampled peak lag")
    ax.plot(velocities_km_s, 1e6 * analytic_shift_s, "-", color="#111827", lw=1.6, label=r"$f_D/\gamma$")
    ax.axvline(velocities_km_s[idx_72], color="#a72222", ls="--", lw=1.2)
    ax.set_xlabel("Monostatic radial speed (km s$^{-1}$)")
    ax.set_ylabel("Matched-filter peak shift ($\\mu$s)")
    ax.grid(True, color="0.88", lw=0.8)
    ax.legend(loc="upper left", frameon=True, framealpha=0.94)

    ax = axes[2]
    labels = ["0", "72"]
    x = np.arange(2, dtype=np.float64)
    snr_db = 10.0 * np.log10(noisy_trials["detected_snr"])
    offset_us = 1e6 * noisy_trials["observed_offset_s"]
    parts = ax.violinplot(
        [snr_db[0], snr_db[1]],
        positions=x,
        widths=0.65,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor("#8bb8ad")
        body.set_edgecolor("#17463d")
        body.set_alpha(0.75)
    parts["cmedians"].set_color("#17463d")
    parts["cmedians"].set_linewidth(1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Simulated monostatic radial speed (km s$^{-1}$)")
    ax.set_ylabel("Detected SNR (dB)")
    ax.grid(True, axis="y", color="0.88", lw=0.8)
    ax_offset = ax.twinx()
    ax_offset.plot(x, np.median(offset_us, axis=1), "o", color="#a72222", ms=6.0, label="Median range offset")
    ax_offset.set_ylabel("Median observed offset ($\\mu$s)")
    ax_offset.tick_params(axis="y", colors="#7f1d1d")
    ax_offset.yaxis.label.set_color("#7f1d1d")
    ax.set_title(
        rf"Noisy full-record detection, input zero-Doppler SNR {args.output_snr_db:.0f} dB"
    )

    fig.savefig(args.output_base + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(args.output_base + ".pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    copied = []
    if args.copy_to_memo:
        os.makedirs(PAPER_MEMO_FIGURE_DIR, exist_ok=True)
        for ext in ("png", "pdf"):
            src = args.output_base + f".{ext}"
            dst = os.path.join(PAPER_MEMO_FIGURE_DIR, f"lfm_doppler_detection_loss_v20260629a.{ext}")
            shutil.copy2(src, dst)
            copied.append(dst)

    print(f"output_h5={args.output_base}.h5")
    print(f"output_png={args.output_base}.png")
    print(f"output_pdf={args.output_base}.pdf")
    print(f"reference_velocity_km_s={velocities_km_s[idx_72]:.6f}")
    print(f"reference_doppler_khz={doppler_hz[idx_72] / 1e3:.6f}")
    print(f"reference_peak_delay_us={1e6 * peak_delay_s[idx_72]:.6f}")
    print(f"reference_analytic_shift_us={1e6 * analytic_shift_s[idx_72]:.6f}")
    print(f"reference_simulated_power_loss_percent={100.0 * (1.0 - peak_power_ratio[idx_72]):.6f}")
    print(f"reference_analytic_power_loss_percent={100.0 * (1.0 - analytic_ratio[idx_72]):.6f}")
    print(f"zero_doppler_noisy_median_snr={detected_snr_median[0]:.6f}")
    print(f"reference_noisy_median_snr={detected_snr_median[1]:.6f}")
    print(f"reference_noisy_power_loss_percent={noisy_power_loss_percent:.6f}")
    print(f"reference_noisy_median_offset_us={1e6 * observed_offset_median_s[1]:.6f}")
    for path in copied:
        print(f"memo_copy={path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--sample-rate-hz", type=float, default=80.0e6)
    parser.add_argument("--duration-s", type=float, default=200.0e-6)
    parser.add_argument("--bandwidth-hz", type=float, default=4.0e6)
    parser.add_argument("--carrier-hz", type=float, default=430.0e6)
    parser.add_argument("--max-velocity-km-s", type=float, default=80.0)
    parser.add_argument("--n-velocity", type=int, default=161)
    parser.add_argument("--reference-velocity-km-s", type=float, default=72.0)
    parser.add_argument("--output-snr-db", type=float, default=20.0)
    parser.add_argument("--n-noise-trials", type=int, default=400)
    parser.add_argument("--true-delay-s", type=float, default=300.0e-6)
    parser.add_argument("--post-echo-s", type=float, default=260.0e-6)
    parser.add_argument("--noise-guard-s", type=float, default=260.0e-6)
    parser.add_argument("--search-half-width-s", type=float, default=40.0e-6)
    parser.add_argument("--random-seed", type=int, default=20260629)
    parser.add_argument("--copy-to-memo", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
