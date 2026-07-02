import argparse
import os
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import estimate_single_pulse_dechirped_fft_doppler as beat
import fit_gcrs_trajectories_lfm_ambiguity as gfit
import sanya_opts as sc


SCRIPT_VERSION = "v20260702b"
DEFAULT_EVENT = os.path.join(
    "results",
    "tristatic_head_echoes",
    "sanya",
    "sanya_1713851737704349518.h5",
)
DEFAULT_OUTPUT_BASE = os.path.join("results", f"appendix_dechirped_beat_example_{SCRIPT_VERSION}")
DEFAULT_PAPER_BASE = os.path.join(
    "/Users/jvi019/src/sanya_tristatic_paper",
    "figures",
    "appendix_dechirped_beat_example",
)
DEFAULT_PULSE_INDEX = 18
DEFAULT_ZERO_PAD_FACTOR = 64
DEFAULT_TIME_PAD_US = 50.0
DEFAULT_GATE_UPSAMPLE_FACTOR = 32
DEFAULT_FIT_H5 = os.path.join(
    "results",
    "tristatic_whipple_jacchia_bootstrap_orbit100_20260701",
    "joint_delay_doppler_fft_tri_0156_1713822937704349518.h5",
)


def load_fit_context(fit_h5, raw_time_ns, site):
    with h5py.File(fit_h5, "r") as h:
        times_ns = np.asarray(h["joint_fit/time_ns"][:], dtype=np.int64)
        row = int(np.argmin(np.abs(times_ns - int(raw_time_ns))))
        link_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in h["joint_fit/link_names"][:]
        ]
        col = link_names.index(site)
        measured_path_m = float(h["joint_fit/measured_total_paths_m"][row, col])
        geometric_path_m = float(h["joint_fit/geometric_total_paths_m"][row, col])
        predicted_path_m = float(h["joint_fit/predicted_total_paths_m"][row, col])
        observed_beat_hz = float(h["joint_fit/observed_fft_beat_hz"][row, col])
        model_beat_hz = float(h["joint_fit/model_fft_peak_hz"][row, col])
        model_doppler_hz = float(h["joint_fit/model_doppler_hz"][row, col])
        fft_keep = bool(h["joint_fit/fft_keep"][row, col])
    return {
        "fit_h5": fit_h5,
        "fit_row": row,
        "fit_time_ns": int(times_ns[row]),
        "fit_time_offset_ns": int(times_ns[row]) - int(raw_time_ns),
        "fit_site_col": col,
        "fit_measured_total_path_m": measured_path_m,
        "fit_geometric_total_path_m": geometric_path_m,
        "fit_predicted_total_path_m": predicted_path_m,
        "fit_observed_beat_hz": observed_beat_hz,
        "fit_model_beat_hz": model_beat_hz,
        "fit_model_doppler_hz": model_doppler_hz,
        "fit_fft_keep": fft_keep,
        "fit_gate_displacement_us": (predicted_path_m - geometric_path_m) / gfit.C * 1e6,
    }


def estimate_with_canonical_pulse_length(
    event_path,
    pulse_index,
    zero_pad_factor,
    time_pad_us,
    gate_upsample_factor,
    fit_h5=None,
):
    result = {}
    with h5py.File(event_path, "r") as h:
        row = np.asarray(h["raw"][pulse_index], dtype=np.complex128)
        source_gate = float(np.asarray(h["range_gate"][pulse_index]))
        snr_db = float(np.asarray(h["snr_peak_db"][pulse_index]))
        sr_mhz = float(np.asarray(h["sr_mhz"]))
        bw_mhz = float(np.asarray(h["bw_mhz"]))
        source_pulse_length_us = float(np.asarray(h["pulse_length_us"]))
        time_ns = int(np.asarray(h["times_ns"][pulse_index]))
        site = h["site"][()]
        event_id = h["event_id"][()]
        if isinstance(site, bytes):
            site = site.decode("utf-8")
        if isinstance(event_id, bytes):
            event_id = event_id.decode("utf-8")

    fit_context = {}
    gate = source_gate
    if fit_h5:
        fit_context = load_fit_context(fit_h5, time_ns, site)
        zero_delay_us = sc.SANYA_CORRECTED_TXRX_DELAY_US
        if site == "danzhou":
            zero_delay_us = gfit.DAN_CENTER_US
        elif site == "wenchang":
            zero_delay_us = gfit.WEN_CENTER_US
        fit_gate = (
            fit_context["fit_measured_total_path_m"] / gfit.C * 1e6 - zero_delay_us
        ) * sr_mhz
        fit_context["fit_refined_range_gate"] = float(fit_gate)

    canonical_pulse_length_us = float(gfit.LFM_DURATION_S * 1e6)
    row_work = np.asarray(row, dtype=np.complex128)
    sr_work_mhz = sr_mhz
    gate_work = gate
    if gate_upsample_factor > 1:
        row_work = sig.resample_poly(row_work, int(gate_upsample_factor), 1).astype(np.complex128)
        sr_work_mhz = sr_mhz * float(gate_upsample_factor)
        gate_work = gate * float(gate_upsample_factor)

    n_code = int(round(canonical_pulse_length_us * sr_mhz))
    n_code_work = int(round(canonical_pulse_length_us * sr_work_mhz))
    center = int(round(gate_work))
    pulse_start = center - n_code_work // 2
    pulse_stop = pulse_start + n_code_work
    pad_samples = int(round(float(time_pad_us) * sr_work_mhz))
    start = max(0, pulse_start - pad_samples)
    stop = min(len(row_work), pulse_stop + pad_samples)
    if pulse_start < 0 or pulse_stop > len(row_work):
        raise ValueError(
            f"Pulse segment [{pulse_start}, {pulse_stop}) falls outside raw row length {len(row_work)}"
        )

    segment = row_work[start:stop]
    sample_offsets = np.arange(start, stop, dtype=np.float64) - float(pulse_start)
    reference, t_s = beat.lfm_reference_for_offsets(
        sample_offsets,
        sr_mhz=sr_work_mhz,
        bandwidth_hz=bw_mhz * 1e6,
        pulse_length_us=canonical_pulse_length_us,
        chirp_rate_scale=gfit.REFERENCE_CHIRP_RATE_SCALE,
    )
    deramped = segment * np.conj(reference)
    metrics = beat.fft_power_metrics(deramped, sr_work_mhz * 1e6, zero_pad_factor)
    fit_corrected_deramped = None
    fit_corrected_peak_hz = np.nan
    fit_corrected_echo_window_shift_us = np.nan
    if fit_context:
        fit_corrected_echo_window_shift_us = float(
            (fit_context["fit_geometric_total_path_m"] - fit_context["fit_measured_total_path_m"])
            / gfit.C
            * 1e6
        )
        fit_refined_gate_shift_us = float(
            (fit_context["fit_refined_range_gate"] - source_gate) / sr_mhz
        )
        corrected_reference_shift_us = fit_corrected_echo_window_shift_us + fit_refined_gate_shift_us
        corrected_sample_offsets = sample_offsets - corrected_reference_shift_us * sr_work_mhz
        corrected_reference, _ = beat.lfm_reference_for_offsets(
            corrected_sample_offsets,
            sr_mhz=sr_work_mhz,
            bandwidth_hz=bw_mhz * 1e6,
            pulse_length_us=canonical_pulse_length_us,
            chirp_rate_scale=gfit.REFERENCE_CHIRP_RATE_SCALE,
        )
        doppler_phase = np.exp(-1j * 2.0 * np.pi * fit_context["fit_model_doppler_hz"] * t_s)
        fit_corrected_deramped = segment * np.conj(corrected_reference) * doppler_phase
        fit_corrected_metrics = beat.fft_power_metrics(
            fit_corrected_deramped,
            sr_work_mhz * 1e6,
            zero_pad_factor,
        )
        fit_corrected_peak_hz = float(fit_corrected_metrics["peak_frequency_hz"])

    result.update(
        {
            "site": site,
            "event_id": event_id,
            "path": event_path,
            "pulse_index": int(pulse_index),
            "time_ns": time_ns,
            "source_range_gate": source_gate,
            "range_gate": gate,
            "snr_peak_db": snr_db,
            "sr_mhz": sr_mhz,
            "sr_work_mhz": sr_work_mhz,
            "bw_mhz": bw_mhz,
            "source_pulse_length_us": source_pulse_length_us,
            "pulse_length_us": canonical_pulse_length_us,
            "uses_canonical_pulse_length": True,
            "n_code": n_code,
            "n_code_work": n_code_work,
            "n_analysis": int(len(deramped)),
            "n_fft": int(metrics["n_fft"]),
            "zero_pad_factor_requested": int(zero_pad_factor),
            "gate_upsample_factor": int(gate_upsample_factor),
            "time_pad_us_requested": float(time_pad_us),
            "reference_chirp_rate_scale": float(gfit.REFERENCE_CHIRP_RATE_SCALE),
            "pad_samples_requested": int(pad_samples),
            "pre_pulse_samples": int(pulse_start - start),
            "post_pulse_samples": int(stop - pulse_stop),
            "analysis_length_us": float(len(deramped) / sr_work_mhz),
            "fft_bin_hz": float(metrics["fft_bin_hz"]),
            "fourier_resolution_hz": float(1.0 / (len(deramped) / (sr_work_mhz * 1e6))),
            "peak_bin_frequency_hz": float(metrics["peak_bin_frequency_hz"]),
            "peak_frequency_hz": float(metrics["peak_frequency_hz"]),
            "beat_inferred_delay_shift_us": float(
                metrics["peak_frequency_hz"] / gfit.CHIRP_RATE_HZ_PER_S * 1e6
            ),
            "peak_delta_bins": float(metrics["peak_delta_bins"]),
            "width_3db_hz": float(metrics["width_3db_hz"]),
            "rms_width_hz": float(metrics["rms_width_hz"]),
            "prominence_db": float(metrics["prominence_db"]),
            "t_us": t_s * 1e6,
            "deramped": deramped,
            "fit_corrected_deramped": (
                fit_corrected_deramped
                if fit_corrected_deramped is not None
                else np.full_like(deramped, np.nan + 1j * np.nan)
            ),
            "fit_corrected_peak_frequency_hz": fit_corrected_peak_hz,
            "fit_corrected_reference_shift_us": (
                corrected_reference_shift_us if fit_context else np.nan
            ),
            "freq_hz": metrics["freq_hz"],
            "power_db_rel": metrics["power_db_rel"],
        }
    )
    result.update(fit_context)
    if fit_context:
        result["fit_echo_window_shift_us"] = fit_corrected_echo_window_shift_us
        result["fit_refined_gate_shift_us"] = float(
            (fit_context["fit_refined_range_gate"] - source_gate) / sr_mhz
        )
        result["fit_predicted_echo_window_shift_us"] = -float(
            fit_context["fit_gate_displacement_us"]
        )
    return result


def write_h5(result, output_base, source_script):
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    with h5py.File(f"{output_base}.h5", "w") as h:
        h.attrs["script"] = os.path.basename(source_script)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["method"] = (
            "Article appendix dechirped beat-frequency example. The raw-voltage "
            "window is padded by 50 us on each side of the nominal pulse and "
            "dechirped with the canonical 200 us LFM phase reference."
        )
        for key, value in result.items():
            if key in {"path", "site", "event_id"}:
                h.attrs[key] = value
            elif isinstance(value, (bool, np.bool_)):
                h.attrs[key] = bool(value)
            elif np.isscalar(value):
                h.attrs[key] = value
            elif key == "deramped":
                h.create_dataset(key, data=np.asarray(value, dtype=np.complex64))
            else:
                h.create_dataset(key, data=np.asarray(value))


def make_plot(result, output_base, paper_base=None):
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    if paper_base:
        os.makedirs(os.path.dirname(paper_base), exist_ok=True)

    with plt.rc_context(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.0, 5.6),
            sharex=False,
            constrained_layout=True,
        )

        deramped = np.asarray(result["deramped"], dtype=np.complex128)
        norm = float(np.nanmax(np.abs(deramped)))
        if not np.isfinite(norm) or norm <= 0.0:
            norm = 1.0
        t_us = np.asarray(result["t_us"], dtype=np.float64)
        pulse_length_us = float(result["pulse_length_us"])
        peak_khz = float(result["peak_frequency_hz"]) / 1e3
        residual_beat_shift_us = float(
            result.get(
                "beat_inferred_delay_shift_us",
                result["peak_frequency_hz"] / gfit.CHIRP_RATE_HZ_PER_S * 1e6,
            )
        )
        fit_echo_start_us = float(result.get("fit_echo_window_shift_us", residual_beat_shift_us))
        fit_echo_stop_us = fit_echo_start_us + pulse_length_us

        ax = axes[0]
        ax.axvline(0.0, color="0.45", lw=1.0, ls="--")
        ax.axvline(pulse_length_us, color="0.45", lw=1.0, ls="--", label="Zero-Doppler range gate")
        ax.axvline(fit_echo_start_us, color="#7b3294", lw=1.7, ls="--", label="Joint-fit echo start/stop")
        ax.axvline(fit_echo_stop_us, color="#7b3294", lw=1.7, ls="--")
        fit_corrected_deramped = np.asarray(
            result.get("fit_corrected_deramped", np.full_like(deramped, np.nan + 1j * np.nan)),
            dtype=np.complex128,
        )
        if np.any(np.isfinite(fit_corrected_deramped)):
            ax.plot(
                t_us,
                np.real(fit_corrected_deramped) / norm,
                color="#1f77b4",
                lw=1.1,
                alpha=0.28,
                ls=":",
                label="Fit-corrected real",
            )
            ax.plot(
                t_us,
                np.imag(fit_corrected_deramped) / norm,
                color="#d95f02",
                lw=1.1,
                alpha=0.28,
                ls=":",
                label="Fit-corrected imaginary",
            )
        ax.plot(t_us, np.real(deramped) / norm, color="#1f77b4", lw=1.1, label="Real")
        ax.plot(t_us, np.imag(deramped) / norm, color="#d95f02", lw=1.1, label="Imaginary")
        ax.set_xlim(float(np.nanmin(t_us)), float(np.nanmax(t_us)))
        ax.set_ylim(-2.0, 2.0)
        ax.set_ylabel("Normalized voltage")
        ax.set_xlabel("Time relative to zero-Doppler gate pulse start (us)")
        ax.set_title(f"High-SNR {result['site'].capitalize()} pulse {result['pulse_index']}")
        ax.legend(loc="upper right", frameon=True, ncol=3, fontsize=9.0)
        ax.grid(True, color="0.85", lw=0.7)
        ax.text(
            0.012,
            0.06,
            rf"fit echo envelope: {fit_echo_start_us:.2f} to {fit_echo_stop_us:.2f} us",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
        )
        ax = axes[1]
        freq_khz = np.asarray(result["freq_hz"], dtype=np.float64) / 1e3
        power_db_rel = np.asarray(result["power_db_rel"], dtype=np.float64)
        use = np.abs(freq_khz - peak_khz) <= 80.0
        ax.plot(freq_khz[use], power_db_rel[use], color="0.12", lw=1.2)
        ax.axvline(peak_khz, color="#c43c39", lw=1.5, label="FFT peak")
        ax.set_ylim(-55.0, 3.0)
        ax.set_xlabel("Dechirped frequency offset (kHz)")
        ax.set_ylabel("Relative power (dB)")
        ax.grid(True, color="0.85", lw=0.7)
        ax.legend(loc="upper right", frameon=True)
        ax.text(
            0.02,
            0.95,
            (
                f"peak = {peak_khz:.2f} kHz\n"
                f"fit-corrected peak = {result['fit_corrected_peak_frequency_hz'] / 1e3:.2f} kHz\n"
                rf"fit echo start = {fit_echo_start_us:.2f} us" "\n"
                rf"fit echo stop = {fit_echo_stop_us:.2f} us" "\n"
                f"FFT interpolation bin = {result['fft_bin_hz']:.1f} Hz\n"
                f"analysis window = {result['analysis_length_us']:.0f} us\n"
                f"SNR = {result['snr_peak_db']:.1f} dB"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.94, "pad": 4.0},
        )

        png = f"{output_base}.png"
        pdf = f"{output_base}.pdf"
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

    if paper_base:
        shutil.copyfile(png, f"{paper_base}.png")
        shutil.copyfile(pdf, f"{paper_base}.pdf")

    return png, pdf


def main():
    parser = argparse.ArgumentParser(
        description="Make an article appendix example of a dechirped LFM echo and beat-frequency estimate."
    )
    parser.add_argument("--event", default=DEFAULT_EVENT)
    parser.add_argument("--pulse-index", type=int, default=DEFAULT_PULSE_INDEX)
    parser.add_argument("--zero-pad-factor", type=int, default=DEFAULT_ZERO_PAD_FACTOR)
    parser.add_argument("--time-pad-us", type=float, default=DEFAULT_TIME_PAD_US)
    parser.add_argument("--gate-upsample-factor", type=int, default=DEFAULT_GATE_UPSAMPLE_FACTOR)
    parser.add_argument("--fit-h5", default=DEFAULT_FIT_H5)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--paper-base", default=DEFAULT_PAPER_BASE)
    args = parser.parse_args()

    result = estimate_with_canonical_pulse_length(
        args.event,
        args.pulse_index,
        args.zero_pad_factor,
        args.time_pad_us,
        args.gate_upsample_factor,
        fit_h5=args.fit_h5,
    )
    write_h5(result, args.output_base, __file__)
    png, pdf = make_plot(result, args.output_base, paper_base=args.paper_base)

    print(f"event_file={args.event}")
    print(f"pulse_index={args.pulse_index}")
    print(f"site={result['site']}")
    print(f"event_id={result['event_id']}")
    print(f"snr_peak_db={result['snr_peak_db']:.2f}")
    print(f"peak_frequency_hz={result['peak_frequency_hz']:.3f}")
    print(f"residual_beat_shift_us={result['beat_inferred_delay_shift_us']:.6f}")
    print(f"fit_echo_window_shift_us={result.get('fit_echo_window_shift_us', np.nan):.6f}")
    print(f"analysis_length_us={result['analysis_length_us']:.3f}")
    print(f"source_pulse_length_us={result.get('source_pulse_length_us', np.nan):.3f}")
    print(f"pulse_length_us={result['pulse_length_us']:.3f}")
    print(f"output_h5={args.output_base}.h5")
    print(f"output_png={png}")
    print(f"output_pdf={pdf}")
    if args.paper_base:
        print(f"paper_png={args.paper_base}.png")
        print(f"paper_pdf={args.paper_base}.pdf")


if __name__ == "__main__":
    main()
