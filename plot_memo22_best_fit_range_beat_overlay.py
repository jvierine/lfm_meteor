import argparse
import os
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import fit_all_ballistic_snr_weighted as base
import fit_event_joint_delay_doppler_fft as joint
import fit_gcrs_trajectories_lfm_ambiguity as gfit
import test_rank02_range_interpolation as interp
from grid_search_delays_beam_axis import DAN_PATTERN, SAN_PATTERN, WEN_PATTERN, load_events, pair_tristatic_events


SCRIPT_VERSION = "v20260625a"
DEFAULT_EVENT_ID = "tri_0110_1713819115444351196"
DEFAULT_RESULTS_DIR = os.path.join("results", "tristatic_calibrated_chirp_v20260624b")
DEFAULT_OUTPUT_BASE = os.path.join("results", f"memo22_best_fit_range_beat_overlay_{SCRIPT_VERSION}")
DEFAULT_PAPER_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/memos/figures"


def quadratic_peak_frequency(freq_hz, power_db, idx):
    if idx <= 0 or idx >= len(power_db) - 1:
        return float(freq_hz[idx])
    ym1 = float(power_db[idx - 1])
    y0 = float(power_db[idx])
    yp1 = float(power_db[idx + 1])
    denom = ym1 - 2.0 * y0 + yp1
    if not np.isfinite(denom) or abs(denom) < 1e-30:
        return float(freq_hz[idx])
    delta = float(np.clip(0.5 * (ym1 - yp1) / denom, -1.0, 1.0))
    return float(freq_hz[idx] + delta * float(freq_hz[1] - freq_hz[0]))


def reconstruct_event_context(event_id, range_upsample_factor=32, system_noise_h5=joint.DEFAULT_SYSTEM_NOISE_H5):
    ref_fits = base.load_reference_fits()
    triplets = pair_tristatic_events(load_events(SAN_PATTERN), load_events(DAN_PATTERN), load_events(WEN_PATTERN))
    _idx, triplet = joint.choose_triplet(event_id, triplets, ref_fits)
    san_event, dan_event, wen_event = triplet
    fit0 = base.match_reference_fit(san_event, ref_fits)
    site_data = {
        "sanya": joint.load_site_h5_with_pulse(san_event.path, fit0, "sanya"),
        "danzhou": joint.load_site_h5_with_pulse(dan_event.path, fit0, "danzhou"),
        "wenchang": joint.load_site_h5_with_pulse(wen_event.path, fit0, "wenchang"),
    }
    refined = {}
    for site in joint.SITE_ORDER:
        gate, range_km, _power_db = joint.refine_site_without_doppler(
            site_data[site],
            upsample_factor=range_upsample_factor,
            same_mode_offset_samples=0.0,
        )
        refined[f"{site}_gate"] = gate
        refined[f"{site}_range_km"] = range_km
    refined["sanya_range_km"] = refined["sanya_range_km"] + joint.sc.SANYA_RANGE_CORRECTION_KM
    noise_power = joint.RawVoltageNoisePower(system_noise_h5)
    snr_by_site = {
        site: joint.normalized_matched_filter_snr_db(site_data[site], refined[f"{site}_gate"], site, noise_power)
        for site in joint.SITE_ORDER
    }
    measured, times_ns, _beijing_ns, snr_db, source_indices = joint.assemble_union_measurements_from_sites(
        {"sanya": san_event, "danzhou": dan_event, "wenchang": wen_event},
        site_data,
        refined,
        snr_by_site,
    )
    order = np.argsort(times_ns)
    return {
        "site_data": site_data,
        "refined": refined,
        "measured": measured[order],
        "times_ns": times_ns[order],
        "snr_db": snr_db[order],
        "source_indices": source_indices[order],
    }


def load_fit(results_dir, event_id):
    path = os.path.join(results_dir, f"joint_delay_doppler_fft_{event_id}.h5")
    with h5py.File(path, "r") as h:
        root_attrs = dict(h.attrs)
        j = h["joint_fit"]
        fit = {key: j[key][:] for key in j.keys()}
        fit_attrs = dict(j.attrs)
        obs = {key: h["fft_observations"][key][:] for key in h["fft_observations"].keys()}
    return path, root_attrs, fit_attrs, fit, obs


def select_samples(fit, obs, context):
    selected = []
    fit_times = np.asarray(fit["time_ns"], dtype=np.int64)
    context_times = np.asarray(context["times_ns"], dtype=np.int64)
    row_by_time = {int(t): i for i, t in enumerate(context_times)}
    for site_col, site in enumerate(joint.SITE_ORDER):
        candidates = []
        for fit_row, time_ns in enumerate(fit_times):
            ctx_row = row_by_time.get(int(time_ns), None)
            if ctx_row is None:
                continue
            src_idx = int(context["source_indices"][ctx_row, site_col])
            if src_idx < 0:
                continue
            if not bool(fit["fft_keep"][fit_row, site_col]) or not bool(fit["path_keep"][fit_row, site_col]):
                continue
            snr_db = float(obs["fft_snr_db"][fit_row, site_col])
            path_resid_m = float(fit["path_residuals_m"][fit_row, site_col])
            beat_resid_hz = float(fit["fft_residuals_hz"][fit_row, site_col])
            if not np.all(np.isfinite([snr_db, path_resid_m, beat_resid_hz])):
                continue
            if abs(path_resid_m) > 10.0 or abs(beat_resid_hz) > 500.0:
                continue
            candidates.append((snr_db, -abs(beat_resid_hz), -abs(path_resid_m), fit_row, ctx_row, src_idx))
        if not candidates:
            raise RuntimeError(f"No retained high-quality sample found for {site}")
        candidates.sort(reverse=True)
        snr_db, _neg_beat, _neg_path, fit_row, ctx_row, src_idx = candidates[0]
        selected.append(
            {
                "site": site,
                "site_col": site_col,
                "fit_row": fit_row,
                "context_row": ctx_row,
                "source_index": src_idx,
                "snr_db": snr_db,
            }
        )
    return selected


def raw_dechirp_diagnostic(site_data, gate, src_idx, root_attrs):
    gate_upsample = int(root_attrs.get("fft_gate_upsample_factor", 32))
    time_pad_us = float(root_attrs.get("fft_time_pad_us", 50.0))
    zero_pad_factor = int(root_attrs.get("zero_pad_factor", 64))
    chirp_rate_scale = float(root_attrs.get("reference_chirp_rate_scale", joint.DEFAULT_REFERENCE_CHIRP_RATE_SCALE))
    row = np.asarray(site_data["raw"][src_idx], dtype=np.complex128)
    row_work = sig.resample_poly(row, gate_upsample, 1).astype(np.complex128)
    sr_work_mhz = float(site_data["sr_mhz"]) * float(gate_upsample)
    center = int(round(float(gate) * float(gate_upsample)))
    pulse_length_us = float(site_data.get("pulse_length_us", 199.0))
    code, _t_s = interp.lfm(
        length_us=pulse_length_us,
        sr_mhz=sr_work_mhz,
        bandwidth_hz=float(site_data["bw_mhz"]) * 1e6,
    )
    n_code = len(code)
    pulse_start = center - n_code // 2
    pulse_stop = pulse_start + n_code
    pad_samples = int(round(time_pad_us * sr_work_mhz))
    start = max(0, pulse_start - pad_samples)
    stop = min(len(row_work), pulse_stop + pad_samples)
    if pulse_start < 0 or pulse_stop > len(row_work):
        raise RuntimeError("Pulse window is outside raw-voltage row")
    sample_offsets = np.arange(start, stop, dtype=np.float64) - float(pulse_start)
    t_us = sample_offsets / sr_work_mhz
    reference = joint.lfm_reference_for_offsets(
        sample_offsets,
        sr_work_mhz,
        float(site_data["bw_mhz"]) * 1e6,
        pulse_length_us,
        chirp_rate_scale=chirp_rate_scale,
    )
    segment = row_work[start:stop]
    deramped = segment * np.conj(reference)
    window = np.hanning(len(deramped))
    n_fft = 1
    while n_fft < int(zero_pad_factor) * len(deramped):
        n_fft *= 2
    sr_hz = sr_work_mhz * 1e6
    spectrum = np.fft.fftshift(np.fft.fft(deramped * window, n=n_fft))
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sr_hz))
    power_db = 10.0 * np.log10(np.maximum(np.abs(spectrum) ** 2.0, 1e-300))
    peak_idx = int(np.nanargmax(power_db))
    peak_hz = quadratic_peak_frequency(freq_hz, power_db, peak_idx)
    power_db_rel = power_db - float(power_db[peak_idx])
    return {
        "t_us": t_us,
        "deramped": deramped,
        "freq_hz": freq_hz,
        "power_db_rel": power_db_rel,
        "raw_fft_peak_hz": peak_hz,
        "pulse_length_us": pulse_length_us,
        "fourier_resolution_hz": float(1.0 / (len(deramped) / sr_hz)),
        "fft_bin_hz": float(sr_hz / n_fft),
        "analysis_length_us": float(len(deramped) / sr_work_mhz),
    }


def plot(samples, output_base, event_id):
    fig, axes = plt.subplots(
        len(samples),
        2,
        figsize=(8.2, 7.8),
        sharex=False,
        constrained_layout=True,
    )
    if len(samples) == 1:
        axes = np.asarray([axes])
    site_colors = {"sanya": "#1b9e77", "danzhou": "#d95f02", "wenchang": "#7570b3"}
    for row_idx, sample in enumerate(samples):
        ax_t, ax_f = axes[row_idx]
        diag = sample["diagnostic"]
        deramped = np.asarray(diag["deramped"], dtype=np.complex128)
        norm = float(np.nanmax(np.abs(deramped)))
        if not np.isfinite(norm) or norm <= 0.0:
            norm = 1.0
        color = site_colors[sample["site"]]
        ax_t.axvspan(0.0, diag["pulse_length_us"], color="0.92", zorder=0)
        ax_t.axvline(0.0, color="0.35", lw=0.8, ls="--")
        ax_t.axvline(diag["pulse_length_us"], color="0.35", lw=0.8, ls="--")
        ax_t.axvline(0.5 * diag["pulse_length_us"], color=color, lw=1.2, label="measured gate")
        ax_t.axvline(sample["model_center_us"], color="#c43c39", lw=1.2, ls=":", label="fit delay")
        ax_t.plot(diag["t_us"], np.real(deramped) / norm, color="#1f77b4", lw=0.8, label="real")
        ax_t.plot(diag["t_us"], np.imag(deramped) / norm, color="#d95f02", lw=0.8, label="imag.")
        ax_t.plot(diag["t_us"], np.abs(deramped) / norm, color="0.10", lw=0.7, alpha=0.65, label="abs")
        ax_t.set_ylabel(f"{sample['site'].capitalize()}\nvoltage")
        ax_t.grid(True, alpha=0.25)
        ax_t.set_xlim(-50.0, diag["pulse_length_us"] + 50.0)
        if row_idx == 0:
            ax_t.legend(loc="upper right", fontsize=7, frameon=True)
        ax_t.text(
            0.02,
            0.05,
            (
                f"SNR {sample['snr_db']:.1f} dB; pulse {sample['source_index']}\n"
                f"delay residual {sample['path_residual_m']:.2f} m"
            ),
            transform=ax_t.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
        )

        freq_khz = diag["freq_hz"] / 1e3
        raw_peak_khz = diag["raw_fft_peak_hz"] / 1e3
        model_khz = sample["model_beat_hz"] / 1e3
        keep = np.abs(freq_khz - model_khz) <= 50.0
        if not np.any(keep):
            keep = np.abs(freq_khz - raw_peak_khz) <= 50.0
        ax_f.plot(freq_khz[keep], diag["power_db_rel"][keep], color="0.12", lw=1.0)
        ax_f.axvline(raw_peak_khz, color=color, lw=1.4, label="raw FFT peak")
        ax_f.axvline(model_khz, color="#c43c39", lw=1.4, ls="--", label="fit model")
        ax_f.set_ylim(-55.0, 3.0)
        ax_f.set_ylabel("Power (dB)")
        ax_f.grid(True, alpha=0.25)
        if row_idx == 0:
            ax_f.legend(loc="upper right", fontsize=7, frameon=True)
        ax_f.text(
            0.02,
            0.05,
            (
                f"raw peak {raw_peak_khz:.3f} kHz\n"
                f"model {model_khz:.3f} kHz\n"
                f"residual {sample['beat_residual_hz']:.1f} Hz"
            ),
            transform=ax_f.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
        )
    axes[-1, 0].set_xlabel("Time relative to measured pulse start (us)")
    axes[-1, 1].set_xlabel("Dechirped beat frequency (kHz)")
    fig.suptitle(f"Range and beat-frequency check from raw voltage: {event_id}", fontsize=12)
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    fig.savefig(f"{output_base}.png", dpi=240)
    fig.savefig(f"{output_base}.pdf")
    plt.close(fig)


def write_h5(samples, output_base, event_id, fit_path):
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(f"{output_base}.h5", "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["event_id"] = event_id
        h.attrs["fit_h5"] = fit_path
        h.attrs["description"] = "Raw-voltage range and dechirped beat-frequency overlay for retained high-SNR pulses."
        for sample in samples:
            g = h.create_group(sample["site"])
            for key in (
                "source_index",
                "fit_row",
                "time_ns",
                "snr_db",
                "measured_total_path_m",
                "predicted_total_path_m",
                "geometric_total_path_m",
                "path_residual_m",
                "observed_beat_hz",
                "model_beat_hz",
                "beat_residual_hz",
                "raw_fft_peak_hz",
                "model_center_us",
            ):
                g.attrs[key] = sample[key]
            g.attrs["site"] = sample["site"]
            g.attrs["raw_event_h5"] = sample["raw_event_h5"]
            diag = sample["diagnostic"]
            g.create_dataset("t_us", data=diag["t_us"])
            g.create_dataset("deramped", data=np.asarray(diag["deramped"], dtype=np.complex64))
            freq_keep = np.abs(np.asarray(diag["freq_hz"]) - float(sample["model_beat_hz"])) <= 75e3
            g.create_dataset("freq_hz", data=np.asarray(diag["freq_hz"])[freq_keep])
            g.create_dataset("power_db_rel", data=np.asarray(diag["power_db_rel"])[freq_keep])
            g.create_dataset("site", data=np.asarray(sample["site"], dtype=object), dtype=string_dtype)


def main():
    parser = argparse.ArgumentParser(description="Plot raw-voltage range and beat-frequency overlays for Memo 22.")
    parser.add_argument("--event-id", default=DEFAULT_EVENT_ID)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--paper-figure-dir", default=DEFAULT_PAPER_FIGURE_DIR)
    parser.add_argument("--range-upsample-factor", type=int, default=32)
    args = parser.parse_args()

    fit_path, root_attrs, _fit_attrs, fit, obs = load_fit(args.results_dir, args.event_id)
    context = reconstruct_event_context(args.event_id, range_upsample_factor=args.range_upsample_factor)
    selected = select_samples(fit, obs, context)
    samples = []
    for selected_sample in selected:
        site = selected_sample["site"]
        site_col = selected_sample["site_col"]
        fit_row = selected_sample["fit_row"]
        ctx_row = selected_sample["context_row"]
        src_idx = selected_sample["source_index"]
        site_data = context["site_data"][site]
        gate = float(context["refined"][f"{site}_gate"][src_idx])
        diag = raw_dechirp_diagnostic(site_data, gate, src_idx, root_attrs)
        path_residual_m = float(fit["path_residuals_m"][fit_row, site_col])
        samples_per_us = float(site_data["sr_mhz"])
        model_center_us = 0.5 * diag["pulse_length_us"] + path_residual_m / gfit.C * 1e6
        sample = dict(selected_sample)
        sample.update(
            {
                "time_ns": int(fit["time_ns"][fit_row]),
                "raw_event_h5": site_data["path"],
                "measured_total_path_m": float(fit["measured_total_paths_m"][fit_row, site_col]),
                "predicted_total_path_m": float(fit["predicted_total_paths_m"][fit_row, site_col]),
                "geometric_total_path_m": float(fit["geometric_total_paths_m"][fit_row, site_col]),
                "path_residual_m": path_residual_m,
                "observed_beat_hz": float(fit["observed_fft_beat_hz"][fit_row, site_col]),
                "model_beat_hz": float(fit["model_fft_peak_hz"][fit_row, site_col]),
                "beat_residual_hz": float(fit["fft_residuals_hz"][fit_row, site_col]),
                "raw_fft_peak_hz": float(diag["raw_fft_peak_hz"]),
                "model_center_us": float(model_center_us),
                "diagnostic": diag,
            }
        )
        samples.append(sample)

    plot(samples, args.output_base, args.event_id)
    write_h5(samples, args.output_base, args.event_id, fit_path)

    os.makedirs(args.paper_figure_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        src = f"{args.output_base}.{ext}"
        dst = os.path.join(args.paper_figure_dir, f"memo22_best_fit_range_beat_overlay_{SCRIPT_VERSION}.{ext}")
        shutil.copy2(src, dst)
        print(f"paper_{ext}={dst}")
    print(f"output_png={args.output_base}.png")
    print(f"output_pdf={args.output_base}.pdf")
    print(f"output_h5={args.output_base}.h5")
    for sample in samples:
        print(
            "sample "
            f"site={sample['site']} pulse={sample['source_index']} "
            f"snr_db={sample['snr_db']:.2f} "
            f"path_residual_m={sample['path_residual_m']:.3f} "
            f"raw_peak_hz={sample['raw_fft_peak_hz']:.3f} "
            f"model_beat_hz={sample['model_beat_hz']:.3f} "
            f"stored_obs_hz={sample['observed_beat_hz']:.3f} "
            f"beat_residual_hz={sample['beat_residual_hz']:.3f}"
        )


if __name__ == "__main__":
    main()
