import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import test_rank02_range_interpolation as interp
from grid_search_delays_beam_axis import (
    DAN_PATTERN,
    SAN_PATTERN,
    WEN_PATTERN,
    load_events,
    pair_tristatic_events,
)


SCRIPT_VERSION = "v20260619a"
C = 299792458.0
SITE_ORDER = ("sanya", "danzhou", "wenchang")
SITE_LABEL = {"sanya": "Sanya", "danzhou": "Danzhou", "wenchang": "Wenchang"}
DEFAULT_OUTPUT_DIR = os.path.join("results", f"tristatic_microdoppler_fft_{SCRIPT_VERSION}")


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def tri_event_id(index, triplet):
    return f"tri_{index:04d}_{int(triplet[0].times_ns[0])}"


def load_raw_event(event):
    with h5py.File(event.path, "r") as h:
        return {
            "site": decode(h["site"][()]).lower(),
            "times_ns": np.asarray(h["times_ns"][:], dtype=np.int64),
            "raw": np.asarray(h["raw"][:], dtype=np.complex64),
            "range_gate": np.asarray(h["range_gate"][:], dtype=np.float64),
            "snr_peak_db": np.asarray(h["snr_peak_db"][:], dtype=np.float64),
            "sr_mhz": float(h["sr_mhz"][()]),
            "bw_mhz": float(h["bw_mhz"][()]),
            "pulse_length_us": float(h["pulse_length_us"][()]) if "pulse_length_us" in h else 199.0,
            "source_file": decode(h["source_file"][()]) if "source_file" in h else os.path.basename(event.path),
        }


def fft_len(n_code, zero_pad_factor):
    n_fft = 1
    target = max(n_code, int(zero_pad_factor) * n_code)
    while n_fft < target:
        n_fft *= 2
    return n_fft


def pulse_spectrum_db(
    row,
    gate,
    sr_mhz,
    bw_mhz,
    pulse_length_us,
    zero_pad_factor,
    gate_upsample_factor,
):
    if gate_upsample_factor > 1:
        row_work = sig.resample_poly(row, gate_upsample_factor, 1).astype(np.complex128)
        sr_work_mhz = float(sr_mhz) * float(gate_upsample_factor)
        center = int(round(float(gate) * float(gate_upsample_factor)))
    else:
        row_work = np.asarray(row, dtype=np.complex128)
        sr_work_mhz = float(sr_mhz)
        center = int(round(float(gate)))

    code, _t_s = interp.lfm(
        length_us=float(pulse_length_us),
        sr_mhz=sr_work_mhz,
        bandwidth_hz=float(bw_mhz) * 1e6,
    )
    n_code = len(code)
    start = center - n_code // 2
    stop = start + n_code
    if start < 0 or stop > len(row_work):
        return None

    segment = np.asarray(row_work[start:stop], dtype=np.complex128)
    deramped = segment * np.conj(code.astype(np.complex128))
    y = deramped * np.hanning(n_code)
    n_fft = fft_len(n_code, zero_pad_factor)
    sr_hz = float(sr_work_mhz) * 1e6
    spectrum = np.fft.fftshift(np.fft.fft(y, n=n_fft))
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sr_hz))
    power_db = 10.0 * np.log10(np.maximum(np.abs(spectrum) ** 2.0, 1e-300))
    peak_idx = int(np.nanargmax(power_db))
    shifted_db = np.roll(power_db, len(power_db) // 2 - peak_idx)
    freq_centered_hz = (np.arange(n_fft, dtype=np.float64) - n_fft // 2) * (sr_hz / n_fft)
    shifted_db -= float(np.nanmax(shifted_db))
    return freq_centered_hz, shifted_db, float(freq_hz[peak_idx]), float(sr_hz / n_fft)


def microdoppler_image(data, args):
    spectra = []
    peak_hz = np.full(len(data["times_ns"]), np.nan, dtype=np.float64)
    fft_bin_hz = np.nan
    freq_hz = None
    for idx, row in enumerate(data["raw"]):
        if data["snr_peak_db"][idx] < args.snr_min_db:
            continue
        out = pulse_spectrum_db(
            row,
            data["range_gate"][idx],
            data["sr_mhz"],
            data["bw_mhz"],
            data["pulse_length_us"],
            args.zero_pad_factor,
            args.gate_upsample_factor,
        )
        if out is None:
            continue
        freq_hz, power_db, peak, fft_bin = out
        spectra.append((idx, power_db))
        peak_hz[idx] = peak
        fft_bin_hz = fft_bin

    if freq_hz is None or not spectra:
        return None

    keep_freq = np.abs(freq_hz) <= 0.5 * args.width_khz * 1e3
    image = np.full((np.count_nonzero(keep_freq), len(data["times_ns"])), np.nan, dtype=np.float32)
    for idx, power_db in spectra:
        image[:, idx] = np.asarray(power_db[keep_freq], dtype=np.float32)
    return {
        "freq_hz": freq_hz[keep_freq],
        "image_db": image,
        "peak_hz": peak_hz,
        "fft_bin_hz": float(fft_bin_hz),
        "n_valid": int(len(spectra)),
    }


def centers_to_edges(values, low=None, high=None):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    if values.size == 1:
        half = 0.5
        edges = np.asarray([values[0] - half, values[0] + half], dtype=np.float64)
    else:
        mid = 0.5 * (values[1:] + values[:-1])
        first = values[0] - 0.5 * (values[1] - values[0])
        last = values[-1] + 0.5 * (values[-1] - values[-2])
        edges = np.concatenate([[first], mid, [last]]).astype(np.float64)
    if low is not None:
        edges[0] = min(edges[0], float(low))
    if high is not None:
        edges[-1] = max(edges[-1], float(high))
    return edges


def write_plot(result, data, event_id, event_index, site, t0_ns, t1_ns, output_dir, args):
    rel_time_s = (data["times_ns"].astype(np.float64) - float(t0_ns)) / 1e9
    x0 = 0.0
    x1 = (float(t1_ns) - float(t0_ns)) / 1e9
    time_edges_s = centers_to_edges(rel_time_s, low=x0, high=x1)
    freq_edges_khz = centers_to_edges(result["freq_hz"] / 1e3)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    shown = np.clip(result["image_db"], args.db_floor, 0.0)
    mesh = ax.pcolormesh(
        time_edges_s,
        freq_edges_khz,
        shown,
        cmap=args.cmap,
        vmin=args.db_floor,
        vmax=0.0,
        shading="flat",
    )
    ax.axhline(0.0, color="white", lw=0.65, alpha=0.55)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.5 * args.width_khz, 0.5 * args.width_khz)
    ax.set_xlabel("Time from event start (s)")
    ax.set_ylabel("Peak-centered frequency (kHz)")
    ax.set_title(
        f"{event_id} {SITE_LABEL.get(site, site)} micro-Doppler FFT\n"
        f"n={result['n_valid']}/{len(data['times_ns'])}, bin={result['fft_bin_hz']:.1f} Hz, "
        f"SNR >= {args.snr_min_db:.1f} dB"
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Relative power (dB)")

    site_dir = os.path.join(output_dir, site)
    os.makedirs(site_dir, exist_ok=True)
    png = os.path.join(site_dir, f"{event_id}_{site}_microdoppler_fft.png")
    fig.savefig(png, dpi=args.dpi)
    plt.close(fig)
    return png


def image_on_common_time_grid(result, data, common_times_ns):
    common_times_ns = np.asarray(common_times_ns, dtype=np.int64)
    image = np.full((result["image_db"].shape[0], len(common_times_ns)), np.nan, dtype=np.float32)
    lookup = {int(t): idx for idx, t in enumerate(common_times_ns)}
    for src_idx, t_ns in enumerate(np.asarray(data["times_ns"], dtype=np.int64)):
        dst_idx = lookup.get(int(t_ns))
        if dst_idx is not None:
            image[:, dst_idx] = result["image_db"][:, src_idx]
    return image


def chirp_rate_hz_per_s(data):
    scale = float(data.get("reference_chirp_rate_scale", interp.REFERENCE_CHIRP_RATE_SCALE))
    return float(data["bw_mhz"]) * 1e6 / (float(data["pulse_length_us"]) * 1e-6) * scale


def freq_khz_to_range_offset_m(freq_khz, data):
    return C * (np.asarray(freq_khz, dtype=np.float64) * 1e3) / chirp_rate_hz_per_s(data)


def range_offset_m_to_freq_khz(range_offset_m, data):
    return np.asarray(range_offset_m, dtype=np.float64) * chirp_rate_hz_per_s(data) / C / 1e3


def draw_microdoppler_panel(ax, result, data, common_times_ns, t0_ns, t1_ns, args, show_xlabel=False):
    rel_time_s = (np.asarray(common_times_ns, dtype=np.float64) - float(t0_ns)) / 1e9
    x0 = 0.0
    x1 = (float(t1_ns) - float(t0_ns)) / 1e9
    time_edges_s = centers_to_edges(rel_time_s, low=x0, high=x1)
    freq_edges_khz = centers_to_edges(result["freq_hz"] / 1e3)
    shown = np.clip(image_on_common_time_grid(result, data, common_times_ns), args.db_floor, 0.0)
    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    mesh = ax.pcolormesh(
        time_edges_s,
        freq_edges_khz,
        np.ma.masked_invalid(shown),
        cmap=cmap,
        vmin=args.db_floor,
        vmax=0.0,
        shading="flat",
    )
    ax.axhline(0.0, color="white", lw=0.6, alpha=0.55)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.5 * args.width_khz, 0.5 * args.width_khz)
    ax.set_ylabel("kHz")
    secax = ax.secondary_yaxis(
        "right",
        functions=(
            lambda y: freq_khz_to_range_offset_m(y, data),
            lambda y: range_offset_m_to_freq_khz(y, data),
        ),
    )
    secax.set_ylabel("m")
    if show_xlabel:
        ax.set_xlabel("Time from event start (s)")
    return mesh


def write_combined_plot(results_by_site, site_data, event_id, event_index, t0_ns, t1_ns, output_dir, args):
    common_times_ns = np.unique(np.concatenate([site_data[site]["times_ns"] for site in SITE_ORDER]).astype(np.int64))
    fig, axes = plt.subplots(
        len(SITE_ORDER),
        1,
        figsize=(8.2, 7.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    mesh = None
    for ax, site in zip(axes, SITE_ORDER):
        result = results_by_site.get(site)
        data = site_data[site]
        if result is None:
            ax.text(0.5, 0.5, "No usable spectra", transform=ax.transAxes, ha="center", va="center")
            ax.set_xlim(0.0, (float(t1_ns) - float(t0_ns)) / 1e9)
            ax.set_ylim(-0.5 * args.width_khz, 0.5 * args.width_khz)
        else:
            mesh = draw_microdoppler_panel(
                ax,
                result,
                data,
                common_times_ns,
                t0_ns,
                t1_ns,
                args,
                show_xlabel=(site == SITE_ORDER[-1]),
            )
        ax.text(
            0.012,
            0.88,
            SITE_LABEL.get(site, site),
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 2.0},
        )
    axes[0].set_title(
        f"{event_id} dechirped FFT micro-Doppler\n"
        f"{-0.5 * args.width_khz:.0f} to {0.5 * args.width_khz:.0f} kHz; right axes show equivalent path offset"
    )
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, pad=0.015)
        cbar.set_label("Relative power (dB)")
    combined_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    png = os.path.join(combined_dir, f"{event_id}_microdoppler_fft_combined.png")
    fig.savefig(png, dpi=args.dpi)
    plt.close(fig)
    return png


def write_h5(summary_rows, output_dir, args):
    path = os.path.join(output_dir, "microdoppler_fft_manifest.h5")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["script"] = os.path.basename(__file__)
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["zero_pad_factor"] = int(args.zero_pad_factor)
        h.attrs["gate_upsample_factor"] = int(args.gate_upsample_factor)
        h.attrs["width_khz"] = float(args.width_khz)
        h.attrs["db_floor"] = float(args.db_floor)
        h.attrs["snr_min_db"] = float(args.snr_min_db)
        h.attrs["cmap"] = str(args.cmap)
        h.attrs["combined"] = bool(args.combined)
        h.attrs["right_axis"] = "equivalent path offset in meters: c * frequency_offset / chirp_rate"
        h.create_dataset("event_id", data=np.asarray([r["event_id"] for r in summary_rows], dtype=object), dtype=string_dtype)
        h.create_dataset("site", data=np.asarray([r["site"] for r in summary_rows], dtype=object), dtype=string_dtype)
        h.create_dataset("png", data=np.asarray([r["png"] for r in summary_rows], dtype=object), dtype=string_dtype)
        for key in ("event_index", "n_pulses", "n_valid", "t0_ns", "t1_ns"):
            h[key] = np.asarray([r[key] for r in summary_rows], dtype=np.int64)
    return path


def run(args):
    triplets = pair_tristatic_events(load_events(SAN_PATTERN), load_events(DAN_PATTERN), load_events(WEN_PATTERN))
    if args.max_events is not None:
        triplets = triplets[: args.max_events]
    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []
    for event_index, triplet in enumerate(triplets):
        event_id = tri_event_id(event_index, triplet)
        site_data = {site: load_raw_event(event) for site, event in zip(SITE_ORDER, triplet)}
        t0_ns = min(int(d["times_ns"].min()) for d in site_data.values())
        t1_ns = max(int(d["times_ns"].max()) for d in site_data.values())
        results_by_site = {site: microdoppler_image(site_data[site], args) for site in SITE_ORDER}
        if args.combined:
            png = write_combined_plot(results_by_site, site_data, event_id, event_index, t0_ns, t1_ns, args.output_dir, args)
            n_valid = int(sum(0 if r is None else r["n_valid"] for r in results_by_site.values()))
            n_pulses = int(sum(len(site_data[site]["times_ns"]) for site in SITE_ORDER))
            summary_rows.append(
                {
                    "event_index": int(event_index),
                    "event_id": event_id,
                    "site": "combined",
                    "png": png,
                    "n_pulses": n_pulses,
                    "n_valid": n_valid,
                    "t0_ns": int(t0_ns),
                    "t1_ns": int(t1_ns),
                }
            )
            print(f"{event_id} combined: {png}")
            continue
        for site in SITE_ORDER:
            result = results_by_site[site]
            if result is None:
                print(f"{event_id} {site}: no usable spectra")
                continue
            png = write_plot(result, site_data[site], event_id, event_index, site, t0_ns, t1_ns, args.output_dir, args)
            summary_rows.append(
                {
                    "event_index": int(event_index),
                    "event_id": event_id,
                    "site": site,
                    "png": png,
                    "n_pulses": int(len(site_data[site]["times_ns"])),
                    "n_valid": int(result["n_valid"]),
                    "t0_ns": int(t0_ns),
                    "t1_ns": int(t1_ns),
                }
            )
            print(f"{event_id} {site}: {png}")
    manifest = write_h5(summary_rows, args.output_dir, args)
    print(f"n_images={len(summary_rows)}")
    print(f"output_dir={os.path.abspath(args.output_dir)}")
    print(f"manifest={os.path.abspath(manifest)}")


def main():
    parser = argparse.ArgumentParser(description="Make peak-centered dechirped-FFT micro-Doppler images for tri-static events.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zero-pad-factor", type=int, default=64)
    parser.add_argument("--gate-upsample-factor", type=int, default=32)
    parser.add_argument("--width-khz", type=float, default=10.0)
    parser.add_argument("--db-floor", type=float, default=-45.0)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--snr-min-db", type=float, default=-np.inf)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
