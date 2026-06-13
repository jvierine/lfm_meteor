import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import test_rank02_range_interpolation as interp


SCRIPT_VERSION = "v20260611f"
OUTPUT_BASE = os.path.join("results", f"rank02_single_pulse_acf_doppler_{SCRIPT_VERSION}")
UPSAMPLE_FACTOR = 4
MAX_LAG_US = 80.0
MIN_LAG_SAMPLES = 2
SNR_MIN_DB = 6.0
COHERENCE_MIN = 0.70


def deramped_echo_segment(row, gate, fd_hz, sr_mhz, bw_mhz, upsample_factor=1):
    if upsample_factor == 1:
        row_work = row
        sr_work_mhz = sr_mhz
        center = int(round(float(gate)))
    else:
        row_work = sig.resample_poly(row, upsample_factor, 1).astype(np.complex64)
        sr_work_mhz = sr_mhz * upsample_factor
        center = int(round(float(gate) * upsample_factor))
    code, t_s = interp.lfm(sr_mhz=sr_work_mhz, bandwidth_hz=bw_mhz * 1e6)
    n_code = len(code)
    start = center - n_code // 2
    stop = start + n_code
    if start < 0 or stop > len(row_work):
        return None, None
    segment = row_work[start:stop].astype(np.complex128, copy=False)
    # Direct raw-pulse deramping uses the received chirp phase convention.
    # The matched-filter implementation later conjugates the reference.
    doppler_code = code.astype(np.complex128) * np.exp(1j * 2.0 * np.pi * fd_hz * t_s)
    deramped = segment * np.conj(doppler_code)
    return deramped, t_s


def acf_residual_doppler_hz(deramped, sr_hz, max_lag_us=MAX_LAG_US):
    max_lag = min(int(round(max_lag_us * 1e-6 * sr_hz)), len(deramped) // 2)
    lags = np.arange(MIN_LAG_SAMPLES, max_lag + 1, dtype=np.int64)
    if len(lags) < 4:
        return np.nan, np.nan, np.nan

    window = np.hanning(len(deramped))
    y = deramped * window
    acf = np.asarray([np.sum(y[lag:] * np.conj(y[:-lag])) for lag in lags], dtype=np.complex128)
    amp = np.abs(acf)
    good = amp > 0.08 * np.nanmax(amp)
    if np.count_nonzero(good) < 4:
        return np.nan, np.nan, np.nan

    lags_good = lags[good].astype(np.float64)
    phase = np.unwrap(np.angle(acf[good]))
    weights = amp[good] / np.nanmax(amp[good])
    x_s = lags_good / sr_hz
    coeff = np.polyfit(x_s, phase, 1, w=np.sqrt(weights))
    slope_rad_s = float(coeff[0])
    residual_fd_hz = -slope_rad_s / (2.0 * np.pi)

    fitted_phase = np.polyval(coeff, x_s)
    phase_rms_rad = float(np.sqrt(np.average((phase - fitted_phase) ** 2.0, weights=weights)))
    coherence = float(np.abs(np.sum(acf[good])) / max(np.sum(amp[good]), 1e-30))
    return residual_fd_hz, phase_rms_rad, coherence


def measure_site(site, site_data, peak_gate, upsample_factor=1):
    n = site_data["raw"].shape[0]
    residual_hz = np.full(n, np.nan, dtype=np.float64)
    measured_hz = np.full(n, np.nan, dtype=np.float64)
    phase_rms_rad = np.full(n, np.nan, dtype=np.float64)
    coherence = np.full(n, np.nan, dtype=np.float64)

    sr_hz = float(site_data["sr_mhz"]) * 1e6
    for idx in range(n):
        if float(site_data["snr_peak_db"][idx]) < SNR_MIN_DB:
            continue
        deramped, _t_s = deramped_echo_segment(
            site_data["raw"][idx],
            peak_gate[idx],
            float(site_data["doppler_hz"][idx]),
            site_data["sr_mhz"],
            site_data["bw_mhz"],
            upsample_factor=upsample_factor,
        )
        if deramped is None:
            continue
        resid, phase_rms, coh = acf_residual_doppler_hz(deramped, sr_hz)
        if not np.isfinite(coh) or coh < COHERENCE_MIN:
            phase_rms_rad[idx] = phase_rms
            coherence[idx] = coh
            continue
        residual_hz[idx] = resid
        measured_hz[idx] = float(site_data["doppler_hz"][idx]) + resid
        phase_rms_rad[idx] = phase_rms
        coherence[idx] = coh

    return {
        "fitted_hz": site_data["doppler_hz"],
        "acf_measured_hz": measured_hz,
        "acf_residual_hz": residual_hz,
        "phase_rms_rad": phase_rms_rad,
        "coherence": coherence,
        "snr_peak_db": site_data["snr_peak_db"],
        "times_ns": site_data["times_ns"],
    }


def plot_results(results):
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.4), sharex=False, constrained_layout=True)
    for ax, site in zip(axes, interp.SITE_ORDER):
        data = results[site]
        t_s = (data["times_ns"].astype(np.float64) - float(np.nanmin(data["times_ns"]))) / 1e9
        good = np.isfinite(data["acf_measured_hz"])
        ax.plot(t_s, data["fitted_hz"] / 1e3, color="0.65", lw=1.3, label="trajectory fitted Doppler")
        sc = ax.scatter(
            t_s[good],
            data["acf_measured_hz"][good] / 1e3,
            c=data["coherence"][good],
            s=24,
            cmap="viridis",
            vmin=0,
            vmax=1,
            label="phase-slope ACF estimate",
            zorder=3,
        )
        ax.set_title(site)
        ax.set_ylabel("Doppler (kHz)")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time since first pulse at station (s)")
    cb = fig.colorbar(sc, ax=axes, pad=0.018, shrink=0.9)
    cb.set_label("ACF coherence")
    fig.suptitle("Rank02 single-pulse phase-slope ACF Doppler diagnostic", fontsize=12)
    fig.savefig(f"{OUTPUT_BASE}.png", dpi=240)
    fig.savefig(f"{OUTPUT_BASE}.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.8), sharex=True, constrained_layout=True)
    bins = np.linspace(-80, 80, 81)
    for ax, site in zip(axes, interp.SITE_ORDER):
        data = results[site]
        good = np.isfinite(data["acf_residual_hz"])
        ax.hist(data["acf_residual_hz"][good] / 1e3, bins=bins, color="#4c78a8", alpha=0.82)
        med = np.nanmedian(data["acf_residual_hz"][good]) / 1e3 if np.any(good) else np.nan
        mad = 1.4826 * np.nanmedian(np.abs(data["acf_residual_hz"][good] - np.nanmedian(data["acf_residual_hz"][good]))) / 1e3 if np.any(good) else np.nan
        ax.axvline(0, color="0.2", lw=1.0)
        ax.axvline(med, color="#f58518", lw=1.6, label=f"median={med:.1f} kHz, robust sigma={mad:.1f} kHz")
        ax.set_ylabel(site)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("ACF residual Doppler after rough correction (kHz)")
    fig.suptitle("Single-pulse phase-slope ACF residual Doppler distribution", fontsize=12)
    fig.savefig(f"{OUTPUT_BASE}_residual_hist.png", dpi=240)
    fig.savefig(f"{OUTPUT_BASE}_residual_hist.pdf")
    plt.close(fig)


def write_h5(results):
    with h5py.File(f"{OUTPUT_BASE}.h5", "w") as h:
        h.attrs["script_version"] = SCRIPT_VERSION
        h.attrs["event_id_local"] = interp.EVENT_ID_LOCAL
        h.attrs["event_id_utc_fit"] = interp.EVENT_ID_UTC
        h.attrs["method"] = "Deramp raw pulse with rough fitted Doppler/LFM code, then fit ACF lag phase slope."
        h.attrs["max_lag_us"] = MAX_LAG_US
        h.attrs["snr_min_db"] = SNR_MIN_DB
        h.attrs["coherence_min"] = COHERENCE_MIN
        for site, data in results.items():
            g = h.create_group(site)
            for key, value in data.items():
                g[key] = value


def main():
    fit = interp.load_reference_fit()
    site_data = {site: interp.load_site(site, fit) for site in interp.SITE_ORDER}
    coarse = interp.precompute_coarse_gates(site_data)
    refined = {}
    for site in interp.SITE_ORDER:
        fine_gate, _fine_range_km, _power_db = interp.refine_site_ranges(site_data[site], UPSAMPLE_FACTOR, coarse[site])
        refined[site] = fine_gate
    results = {site: measure_site(site, site_data[site], refined[site], upsample_factor=UPSAMPLE_FACTOR) for site in interp.SITE_ORDER}
    write_h5(results)
    plot_results(results)

    print(f"wrote {OUTPUT_BASE}.h5")
    print(f"wrote {OUTPUT_BASE}.png")
    for site in interp.SITE_ORDER:
        data = results[site]
        good = np.isfinite(data["acf_residual_hz"])
        if not np.any(good):
            print(f"{site}: no valid ACF estimates")
            continue
        residual_khz = data["acf_residual_hz"][good] / 1e3
        print(
            f"{site}: valid={np.count_nonzero(good)}/{len(good)}, "
            f"median residual={np.nanmedian(residual_khz):.2f} kHz, "
            f"robust sigma={1.4826 * np.nanmedian(np.abs(residual_khz - np.nanmedian(residual_khz))):.2f} kHz, "
            f"median coherence={np.nanmedian(data['coherence'][good]):.2f}"
        )


if __name__ == "__main__":
    main()
