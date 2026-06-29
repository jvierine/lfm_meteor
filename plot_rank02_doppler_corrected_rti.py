import os

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time

import sanya_opts as sc
from fit_gcrs_trajectories_lfm_ambiguity import REFERENCE_CHIRP_RATE_SCALE


C = 299792458.0
RADAR_FREQUENCY_HZ = sc.RADAR_FREQUENCY_HZ
RADAR_WAVELENGTH_M = sc.RADAR_WAVELENGTH_M
EVENT_ID_LOCAL = "tri_0134_1713850083054349899"
EVENT_ID_UTC = "tri_0134_1713821283054349899"
FIT_H5 = "results/gcrs_trajectory_fits_lfm_ambiguity_v20260613b.h5"
ARTICLE_FIGURE_DIR = "/Users/jvi019/src/sanya_tristatic_paper/figures"
OUTPUT_BASE = os.path.join(ARTICLE_FIGURE_DIR, "tristatic_rank02_doppler_corrected_rti")

SITE_ORDER = ["sanya", "danzhou", "wenchang"]
SITE_LABELS = {
    "sanya": "Sanya TX",
    "danzhou": "Danzhou RX",
    "wenchang": "Wenchang RX",
}
EVENT_PATHS = {
    "sanya": "results/tristatic_head_echoes/sanya/sanya_1713850083054349899.h5",
    "danzhou": "results/tristatic_head_echoes/danzhou/danzhou_1713850083119349957.h5",
    "wenchang": "results/tristatic_head_echoes/wenchang/wenchang_1713850083129349947.h5",
}
LINK_RX_POSITIONS_M = {
    "sanya": np.asarray(sc.p_san, dtype=np.float64),
    "danzhou": np.asarray(sc.p_dan, dtype=np.float64),
    "wenchang": np.asarray(sc.p_wen, dtype=np.float64),
}
TX_POSITION_M = np.asarray(sc.p_san, dtype=np.float64)


def lfm(length_us=199, sr_mhz=4.0, bandwidth_hz=4e6, chirp_rate_scale=REFERENCE_CHIRP_RATE_SCALE):
    t_s = np.arange(int(length_us * sr_mhz), dtype=np.float64) / (sr_mhz * 1e6)
    sweep_rate = bandwidth_hz * 1e6 / length_us / 2.0 * float(chirp_rate_scale)
    code = np.exp(1j * 2 * np.pi * (t_s * bandwidth_hz / 2.0 - sweep_rate * t_s**2.0))
    return code.astype(np.complex64), t_s


def load_fit():
    with h5py.File(FIT_H5, "r") as h:
        event_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h["event_id"][:]]
        idx = event_ids.index(EVENT_ID_UTC)
        return {
            "r0_gcrs_m": h["r0_gcrs_m"][idx],
            "v0_gcrs_mps": h["v0_gcrs_mps"][idx],
            "t0_ns": int(h["t0_ns"][idx]),
            "speed_km_s": float(h["speed_km_s"][idx]),
            "duration_s": float(h["duration_s"][idx]),
            "n_points": int(h["n_points"][idx]),
        }


def gcrs_state_to_itrs(r0_gcrs_m, v0_gcrs_mps, t_rel_s, times_ns):
    obstime = Time(np.asarray(times_ns, dtype=np.float64) / 1e9, format="unix", scale="utc")
    positions = r0_gcrs_m[None, :] + t_rel_s[:, None] * v0_gcrs_mps[None, :]
    representation = CartesianRepresentation(
        positions[:, 0] * u.m,
        positions[:, 1] * u.m,
        positions[:, 2] * u.m,
        differentials=CartesianDifferential(
            np.repeat(v0_gcrs_mps[0], len(t_rel_s)) * u.m / u.s,
            np.repeat(v0_gcrs_mps[1], len(t_rel_s)) * u.m / u.s,
            np.repeat(v0_gcrs_mps[2], len(t_rel_s)) * u.m / u.s,
        ),
    )
    gcrs = GCRS(representation, obstime=obstime)
    itrs = gcrs.transform_to(ITRS(obstime=obstime))
    positions_itrs = itrs.cartesian.without_differentials().xyz.to_value(u.m).T
    velocities_itrs = itrs.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s).T
    return positions_itrs, velocities_itrs


def fitted_doppler_hz(site, fit, times_ns):
    t_rel_s = (np.asarray(times_ns, dtype=np.float64) - float(fit["t0_ns"])) / 1e9
    positions, velocities = gcrs_state_to_itrs(fit["r0_gcrs_m"], fit["v0_gcrs_mps"], t_rel_s, times_ns)
    tx_vectors = positions - TX_POSITION_M[None, :]
    rx_vectors = positions - LINK_RX_POSITIONS_M[site][None, :]
    tx_unit = tx_vectors / np.linalg.norm(tx_vectors, axis=1)[:, None]
    rx_unit = rx_vectors / np.linalg.norm(rx_vectors, axis=1)[:, None]
    path_rate_mps = np.sum((tx_unit + rx_unit) * velocities, axis=1)
    return -path_rate_mps / RADAR_WAVELENGTH_M


def doppler_corrected_filter(raw, doppler_hz, sr_mhz, bandwidth_mhz):
    code, t_s = lfm(sr_mhz=sr_mhz, bandwidth_hz=bandwidth_mhz * 1e6)
    corrected = np.empty_like(raw, dtype=np.complex64)
    for idx, (row, fd_hz) in enumerate(zip(raw, doppler_hz)):
        # Use the received-chirp phase convention; the matched filter applies
        # conj(doppler_code), matching the single-pulse ACF Doppler diagnostic.
        doppler_code = code * np.exp(1j * 2.0 * np.pi * fd_hz * t_s).astype(np.complex64)
        corrected[idx, :] = np.convolve(row, np.conj(doppler_code), mode="same")
    return corrected


def load_and_filter_site(site, fit):
    with h5py.File(EVENT_PATHS[site], "r") as h:
        raw = h["raw"][()]
        old_echoes = h["echoes"][()]
        times_ns = h["times_ns"][()].astype(np.int64)
        ranges_km = h["ranges_km_axis"][()]
        old_range_km = h["range_km"][()]
        sr_mhz = float(h["sr_mhz"][()])
        bandwidth_mhz = float(h["bw_mhz"][()])

    doppler_hz = fitted_doppler_hz(site, fit, times_ns)
    corrected = doppler_corrected_filter(raw, doppler_hz, sr_mhz, bandwidth_mhz)
    power_db = 10.0 * np.log10(np.maximum(np.abs(corrected.T) ** 2.0, 1e-12))
    power_db -= np.nanmedian(power_db)
    old_power_db = 10.0 * np.log10(np.maximum(np.abs(old_echoes.T) ** 2.0, 1e-12))
    old_power_db -= np.nanmedian(old_power_db)
    peak_gate = np.argmax(power_db, axis=0)
    return {
        "times_ns": times_ns,
        "ranges_km": ranges_km,
        "power_db": power_db,
        "old_power_db": old_power_db,
        "peak_range_km": ranges_km[peak_gate],
        "old_range_km": old_range_km,
        "doppler_hz": doppler_hz,
    }


def plot_panel(ax, site, data, t0_ns, vmin, vmax):
    t_s = (data["times_ns"].astype(np.float64) - float(t0_ns)) / 1e9
    mesh = ax.pcolormesh(t_s, data["ranges_km"], data["power_db"], shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.plot(t_s, data["peak_range_km"], ".", color="white", ms=2.4, alpha=0.96, label="Doppler-corrected peak")
    ax.plot(t_s, data["old_range_km"], ".", color="#7ec8ff", ms=1.7, alpha=0.70, label="Original peak")
    center = float(np.nanmedian(data["old_range_km"]))
    spread = float(np.nanmax(data["old_range_km"]) - np.nanmin(data["old_range_km"]))
    ax.set_ylim(center - max(4.0, 0.5 * spread + 2.0), center + max(4.0, 0.5 * spread + 2.0))
    fd0 = np.nanmedian(data["doppler_hz"]) / 1e3
    ax.set_title(f"{SITE_LABELS[site]}  median fitted Doppler={fd0:.1f} kHz")
    ax.set_ylabel("Range (km)")
    return mesh


def plot_corrected(data, fit):
    os.makedirs(ARTICLE_FIGURE_DIR, exist_ok=True)
    t0_ns = min(int(site_data["times_ns"].min()) for site_data in data.values())
    vmax = max(float(np.nanpercentile(site_data["power_db"], 99.7)) for site_data in data.values())
    vmax = min(max(vmax, 20.0), 65.0)
    vmin = 0.0

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True, constrained_layout=True)
    mesh = None
    for ax, site in zip(axes, SITE_ORDER):
        mesh = plot_panel(ax, site, data[site], t0_ns, vmin, vmax)
    axes[-1].set_xlabel("Time since first station detection (s)")
    axes[0].legend(loc="upper right", frameon=True, fontsize=8)
    fig.suptitle(
        f"Doppler-corrected RTI, {EVENT_ID_LOCAL} "
        f"({fit['n_points']} fitted points, {fit['duration_s']:.3f} s, {fit['speed_km_s']:.1f} km/s)",
        fontsize=12,
    )
    cb = fig.colorbar(mesh, ax=axes, pad=0.018, shrink=0.88)
    cb.set_label("Doppler-corrected matched-filter power (dB)")
    fig.savefig(f"{OUTPUT_BASE}.png", dpi=260)
    fig.savefig(f"{OUTPUT_BASE}.pdf")
    plt.close(fig)

    for site in SITE_ORDER:
        fig, ax = plt.subplots(figsize=(6.7, 4.1), constrained_layout=True)
        mesh = plot_panel(ax, site, data[site], t0_ns, vmin, vmax)
        ax.set_xlabel("Time since first station detection (s)")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        cb = fig.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label("Doppler-corrected matched-filter power (dB)")
        fig.savefig(f"{OUTPUT_BASE}_{site}.png", dpi=260)
        fig.savefig(f"{OUTPUT_BASE}_{site}.pdf")
        plt.close(fig)


def write_h5(data, fit):
    with h5py.File(f"{OUTPUT_BASE}.h5", "w") as h:
        h.attrs["event_id_local"] = EVENT_ID_LOCAL
        h.attrs["event_id_utc_fit"] = EVENT_ID_UTC
        h.attrs["fit_h5"] = FIT_H5
        h.attrs["radar_frequency_hz"] = RADAR_FREQUENCY_HZ
        h.attrs["filter"] = "Per-pulse LFM template multiplied by exp(-i 2 pi fitted_doppler_hz t)"
        h.attrs["fit_duration_s"] = fit["duration_s"]
        h.attrs["fit_speed_km_s"] = fit["speed_km_s"]
        for site in SITE_ORDER:
            g = h.create_group(site)
            g["times_ns"] = data[site]["times_ns"]
            g["ranges_km"] = data[site]["ranges_km"]
            g["power_db"] = data[site]["power_db"]
            g["old_power_db"] = data[site]["old_power_db"]
            g["peak_range_km"] = data[site]["peak_range_km"]
            g["old_range_km"] = data[site]["old_range_km"]
            g["doppler_hz"] = data[site]["doppler_hz"]


def main():
    fit = load_fit()
    data = {site: load_and_filter_site(site, fit) for site in SITE_ORDER}
    write_h5(data, fit)
    plot_corrected(data, fit)
    print(f"event: {EVENT_ID_LOCAL}")
    print(f"fit: {EVENT_ID_UTC}")
    for site in SITE_ORDER:
        print(
            f"{site}: doppler kHz median/range "
            f"{np.nanmedian(data[site]['doppler_hz'])/1e3:.2f} / "
            f"{np.nanmin(data[site]['doppler_hz'])/1e3:.2f} to {np.nanmax(data[site]['doppler_hz'])/1e3:.2f}"
        )
    print(f"wrote: {OUTPUT_BASE}.png")
    print(f"wrote: {OUTPUT_BASE}.pdf")
    print(f"wrote: {OUTPUT_BASE}.h5")


if __name__ == "__main__":
    main()
