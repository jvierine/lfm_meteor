#!/usr/bin/env python3
"""Plot gain-weighted pygdsm sky-noise temperature for fixed Sanya beams."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import astropy.units as u
import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pygdsm
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

import plot_memo09_antenna_gain_patterns as gain_model
import rangedelay
import sanya_opts as sc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--system-noise-h5",
        default="/Users/jvi019/src/lfm_meteor/results/sanya_4mhz_system_noise_power_100pulse.h5",
        help="Reduced low-rate system-noise HDF5 product used only to define the time interval.",
    )
    p.add_argument("--frequency-mhz", type=float, default=sc.RADAR_FREQUENCY_MHZ, help="Frequency for the pygdsm model.")
    p.add_argument("--cadence-min", type=float, default=2.5, help="Time cadence for the sky-noise model.")
    p.add_argument("--beam-radius-deg", type=float, default=5.0, help="Angular radius around beam axis for gain-weighted averaging.")
    p.add_argument("--beam-grid-step-deg", type=float, default=0.1, help="Grid step for beam-pattern integration.")
    p.add_argument(
        "--remote-effective-aperture-scale",
        type=float,
        default=1.0,
        help="Scale Danzhou/Wenchang aperture dimensions for an empirically broadened sky-noise receive pattern.",
    )
    p.add_argument(
        "--output-dir",
        default="/Users/jvi019/src/sanya_tristatic_paper/memos/figures",
        help="Directory for PDF and PNG outputs.",
    )
    p.add_argument(
        "--basename",
        default=f"memo20_pygdsm_station_sky_noise_gain_weighted_{int(round(sc.RADAR_FREQUENCY_MHZ))}mhz",
        help="Output filename stem.",
    )
    p.add_argument("--include-cmb", action="store_true", help="Add the 2.725 K CMB contribution.")
    return p.parse_args()


def ns_interval(path: str) -> tuple[int, int]:
    with h5py.File(path, "r") as h:
        t = h["bins/time_utc_mid_ns"][:]
    return int(np.nanmin(t)), int(np.nanmax(t))


def make_times(start_ns: int, stop_ns: int, cadence_min: float) -> Time:
    step_ns = int(round(cadence_min * 60.0 * 1e9))
    if step_ns <= 0:
        raise ValueError("--cadence-min must be positive")
    ns = np.arange(start_ns, stop_ns + step_ns, step_ns, dtype=np.int64)
    ns[-1] = min(ns[-1], stop_ns)
    unix_s = ns.astype(np.float64) / 1e9
    return Time(unix_s, format="unix", scale="utc")


def time_to_mpl(time: Time) -> np.ndarray:
    return mdates.date2num(time.to_datetime())


def effective_site(site: gain_model.SitePattern, remote_scale: float) -> gain_model.SitePattern:
    if site.name == "Sanya":
        return site
    if remote_scale <= 0.0:
        raise ValueError("--remote-effective-aperture-scale must be positive")
    return dataclasses.replace(
        site,
        dim_tilt_plane_m=site.dim_tilt_plane_m * remote_scale,
        dim_cross_tilt_m=site.dim_cross_tilt_m * remote_scale,
    )


def integration_grid(site: gain_model.SitePattern, radius_deg: float, step_deg: float, remote_scale: float) -> tuple[np.ndarray, np.ndarray]:
    if radius_deg <= 0.0:
        raise ValueError("--beam-radius-deg must be positive")
    if step_deg <= 0.0:
        raise ValueError("--beam-grid-step-deg must be positive")
    site = effective_site(site, remote_scale)

    offsets = np.arange(-radius_deg, radius_deg + 0.5 * step_deg, step_deg, dtype=np.float64)
    scan_offset_deg, cross_offset_deg = np.meshgrid(offsets, offsets)
    tx = np.tan(np.deg2rad(scan_offset_deg))
    ty = np.tan(np.deg2rad(cross_offset_deg))
    off_axis_deg = np.rad2deg(np.arccos(np.clip(1.0 / np.sqrt(1.0 + tx**2 + ty**2), -1.0, 1.0)))
    inside = off_axis_deg <= radius_deg

    pointing = gain_model.unit(gain_model.azel_to_enu(site.pointing_az_deg, site.pointing_el_deg))
    _normal, tilt_axis, panel_cross_axis = gain_model.panel_axes(site)
    scan_axis, plot_cross_axis = gain_model.offset_basis(pointing, tilt_axis, panel_cross_axis)
    directions = gain_model.directions_from_offsets(pointing, scan_axis, plot_cross_axis, scan_offset_deg, cross_offset_deg)
    aperture_power = gain_model.aperture_power(
        directions,
        pointing,
        tilt_axis,
        panel_cross_axis,
        site.dim_tilt_plane_m,
        site.dim_cross_tilt_m,
    )

    solid_angle_factor = (
        (1.0 / np.cos(np.deg2rad(scan_offset_deg)) ** 2)
        * (1.0 / np.cos(np.deg2rad(cross_offset_deg)) ** 2)
        / (1.0 + tx**2 + ty**2) ** 1.5
    )
    weights = np.where(inside, aperture_power * solid_angle_factor, 0.0).ravel()
    directions = directions.reshape(-1, 3)
    keep = weights > 0.0
    weights = weights[keep]
    weights = weights / np.sum(weights)
    return directions[keep], weights


def directions_to_az_el(directions_enu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    east = directions_enu[:, 0]
    north = directions_enu[:, 1]
    up = directions_enu[:, 2]
    az_deg = np.rad2deg(np.arctan2(east, north)) % 360.0
    el_deg = np.rad2deg(np.arcsin(np.clip(up, -1.0, 1.0)))
    return az_deg, el_deg


def station_sky_temperature(
    gsm: pygdsm.GlobalSkyModel,
    name: str,
    times: Time,
    frequency_mhz: float,
    include_cmb: bool,
    radius_deg: float,
    step_deg: float,
    remote_scale: float,
) -> tuple[np.ndarray, int]:
    lat_deg, lon_deg, alt_km = rangedelay.SITE_COORDS[name]
    site = next(s for s in gain_model.SITES if s.name == name)
    directions_enu, weights = integration_grid(site, radius_deg, step_deg, remote_scale)
    az_deg, el_deg = directions_to_az_el(directions_enu)
    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_km * 1000.0 * u.m)
    weighted = np.empty(len(times), dtype=np.float64)
    for i, time in enumerate(times):
        altaz = AltAz(obstime=time, location=loc)
        coords = SkyCoord(az=az_deg * u.deg, alt=el_deg * u.deg, frame=altaz)
        sky = np.asarray(gsm.get_sky_temperature(coords, freqs=frequency_mhz, include_cmb=include_cmb), dtype=np.float64)
        weighted[i] = float(np.sum(weights * sky))
    return weighted, len(weights)


def main() -> None:
    args = parse_args()
    iers.conf.auto_download = False
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    start_ns, stop_ns = ns_interval(args.system_noise_h5)
    times = make_times(start_ns, stop_ns, args.cadence_min)
    x = time_to_mpl(times)

    gsm = pygdsm.GlobalSkyModel(freq_unit="MHz", include_cmb=False)
    colors = {
        "Sanya": "#1f77b4",
        "Danzhou": "#2ca02c",
        "Wenchang": "#d62728",
    }

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    n_samples_by_station = {}
    for name in ("Sanya", "Danzhou", "Wenchang"):
        temp_k, n_samples = station_sky_temperature(
            gsm,
            name,
            times,
            args.frequency_mhz,
            args.include_cmb,
            args.beam_radius_deg,
            args.beam_grid_step_deg,
            args.remote_effective_aperture_scale,
        )
        n_samples_by_station[name] = n_samples
        ax.plot(x, temp_k, lw=1.8, color=colors[name], label=name)

    ax.set_title(f"Gain-weighted pygdsm sky noise at {args.frequency_mhz:g} MHz")
    ax.set_ylabel(r"$T_{\mathrm{sky}}$ (K)")
    ax.set_xlabel("UTC time")
    ax.grid(True, color="0.88", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, frameon=False)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    pdf = outdir / f"{args.basename}.pdf"
    png = outdir / f"{args.basename}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)

    print(f"time_start_utc={times[0].isot}")
    print(f"time_stop_utc={times[-1].isot}")
    for name, n_samples in n_samples_by_station.items():
        print(f"{name}_beam_samples={n_samples}")
    print(f"remote_effective_aperture_scale={args.remote_effective_aperture_scale}")
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
