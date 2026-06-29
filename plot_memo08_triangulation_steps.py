#!/usr/bin/env python3
"""Make step-by-step figures for the tri-static triangulation memo."""

from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

import sanya_opts as sc

EVENT_ID_LOCAL = "tri_0134_1713850083054349899"
EVENT_PATHS = {
    "sanya": "results/tristatic_head_echoes/sanya/sanya_1713850083054349899.h5",
    "danzhou": "results/tristatic_head_echoes/danzhou/danzhou_1713850083119349957.h5",
    "wenchang": "results/tristatic_head_echoes/wenchang/wenchang_1713850083129349947.h5",
}
SITE_LABELS = {
    "sanya": "Sanya transmit/receive path",
    "danzhou": "Sanya transmit--Danzhou receive path",
    "wenchang": "Sanya transmit--Wenchang receive path",
}
SITE_ORDER = ("sanya", "danzhou", "wenchang")
C_KM_PER_S = 299792458.0 / 1e3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="/Users/jvi019/src/sanya_tristatic_paper/figures")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--event-id", default=EVENT_ID_LOCAL)
    return p.parse_args()


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return value


def total_path_axis_km(ranges_km_axis: np.ndarray) -> np.ndarray:
    """Convert stored half-path RTI axis to Memo 3 total tx-target-rx path."""
    return 2.0 * (np.asarray(ranges_km_axis, dtype=np.float64) + sc.SANYA_RANGE_CORRECTION_KM)


def gate_to_total_path_km(gate: np.ndarray, sr_mhz: float) -> np.ndarray:
    delay_us = sc.SANYA_CORRECTED_TXRX_DELAY_US + np.asarray(gate, dtype=np.float64) / float(sr_mhz)
    return C_KM_PER_S * delay_us * 1e-6


def load_site(path: str) -> dict:
    with h5py.File(path, "r") as h:
        echoes = h["echoes"][()]
        power_db = 10.0 * np.log10(np.maximum(np.abs(echoes.T) ** 2.0, 1e-12))
        power_db -= np.nanmedian(power_db)
        times_ns = h["times_ns"][()].astype(np.int64)
        t_s = (times_ns.astype(np.float64) - float(times_ns.min())) / 1e9
        ranges_km_axis = h["ranges_km_axis"][()].astype(np.float64)
        range_gate = h["range_gate"][()].astype(np.float64)
        sr_mhz = float(h["sr_mhz"][()])
        return {
            "site": str(decode_scalar(h["site"][()])),
            "t_s": t_s,
            "total_path_axis_km": total_path_axis_km(ranges_km_axis),
            "power_db": power_db,
            "peak_total_path_km": gate_to_total_path_km(range_gate, sr_mhz),
            "snr_peak_db": h["snr_peak_db"][()].astype(np.float64),
            "sr_mhz": sr_mhz,
            "az_deg": float(h["az"][()]),
            "el_deg": float(h["el"][()]),
        }


def plot_site(site: str, data: dict, output_dir: str, event_id: str) -> str:
    y = data["total_path_axis_km"]
    p = data["power_db"]
    finite = np.isfinite(p)
    vmax = float(np.nanpercentile(p[finite], 99.7)) if np.any(finite) else 30.0
    vmax = min(max(vmax, 18.0), 65.0)
    vmin = 0.0
    peak = data["peak_total_path_km"]
    center = float(np.nanmedian(peak))
    spread = float(np.nanmax(peak) - np.nanmin(peak))
    half_width = max(6.0, 0.5 * spread + 4.0)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    mesh = ax.pcolormesh(data["t_s"], y, p, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.plot(data["t_s"], peak, ".", color="#7ec8ff", ms=2.4, alpha=0.92, label="Detected peak")
    ax.set_ylim(center - half_width, center + half_width)
    ax.set_xlabel("Time since first pulse at this station (s)")
    ax.set_ylabel("Total tx-target-rx path, $L=c\\tau$ (km)")
    ax.set_title(f"{SITE_LABELS[site]}\n{event_id}; az/el={data['az_deg']:.1f}/{data['el_deg']:.1f} deg")
    ax.legend(loc="upper right", fontsize=8)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Matched-filter power above median (dB)")
    out = os.path.join(output_dir, f"memo08_{event_id}_{site}_rti_total_path.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for site in SITE_ORDER:
        data = load_site(EVENT_PATHS[site])
        out = plot_site(site, data, args.output_dir, args.event_id)
        print(out)
        print(
            f"{site}: total path peak median/range "
            f"{np.nanmedian(data['peak_total_path_km']):.3f} / "
            f"{np.nanmin(data['peak_total_path_km']):.3f} to "
            f"{np.nanmax(data['peak_total_path_km']):.3f} km; "
            f"SNR median={np.nanmedian(data['snr_peak_db']):.1f} dB"
        )


if __name__ == "__main__":
    main()
