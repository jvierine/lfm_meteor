#!/usr/bin/env python3
"""Measured system-noise model for Sanya tri-static analysis."""

from __future__ import annotations

import h5py
import numpy as np


SYSTEM_NOISE_H5 = "/Users/jvi019/src/lfm_meteor/results/sanya_4mhz_system_noise_power_100pulse.h5"
SITE_NAMES = ("Sanya", "Danzhou", "Wenchang")

# Fitted by /Users/jvi019/src/lfm_meteor/plot_system_noise_floor_gdsm_fit.py
# from P(t) = C * [T_sky(t) + T_rec], using the Memo 20 settings.
POWER_PER_K = {
    "Sanya": 4.353530e12,
    "Danzhou": 2.461363e13,
    "Wenchang": 2.463654e13,
}
FITTED_T_REC_K = {
    "Sanya": 249.901,
    "Danzhou": 404.922,
    "Wenchang": 390.549,
}
MEDIAN_TSYS_K = {
    "Sanya": 267.540,
    "Danzhou": 434.333,
    "Wenchang": 411.363,
}


def decode_strings(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


class MeasuredSystemNoise:
    """Interpolate measured low-rate raw noise power as equivalent T_sys."""

    def __init__(self, path: str = SYSTEM_NOISE_H5):
        self.path = path
        self.time_ns_by_site: dict[str, np.ndarray] = {}
        self.tsys_k_by_site: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as h:
            names = decode_strings(h["site_names"][:])
            station_id = h["bins/station_id"][:]
            time_ns = h["bins/time_utc_mid_ns"][:].astype(np.int64)
            power = h["bins/noise_power_mean_raw_voltage"][:].astype(np.float64)

        for sid, name in enumerate(names):
            if name not in POWER_PER_K:
                continue
            good = (station_id == sid) & np.isfinite(time_ns) & np.isfinite(power) & (power > 0.0)
            t = time_ns[good]
            tsys = power[good] / POWER_PER_K[name]
            order = np.argsort(t)
            self.time_ns_by_site[name] = t[order]
            self.tsys_k_by_site[name] = tsys[order]

    def tsys_k(self, site: str, time_ns: np.ndarray) -> np.ndarray:
        """Return interpolated equivalent system temperature in kelvin."""

        if site not in self.time_ns_by_site:
            raise KeyError(f"unknown site {site!r}")
        t = self.time_ns_by_site[site].astype(np.float64)
        tsys = self.tsys_k_by_site[site]
        query = np.asarray(time_ns, dtype=np.float64)
        out = np.interp(query, t, tsys, left=np.nan, right=np.nan)
        return out.astype(np.float64)

    def summary_text(self) -> str:
        parts = []
        for name in SITE_NAMES:
            parts.append(
                f"{name}: C={POWER_PER_K[name]:.6e} raw-power K^-1, "
                f"Trec={FITTED_T_REC_K[name]:.1f} K, median Tsys={MEDIAN_TSYS_K[name]:.1f} K"
            )
        return "; ".join(parts)
