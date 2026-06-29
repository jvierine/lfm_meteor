#!/usr/bin/env python
"""Calculate Sanya tri-static station baseline distances."""

from __future__ import annotations

import numpy as np
from pyproj import Geod

import jcoord
import sanya_opts as so


STATION_NAMES = ("Sanya", "Danzhou", "Wenchang")
WGS84_GEOD = Geod(ellps="WGS84")


def station_ecef_m(station_index: int) -> np.ndarray:
    """Return station position in WGS84 ECEF meters."""
    return np.asarray(
        jcoord.geodetic2ecef(
            so.lat0[station_index],
            so.lon0[station_index],
            so.alt0[station_index] * 1e3,
        ),
        dtype=float,
    )


def baseline_from_sanya(remote_index: int) -> dict[str, float | str]:
    """Calculate surface and straight-line baselines from Sanya."""
    sanya_index = 0
    _, _, surface_distance_m = WGS84_GEOD.inv(
        so.lon0[sanya_index],
        so.lat0[sanya_index],
        so.lon0[remote_index],
        so.lat0[remote_index],
    )

    sanya_ecef_m = station_ecef_m(sanya_index)
    remote_ecef_m = station_ecef_m(remote_index)
    chord_distance_m = float(np.linalg.norm(remote_ecef_m - sanya_ecef_m))

    return {
        "remote": STATION_NAMES[remote_index],
        "surface_distance_km": surface_distance_m / 1e3,
        "chord_distance_km": chord_distance_m / 1e3,
        "altitude_difference_m": (so.alt0[remote_index] - so.alt0[sanya_index]) * 1e3,
    }


def main() -> None:
    print("Station coordinates from sanya_opts.py")
    for name, lat_deg, lon_deg, alt_km in zip(STATION_NAMES, so.lat0, so.lon0, so.alt0):
        print(f"{name:8s}: lat {lat_deg:9.4f} deg, lon {lon_deg:10.4f} deg, alt {alt_km:7.4f} km")

    print("\nBaselines from Sanya")
    print("Remote    WGS84 surface distance (km)    ECEF chord distance (km)    Remote-Sanya alt (m)")
    for remote_index in (1, 2):
        baseline = baseline_from_sanya(remote_index)
        print(
            f"{baseline['remote']:8s}"
            f"{baseline['surface_distance_km']:28.3f}"
            f"{baseline['chord_distance_km']:27.3f}"
            f"{baseline['altitude_difference_m']:22.1f}"
        )


if __name__ == "__main__":
    main()
