"""Python translation of ``rangedelay.m``.

The public ``rangedelay`` function keeps the MATLAB calling convention:
latitudes, longitudes, azimuths, and elevations are in degrees, altitudes and
returned ranges are in kilometers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


SPEED_OF_LIGHT_KM_S = 299792.458
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

SITE_COORDS = {
    "Sanya": (18.3492, 109.6222, 0.05),
    "Danzhou": (19.5281, 109.1322, 0.0999),
    "Wenchang": (19.5982, 110.7908, 0.0249),
}

POINTINGS = {
    "Sanya": (14.996337890625, 74.9981689453125),
    "Danzhou": (151.2652587890625, 37.3260498046875),
    "Wenchang": (225.7855224609375, 29.2950439453125),
}


@dataclass(frozen=True)
class RangeDelayResult:
    alt_cross: float
    rg_t: float
    rg_r: float
    rg_delay: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.alt_cross, self.rg_t, self.rg_r, self.rg_delay


@dataclass(frozen=True)
class StationPrediction:
    station: str
    alt_cross: float
    rg_t: float
    rg_r: float
    rg_delay: float
    delay_s: float
    total_path_km: float


def geodetic_to_ecef_km(lat_deg, lon_deg, alt_km):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    alt = np.asarray(alt_km, dtype=np.float64)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (n + alt) * cos_lat * np.cos(lon)
    y = (n + alt) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return x, y, z


def ecef_to_geodetic_km(x_km, y_km, z_km):
    x = np.asarray(x_km, dtype=np.float64)
    y = np.asarray(y_km, dtype=np.float64)
    z = np.asarray(z_km, dtype=np.float64)

    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    alt = np.zeros_like(lat, dtype=np.float64)

    for _ in range(8):
        sin_lat = np.sin(lat)
        n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))

    sin_lat = np.sin(lat)
    n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / np.cos(lat) - n

    return np.rad2deg(lat), np.rad2deg(lon), alt


def aer_to_geodetic_km(az_deg, el_deg, range_km, lat0_deg, lon0_deg, alt0_km):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    rg = np.asarray(range_km, dtype=np.float64)

    east = rg * np.cos(el) * np.sin(az)
    north = rg * np.cos(el) * np.cos(az)
    up = rg * np.sin(el)

    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    x0, y0, z0 = geodetic_to_ecef_km(lat0_deg, lon0_deg, alt0_km)

    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)

    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up

    return ecef_to_geodetic_km(x0 + dx, y0 + dy, z0 + dz)


def geodetic_to_aer_km(lat_deg, lon_deg, alt_km, lat0_deg, lon0_deg, alt0_km):
    x, y, z = geodetic_to_ecef_km(lat_deg, lon_deg, alt_km)
    x0, y0, z0 = geodetic_to_ecef_km(lat0_deg, lon0_deg, alt0_km)
    dx = x - x0
    dy = y - y0
    dz = z - z0

    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    horizontal = np.hypot(east, north)
    rg = np.sqrt(east * east + north * north + up * up)
    az = np.mod(np.rad2deg(np.arctan2(east, north)), 360.0)
    el = np.rad2deg(np.arctan2(up, horizontal))
    return az, el, rg


def _interp1_monotonic(x, y, xq):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = float(xq)

    if np.all(np.diff(x) < 0.0):
        x = x[::-1]
        y = y[::-1]
    elif not np.all(np.diff(x) > 0.0):
        raise ValueError("interpolation coordinate is not monotonic")

    if xq < x[0] or xq > x[-1]:
        return float("nan")
    return float(np.interp(xq, x, y))


def _interp_lon_by_alt(alt_km, lon_deg, alt_cross_km):
    lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon_deg)))
    lon_cross = _interp1_monotonic(alt_km, lon_unwrapped, alt_cross_km)
    return float((lon_cross + 180.0) % 360.0 - 180.0)


def rangedelay(
    lat_t,
    lon_t,
    alt_t,
    az_t,
    el_t,
    lat_r,
    lon_r,
    alt_r,
    az_r,
    el_r,
    ranges_km=None,
    *,
    as_dataclass=False,
):
    """Estimate common-volume altitude, ranges, and receiver delay.

    This is a direct translation of ``rangedelay.m``. It samples the
    transmitter beam, views those samples from the receiver, finds the first
    monotonic coordinate that crosses the receiver beam pointing, and computes
    the bistatic range-delay term ``rg_r - (rg_r + rg_t) / 2``.
    """

    if ranges_km is None:
        ranges_km = np.arange(50.0, 2000.0 + 2.5, 5.0)
    ranges_km = np.asarray(ranges_km, dtype=np.float64)

    lat0, lon0, alt0 = aer_to_geodetic_km(az_t, el_t, ranges_km, lat_t, lon_t, alt_t)
    az_from_r, el_from_r, _ = geodetic_to_aer_km(lat0, lon0, alt0, lat_r, lon_r, alt_r)

    if np.all(np.diff(el_from_r) > 0.0) or np.all(np.diff(el_from_r) < 0.0):
        alt_cross = _interp1_monotonic(el_from_r, alt0, el_r)
    elif np.all(np.diff(az_from_r) > 0.0) or np.all(np.diff(az_from_r) < 0.0):
        alt_cross = _interp1_monotonic(az_from_r, alt0, az_r)
    else:
        raise ValueError("neither receiver elevation nor azimuth is monotonic along transmitter beam")

    lat_cross = _interp1_monotonic(alt0, lat0, alt_cross)
    lon_cross = _interp_lon_by_alt(alt0, lon0, alt_cross)

    _, _, rg_t = geodetic_to_aer_km(lat_cross, lon_cross, alt_cross, lat_t, lon_t, alt_t)
    _, _, rg_r = geodetic_to_aer_km(lat_cross, lon_cross, alt_cross, lat_r, lon_r, alt_r)
    rg_t = float(rg_t)
    rg_r = float(rg_r)
    rg_delay = rg_r - (rg_r + rg_t) / 2.0

    result = RangeDelayResult(float(alt_cross), rg_t, rg_r, float(rg_delay))
    if as_dataclass:
        return result
    return result.as_tuple()


def range_delay_km_to_bistatic_delay_s(rg_delay_km):
    """Convert the MATLAB range-delay offset to the delay term in the path formula."""

    return 2.0 * float(rg_delay_km) / SPEED_OF_LIGHT_KM_S


def total_path_from_delay_s(r0_km, gate, fs_hz, delay_s):
    """Return ``2*r0 + c*gate/fs + delay*c`` in km."""

    return (
        2.0 * float(r0_km)
        + SPEED_OF_LIGHT_KM_S * float(gate) / float(fs_hz)
        + SPEED_OF_LIGHT_KM_S * float(delay_s)
    )


def total_path_from_range_delay_km(r0_km, gate, fs_hz, rg_delay_km):
    """Return the bistatic total path using ``rangedelay``'s km offset."""

    return total_path_from_delay_s(
        r0_km,
        gate,
        fs_hz,
        range_delay_km_to_bistatic_delay_s(rg_delay_km),
    )


def predict_station_total_path(station, r0_km, gate, fs_hz):
    """Predict Wenchang or Danzhou total path length from the nominal beam geometry."""

    if station not in ("Danzhou", "Wenchang"):
        raise ValueError("station must be 'Danzhou' or 'Wenchang'")

    lat_t, lon_t, alt_t = SITE_COORDS["Sanya"]
    az_t, el_t = POINTINGS["Sanya"]
    lat_r, lon_r, alt_r = SITE_COORDS[station]
    az_r, el_r = POINTINGS[station]

    result = rangedelay(
        lat_t,
        lon_t,
        alt_t,
        az_t,
        el_t,
        lat_r,
        lon_r,
        alt_r,
        az_r,
        el_r,
        as_dataclass=True,
    )
    delay_s = range_delay_km_to_bistatic_delay_s(result.rg_delay)
    total_path_km = total_path_from_delay_s(r0_km, gate, fs_hz, delay_s)
    return StationPrediction(
        station=station,
        alt_cross=result.alt_cross,
        rg_t=result.rg_t,
        rg_r=result.rg_r,
        rg_delay=result.rg_delay,
        delay_s=delay_s,
        total_path_km=total_path_km,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Python translation of rangedelay.m; all angles are degrees and altitudes/ranges are km."
    )
    parser.add_argument("lat_t", type=float, nargs="?", help="transmitter geodetic latitude")
    parser.add_argument("lon_t", type=float, nargs="?", help="transmitter geodetic longitude")
    parser.add_argument("alt_t", type=float, nargs="?", help="transmitter geodetic altitude in km")
    parser.add_argument("az_t", type=float, nargs="?", help="transmitter azimuth")
    parser.add_argument("el_t", type=float, nargs="?", help="transmitter elevation")
    parser.add_argument("lat_r", type=float, nargs="?", help="receiver geodetic latitude")
    parser.add_argument("lon_r", type=float, nargs="?", help="receiver geodetic longitude")
    parser.add_argument("alt_r", type=float, nargs="?", help="receiver geodetic altitude in km")
    parser.add_argument("az_r", type=float, nargs="?", help="receiver azimuth")
    parser.add_argument("el_r", type=float, nargs="?", help="receiver elevation")
    parser.add_argument(
        "--station",
        choices=["Danzhou", "Wenchang", "all"],
        help="predict nominal bistatic path for Danzhou, Wenchang, or both",
    )
    parser.add_argument("--r0-km", type=float, default=69.9, help="monostatic first-sample one-way range in km")
    parser.add_argument("--gate", type=float, default=0.0, help="range gate index")
    parser.add_argument("--fs-hz", type=float, help="sample rate in Hz for the c*gate/fs term")
    parser.add_argument("--fs-mhz", type=float, help="sample rate in MHz for the c*gate/fs term")
    args = parser.parse_args()

    fs_hz = args.fs_hz
    if args.fs_mhz is not None:
        fs_hz = args.fs_mhz * 1e6

    if args.station is not None:
        if fs_hz is None:
            parser.error("--station requires --fs-hz or --fs-mhz")
        stations = ("Danzhou", "Wenchang") if args.station == "all" else (args.station,)
        for station in stations:
            prediction = predict_station_total_path(station, args.r0_km, args.gate, fs_hz)
            print(station)
            print(f"  alt_cross_km  {prediction.alt_cross:.9f}")
            print(f"  rg_t_km        {prediction.rg_t:.9f}")
            print(f"  rg_r_km        {prediction.rg_r:.9f}")
            print(f"  rg_delay_km    {prediction.rg_delay:.9f}")
            print(f"  delay_s        {prediction.delay_s:.12e}")
            print(f"  total_path_km  {prediction.total_path_km:.9f}")
        return

    manual_args = {
        name: getattr(args, name)
        for name in ("lat_t", "lon_t", "alt_t", "az_t", "el_t", "lat_r", "lon_r", "alt_r", "az_r", "el_r")
    }
    if any(value is None for value in manual_args.values()):
        parser.error("provide either --station or all ten manual geometry arguments")

    result = rangedelay(**manual_args, as_dataclass=True)
    print(f"alt_cross_km {result.alt_cross:.9f}")
    print(f"rg_t_km       {result.rg_t:.9f}")
    print(f"rg_r_km       {result.rg_r:.9f}")
    print(f"rg_delay_km   {result.rg_delay:.9f}")
    print(f"delay_s       {range_delay_km_to_bistatic_delay_s(result.rg_delay):.12e}")
    if fs_hz is not None:
        total_path_km = total_path_from_range_delay_km(args.r0_km, args.gate, fs_hz, result.rg_delay)
        print(f"total_path_km {total_path_km:.9f}")


if __name__ == "__main__":
    main()
