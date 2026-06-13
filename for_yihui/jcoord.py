"""
Small local copy of Juha Vierinen's jcoord helpers used by for_yihui.py.

This file is vendored here so the rank-02 example does not require installing
the separate `jcoord` Python package.
"""

import math

import numpy
from numpy import (
    abs,
    arccos,
    arctan,
    arctan2,
    array,
    cos,
    degrees,
    dot,
    power,
    radians,
    sign,
    sin,
    sqrt,
)


def cbrt(x):
    if x >= 0:
        return power(x, 1.0 / 3.0)
    return -power(abs(x), 1.0 / 3.0)


# Constants defined by the World Geodetic System 1984 (WGS84).
a = 6378.137 * 1e3
b = 6356.7523142 * 1e3
esq = 6.69437999014 * 0.001
e1sq = 6.73949674228 * 0.001


def geodetic2ecef(lat, lon, alt):
    """Convert geodetic coordinates to ECEF; lat/lon in degrees, alt in m."""
    lat, lon = radians(lat), radians(lon)
    xi = sqrt(1 - esq * sin(lat) ** 2)
    x = (a / xi + alt) * cos(lat) * cos(lon)
    y = (a / xi + alt) * cos(lat) * sin(lon)
    z = (a / xi * (1 - esq) + alt) * sin(lat)
    return numpy.array([x, y, z])


def ned2ecef(lat, lon, alt, n, e, d):
    """NED (north/east/down) to ECEF coordinate-system conversion."""
    x, y, z = e, n, -1.0 * d
    lat, lon = radians(lat), radians(lon)
    mx = array(
        [
            [-sin(lon), -sin(lat) * cos(lon), cos(lat) * cos(lon)],
            [cos(lon), -sin(lat) * sin(lon), cos(lat) * sin(lon)],
            [0, cos(lat), sin(lat)],
        ]
    )
    enu = array([x, y, z])
    return dot(mx, enu)


def azel_ecef(lat, lon, alt, az, el):
    """Radar pointing az/el in degrees to a unit vector in ECEF."""
    return ned2ecef(
        lat,
        lon,
        alt,
        cos(-radians(az)) * cos(radians(el)),
        -sin(-radians(az)) * cos(radians(el)),
        -sin(radians(el)),
    )


def ecef2geodetic(x, y, z):
    """Convert ECEF coordinates to geodetic coordinates.

    J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates to geodetic
    coordinates," IEEE Transactions on Aerospace and Electronic Systems,
    vol. 30, pp. 957-961, 1994.
    """
    r = sqrt(x * x + y * y)
    Esq = a * a - b * b
    F = 54 * b * b * z * z
    G = r * r + (1 - esq) * z * z - esq * Esq
    C = (esq * esq * F * r * r) / (pow(G, 3))
    S = cbrt(1 + C + sqrt(C * C + 2 * C))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = sqrt(1 + 2 * esq * esq * P)
    r_0 = -(P * esq * r) / (1 + Q) + sqrt(
        0.5 * a * a * (1 + 1.0 / Q)
        - P * (1 - esq) * z * z / (Q * (1 + Q))
        - 0.5 * P * r * r
    )
    U = sqrt(pow((r - esq * r_0), 2) + z * z)
    V = sqrt(pow((r - esq * r_0), 2) + (1 - esq) * z * z)
    Z_0 = b * b * z / (a * V)
    h = U * (1 - b * b / (a * V))
    lat = arctan((z + e1sq * Z_0) / r)
    lon = arctan2(y, x)
    return array([degrees(lat), degrees(lon), h])


def geodetic_to_az_el_r(obs_lat, obs_lon, obs_h, target_lat, target_lon, target_h):
    """Return azimuth, elevation, and range from observer to target."""
    up = ned2ecef(obs_lat, obs_lon, obs_h, 0.0, 0.0, -1.0)
    north = ned2ecef(obs_lat, obs_lon, obs_h, 1.0, 0.0, 0.0)
    east = ned2ecef(obs_lat, obs_lon, obs_h, 0.0, 1.0, 0.0)
    obs = array(geodetic2ecef(obs_lat, obs_lon, obs_h))
    target = array(geodetic2ecef(target_lat, target_lon, target_h))
    p_vec = target - obs
    az_p = dot(p_vec, north) * north + dot(p_vec, east) * east
    azs = sign(dot(p_vec, east))

    elevation = 90.0 - 180.0 * arccos(dot(p_vec, up) / (sqrt(dot(p_vec, p_vec)) * sqrt(dot(up, up)))) / math.pi
    azimuth = azs * 180.0 * arccos(dot(az_p, north) / (sqrt(dot(az_p, az_p)) * sqrt(dot(north, north)))) / math.pi
    target_range = sqrt(dot(p_vec, p_vec))

    return array([azimuth, elevation, target_range])


def az_el_r2geodetic(obs_lat, obs_lon, obs_h, az, el, r):
    """Return target lat/lon/height from observer lat/lon/height and az/el/r."""
    x = geodetic2ecef(obs_lat, obs_lon, obs_h) + azel_ecef(obs_lat, obs_lon, obs_h, az, el) * r
    llh = ecef2geodetic(x[0], x[1], x[2])
    if llh[1] < 0.0:
        llh[1] = llh[1] + 360.0
    return llh
