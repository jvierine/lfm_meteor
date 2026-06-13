import jcoord
import numpy as np

import sanya_opts as sc


POINTINGS = {
    "Sanya": (14.996337890625, 74.9981689453125),
    "Danzhou": (151.2652587890625, 37.3260498046875),
    "Wenchang": (225.7855224609375, 29.2950439453125),
}

REMOTE_BORESIGHTS = {
    "Danzhou": (158.3, 70.0),
    "Wenchang": (221.9, 70.0),
}
REMOTE_NORMAL_BEAMWIDTH_DEG = {
    "short_axis": 1.1,
    "long_axis": 1.5,
}


def beam_line(site_index, az_deg, el_deg):
    origin = np.asarray(
        jcoord.geodetic2ecef(sc.lat0[site_index], sc.lon0[site_index], sc.alt0[site_index] * 1e3),
        dtype=np.float64,
    )
    point_llh = jcoord.az_el_r2geodetic(
        sc.lat0[site_index],
        sc.lon0[site_index],
        sc.alt0[site_index] * 1e3,
        az_deg,
        el_deg,
        100e3,
    )
    point = np.asarray(jcoord.geodetic2ecef(point_llh[0], point_llh[1], point_llh[2]), dtype=np.float64)
    direction = point - origin
    direction /= np.linalg.norm(direction)
    return origin, direction


def closest_common_point(origins, directions):
    lhs = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for origin, direction in zip(origins, directions):
        projector = np.eye(3) - np.outer(direction, direction)
        lhs += projector
        rhs += projector @ origin
    return np.linalg.solve(lhs, rhs)


def azel_to_enu(az_deg, el_deg):
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    return np.asarray(
        [
            np.cos(el_rad) * np.sin(az_rad),
            np.cos(el_rad) * np.cos(az_rad),
            np.sin(el_rad),
        ],
        dtype=np.float64,
    )


def main():
    names = list(POINTINGS)
    origins = []
    directions = []
    for site_index, name in enumerate(names):
        az_deg, el_deg = POINTINGS[name]
        origin, direction = beam_line(site_index, az_deg, el_deg)
        origins.append(origin)
        directions.append(direction)

    point = closest_common_point(origins, directions)
    lat_deg, lon_deg, alt_m = jcoord.ecef2geodetic(point[0], point[1], point[2])
    print(f"Common beam-axis point: {lat_deg:.6f} deg N, {lon_deg:.6f} deg E, {alt_m / 1e3:.3f} km")

    for name, origin, direction in zip(names, origins, directions):
        slant_m = float(np.dot(point - origin, direction))
        closest = origin + slant_m * direction
        miss_m = float(np.linalg.norm(point - closest))
        print(f"{name}: slant range {slant_m / 1e3:.3f} km, miss distance {miss_m:.2f} m")

    for name in ["Danzhou", "Wenchang"]:
        boresight = azel_to_enu(*REMOTE_BORESIGHTS[name])
        pointing = azel_to_enu(*POINTINGS[name])
        scan_angle_deg = float(np.rad2deg(np.arccos(np.clip(np.dot(boresight, pointing), -1.0, 1.0))))
        secant = 1.0 / np.cos(np.deg2rad(scan_angle_deg))
        short_axis_deg = REMOTE_NORMAL_BEAMWIDTH_DEG["short_axis"] * secant
        long_axis_deg = REMOTE_NORMAL_BEAMWIDTH_DEG["long_axis"] * secant
        site_index = names.index(name)
        slant_km = float(np.dot(point - origins[site_index], directions[site_index]) / 1e3)
        short_axis_km = 2.0 * slant_km * np.tan(0.5 * np.deg2rad(short_axis_deg))
        long_axis_km = 2.0 * slant_km * np.tan(0.5 * np.deg2rad(long_axis_deg))
        print(
            f"{name}: receiver boresight az/el {REMOTE_BORESIGHTS[name][0]:.1f}/{REMOTE_BORESIGHTS[name][1]:.1f} deg, "
            f"scan angle {scan_angle_deg:.2f} deg, scaled 3-dB beam {short_axis_deg:.2f}/{long_axis_deg:.2f} deg, "
            f"footprint {short_axis_km:.1f}/{long_axis_km:.1f} km"
        )


if __name__ == "__main__":
    main()
