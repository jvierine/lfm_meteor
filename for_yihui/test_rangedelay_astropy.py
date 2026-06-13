import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rangedelay import aer_to_geodetic_km


HAVE_ASTROPY = importlib.util.find_spec("astropy") is not None

if HAVE_ASTROPY:
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, ITRS, SkyCoord
    from astropy.time import Time


def astropy_aer_to_geodetic_km(az_deg, el_deg, range_km, lat0_deg, lon0_deg, alt0_km):
    location = EarthLocation.from_geodetic(
        lon=lon0_deg * u.deg,
        lat=lat0_deg * u.deg,
        height=alt0_km * u.km,
        ellipsoid="WGS84",
    )
    obstime = Time("2024-04-22T16:00:00", scale="utc")
    altaz_frame = AltAz(obstime=obstime, location=location, pressure=0.0 * u.hPa)
    target_altaz = SkyCoord(
        az=np.asarray(az_deg) * u.deg,
        alt=np.asarray(el_deg) * u.deg,
        distance=np.asarray(range_km) * u.km,
        frame=altaz_frame,
    )
    target_itrs_topocentric = target_altaz.transform_to(ITRS(obstime=obstime))
    observer_itrs = location.get_itrs(obstime=obstime)
    target_location = EarthLocation.from_geocentric(
        target_itrs_topocentric.x + observer_itrs.x,
        target_itrs_topocentric.y + observer_itrs.y,
        target_itrs_topocentric.z + observer_itrs.z,
    )
    return (
        target_location.lat.to_value(u.deg),
        target_location.lon.to_value(u.deg),
        target_location.height.to_value(u.km),
    )


def lon_error_deg(lon_a, lon_b):
    return (np.asarray(lon_a) - np.asarray(lon_b) + 180.0) % 360.0 - 180.0


def log(message):
    print(f"[test_rangedelay_astropy] {message}", flush=True)


def log_table(name, lat, lon, alt):
    log(name)
    for i, (lat_i, lon_i, alt_i) in enumerate(zip(lat, lon, alt)):
        log(f"  ray {i}: lat={lat_i:.12f} deg lon={lon_i:.12f} deg alt={alt_i:.9f} km")


@unittest.skipUnless(HAVE_ASTROPY, "astropy is required for this independent geodesy check")
class AerToGeodeticAstropyTest(unittest.TestCase):
    def test_aer_to_geodetic_matches_astropy_wgs84(self):
        lat0_deg = 18.3492
        lon0_deg = 109.6222
        alt0_km = 0.05
        az_deg = np.asarray([14.996337890625, 40.0, 225.0])
        el_deg = np.asarray([74.9981689453125, 35.0, 12.0])
        range_km = np.asarray([98.0, 250.0, 900.0])

        log("setting up Sanya observer and three az/el/range test rays")
        log(f"observer: lat={lat0_deg:.6f} deg lon={lon0_deg:.6f} deg alt={alt0_km:.6f} km")
        for i, (az_i, el_i, range_i) in enumerate(zip(az_deg, el_deg, range_km)):
            log(f"  ray {i}: az={az_i:.12f} deg el={el_i:.12f} deg range={range_i:.6f} km")

        log("running rangedelay.aer_to_geodetic_km")
        lat, lon, alt = aer_to_geodetic_km(az_deg, el_deg, range_km, lat0_deg, lon0_deg, alt0_km)
        log("running independent Astropy AltAz/ITRS/WGS84 reference conversion")
        ref_lat, ref_lon, ref_alt = astropy_aer_to_geodetic_km(
            az_deg,
            el_deg,
            range_km,
            lat0_deg,
            lon0_deg,
            alt0_km,
        )

        log_table("rangedelay output:", lat, lon, alt)
        log_table("Astropy reference:", ref_lat, ref_lon, ref_alt)

        lat_err = lat - ref_lat
        lon_err = lon_error_deg(lon, ref_lon)
        alt_err = alt - ref_alt
        log(f"max latitude error:  {np.max(np.abs(lat_err)):.3e} deg")
        log(f"max longitude error: {np.max(np.abs(lon_err)):.3e} deg")
        log(f"max altitude error:  {np.max(np.abs(alt_err)):.3e} km")
        log("checking tolerances: lat/lon 2e-7 deg, altitude 2e-5 km")

        np.testing.assert_allclose(lat, ref_lat, atol=2e-7)
        np.testing.assert_allclose(lon_err, 0.0, atol=2e-7)
        np.testing.assert_allclose(alt, ref_alt, atol=2e-5)
        log("aer_to_geodetic_km agrees with Astropy reference")


if __name__ == "__main__":
    unittest.main()
