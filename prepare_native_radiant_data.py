"""Scientific visibility calculation only. No plotting or Matplotlib imports."""
from pathlib import Path
import sys
import h5py
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, GeocentricTrueEcliptic, SkyCoord, get_sun
from astropy.time import Time

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT/"results/sun_centered_ecliptic_radiants.h5"
    output = ROOT/"results/native_radiant_scientific_data.h5"
    with h5py.File(source) as original:
        arrays = {name: original[name][()] for name in (
            "lambda_minus_sun_deg", "beta_ecliptic_deg", "speed_km_s", "t0_ns")}
        equinox = original.attrs.get("fixed_ecliptic_equinox_utc")
    time = arrays["t0_ns"]/1e9
    t0, t1 = Time(np.min(time), format="unix"), Time(np.max(time), format="unix")
    duration = (t1-t0).to_value(u.hour)
    intervals = max(1, int(np.ceil(duration*60/5)))
    samples = t0+(np.arange(intervals)+0.5)*duration/intervals*u.hour
    if equinox is None:
        equinox = Time(np.median(time),format="unix").isot
    # Same Sanya location, fixed ecliptic frame, cadence and gamma as the article.
    site = EarthLocation(lat=18.3492*u.deg, lon=109.6222*u.deg, height=50*u.m)
    x = np.linspace(-180., 180., 145)
    latitude = np.linspace(-90., 90., 73)
    lon, lat = np.meshgrid((270.-x)%360., latitude)
    visible = np.zeros(lon.shape)
    effective = np.zeros(lon.shape)
    frame_time = Time(equinox, format="isot", scale="utc")
    frame = GeocentricTrueEcliptic(obstime=samples, equinox=frame_time)
    sun = get_sun(samples).transform_to(frame)
    for epoch, sun_lon in zip(samples, sun.lon.to_value(u.deg)):
        directions = SkyCoord(lon=((lon+sun_lon)%360.).ravel()*u.deg,
            lat=lat.ravel()*u.deg,
            frame=GeocentricTrueEcliptic(obstime=epoch,equinox=frame_time))
        elevation = directions.transform_to(AltAz(obstime=epoch,location=site)).alt.to_value(u.deg).reshape(lon.shape)
        visible += (elevation > 0.)
        effective += np.sin(np.deg2rad(np.clip(elevation,0.,90.)))**1.47
    dt = (samples[1]-samples[0]).to_value(u.hour) if len(samples)>1 else 5/60
    with h5py.File(output,"w") as data:
        data.attrs["source"] = str(source)
        data.attrs["script"] = __file__
        data.attrs["fixed_ecliptic_equinox_utc"] = equinox
        data.attrs["gamma"] = 1.47
        for name, array in arrays.items():
            data[name] = array
        data["signed_westward_apex_longitude_deg"] = x
        data["grid_latitude_deg"] = latitude
        data["visible_hours"] = visible*dt
        data["effective_hours"] = effective*dt
    assert "matplotlib" not in sys.modules
    print(output, flush=True)


if __name__ == "__main__":
    main()
