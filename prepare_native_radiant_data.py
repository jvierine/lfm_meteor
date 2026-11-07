"""Current direct-fit radiants and exposure; no historical catalogue fallback."""
from pathlib import Path
import h5py
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, GeocentricTrueEcliptic, SkyCoord, get_sun
from astropy.time import Time
from astropy.utils import iers
from prepare_current_kinematics import OUTPUT, AREA, MANIFEST, START, STOP, check_export, sha
from plot_sun_centered_ecliptic_radiants import load_fits, calculate_radiants

iers.conf.auto_download = False
ROOT = Path(__file__).resolve().parent


def exposure(lon, lat, samples, equinox, dt):
    visible = np.zeros(np.shape(lon))
    effective = np.zeros(np.shape(lon))
    site = EarthLocation(lat=18.3492*u.deg, lon=109.6222*u.deg, height=50*u.m)
    frame_time = Time(equinox, format="isot", scale="utc")
    sun = get_sun(samples).transform_to(GeocentricTrueEcliptic(obstime=samples, equinox=frame_time))
    for epoch, sun_lon in zip(samples, sun.lon.to_value(u.deg)):
        directions = SkyCoord(lon=((lon+sun_lon)%360.).ravel()*u.deg,
            lat=np.asarray(lat).ravel()*u.deg,
            frame=GeocentricTrueEcliptic(obstime=epoch,equinox=frame_time))
        elevation = directions.transform_to(AltAz(obstime=epoch,location=site)).alt.to_value(u.deg).reshape(np.shape(lon))
        visible += (elevation > 0.)
        effective += np.sin(np.deg2rad(np.clip(elevation,0.,90.)))**1.47
    return visible*dt, effective*dt


def main():
    check_export()
    ids, times, positions, velocity, speed, rms, count, frame = load_fits(OUTPUT)
    lon, lat, sun, relative, equinox = calculate_radiants(times, positions, velocity, frame)
    start, stop = Time(START), Time(STOP)
    duration = (stop-start).to_value(u.hour)
    intervals = int(np.ceil(duration*60/5))
    dt = duration/intervals
    samples = start+(np.arange(intervals)+.5)*dt*u.hour
    x = np.linspace(-180.,180.,145)
    y = np.linspace(-90.,90.,73)
    grid_lon, grid_lat = np.meshgrid((270.-x)%360.,y)
    visible, effective = exposure(grid_lon,grid_lat,samples,equinox,dt)
    # Four symmetry members; numerator four is NOT included in these hours.
    family_lon = np.column_stack([relative,(540.-relative)%360.,relative,(540.-relative)%360.])
    family_lat = np.column_stack([lat,lat,-lat,-lat])
    _, family_hours = exposure(family_lon,family_lat,samples,equinox,dt)
    with h5py.File(AREA) as area:
        area_km2 = float(area.attrs['area_horizontal_km2'])
    target = OUTPUT.parent/'radiants.h5'
    with h5py.File(target,'w') as h:
        h.attrs.update(source=str(OUTPUT),source_sha256=sha(OUTPUT),manifest_sha256=sha(MANIFEST),
            source_program=__file__, source_program_sha256=sha(__file__),
            model='direct shrinking radius',fixed_ecliptic_equinox_utc=equinox,gamma=1.47,
            observation_start_utc=START,observation_stop_utc=STOP,observed_hours=duration,
            sampling_interval_h=dt,area_horizontal_km2=area_km2,area_source=str(AREA),
            area_sha256=sha(AREA),altitude_correction_included=False,velocity_correction_included=False,
            exposure_convention='sum of four apex/ecliptic symmetry-member effective hours; no factor four in hours; rate weight is 4*24/(area*sum_hours)',
            uptime_assumption='continuous within documented experiment start and stop; internal gaps not modeled',
            longitude_convention='lambda-lambda_sun modulo 360; west-positive apex-centered display')
        h['event_id'] = np.asarray(ids,dtype=h5py.string_dtype())
        for name,array in dict(t0_ns=times,speed_km_s=speed,lambda_ecliptic_deg=lon,
            beta_ecliptic_deg=lat,sun_lambda_ecliptic_deg=sun,lambda_minus_sun_deg=relative,
            state_position_gcrs_m=positions,state_velocity_gcrs_mps=velocity,
            rms_total_path_residual_m=rms,n_points=count,
            signed_westward_apex_longitude_deg=x,grid_latitude_deg=y,
            visible_hours=visible,effective_hours=effective,
            family_lambda_minus_sun_deg=family_lon,family_beta_ecliptic_deg=family_lat,
            family_effective_exposure_h=family_hours,
            family_effective_exposure_sum_h=family_hours.sum(axis=1)).items():
            h[name]=array
    print(target,flush=True)
    print('events',len(ids),'speed',speed.min(),speed.max(),'area',area_km2,flush=True)


if __name__ == '__main__':
    main()
