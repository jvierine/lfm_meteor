"""Select meteor-like monostatic tracks without deleting original cuts."""
import argparse
import concurrent.futures
from pathlib import Path
import h5py
import numpy as np
from analyze import C, STATIONS, ENU, remaining_delay_correction, save
from derived import RCS_AZ_EL, beam_frame


def meteor_mask(height,duration,rate):
    height=np.asarray(height);duration=np.asarray(duration);rate=np.asarray(rate)
    outside=~np.isfinite(height)|(height<75.)|(height>120.)
    slow_long=(duration>=.5)&np.isfinite(rate)&(abs(rate)<8.)
    return ~(outside|slow_long),outside,slow_long


def read_rate(path):
    with h5py.File(path) as h:return float(h['track_range_rate_km_s'][0])


def selection(root,workers=1):
    from astropy.coordinates import EarthLocation
    from astropy import units as u
    with h5py.File(root/'catalogue.h5') as h:
        ranges=h['event_median_apparent_range_km'][:]+C*remaining_delay_correction(h)*1e-9/2
        start=h['event_start_utc_ns'][:];stop=h['event_stop_utc_ns'][:]
    duration=(stop-start)*1e-9
    axis=ENU.T@beam_frame(*RCS_AZ_EL)[0]
    xyz=STATIONS[0]+1000*ranges[:,None]*axis
    height=EarthLocation.from_geocentric(xyz[:,0]*u.m,xyz[:,1]*u.m,xyz[:,2]*u.m).height.to_value(u.km)
    rate=np.full(len(ranges),np.nan)
    ids=np.flatnonzero(duration>=.5)
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        rate[ids]=list(pool.map(read_rate,[root/'event_cuts'/('event_%06d.h5'%i) for i in ids],chunksize=10))
    keep,outside,slow=meteor_mask(height,duration,rate)
    save(root/'meteor_selection.h5',dict(event_index=np.arange(len(keep)),accepted=keep,
        apparent_beam_height_km=height,duration_s=duration,range_rate_km_s=rate,
        outside_meteor_layer=outside,slow_long_track=slow),
        {'height_lower_km':75.,'height_upper_km':120.,'long_duration_s':.5,
         'slow_abs_range_rate_km_s':8.,'beam_azimuth_deg':RCS_AZ_EL[0],
         'beam_elevation_deg':RCS_AZ_EL[1],
         'interpretation':'meteor-like monostatic selection; height along fitted beam, not independent height measurement; slow-long tracks are non-meteor candidates, not ephemeris identifications'})
    print('MONOSTATIC total',len(keep),'accepted',keep.sum(),'outside_layer',outside.sum(),
          'slow_long',slow.sum(),'both',np.count_nonzero(outside&slow),flush=True)


def load_selection(root):
    with h5py.File(root/'meteor_selection.h5') as h:
        return h['accepted'][:],h['apparent_beam_height_km'][:]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/gpfs/pub/data_other/juha/20261117_sanya'))
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();selection(a.root,a.workers)
