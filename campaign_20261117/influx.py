"""November-only pilot influx; no April products or correction constants."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from analyze import STATIONS, ENU, LLH, save
from derived import RCS_AZ_EL, beam_frame, beam_power, rcs_beam
from monostatic_selection import load_selection


def merge_intervals(start,stop):
    result=[]
    for lo,hi in sorted(zip(start,stop)):
        if result and lo<=result[-1][1]+10000000:
            result[-1][1]=max(result[-1][1],int(hi))
        else:result.append([int(lo),int(hi)])
    return np.asarray(result,dtype=np.int64)


def intersect(a,b):
    result=[];i=j=0
    while i<len(a) and j<len(b):
        lo=max(a[i,0],b[j,0]);hi=min(a[i,1],b[j,1])
        if hi>lo:result.append([lo,hi])
        if a[i,1]<b[j,1]:i+=1
        else:j+=1
    return np.asarray(result,dtype=np.int64).reshape(-1,2)


def in_intervals(times,intervals):
    return np.logical_or.reduce([(times>=lo)&(times<hi) for lo,hi in intervals])


def weights(area,altitude,exposure,speed,p,altitude_factor):
    earth=4*np.pi*(6371.+altitude)**2
    return 4*earth/area*24/np.asarray(exposure)*(72/np.asarray(speed))**p*altitude_factor


def collecting_area(cutoff,altitude,step=.05):
    axis=beam_frame(*RCS_AZ_EL)[0]
    up=altitude-LLH[0,2]/1000
    center=axis/axis[2]*up
    coordinates=np.arange(-40.,40.+step/2,step)
    x,y=np.meshgrid(coordinates+center[0],coordinates+center[1])
    v=np.stack([x,y,np.full_like(x,up)],axis=-1)
    v/=np.linalg.norm(v,axis=-1,keepdims=True)
    gain=10*np.log10(np.maximum(beam_power(v,axis),1e-30))
    inside=gain>=cutoff
    if inside[0].any() or inside[-1].any() or inside[:,0].any() or inside[:,-1].any():
        raise ValueError('Collecting-area contour touches grid edge')
    return float(inside.sum()*step**2)


def main():
    from astropy.coordinates import SkyCoord,GCRS,GeocentricTrueEcliptic,get_sun
    from astropy.time import Time
    from astropy import units as u
    from astropy.utils import iers
    iers.conf.auto_download=False
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/gpfs/pub/data_other/juha'))
    a=p.parse_args();root=a.root/'20261117_tristatic'
    intervals=None
    for site in ('sanya','danzhou','wenchang'):
        with h5py.File(a.root/('20261117_'+site)/'noise.h5') as h:
            recording=merge_intervals(h['averages/start_utc_ns'][:],h['averages/stop_utc_ns'][:]+5000000)
        intervals=recording if intervals is None else intersect(intervals,recording)
    live_h=np.sum(intervals[:,1]-intervals[:,0])/3.6e12
    with h5py.File(root/'association_selection.h5') as h:
        names=[n.decode() for n,k in zip(h['event_id'][:],h['accepted'][:]) if k]
    rows=[]
    for name in names:
        folder=root/'events'/name
        with h5py.File(folder/'physical.h5') as h:
            if not h.attrs.get('path_quality_pass',False):continue
            epoch=int(h['time_ns'][0])
            if not in_intervals(np.array([epoch]),intervals)[0]:continue
            mass=float(h['initial_mass_kg'][()]) if h.attrs['mass_resolved_profile'] else np.nan
            bounds=h['mass_local95_kg'][:] if np.isfinite(mass) else np.full(2,np.nan)
            mask=h['dynamic_mass_fit_mask'][:]
            gain=rcs_beam(h['position_ecef_m'][:][mask])[0]
            velocity=h['initial_gcrs_velocity_mps'][:]
            height=float(np.median(h['height_km'][:][mask]))
        with h5py.File(folder/'rcs.h5') as h:rcs=float(h['integrated_mass_loss_kg'][()])
        rows.append((name,epoch,velocity,height,np.max(gain),mass,bounds,rcs))
    names=np.array([r[0] for r in rows],dtype='S32')
    ns=np.array([r[1] for r in rows],dtype=np.int64)
    v=np.array([r[2] for r in rows]);speed=np.linalg.norm(v,axis=1)/1000
    height=np.array([r[3] for r in rows]);max_gain=np.array([r[4] for r in rows])
    mass=np.array([r[5] for r in rows]);bounds=np.array([r[6] for r in rows]);rcs=np.array([r[7] for r in rows])
    with h5py.File(root/'figure_data/altitude_correction.h5') as h:
        lower=float(h.attrs['common_lower_km']);upper=float(h.attrs['common_upper_km'])
    altitude=(lower+upper)/2
    mono_keep,mono_height=load_selection(a.root/'20261117_sanya')
    with h5py.File(a.root/'20261117_sanya/catalogue.h5') as h:
        mono_time=h['event_start_utc_ns'][:]+(h['event_stop_utc_ns'][:]-h['event_start_utc_ns'][:])//2
    mono_keep &= in_intervals(mono_time,intervals)
    mono_total=int(mono_keep.sum());mono_common=int(np.sum(mono_keep&(mono_height>=lower)&(mono_height<=upper)))
    tri_common=int(np.sum((height>=lower)&(height<=upper)))
    if not mono_common or not tri_common:raise ValueError('No common-volume reference counts')
    falt=mono_total/mono_common*tri_common/len(rows)
    cutoff=float(np.quantile(max_gain,.05))
    area=collecting_area(cutoff,altitude)
    fine_area=collecting_area(cutoff,altitude,.025)
    if abs(fine_area/area-1)>.02:raise ValueError('Collecting-area grid has not converged to 2 percent')
    area=fine_area
    times=Time(ns.astype(np.longdouble)/1e9,format='unix',scale='utc')
    sky=SkyCoord(x=-v[:,0]*u.one,y=-v[:,1]*u.one,z=-v[:,2]*u.one,
                 representation_type='cartesian',frame=GCRS(obstime=times))
    ecliptic=sky.transform_to(GeocentricTrueEcliptic(equinox=times))
    sun=get_sun(times).transform_to(GeocentricTrueEcliptic(equinox=times))
    longitude=(ecliptic.lon.deg-sun.lon.deg)%360;beta=ecliptic.lat.deg
    midpoint=Time(np.longdouble(intervals[0,0]+(intervals[-1,1]-intervals[0,0])//2)/1e9,format='unix',scale='utc')
    sun_mid=get_sun(midpoint).transform_to(GeocentricTrueEcliptic(equinox=midpoint)).lon.deg
    family_lon=np.stack([longitude,(540-longitude)%360,longitude,(540-longitude)%360],axis=1)
    family_beta=np.stack([beta,beta,-beta,-beta],axis=1)
    family=SkyCoord(lon=((family_lon+sun_mid)%360)*u.deg,lat=family_beta*u.deg,
                    frame=GeocentricTrueEcliptic(equinox=midpoint)).transform_to(GCRS(obstime=midpoint))
    ticks=[];durations=[]
    for lo,hi in intervals:
        starts=np.arange(lo,hi,60000000000,dtype=np.int64)
        ends=np.minimum(starts+60000000000,hi)
        ticks.extend(starts+(ends-starts)//2);durations.extend((ends-starts)/3.6e12)
    t=Time(np.asarray(ticks,dtype=np.longdouble)/1e9,format='unix',scale='utc')
    lst=t.sidereal_time('apparent',longitude=LLH[0,1]*u.deg).rad
    dec=family.dec.rad;ra=family.ra.rad;phi=np.deg2rad(LLH[0,0])
    sine=np.sin(phi)*np.sin(dec[...,None])+np.cos(phi)*np.cos(dec[...,None])*np.cos(lst-ra[...,None])
    exposure=(np.maximum(sine,0)**1.47*np.asarray(durations)).sum(axis=(1,2))
    if not np.all(exposure>0):raise ValueError('Zero radiant-family exposure')
    out=root/'november_mass_influx.h5'
    rng=np.random.default_rng(20261117)
    with h5py.File(out,'w') as h:
        h.attrs.update(dataset='2026-11-17 only; no April data',live_hours=live_h,
            common_lower_km=lower,common_upper_km=upper,reference_altitude_km=altitude,
            collecting_area_km2=area,coarse_area_km2=collecting_area(cutoff,altitude),
            containment_cutoff_db=cutoff,altitude_factor=falt,monostatic_total=mono_total,
            monostatic_common=mono_common,tristatic_common=tri_common,quality_events=len(rows),
            interpretation='pilot extrapolation using reference atmosphere and 120 K RCS calibration',
            uncertainty='p=3..3.5 and conditional dynamic-mass endpoints are sensitivity envelopes, not global confidence intervals; RCS mass uncertainty not available; sampling intervals exclude systematic errors',
            exposure_convention='fourfold reflected Sun-centred ecliptic radiants; sum of above-horizon sin(elevation)^1.47 over simultaneous recorded intervals',
            area_convention='horizontal plane at common altitude; 95-percent event-maximum gain containment; fitted pointing; no factor inherited from April')
        h['event_id']=names;h['recording_intervals_utc_ns']=intervals
        h['speed_km_s']=speed;h['family_effective_hours']=exposure
        h['event_max_gain_db']=max_gain;h['dynamic_mass_kg']=mass;h['rcs_mass_loss_kg']=rcs
        for kind,values in [('dynamic',mass),('rcs',rcs)]:
            keep=np.isfinite(values)&(values>0);missing=len(rows)-int(keep.sum())
            g=h.create_group(kind);g['event_id']=names[keep]
            g['mass_kg']=values[keep]
            g.attrs.update(measured_events=int(keep.sum()),missing_events=missing,
                mass_min_kg=float(values[keep].min()),mass_max_kg=float(values[keep].max()),
                missing_assumption='optional: each missing event contributes the median selected weighted contribution')
            pgrid=np.linspace(3.,3.5,101);estimates=[];envelopes=[];imputed=[]
            for exponent in pgrid:
                w=weights(area,altitude,exposure[keep],speed[keep],exponent,falt)/1000
                contribution=values[keep]*w
                estimates.append(contribution.sum());imputed.append(contribution.sum()+missing*np.median(contribution))
                if kind=='dynamic':
                    weighted=bounds[keep]*w[:,None]
                    envelopes.append(weighted.sum(axis=0)+missing*np.median(weighted,axis=0))
            w=weights(area,altitude,exposure[keep],speed[keep],3.25,falt)/1000
            contribution=values[keep]*w;order=np.argsort(contribution)[::-1]
            poisson=rng.poisson(1.,size=(20000,len(contribution)))
            sampled=poisson@contribution
            g['p']=pgrid;g['measured_rate_tonne_day']=estimates;g['median_imputed_rate_tonne_day']=imputed
            g['weight_tonne_day_per_kg']=w;g['contribution_tonne_day']=contribution
            g['top_event_id']=names[keep][order[:10]];g['top_contribution_tonne_day']=contribution[order[:10]]
            g['sampling_interval95_tonne_day']=np.quantile(sampled,[.025,.975])
            g.attrs.update(reference_p=3.25,reference_measured_tonne_day=float(contribution.sum()),
                reference_imputed_tonne_day=float(contribution.sum()+missing*np.median(contribution)),
                omit_two_largest_tonne_day=float(contribution[order[2:]].sum()),
                largest_fraction=float(contribution.max()/contribution.sum()),
                top_two_fraction=float(contribution[order[:2]].sum()/contribution.sum()))
            if envelopes:
                g['mass_bound_imputed_envelope_tonne_day']=envelopes
                g['mass_and_p_sensitivity_tonne_day']=[np.min(np.array(envelopes)[:,0]),np.max(np.array(envelopes)[:,1])]
            print(kind,dict(g.attrs),'p-endpoints',np.asarray(estimates)[[0,-1]],
                  'imputed',np.asarray(imputed)[[0,-1]],flush=True)
        print('GEOMETRY',dict(h.attrs),flush=True)
    print(out,flush=True)


if __name__=='__main__':main()
