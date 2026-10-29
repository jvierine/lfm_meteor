"""Export calibrated Sanya diagnostics without changing fits or review flags."""
import argparse
import hashlib
import os
from pathlib import Path

import contourpy
import h5py
import numpy as np
# The installed legacy jcoord still imports numpy.mat.
if not hasattr(np,'mat'):np.mat=np.asmatrix
import jcoord

import noise_model
import plot_memo09_antenna_gain_patterns as gain
from plot_sanya_beam_position_histogram import ecef_to_enu_vectors, local_sky_axes


def positions(q, d):
    x0=d['center'][:3]+1000*q[:3];v0=d['center'][3:]+1000*q[3:6]
    speed=np.linalg.norm(v0);u=v0/speed;a=10**q[6];b=10**q[7];t=d['time_s']
    v=(speed-a*np.exp(b*t))[:,None]*u
    x=x0+(speed*t-a*np.expm1(b*t)/b)[:,None]*u
    f=d['frames'];rp=f[:,6:15].reshape(-1,3,3);rv=f[:,24:33].reshape(-1,3,3)
    transform=lambda p:f[:,:3]+np.einsum('nij,nj->ni',rp,p)+np.einsum('nij,nj->ni',rv,v)
    retarded=x.copy()
    for _ in range(2):retarded=x-v*np.linalg.norm(transform(retarded)-d['tx'],axis=1)[:,None]/299792458.
    return transform(retarded)


def export(folder, source_root, noise):
    folder=Path(folder)
    target=folder/'beam_rcs.h5'
    digest=hashlib.sha256((folder/'fit.h5').read_bytes()).hexdigest()
    if target.exists():
        with h5py.File(target) as h:
            if h.attrs.get('fit_sha256')==digest:return
    with h5py.File(folder/'input.h5') as h:
        original=Path(h.attrs.get('original_input',folder/'input.h5'))
    with h5py.File(original) as h:
        d={k:h[k][()] for k in ('center','time_s','time_ns','station','frames','tx','wavelength','duration')}
        source=str(h.attrs['source_event_files']).split(';')[0]
    with h5py.File(folder/'fit.h5') as h:q=h['params'][:]
    keep=d['station']==0
    for k in ('time_s','time_ns','station','frames'):d[k]=d[k][keep]
    source=Path(source_root)/'results/tristatic_head_echoes/sanya'/Path(source).name
    with h5py.File(source) as h:
        raw_times=h['times_ns'][:].astype(np.int64);snr=h['snr_peak_db'][:]
    # These event files use China local timestamps; require exact ID matches.
    candidates=[raw_times,raw_times-8*3600*10**9]
    matches=[t for t in candidates if np.isin(d['time_ns'],t).all()]
    if len(matches)!=1:raise ValueError('Cannot uniquely match Sanya UTC pulse IDs')
    index={int(t):i for i,t in enumerate(matches[0])}
    snr=np.array([snr[index[int(t)]] for t in d['time_ns']])
    xyz=positions(q,d);los=xyz-d['tx'];distance=np.linalg.norm(los,axis=1)
    lat,lon,_=jcoord.ecef2geodetic(*d['tx'])
    enu=ecef_to_enu_vectors(los,lat,lon)
    site=gain.SITES[0];p=gain.azel_to_enu(site.pointing_az_deg,site.pointing_el_deg)
    east,north=local_sky_axes(p);_,tilt,cross=gain.panel_axes(site)
    power=lambda directions:gain.aperture_power(directions,p,tilt,cross,site.dim_tilt_plane_m,site.dim_cross_tilt_m)
    relative=10*np.log10(np.maximum(power(enu/distance[:,None]),1e-12))
    peak=gain.site_summary(site)['steered_peak_gain_dbi'];linear=10**((peak+relative)/10)
    tsys=noise.tsys_k('Sanya',d['time_ns'])
    sigma=10**(snr/10)*1.380649e-23*tsys/d['duration']*(4*np.pi)**3*distance**4/(4.7e6*linear**2*d['wavelength']**2)
    rcs=np.where((relative>=-3)&np.isfinite(tsys)&(sigma>0),10*np.log10(sigma),np.nan)
    # Project rays onto one transverse plane: contours and points share a range.
    plane=float(np.median(enu@p));scale=plane/(enu@p)
    e=(enu@east)*scale;n=(enu@north)*scale
    extent=max(2500.,float(np.max(np.abs(np.r_[e,n])))*1.15)
    axis=np.linspace(-extent,extent,181);xx,yy=np.meshgrid(axis,axis)
    dirs=plane*p+xx[...,None]*east+yy[...,None]*north
    dirs/=np.linalg.norm(dirs,axis=-1)[...,None]
    field=10*np.log10(np.maximum(power(dirs),1e-12))
    contour=contourpy.contour_generator(x=axis,y=axis,z=field)
    with h5py.File(target.with_suffix('.h5.partial'),'w') as h:
        for key,val in dict(time_ns=d['time_ns'],time_s=d['time_s'],snr_db=snr,relative_gain_db=relative,rcs_dbsm=rcs,east_m=e,north_m=n,system_temperature_k=tsys).items():h[key]=val
        cg=h.create_group('contours')
        for level in (-3.,-10.,-20.):
            g=cg.create_group(str(int(level)));g.attrs['gain_db']=level
            for i,line in enumerate(contour.lines(level)):g[str(i)]=line
        h.attrs.update(plane_range_m=plane,extent_m=extent,fit_sha256=digest,
                       gain_convention='one-way power relative to steered peak; monostatic RCS uses G_tx*G_rx',
                       coordinate_convention='orthogonal projected east and north; ray intersection with plane at median axial range',
                       rcs_selection='Sanya one-way gain >= -3 dB; measured noise temperature; 1/pulse_duration noise bandwidth',source_program=__file__)
    os.replace(target.with_suffix('.h5.partial'),target)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+');p.add_argument('--source-root',required=True);args=p.parse_args()
    noise=noise_model.MeasuredSystemNoise(str(Path(args.source_root)/'results/sanya_4mhz_system_noise_power_100pulse.h5'))
    for root in args.roots:
        for fit in sorted(Path(root).glob('tri_*/fit.h5')):
            try:export(fit.parent,args.source_root,noise);print('OK',fit.parent.name,flush=True)
            except Exception as e:print('ERROR',fit.parent.name,str(e),flush=True)
