"""Observed RCS mass loss with auditable altitude/velocity/source interpolation."""
import argparse
import hashlib
import os
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

RATE_FACTOR = 4*np.sqrt(2*np.pi)*951.*3.8e-26


def invert(curve, measured, source):
    measured=np.asarray(measured)
    roots=np.full(measured.shape+(len(source)-1,),np.nan)
    for j,(a,b) in enumerate(zip(curve[:-1],curve[1:])):
        if not np.isfinite(a+b) or a==b:continue
        f=(measured-a)/(b-a)
        valid=(f>=0)&(f<=1 if j==len(source)-2 else f<1)
        roots[...,j]=np.where(valid,source[j]+f*(source[j+1]-source[j]),np.nan)
    count=np.isfinite(roots).sum(axis=-1)
    return np.where(count==1,np.nansum(roots,axis=-1),np.nan),count


def integrate(t, rcs, altitude, speed, gain, rejected, table, rng):
    """Interpolate bounded gaps in dBsm; never bridge an out-of-beam interval.

    Noise draws belong to measured anchors, not interpolated pseudo-measurements.
    Nonunique/out-of-table inversions break the integration instead of clipping.
    """
    ah, av, source, values = table
    interpolator = RegularGridInterpolator((ah, av), values, bounds_error=False, fill_value=np.nan)
    beam = np.isfinite(gain) & (gain >= -10)
    observed = beam & np.isfinite(rcs) & ~rejected
    pieces = np.split(np.arange(len(t)), np.flatnonzero(np.diff(beam.astype(int)))+1)
    result = []; masses = []; draws = []; requested = 0.
    for indices in pieces:
        anchors = indices[observed[indices]]
        if len(anchors) < 2: continue
        start, end = t[anchors[0]], t[anchors[-1]]
        if end <= start: continue
        requested += end-start
        grid = np.unique(np.r_[t[anchors], np.linspace(start, end, max(33, int(np.ceil((end-start)/.001))+1))])
        a = np.interp(grid, t, altitude); v = np.interp(grid, t, speed)
        curves = interpolator(np.c_[a,v])
        power = np.interp(grid, t[anchors], rcs[anchors])
        roots = np.array([invert(c, p, source)[0] for c,p in zip(curves,power)])
        rate = RATE_FACTOR*10**roots
        ok = np.isfinite(rate)
        pairs = ok[:-1]&ok[1:]
        weights = np.zeros(len(grid)); dt=np.diff(grid)*pairs
        weights[:-1] += dt/2; weights[1:] += dt/2
        active = weights > 0
        if not active.any(): continue
        masses.append(weights[active]@rate[active])
        # Conditional, receiver-noise-only intervals; shared anchor draws retain
        # the strong correlation introduced by interpolation across a gap.
        snr = 10**(rng['snr'][anchors]/10)
        error = 10/np.log(10)*np.sqrt(1+2*snr)/snr
        noisy = rcs[anchors]+rng['generator'].normal(size=(256,len(anchors)))*error
        noisy_grid = np.array([np.interp(grid,t[anchors],row) for row in noisy])
        samples = np.array([RATE_FACTOR*10**invert(c,noisy_grid[:,j],source)[0] for j,c in enumerate(curves)]).T
        draws.append(samples[:,active]@weights[active])
        result.append(dict(time_s=grid, altitude_km=a, speed_km_s=v, rcs_dbsm=power,
                           log10_source=roots, rate_kg_s=rate, integration_weights_s=weights,
                           interpolated_curves_dbsm=curves, anchor_indices=anchors))
    mass=float(sum(masses)) if masses else np.nan
    samples=np.sum(draws,axis=0) if draws else np.full(256,np.nan)
    fraction=float(np.isfinite(samples).mean())
    interval=np.nanpercentile(samples,[2.5,97.5]) if fraction>=.95 else [np.nan,np.nan]
    duration=sum(r['integration_weights_s'].sum() for r in result)
    return result, dict(mass_kg=mass, lo95_noise_kg=interval[0],hi95_noise_kg=interval[1],
                       noise_valid_fraction=fraction,covered_duration_s=duration,
                       requested_duration_s=requested,n_measured=int(observed.sum()),
                       n_missing_in_beam=int((beam&~observed).sum())), samples


def export(folder, lookup, flags):
    from export_voltage_beam import positions, jcoord
    target=folder/'rcs_mass.h5'
    flag=flags/(folder.name+'.h5')
    paths=[folder/'fit.h5',folder/'beam_rcs.h5',lookup]
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in paths)+(flag.read_bytes() if flag.exists() else b'')+Path(__file__).read_bytes()).hexdigest()
    if target.exists():
        with h5py.File(target) as h:
            if h.attrs.get('source_sha256')==digest:return
    with h5py.File(lookup) as h:
        table=(h['altitude_km'][:],h['velocity_km_s'][:],np.log10(h['source_n0_rM2_m_inv'][:]),h['rcs_dbsm'][:])
    with h5py.File(folder/'beam_rcs.h5') as h:
        beam={k:h[k][:] for k in ('time_ns','time_s','rcs_dbsm','relative_gain_db','snr_db')}
    with h5py.File(folder/'input.h5') as h:original=Path(h.attrs.get('original_input',folder/'input.h5'))
    with h5py.File(original) as h:
        keep=h['station'][:]==0
        d={k:h[k][:] for k in ('center','tx')}
        d.update({k:h[k][:][keep] for k in ('frames','time_s','time_ns')})
    np.testing.assert_array_equal(d['time_ns'],beam['time_ns'])
    with h5py.File(folder/'fit.h5') as h:q=h['params'][:]
    altitude=np.array([jcoord.ecef2geodetic(*p)[2]/1000 for p in positions(q,d)])
    speed=(np.linalg.norm(d['center'][3:]+1000*q[3:6])-10**q[6]*np.exp(10**q[7]*d['time_s']))/1000
    excluded=set()
    if flag.exists():
        with h5py.File(flag) as h:excluded=set(h['excluded'].asstr()[:])
    rejected=np.array([f'0:{t}' in excluded for t in beam['time_ns']])
    pieces,summary,samples=integrate(beam['time_s'],beam['rcs_dbsm'],altitude,speed,beam['relative_gain_db'],rejected,table,
        dict(snr=beam['snr_db'],generator=np.random.default_rng(20261029)))
    with h5py.File(target.with_suffix('.h5.partial'),'w') as h:
        h.attrs.update(**summary,source_sha256=digest,event=folder.name,minimum_one_way_gain_db=-10.,
            fit_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            beam_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest(),
            flag_sha256=hashlib.sha256(flag.read_bytes() if flag.exists() else b'').hexdigest(),
            meaning='Observed ablated mass over bounded in-beam intervals, not initial mass',
            uncertainty='Conditional receiver noise only; excludes calibration, trajectory, interpolation and scattering-model uncertainty',
            lookup_warning='Pilot head-on collisionless FDTD; mesh/time convergence not certified',
            source_program=Path(__file__).read_text(),lookup_file=str(lookup),rate_factor=RATE_FACTOR)
        h['mass_noise_samples_kg']=samples
        for k,val in beam.items():h['measurements/'+k]=val
        h['measurements/altitude_km']=altitude;h['measurements/speed_km_s']=speed;h['measurements/rejected']=rejected
        for i,piece in enumerate(pieces):
            for k,val in piece.items():h[f'intervals/{i}/{k}']=val
        for k,val in zip(('altitude_km','velocity_km_s','log10_source','rcs_dbsm'),table):h['lookup/'+k]=val
    os.replace(target.with_suffix('.h5.partial'),target)
    print(folder.name,summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--lookup',required=True,type=Path);p.add_argument('--flags',required=True,type=Path);p.add_argument('--cycles',default=1,type=int);args=p.parse_args()
    for cycle in range(args.cycles):
        for beam in sorted(args.root.glob('tri_*/beam_rcs.h5')):
            try:export(beam.parent,args.lookup,args.flags)
            except Exception as e:print('ERROR',beam.parent.name,repr(e),flush=True)
        if cycle+1<args.cycles:time.sleep(60)
