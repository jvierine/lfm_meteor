"""Conditional mass intervals for calibrated Sanya RCS pairs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

RATE = 4*np.sqrt(2*np.pi)*951.*3.8e-26


def setup(root, data):
    global ROOT, SKY, INTERP, SOURCE
    ROOT = root
    with h5py.File(data/"sanya_noise.h5") as h:
        SKY = {d: (h[d+"/unix_s"][:], h[d+"/sky_k"][:]) for d in ("17", "21")}
    with h5py.File(data/"lookup.h5") as h:
        SOURCE = np.log10(h["source_n0_rM2_m_inv"][:])
        INTERP = RegularGridInterpolator((h["altitude_km"][:], h["velocity_km_s"][:]),
                                         h["rcs_dbsm"][:], bounds_error=False, fill_value=np.nan)


def event(key):
    day, name = key.split(":")
    folder = ROOT/(day+"_tristatic")/"fitted_beam/events"/name
    with h5py.File(folder/"physical.h5") as h:
        dyn = h["mass_local95_kg"][:] if h.attrs.get("path_quality_pass", False) else [np.nan]*2
        mass = float(h["initial_mass_kg"][()])
    with h5py.File(folder/"rcs.h5") as h:
        dt = np.diff(h["time_s"][:])
        t, sky = SKY[day[-2:]]
        pulse = h["time_ns"][:]/1e9
        assert pulse.min() >= t[0] and pulse.max() <= t[-1]
        rcs = h["rcs_dbsm"][:]+10*np.log10((250+np.interp(pulse,t,sky))/h.attrs["system_temperature_k"])
        beam = h["relative_gain_db"][:] >= -10.
        curves = INTERP(np.c_[h["height_km"][:],h["speed_km_s"][:]])
    good = beam & np.isfinite(rcs)
    if good.sum() < 2:
        return key, mass, dyn, np.nan, [np.nan]*2, 0.
    scale = np.std(rcs[good])
    offsets = np.r_[0.,np.random.default_rng(int(name.split("_")[1])+int(day)).normal(0.,scale,2048)]
    rate = np.full((len(offsets), len(rcs)),np.nan)
    for i in np.flatnonzero(good):
        c = curves[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = (rcs[i]+offsets[:,None]-c[:-1])/np.diff(c)
        ok = np.isfinite(w) & (w >= 0) & (w <= 1)
        ok[:,1:] &= w[:,1:] != 0
        root = SOURCE[:-1]+w*np.diff(SOURCE)
        rate[:,i] = np.where(ok.sum(axis=1)==1, RATE*10**np.where(ok,root,0.).sum(axis=1),np.nan)
    # Fix integration support to the nominal estimator. Never manufacture
    # a narrow interval by dropping missing intervals from individual draws.
    nominal = np.isfinite(rate[0,:-1]) & np.isfinite(rate[0,1:]) & (dt <= .050001)
    masses = ((rate[:,:-1]+rate[:,1:])[:,nominal] @ (dt[nominal]/2)
              if nominal.any() else np.full(len(offsets),np.nan))
    samples = masses[1:]
    fraction = np.isfinite(samples).mean()
    bounds = np.quantile(samples,[.025,.975]) if fraction==1. else [np.nan]*2
    return key,mass,dyn,masses[0],bounds,fraction


def main():
    p = argparse.ArgumentParser()
    for key in ("root","data","out"):
        p.add_argument(key,type=Path)
    p.add_argument("--workers",type=int,default=1)
    a=p.parse_args()
    with h5py.File(a.data/"figure8.h5") as h:
        ids=[k for k in h["mass_comparison/event_id"].asstr()[:] if k.startswith("202611")]
    with ProcessPoolExecutor(a.workers,initializer=setup,initargs=(a.root,a.data)) as pool:
        rows=list(pool.map(event,ids,chunksize=4))
    with h5py.File(a.out,"w") as h:
        for k,v in zip(("event_id","dynamic_mass_kg","dynamic_95_kg","rcs_mass_kg","rcs_95_kg","rcs_draw_valid_fraction"),zip(*rows)):
            h[k]=np.asarray(v,dtype=h5py.string_dtype()) if k=="event_id" else v
        h.attrs["rcs_interval"]="2048 shared Gaussian dBsm offsets with observed in-beam RMS; fixed trajectory and nominal integration support; unresolved if a draw leaves lookup"
    print("Events",len(rows),"resolved RCS intervals",sum(np.isfinite(r[4]).all() for r in rows),flush=True)


if __name__=="__main__":
    main()
