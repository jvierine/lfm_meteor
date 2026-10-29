"""Conditional measurement intervals for the displayed chirp-coupled paths."""
import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
from run_native_voltage_catalog import beat_peak
from voltage_noise_covariance import colored_noise


def path_intervals(data, predicted, observed, draws=128):
    anchor=299792458.*(359.45540369317763e-6+data['gate']/data['fs'])
    t=data['sample_time_s']-anchor[:,None]/299792458.
    chirp_rate=data['bandwidth']/data['duration']
    reference=np.exp(2j*np.pi*(data['bandwidth']/2*t-chirp_rate/2*t*t))
    rng=np.random.default_rng(20261030)
    intervals=[]
    for i,signal in enumerate(predicted):
        reference_conj=np.conj(reference[i])
        noiseless=beat_peak(signal*reference_conj,data['fs'])
        coeff=-(data['whitener_real'][i,1:]+1j*data['whitener_imag'][i,1:])
        error=np.empty(draws)
        for j in range(draws):
            noise=colored_noise(rng,len(signal),coeff,data['noise_power'][i])
            error[j]=(beat_peak((signal+noise)*reference_conj,data['fs'])-noiseless)*299792458./chirp_rate
        lo,hi=np.quantile(error,[.025,.975])
        intervals.append([observed[i]-hi,observed[i]-lo])
    return np.asarray(intervals)


def add_intervals(folder):
    folder=Path(folder)
    with h5py.File(folder/'input.h5') as h:data={k:h[k][()] for k in h}
    with h5py.File(folder/'fit.h5') as h:
        signal=h['predicted_real'][:]+1j*h['predicted_imag'][:]
        observed=h['seed_apparent_path_m'][:]
    interval=path_intervals(data,signal,observed)
    # A sidecar avoids races with the running sampler's plot bundle exporter.
    target=folder/'path_measurement_uncertainty.h5'
    with h5py.File(target.with_suffix('.h5.partial'),'w') as h:
        h['path_measurement_lower95_m']=interval[:,0]
        h['path_measurement_upper95_m']=interval[:,1]
        h['time_ns']=data['time_ns'];h['station']=data['station']
        h.attrs.update(draws=128,confidence=.95,
            fit_sha256=hashlib.sha256((folder/'fit.h5').read_bytes()).hexdigest(),
            interval='Conditional measurement interval from 128 correlated complex-noise realizations using the same fixed-reference dechirp and peak estimator as displayed data',
            observable='Apparent chirp-coupled tx-target-rx path, not independently Doppler-corrected geometric path',
            excludes='Calibration and trajectory-model discrepancy; not extra independent likelihood data')
    os.replace(target.with_suffix('.h5.partial'),target)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args()
    root=Path(args.root)
    folders=[root] if (root/'fit.h5').exists() else sorted(x.parent for x in root.glob('*/fit.h5'))
    for folder in folders:
        if (folder/'path_measurement_uncertainty.h5').exists():continue
        add_intervals(folder)
        print(folder.name,flush=True)
