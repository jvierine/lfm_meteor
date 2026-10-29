"""Evaluate rejected measurements against the new fit, never fit them."""
import ctypes as ct
import os
from pathlib import Path
import h5py
import numpy as np
from run_native_voltage_catalog import load_library, apparent_paths, beat_peak


def export(folder, library):
    folder=Path(folder)
    with h5py.File(folder/'input.h5') as h:
        source=h.attrs.get('original_input')
        present=set(zip(h['station'][:],h['time_ns'][:]))
    if not source:return
    with h5py.File(source) as h:data={k:h[k][()] for k in h}
    missing=np.array([i for i,k in enumerate(zip(data['station'],data['time_ns'])) if k not in present],dtype=int)
    if not len(missing):return
    with h5py.File(folder/'fit.h5') as h:q=np.ascontiguousarray(h['params'][:])
    handle=library.voltage_load(os.fsencode(source))
    if not handle:raise RuntimeError('Cannot load original measurements')
    try:
        output=np.empty(library.voltage_len(handle));ptr=ct.POINTER(ct.c_double)
        valid=bool(library.voltage_prediction(handle,q.ctypes.data_as(ptr),output.ctypes.data_as(ptr)))
        prediction=output.reshape(data['real'].shape+(2,));prediction=prediction[:,:,0]+1j*prediction[:,:,1]
    finally:library.voltage_free(handle)
    model=apparent_paths(q,data)
    anchor=299792458.*(359.45540369317763e-6+data['gate']/data['fs'])
    t=data['sample_time_s']-anchor[:,None]/299792458.;rate=data['bandwidth']/data['duration']
    reference=np.exp(2j*np.pi*(data['bandwidth']/2*t-rate/2*t*t))
    raw=data['real']+1j*data['imag'];paths=[];beats=[]
    for i in missing:
        paths.append(anchor[i]+299792458./rate*beat_peak(raw[i]*np.conj(reference[i]),data['fs']))
        if not valid:
            beats.append(np.nan);continue
        mask=abs(prediction[i])>.2*np.max(abs(prediction[i]));phase=np.zeros(raw.shape[1],complex)
        phase[mask]=np.conj(prediction[i,mask])/abs(prediction[i,mask])
        beats.append(beat_peak(raw[i]*phase,data['fs']))
    target=folder/'excluded_measurements.h5'
    with h5py.File(target.with_suffix('.h5.partial'),'w') as h:
        for k in ('station','time_ns','time_s'):h[k]=data[k][missing]
        h['path_m']=paths;h['modeled_path_m']=model[missing];h['residual_beat_hz']=beats
        h.attrs['used_in_fit']=False
    os.replace(target.with_suffix('.h5.partial'),target)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--library',required=True);args=p.parse_args()
    lib=load_library(args.library)
    for fit in sorted(Path(args.root).glob('*/fit.h5')):
        export(fit.parent,lib)
        print(fit.parent.name,flush=True)
