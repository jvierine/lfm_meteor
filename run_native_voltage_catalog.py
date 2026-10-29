"""MPI catalogue driver for the Rust waveform kernel and MH sampler.

Provisional ideal-waveform likelihood, not a validated replacement catalogue.
"""
import argparse
import ctypes as ct
import os
from pathlib import Path
import time
import traceback
import h5py
import numpy as np
# Legacy jcoord imports numpy.mat, removed in NumPy 2. Keep its matrix API local
# to this process without changing the shared server installation.
if not hasattr(np, 'mat'):
    np.mat = np.asmatrix
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import fit_gcrs_trajectories_lfm_ambiguity as gfit
import sanya_opts as sc
import fit_all_ballistic_snr_weighted as base
from grid_search_delays_beam_axis import load_events, pair_tristatic_events, SAN_PATTERN, DAN_PATTERN, WEN_PATTERN


def frame_coefficients(times):
    # Same affine transform convention as fit_event_joint_delay_doppler_fft.
    zero=np.zeros((len(times),3));x0,v0=base.gcrs_state_samples_to_itrs(zero,zero,times)
    rp=np.empty((len(times),3,3));vp=rp.copy();rv=rp.copy();vv=rp.copy()
    for j in range(3):
        basis=zero.copy();basis[:,j]=1.
        x,v=base.gcrs_state_samples_to_itrs(basis,zero,times);rp[:,:,j]=x-x0;vp[:,:,j]=v-v0
        x,v=base.gcrs_state_samples_to_itrs(zero,basis,times);rv[:,:,j]=x-x0;vv[:,:,j]=v-v0
    return x0,v0,rp,vp,rv,vv


def prepare(path, events, reference):
    rows, times, stations, sample_times, noise, gates = [], [], [], [], [], []
    n = 1280
    for station, event in enumerate(events):
        with h5py.File(event.path, 'r') as h:
            raw = h['raw'][()]
            ts = h['times_ns'][()].astype(np.int64)
            if not event.times_ns_are_utc:
                ts -= int(event.source_timezone_offset_hours*3600*1e9)
            gate = h['range_gate'][()]
            snr = h['snr_peak_db'][()]
            fs = float(h['sr_mhz'][()])*1e6
            bandwidth = float(h['bw_mhz'][()])*1e6
            if fs != 4e6 or bandwidth != 4e6:
                raise ValueError('Only the 4 MHz waveform is supported by this pilot')
            for i in range(len(ts)):
                if not np.isfinite(snr[i]) or snr[i]<15:
                    continue
                start = int(gate[i])-n//2
                if start<0 or start+n>raw.shape[1]:
                    continue
                idx = np.arange(raw.shape[1])
                quiet = abs(idx-gate[i])>600
                if quiet.sum()<100:
                    raise ValueError('Insufficient off-pulse voltage for noise estimate')
                variance = np.median(abs(raw[i,quiet])**2)/np.log(2)
                if not np.isfinite(variance) or variance<=0:
                    continue
                rows.append(raw[i,start:start+n]); times.append(ts[i]); stations.append(station)
                # Catalogue gates represent chirp centres, not envelope leading edges.
                sample_times.append(359.45540369317763e-6+(start+np.arange(n))/fs+100e-6)
                noise.append(variance); gates.append(gate[i])
    if len(rows)<12 or len(set(stations))!=3:
        raise ValueError('Insufficient usable pulses across all three receivers')
    times = np.array(times, dtype=np.int64)
    transforms = frame_coefficients(times)
    frames = np.concatenate([a.reshape(len(times),-1) for a in transforms], axis=1)
    center = np.r_[reference['r0_gcrs_m'], reference['v0_gcrs_mps']]
    # Initial asymptotic speed includes a=100 m/s, preserving seed detection speed.
    center[3:] += 100*center[3:]/np.linalg.norm(center[3:])
    rx = np.array(gfit.LINK_RX_POSITIONS_M)[stations]
    data = dict(center=center,time_s=(times-reference['t0_ns'])*1e-9,time_ns=times,
                frames=frames,tx=np.array(sc.p_san),rx=rx,station=np.array(stations),
                real=np.array(rows).real,imag=np.array(rows).imag,noise_power=np.array(noise),
                sample_time_s=np.array(sample_times),fs=fs,bandwidth=bandwidth,duration=200e-6,
                wavelength=float(sc.RADAR_WAVELENGTH_M),gate=np.array(gates))
    with h5py.File(path,'w') as h:
        for key,value in data.items(): h[key]=value
        h.attrs['source_event_files']=';'.join(e.path for e in events)
        h.attrs['seed_event']=reference['event_id']
        h.attrs['provisional']=True
    return data


def load_library(path):
    lib=ct.CDLL(str(path))
    pointer=ct.POINTER(ct.c_double)
    lib.voltage_load.argtypes=[ct.c_char_p];lib.voltage_load.restype=ct.c_void_p
    lib.voltage_len.argtypes=[ct.c_void_p];lib.voltage_len.restype=ct.c_size_t
    lib.voltage_residual.argtypes=[ct.c_void_p,pointer,pointer];lib.voltage_residual.restype=ct.c_int
    lib.voltage_free.argtypes=[ct.c_void_p]
    lib.voltage_sample.argtypes=[ct.c_void_p,pointer,pointer,ct.c_size_t,ct.c_size_t,ct.c_uint64,ct.c_char_p]
    lib.voltage_sample.restype=ct.c_int
    return lib


def beat_peak(row, fs):
    nfft=1<<int(np.ceil(np.log2(len(row)*64)))
    power=abs(np.fft.fftshift(np.fft.fft(row*np.hanning(len(row)),nfft)))**2
    k=int(np.argmax(power)); frequency=(k-nfft//2)*fs/nfft
    if 0<k<len(power)-1:
        a,b,c=np.log(np.maximum(power[k-1:k+2],1e-300))
        if a-2*b+c<0: frequency += .5*(a-c)/(a-2*b+c)*fs/nfft
    return frequency


def run_one(args, event, events, reference, lib):
    out=Path(args.output)/event;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.h5').exists():
        return 'already_complete'
    data=prepare(out/'input.h5',events,reference)
    handle=lib.voltage_load(os.fsencode(out/'input.h5'))
    if not handle: raise RuntimeError('Rust input load failed')
    ptr=ct.POINTER(ct.c_double)
    try:
        size=lib.voltage_len(handle)
        def residual(q):
            q=np.ascontiguousarray(q,dtype=np.float64);r=np.empty(size)
            if not lib.voltage_residual(handle,q.ctypes.data_as(ptr),r.ctypes.data_as(ptr)):
                return np.full(size,1e10)
            return r
        initial=np.r_[np.zeros(6),2.,0.]
        start=time.monotonic()
        result=least_squares(residual,initial,x_scale='jac',max_nfev=args.max_nfev,
                             ftol=1e-8,xtol=1e-8,gtol=1e-7)
        r=residual(result.x).reshape(data['real'].shape+(2,))
        raw=data['real']+1j*data['imag']
        predicted=raw+(r[:,:,0]+1j*r[:,:,1])*np.sqrt(data['noise_power'][:,None]/2)
        beats=[];control_std=[]
        rng=np.random.default_rng(20261029)
        for i in range(len(raw)):
            mask=abs(predicted[i])>.2*np.max(abs(predicted[i]))
            phase=np.zeros(raw.shape[1],complex)
            phase[mask]=np.conj(predicted[i,mask])/abs(predicted[i,mask])
            beats.append(beat_peak(raw[i]*phase,data['fs']))
            # Conditional estimator precision; does not include model/calibration errors.
            draws=[]
            for _ in range(32):
                noise=np.sqrt(data['noise_power'][i]/2)*(rng.normal(size=len(phase))+1j*rng.normal(size=len(phase)))
                draws.append(beat_peak((predicted[i]+noise)*phase,data['fs']))
            control_std.append(np.std(draws,ddof=1))
        beats=np.array(beats);control_std=np.array(control_std)
        z=beats/np.maximum(control_std,1e-6)
        beat_guard=bool(np.all(np.abs(z)<5) and abs(np.mean(z))<3/np.sqrt(len(z)))
        with h5py.File(out/'fit.h5.partial','w') as h:
            h['params']=result.x;h['initial']=initial;h['residual_beat_hz']=beats
            h['conditional_beat_std_hz']=control_std;h['beat_z']=z;h['station']=data['station'];h['time_s']=data['time_s']
            h['predicted_real']=predicted.real;h['predicted_imag']=predicted.imag
            h.attrs.update(optimizer_success=result.success,optimizer_message=result.message,
                           cost=result.cost,nfev=result.nfev,elapsed_s=time.monotonic()-start,
                           beat_guard_pass=beat_guard,provisional=True,
                           likelihood='Gaussian I/Q noise; per-pulse complex amplitude profiled',
                           limitations='Ideal waveform; cell-averaged rectangular support; receiver filtering and intrapulse motion unvalidated. Beat guard is diagnostic, not independent validation.')
        os.replace(out/'fit.h5.partial',out/'fit.h5')
        fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True,constrained_layout=True)
        for station,ax in enumerate(axes):
            k=data['station']==station
            ax.errorbar(data['time_s'][k],beats[k],yerr=1.96*control_std[k],fmt='.',capsize=2)
            ax.axhline(0,color='black',lw=.8);ax.set_ylabel(['Sanya','Danzhou','Wenchang'][station]+'\nResidual beat (Hz)')
        axes[-1].set_xlabel('Time from reference epoch (s)')
        fig.savefig(out/'beat_guard.png',dpi=160);fig.savefig(out/'beat_guard.pdf');plt.close(fig)
        if args.draws>0:
            widths=np.r_[np.full(3,1e-4),np.full(3,1e-3),.01,.01].astype(float)
            q=np.ascontiguousarray(result.x)
            for chain in range(4):
                target=out/f'chain_{chain}.h5.partial'
                ok=lib.voltage_sample(handle,q.ctypes.data_as(ptr),widths.ctypes.data_as(ptr),args.warmup,args.draws,
                                      20261029+chain,os.fsencode(target))
                if not ok: raise RuntimeError('Native MH failed')
                os.replace(target,out/f'chain_{chain}.h5')
        with h5py.File(out/'complete.h5','w') as h:
            h.attrs.update(provisional=True,draws_per_chain=args.draws,chains=4 if args.draws else 0,
                           beat_guard_pass=beat_guard,optimizer_success=result.success,
                           sampler='Rust random-walk Metropolis-Hastings; proposal frozen after warmup',
                           amplitude_prior='Flat complex Re/Im prior, amplitudes analytically marginalized in MCMC',
                           convergence_verified=False)
        return f'optimizer={result.success} beat_guard={beat_guard} median_abs_beat={np.median(abs(beats)):.2f}Hz'
    finally:
        lib.voltage_free(handle)


def main():
    os.chdir(Path(__file__).resolve().parent)
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--library',required=True)
    p.add_argument('--event');p.add_argument('--max-nfev',type=int,default=150)
    p.add_argument('--warmup',type=int,default=4000);p.add_argument('--draws',type=int,default=8000)
    args=p.parse_args();comm=MPI.COMM_WORLD
    reference=base.load_reference_fits()
    triplets=pair_tristatic_events(load_events(SAN_PATTERN),load_events(DAN_PATTERN),load_events(WEN_PATTERN))
    jobs=[]
    for events in triplets:
        ref=base.match_reference_fit(events[0],reference)
        if ref and (not args.event or ref['event_id']==args.event):jobs.append((ref['event_id'],events,ref))
    if not jobs:raise RuntimeError('No matching events')
    if comm.rank==0:print(f'events={len(jobs)} mpi_ranks={comm.size}',flush=True)
    lib=load_library(args.library)
    for index,(event,events,ref) in enumerate(jobs):
        if index%comm.size!=comm.rank:continue
        try:
            print(f'START {event}',flush=True)
            status=run_one(args,event,events,ref,lib)
            print(f'DONE {event} {status}',flush=True)
        except Exception:
            out=Path(args.output)/event;out.mkdir(parents=True,exist_ok=True)
            (out/'error.log').write_text(traceback.format_exc())
            print(f'FAILED {event}\n{traceback.format_exc()}',flush=True)


if __name__=='__main__':main()
