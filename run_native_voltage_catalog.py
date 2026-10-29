"""MPI catalogue driver for the Rust waveform kernel and MH sampler.

Provisional ideal-waveform likelihood, not a validated replacement catalogue.
"""
import argparse
import ctypes as ct
import os
from pathlib import Path
import time
import traceback
import subprocess
import hashlib
import h5py
import numpy as np
from scipy.signal import lfilter
from voltage_noise_covariance import fit_noise, colored_noise
# Legacy jcoord imports numpy.mat, removed in NumPy 2. Keep its matrix API local
# to this process without changing the shared server installation.
if not hasattr(np, 'mat'):
    np.mat = np.asmatrix
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
from astropy.utils import iers
iers.conf.auto_download=False
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
    filters=[];noise_audits=[]
    for station, event in enumerate(events):
        with h5py.File(event.path, 'r') as h:
            raw = h['raw'][()]
            ts = h['times_ns'][()].astype(np.int64)
            if not event.times_ns_are_utc:
                ts -= int(event.source_timezone_offset_hours*3600*1e9)
            gate = h['range_gate'][()]
            audit=fit_noise(raw,gate);noise_audits.append(audit)
            whitening=np.zeros(65,complex);whitening[:len(audit['whitener'])]=audit['whitener']
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
                quiet = (abs(idx-gate[i])>714)&(idx>=64)
                if quiet.sum()<100:
                    raise ValueError('Insufficient off-pulse voltage for noise estimate')
                innovations=lfilter(whitening,[1.],raw[i])
                variance = np.median(abs(innovations[quiet])**2)/np.log(2)
                if not np.isfinite(variance) or variance<=0:
                    continue
                rows.append(raw[i,start:start+n]); times.append(ts[i]); stations.append(station)
                # Catalogue gates represent chirp centres, not envelope leading edges.
                sample_times.append(359.45540369317763e-6+(start+np.arange(n))/fs+100e-6)
                noise.append(variance); gates.append(gate[i]);filters.append(whitening)
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
                whitener_real=np.array(filters).real,whitener_imag=np.array(filters).imag,
                noise_covariance_pass=np.array([a['whiteness_pass'] for a in noise_audits]),
                noise_ar_order=np.array([a['order'] for a in noise_audits]),
                noise_heldout_correlation=np.array([a['heldout_correlation'] for a in noise_audits]),
                noise_temporal_correlation=np.array([a['temporal_correlation'] for a in noise_audits]),
                noise_pseudocorrelation=np.array([a['heldout_pseudocorrelation'] for a in noise_audits]),
                noise_mean_z=np.array([a['mean_z'] for a in noise_audits]),
                noise_heldout_tolerance=np.array([a['heldout_tolerance'] for a in noise_audits]),
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
    lib.voltage_prediction.argtypes=[ct.c_void_p,pointer,pointer];lib.voltage_prediction.restype=ct.c_int
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


def apparent_paths(q,data):
    x0=data['center'][:3]+1000*q[:3];v0=data['center'][3:]+1000*q[3:6]
    speed=np.linalg.norm(v0);u=v0/speed;a=10**q[6];b=10**q[7];t=data['time_s']
    with np.errstate(over='ignore',invalid='ignore'):
        velocity=(speed-a*np.exp(b*t))[:,None]*u
        nominal=x0+(speed*t-a*np.expm1(b*t)/b)[:,None]*u
    if not np.all(np.isfinite(nominal)) or np.any(speed-a*np.exp(b*t)<=100):
        raise ValueError('Nonphysical WJ state')
    f=data['frames'];rp=f[:,6:15].reshape(-1,3,3);vp=f[:,15:24].reshape(-1,3,3)
    rv=f[:,24:33].reshape(-1,3,3);vv=f[:,33:42].reshape(-1,3,3)
    def frame(x,v):
        return (f[:,:3]+np.einsum('nij,nj->ni',rp,x)+np.einsum('nij,nj->ni',rv,v),
                f[:,3:6]+np.einsum('nij,nj->ni',vp,x)+np.einsum('nij,nj->ni',vv,v))
    retarded=nominal.copy()
    for _ in range(2):
        x,_=frame(retarded,velocity)
        retarded=nominal-velocity*np.linalg.norm(x-data['tx'],axis=1)[:,None]/299792458.
    x,v=frame(retarded,velocity);tx=x-data['tx'];rx=x-data['rx']
    a=np.linalg.norm(tx,axis=1);b=np.linalg.norm(rx,axis=1)
    fd=-np.sum((tx/a[:,None]+rx/b[:,None])*v,axis=1)/data['wavelength']
    return a+b+299792458.*fd/(data['bandwidth']/data['duration'])


def beat_seed(data,initial):
    anchor=299792458.*(359.45540369317763e-6+data['gate']/data['fs'])
    t=data['sample_time_s']-anchor[:,None]/299792458.
    k=data['bandwidth']/data['duration']
    reference=np.exp(2j*np.pi*(data['bandwidth']/2*t-k/2*t*t))
    raw=data['real']+1j*data['imag']
    observed=np.array([beat_peak(row,data['fs']) for row in raw*np.conj(reference)])
    measured=anchor+299792458.*observed/k
    def residual(q):
        try:return (apparent_paths(q,data)-measured)/3.
        except ValueError:return np.full(len(measured),1e10)
    bounds=(np.r_[np.full(3,-20.),np.full(3,-40.),-5.,-4.],
            np.r_[np.full(3,20.),np.full(3,40.),5.,3.])
    result=least_squares(residual,initial,bounds=bounds,x_scale='jac',loss='soft_l1',
                         max_nfev=400,ftol=1e-10,xtol=1e-10,gtol=1e-9)
    return result.x,observed,measured,residual(result.x)*3


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
        seeded,seed_beats,seed_paths,seed_residual=beat_seed(data,initial)
        result=least_squares(residual,seeded,x_scale='jac',max_nfev=args.max_nfev,
                             ftol=1e-8,xtol=1e-8,gtol=1e-7)
        raw=data['real']+1j*data['imag']
        q=np.ascontiguousarray(result.x);prediction=np.empty(size)
        if not lib.voltage_prediction(handle,q.ctypes.data_as(ptr),prediction.ctypes.data_as(ptr)):
            raise RuntimeError('Native waveform prediction failed')
        prediction=prediction.reshape(raw.shape+(2,))
        predicted=prediction[:,:,0]+1j*prediction[:,:,1]
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
                coeff=-(data['whitener_real'][i,1:]+1j*data['whitener_imag'][i,1:])
                noise=colored_noise(rng,len(phase),coeff,data['noise_power'][i])
                draws.append(beat_peak((predicted[i]+noise)*phase,data['fs']))
            control_std.append(np.std(draws,ddof=1))
        beats=np.array(beats);control_std=np.array(control_std)
        z=beats/np.maximum(control_std,1e-6)
        trend_coefficients=[];trend_z=[]
        for station in range(3):
            mask=data['station']==station
            t=data['time_s'][mask];t=t-np.mean(t)
            design=np.column_stack([np.ones(len(t)),t])/control_std[mask,None]
            covariance=np.linalg.pinv(design.T@design)
            coef=covariance@design.T@(beats[mask]/control_std[mask])
            trend_coefficients.append(coef)
            trend_z.append(coef/np.sqrt(np.diag(covariance)))
        beat_guard=bool(np.all(np.abs(z)<5) and np.all(np.abs(trend_z)<3))
        with h5py.File(out/'fit.h5.partial','w') as h:
            h['params']=result.x;h['initial']=initial;h['residual_beat_hz']=beats
            h['beat_seed_params']=seeded;h['seed_observed_beat_hz']=seed_beats
            h['seed_apparent_path_m']=seed_paths;h['seed_path_residual_m']=seed_residual
            h['conditional_beat_std_hz']=control_std;h['beat_z']=z;h['station']=data['station'];h['time_s']=data['time_s']
            h['station_beat_offset_and_slope']=trend_coefficients;h['station_beat_trend_z']=trend_z
            h['predicted_real']=predicted.real;h['predicted_imag']=predicted.imag
            h.attrs.update(optimizer_success=result.success,optimizer_message=result.message,
                           cost=result.cost,nfev=result.nfev,elapsed_s=time.monotonic()-start,
                           beat_guard_pass=beat_guard,provisional=True,
                           likelihood='Proper complex Gaussian AR innovations; shared whitening of waveform and data; amplitude profiled for optimization and marginalized for MCMC',
                           noise_covariance_pass=bool(np.all(data['noise_covariance_pass'])),
                           limitations='Ideal waveform; cell-averaged rectangular support; receiver filtering and intrapulse motion unvalidated. Beat guard is diagnostic, not independent validation.')
        os.replace(out/'fit.h5.partial',out/'fit.h5')
        fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True,constrained_layout=True)
        for station,ax in enumerate(axes):
            k=data['station']==station
            ax.errorbar(data['time_s'][k],beats[k],yerr=1.96*control_std[k],fmt='.',capsize=2)
            ax.axhline(0,color='black',lw=.8);ax.set_ylabel(['Sanya','Danzhou','Wenchang'][station]+'\nResidual beat (Hz)')
        axes[-1].set_xlabel('Time from reference epoch (s)')
        fig.savefig(out/'beat_guard.png',dpi=160);fig.savefig(out/'beat_guard.pdf');plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        modeled=apparent_paths(result.x,data)
        for station,label in enumerate(['Sanya','Danzhou','Wenchang']):
            k=data['station']==station;t=data['time_s'][k];order=np.argsort(t)
            offset=seed_paths[k][0]
            axes[0,0].plot(t,(seed_paths[k]-offset)/1000,'.',alpha=.6)
            axes[0,0].plot(t[order],(modeled[k][order]-offset)/1000,label=label)
            axes[1,0].plot(t,modeled[k]-seed_paths[k],'.',label=label)
            axes[1,1].errorbar(t,beats[k],yerr=1.96*control_std[k],fmt='.',capsize=2,label=label)
        t=np.linspace(data['time_s'].min(),data['time_s'].max(),200)
        vinf=np.linalg.norm(data['center'][3:]+1000*result.x[3:6]);a=10**result.x[6];b=10**result.x[7]
        axes[0,1].plot(t,(vinf-a*np.exp(b*t))/1000,color='black')
        axes[0,1].text(.03,.05,'Beat guard: '+('PASS' if beat_guard else 'FAIL')+'\nProvisional waveform model',
                       transform=axes[0,1].transAxes,fontsize=10)
        axes[0,0].set_ylabel('Apparent path offset (km)');axes[0,1].set_ylabel('Speed (km/s)')
        axes[1,0].set_ylabel('Apparent path residual (m)');axes[1,1].set_ylabel('Residual beat (Hz)')
        for ax in axes[1]:ax.axhline(0,color='black',lw=.8)
        for ax in axes.flat:ax.set_xlabel('Time from reference epoch (s)')
        axes[0,0].legend();fig.suptitle(event)
        fig.savefig(out/'event.png',dpi=160);fig.savefig(out/'event.pdf');plt.close(fig)
        sampling_allowed=bool(args.draws>0 and beat_guard and result.success and np.all(data['noise_covariance_pass']))
        if sampling_allowed:
            widths=np.r_[np.full(3,1e-4),np.full(3,1e-3),.01,.01].astype(float)
            q=np.ascontiguousarray(result.x)
            for chain in range(4):
                target=out/f'chain_{chain}.h5.partial'
                ok=lib.voltage_sample(handle,q.ctypes.data_as(ptr),widths.ctypes.data_as(ptr),args.warmup,args.draws,
                                      20261029+chain,os.fsencode(target))
                if not ok: raise RuntimeError('Native MH failed')
                os.replace(target,out/f'chain_{chain}.h5')
        with h5py.File(out/'complete.h5','w') as h:
            h.attrs.update(provisional=True,draws_per_chain=args.draws if sampling_allowed else 0,chains=4 if sampling_allowed else 0,
                           beat_guard_pass=beat_guard,optimizer_success=result.success,
                           noise_covariance_pass=bool(np.all(data['noise_covariance_pass'])),
                           sampling_blocked_by_guard=bool(args.draws>0 and not sampling_allowed),
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
    if comm.rank==0:
        print(f'events={len(jobs)} mpi_ranks={comm.size}',flush=True)
        Path(args.output).mkdir(parents=True,exist_ok=True)
        with h5py.File(Path(args.output)/'launch.h5','w') as h:
            h['event_ids']=np.array([j[0] for j in jobs],dtype=h5py.string_dtype())
            h.attrs.update(mpi_ranks=comm.size,draws=args.draws,warmup=args.warmup,
                           git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                           native_library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
                           provisional=True,started_unix=time.time())
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
