"""Direct voltage fit of x0, v0 and log10(r0); WJ supplies an initial guess only."""
import argparse
import ctypes as ct
import os
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares
from mpi4py import MPI
from arviz_stats.base import array_stats
from refit_voltage_reviewed import masked_input
from run_native_voltage_catalog import load_library, beat_peak, colored_noise
import fit_all_ballistic_snr_weighted as atmosphere
import jcoord

C=299792458.
PTR=ct.POINTER(ct.c_double)


def save(path,fields,attrs):
    with h5py.File(str(path)+'.partial','w') as h:
        for k,v in fields.items():h[k]=v
        h.attrs.update(attrs)
    os.replace(str(path)+'.partial',path)


def prepare(args,event,folder):
    d=masked_input(args.source/event/'input.h5',args.flags/(event+'.h5'),folder/'input.h5')
    t0=float(d['initial_detection_time_s'])
    guess=args.initial/event/'fit.h5'
    if guess.exists():
        with h5py.File(guess) as h:q=h['params'][:]
    else:q=np.r_[np.zeros(6),2.,0.]
    x=d['center'][:3]+1000*q[:3];v=d['center'][3:]+1000*q[3:6]
    speed=np.linalg.norm(v);u=v/speed;a=10**q[6];b=10**q[7]
    x=x+u*(speed*t0-a*np.expm1(b*t0)/b);v=u*(speed-a*np.exp(b*t0))
    d['center']=np.r_[x,v];d['time_s']=d['time_s']-t0
    f=d['initial_detection_frame'];p=f[:3]+f[6:15].reshape(3,3)@x+f[24:33].reshape(3,3)@v
    rho,meta=atmosphere.density_interpolator(np.array([d['initial_detection_time_ns']]),p[None,:])
    d['height_m']=atmosphere.MSIS_ALT_GRID_KM*1000
    d['density_kg_m3']=rho(d['height_m']);d['initial_detection_time_s']=0.
    density=float(rho(np.array([jcoord.ecef2geodetic(*p)[2]]))[0])
    radius=np.clip(3*density*np.dot(v,v)/(4*3000*max(a*b*np.exp(b*t0),1e-6)),1e-7,1e-2)
    with h5py.File(folder/'input.h5','a') as h:
        for k in ('center','time_s','height_m','density_kg_m3','initial_detection_time_s'):
            if k in h:del h[k]
            h[k]=d[k]
        h.attrs.update(model='shrinking_radius_voltage',Gamma=1.,Cd=2.,material_density_kg_m3=3000.,sigma_s2_m2=1e-8,
                       initial_guess_only='Whipple-Jacchia',time_origin_shift_s=t0,
                       model_limitations='Fixed direction; drag-only shrinking ODE; gravity, wind and receiver-filter calibration not included',**meta)
    return d,np.r_[np.zeros(6),np.log10(radius)]


class Model:
    def __init__(self,lib,folder,d):
        self.lib=lib;self.d=d;self.handle=lib.voltage_load(os.fsencode(folder/'input.h5'))
        if not self.handle:raise RuntimeError('Native physical model load failed')
        self.size=lib.voltage_len(self.handle)
    def residual(self,q):
        q=np.ascontiguousarray(q);r=np.empty(self.size)
        if not self.lib.voltage_residual(self.handle,q.ctypes.data_as(PTR),r.ctypes.data_as(PTR)):
            return np.full(self.size,1e10)
        return r
    def states(self,q):
        q=np.ascontiguousarray(q);s=np.empty((len(self.d['time_s']),4))
        if not self.lib.voltage_states(self.handle,q.ctypes.data_as(PTR),s.ctypes.data_as(PTR)):
            raise ValueError('Physical trajectory integration failed')
        return s
    def prediction(self,q):
        q=np.ascontiguousarray(q);p=np.empty(self.size)
        if not self.lib.voltage_prediction(self.handle,q.ctypes.data_as(PTR),p.ctypes.data_as(PTR)):
            raise ValueError('Physical waveform prediction failed')
        p=p.reshape(self.d['real'].shape+(2,));return p[...,0]+1j*p[...,1]
    def paths(self,q,s):
        d=self.d;v=d['center'][3:]+1000*q[3:6];u=v/np.linalg.norm(v)
        nominal=d['center'][:3]+1000*q[:3]+s[:,0,None]*u;vel=s[:,1,None]*u
        f=d['frames']
        def frame(x,v):
            return (f[:,:3]+np.einsum('nij,nj->ni',f[:,6:15].reshape(-1,3,3),x)+np.einsum('nij,nj->ni',f[:,24:33].reshape(-1,3,3),v),
                    f[:,3:6]+np.einsum('nij,nj->ni',f[:,15:24].reshape(-1,3,3),x)+np.einsum('nij,nj->ni',f[:,33:42].reshape(-1,3,3),v))
        x=nominal.copy()
        for _ in range(2):
            p,_=frame(x,vel);x=nominal-vel*np.linalg.norm(p-d['tx'],axis=1)[:,None]/C
        p,v=frame(x,vel);a=p-d['tx'];b=p-d['rx'];la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1)
        fd=-np.sum((a/la[:,None]+b/lb[:,None])*v,axis=1)/d['wavelength']
        return la+lb+C*fd/(d['bandwidth']/d['duration'])


def optimize(model,initial):
    bounds=(np.r_[np.full(3,-20.),np.full(3,-40.),-7.],np.r_[np.full(3,20.),np.full(3,40.),-2.])
    candidates=[]
    for logr in np.unique(np.clip([initial[6],-5.,-4.,-3.],-6.9,-2.1)):
        q=initial.copy();q[6]=logr
        if np.max(np.abs(model.residual(q)))>=1e9:continue
        fit=least_squares(model.residual,q,bounds=bounds,jac='3-point',x_scale='jac',max_nfev=800,ftol=1e-11,xtol=1e-11,gtol=1e-9)
        if np.max(np.abs(model.residual(fit.x)))<1e9:candidates.append(fit)
    if not candidates:raise RuntimeError('No valid physical waveform seed')
    return min(candidates,key=lambda f:f.cost)


def export(folder,model,q,fit,chains=None):
    d=model.d;s=model.states(q);pred=model.prediction(q);raw=d['real']+1j*d['imag']
    beats=[];std=[];rng=np.random.default_rng(20261029)
    for i,(row,reference) in enumerate(zip(raw,pred)):
        phase=np.divide(reference,np.abs(reference),out=np.zeros_like(reference),where=np.abs(reference)>.2*np.max(np.abs(reference)))
        beats.append(beat_peak(row*np.conj(phase),d['fs']))
        coeff=-(d['whitener_real'][i,1:]+1j*d['whitener_imag'][i,1:])
        noise_beats=[beat_peak((reference+colored_noise(rng,len(row),coeff,d['noise_power'][i]))*np.conj(phase),d['fs']) for _ in range(32)]
        std.append(float(np.std(noise_beats,ddof=1)))
    anchor=C*(359.45540369317763e-6+d['gate']/d['fs']);t=d['sample_time_s']-anchor[:,None]/C;k=d['bandwidth']/d['duration']
    reference=np.exp(2j*np.pi*(d['bandwidth']/2*t-k/2*t*t))
    measured=anchor+C*np.array([beat_peak(row,d['fs']) for row in raw*np.conj(reference)])/k
    order=np.argsort(d['time_s']);radius=10**q[6];mass=4*np.pi/3*3000*radius**3
    fields=dict(time_s=d['time_s'],time_ns=d['time_ns'],station=d['station'],seed_apparent_path_m=measured,
                modeled_path_m=model.paths(q,s),residual_beat_hz=beats,conditional_beat_std_hz=std,
                speed_time_s=d['time_s'][order],speed_km_s=s[order,1]/1000)
    with h5py.File(folder/'input.h5') as h:flags={k:h.attrs[k] for k in ('manual_flag_revision','manually_excluded_count')}
    attrs=dict(event_id=folder.name,trajectory_model='Direct shrinking-radius voltage fit',beat_guard_pass=bool(np.all(np.abs(beats)<5*np.maximum(std,1e-6))),
               fit_unix_time=fit['fit_unix_time'],fit_timestamp_source='recorded fit completion',fit_flag_revision=flags['manual_flag_revision'],
               fit_excluded_count=flags['manually_excluded_count'],best_initial_mass_kg=mass,best_initial_radius_m=radius,
               uncertainty_status='Uncertainty pending; four physical-model chains required',mass_status='Direct shrinking-radius mass; posterior sampling pending')
    if chains is not None:
        a=chains.transpose(2,0,1)
        rh=float(np.max(array_stats.rhat(a,chain_axis=1,draw_axis=2)));ess=float(np.min(array_stats.ess(a,chain_axis=1,draw_axis=2)))
        tail=float(np.min(array_stats.ess(a,method='tail',prob=.025,chain_axis=1,draw_axis=2)))
        passed=rh<1.01 and min(ess,tail)>=400
        draws=chains.reshape(-1,7);r=10**draws[:,6];m=4*np.pi/3*3000*r**3
        save(folder/'dynamic_mass_posterior.h5',dict(radius_m=r,mass_kg=m),dict(model='direct shrinking radius',convergence_verified=passed))
        fields.update(initial_radius_m_quantiles=np.quantile(r,[.025,.5,.975]),initial_mass_kg_quantiles=np.quantile(m,[.025,.5,.975]))
        # Deterministic subsampling for pointwise bands only; mass uses every draw.
        paths=[];speeds=[]
        for draw in draws[np.linspace(0,len(draws)-1,min(2048,len(draws)),dtype=int)]:
            state=model.states(draw);paths.append(model.paths(draw,state));speeds.append(state[order,1]/1000)
        fields['path_lower95_m'],fields['path_upper95_m']=np.quantile(paths,[.025,.975],axis=0)
        fields['speed_lower95_km_s'],fields['speed_upper95_km_s']=np.quantile(speeds,[.025,.975],axis=0)
        attrs.update(uncertainty_converged=passed,posterior_max_rhat=rh,posterior_min_bulk_ess=ess,posterior_min_tail_ess=tail,
                     uncertainty_status='95% physical-model interval' if passed else 'Physical-model uncertainty requires further sampling',
                     mass_status='Direct shrinking-radius mass; density 3000 kg/m3, Gamma=1, sigma=1e-8; conditional uncertainty',uncertainty_alpha=.5)
    save(folder/'plot.h5',fields,attrs)
    return attrs


def full_products(args,event,folder,model,q):
    """Evaluate excluded pulses too, without including them in the likelihood."""
    with h5py.File(args.source/event/'input.h5') as h:full={k:h[k][()] for k in h}
    with h5py.File(folder/'input.h5') as h:shift=float(h.attrs['time_origin_shift_s'])
    full['time_s']-=shift
    for k in ('center','height_m','density_kg_m3'):full[k]=model.d[k]
    temp=folder/'full_prediction';temp.mkdir(exist_ok=True)
    save(temp/'input.h5',full,{})
    other=Model(model.lib,temp,full);s=other.states(q)
    v0=full['center'][3:]+1000*q[3:6];u=v0/np.linalg.norm(v0)
    x=full['center'][:3]+1000*q[:3]+s[:,0,None]*u;v=s[:,1,None]*u
    f=full['frames']
    xyz=f[:,:3]+np.einsum('nij,nj->ni',f[:,6:15].reshape(-1,3,3),x)+np.einsum('nij,nj->ni',f[:,24:33].reshape(-1,3,3),v)
    save(folder/'physical_states.h5',dict(time_s=full['time_s'],time_ns=full['time_ns'],station=full['station'],position_ecef_m=xyz,speed_km_s=s[:,1]/1000),dict(model='direct shrinking radius'))
    retained=set(zip(model.d['station'],model.d['time_ns']))
    missing=np.array([i for i,k in enumerate(zip(full['station'],full['time_ns'])) if k not in retained],dtype=int)
    if len(missing):
        prediction=other.prediction(q);paths=other.paths(q,s)
        anchor=C*(359.45540369317763e-6+full['gate']/full['fs']);t=full['sample_time_s']-anchor[:,None]/C;rate=full['bandwidth']/full['duration']
        reference=np.exp(2j*np.pi*(full['bandwidth']/2*t-rate/2*t*t));raw=full['real']+1j*full['imag']
        measured=[];beats=[]
        for i in missing:
            measured.append(anchor[i]+C/rate*beat_peak(raw[i]*np.conj(reference[i]),full['fs']))
            phase=np.divide(prediction[i],abs(prediction[i]),out=np.zeros_like(prediction[i]),where=abs(prediction[i])>.2*max(abs(prediction[i])))
            beats.append(beat_peak(raw[i]*np.conj(phase),full['fs']))
        fields={k:full[k][missing] for k in ('station','time_ns','time_s')}
        fields.update(path_m=measured,modeled_path_m=paths[missing],residual_beat_hz=beats)
        save(folder/'excluded_measurements.h5',fields,dict(used_in_fit=False))
    model.lib.voltage_free(other.handle)


def main():
    p=argparse.ArgumentParser()
    for key in ('source','initial','flags','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--library',required=True);p.add_argument('--event');p.add_argument('--warmup',type=int,default=16000);p.add_argument('--draws',type=int,default=32000)
    p.add_argument('--synthetic',action='store_true');args=p.parse_args();comm=MPI.COMM_WORLD
    lib=load_library(args.library);lib.voltage_states.argtypes=[ct.c_void_p,PTR,PTR];lib.voltage_states.restype=ct.c_int
    events=[args.event] if args.event else sorted(f.parent.name for f in args.source.glob('tri_*/input.h5'))
    for event in events[comm.rank::comm.size]:
        folder=args.output/event;folder.mkdir(parents=True,exist_ok=True)
        try:
            d,initial=prepare(args,event,folder);model=Model(lib,folder,d)
            if args.synthetic:
                initial[6]=-4.;pred=model.prediction(initial)
                lib.voltage_free(model.handle)
                with h5py.File(folder/'input.h5','a') as h:
                    h['real'][:]=pred.real;h['imag'][:]=pred.imag
                d['real']=pred.real;d['imag']=pred.imag;model=Model(lib,folder,d)
            start=initial.copy()
            if args.synthetic:start+=np.array([.002,-.003,.002,.005,-.004,.003,.15])
            fit=optimize(model,start);q=np.ascontiguousarray(fit.x)
            meta=dict(fit_unix_time=time.time(),cost=fit.cost,optimizer_success=fit.success,model='direct shrinking-radius voltage',
                      Gamma=1.,Cd=2.,sigma_s2_m2=1e-8,material_density_kg_m3=3000.,prior='Uniform x0,v0,log10 r0; radius limits 1e-7 to 1e-2 m')
            save(folder/'fit.h5',dict(params=q,initial=initial),meta);export(folder,model,q,meta)
            print('FIT',event,'radius_m',10**q[6],'cost',fit.cost,flush=True)
            if not args.synthetic:
                full_products(args,event,folder,model,q)
                from voltage_path_uncertainty import path_intervals
                with h5py.File(folder/'plot.h5') as h:observed=h['seed_apparent_path_m'][:]
                interval=path_intervals(d,model.prediction(q),observed)
                save(folder/'path_measurement_uncertainty.h5',dict(time_ns=d['time_ns'],station=d['station'],path_measurement_lower95_m=interval[:,0],path_measurement_upper95_m=interval[:,1]),dict(confidence=.95,conditional=True))
            if args.synthetic:
                recovered=abs(q[6]-initial[6])<.005 and np.linalg.norm(q[:3]-initial[:3])<.001 and np.linalg.norm(q[3:6]-initial[3:6])<.001
                save(folder/'validation.h5',dict(truth=initial,recovered=q),dict(passed=recovered,kind='Noiseless waveform recovery; not statistical coverage validation'))
                if not recovered:raise RuntimeError('Synthetic radius/state recovery failed')
                print('VALIDATED noiseless recovery',event,flush=True)
            elif args.draws:
                widths=np.r_[np.full(3,1e-4),np.full(3,1e-3),.01].astype(float);chains=[]
                for i in range(4):
                    target=folder/f'chain_{i}.h5'
                    ok=lib.voltage_sample(model.handle,q.ctypes.data_as(PTR),widths.ctypes.data_as(PTR),args.warmup,args.draws,20261100+i,os.fsencode(str(target)+'.partial'))
                    if not ok:raise RuntimeError('Physical-model MCMC failed')
                    os.replace(str(target)+'.partial',target)
                    with h5py.File(target) as h:chains.append(h['samples'][:])
                attrs=export(folder,model,q,meta,np.stack(chains));save(folder/'complete.h5',{},attrs)
            lib.voltage_free(model.handle)
            print('DONE',event,flush=True)
        except Exception:
            error=traceback.format_exc();(folder/'error.log').write_text(error);print('ERROR',event,error,flush=True)


if __name__=='__main__':main()
