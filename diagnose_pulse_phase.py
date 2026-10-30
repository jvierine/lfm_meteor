"""Isolated best-fit-guided pulse phase diagnostic; never modifies catalogue fits."""
import argparse
import ctypes as ct
from pathlib import Path
import subprocess
import h5py
import numpy as np
from scipy.signal import lfilter


def unwrap_guided(observed, predicted):
    residual = np.unwrap(np.angle(np.exp(1j*(observed-predicted))))
    return predicted+residual, residual


def run(folder, library, output):
    from fit_shrinking_voltage import Model, PTR, C
    from run_native_voltage_catalog import load_library
    lib = load_library(library)
    lib.voltage_states.argtypes = [ct.c_void_p, PTR, PTR]
    lib.voltage_states.restype = ct.c_int
    with h5py.File(folder/'input.h5') as h:
        d = {k:h[k][()] for k in h}
    with h5py.File(folder/'fit.h5') as h:
        q = h['params'][:]
    if len(q) != 7:
        raise ValueError('Requires direct shrinking-radius fit')
    model = Model(lib, folder, d)
    try:
        s = model.states(q)
        prediction = model.prediction(q)
    finally:
        lib.voltage_free(model.handle)
    u = d['center'][3:]+1000*q[3:6]
    u /= np.linalg.norm(u)
    nominal = d['center'][:3]+1000*q[:3]+s[:,0,None]*u
    velocity = s[:,1,None]*u
    f = d['frames']
    def frame(x):
        return (f[:,:3]+np.einsum('nij,nj->ni',f[:,6:15].reshape(-1,3,3),x)+np.einsum('nij,nj->ni',f[:,24:33].reshape(-1,3,3),velocity),
                f[:,3:6]+np.einsum('nij,nj->ni',f[:,15:24].reshape(-1,3,3),x)+np.einsum('nij,nj->ni',f[:,33:42].reshape(-1,3,3),velocity))
    x = nominal.copy()
    for _ in range(2):
        p,_ = frame(x)
        x = nominal-velocity*np.linalg.norm(p-d['tx'],axis=1)[:,None]/C
    p,v = frame(x)
    a,b = p-d['tx'],p-d['rx']
    la,lb = np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1)
    path = la+lb
    fd = -np.sum((a/la[:,None]+b/lb[:,None])*v,axis=1)/d['wavelength']
    tau = d['sample_time_s']-path[:,None]/C
    envelope = np.clip((tau+.5/d['fs'])*d['fs'],0,1)*np.clip((d['duration']-tau+.5/d['fs'])*d['fs'],0,1)
    template = envelope*np.exp(2j*np.pi*(d['bandwidth']/2*tau-d['bandwidth']/(2*d['duration'])*tau**2+fd[:,None]*d['sample_time_s']))
    amplitudes=[];phase_std=[];parity=[]
    for i in range(len(path)):
        filt = d['whitener_real'][i]+1j*d['whitener_imag'][i]
        m = lfilter(filt,[1],template[i])[len(filt)-1:]
        y = lfilter(filt,[1],d['real'][i]+1j*d['imag'][i])[len(filt)-1:]
        norm = np.vdot(m,m).real
        amp = np.vdot(m,y)/norm
        amplitudes.append(amp)
        phase_std.append(np.sqrt(d['noise_power'][i]/(2*norm*abs(amp)**2)))
        parity.append(np.max(abs(amp*template[i]-prediction[i]))/max(np.max(abs(prediction[i])),1e-30))
    if max(parity)>1e-6:
        raise ValueError(f'Native waveform parity failed: {max(parity)}')
    output.mkdir(parents=True,exist_ok=True)
    with h5py.File(output/(folder.name+'.h5'),'w') as h:
        h.attrs.update(source_program=__file__,git_hash=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            diagnostic_only=True,native_waveform_relative_error=max(parity),
            caveat='Best-fit-guided branch selection is not independent validation; unknown TX phase, oscillator drift, scattering changes and gaps may invalidate coherence. Phase errors are conditional thermal errors only.')
        for station in np.unique(d['station']):
            idx=np.flatnonzero(d['station']==station)
            idx=idx[np.argsort(d['time_ns'][idx])]
            t=(d['time_ns'][idx]-d['time_ns'][idx][0])*1e-9
            predicted=-2*np.pi*(path[idx]-path[idx][0])/d['wavelength']
            observed=np.angle(np.asarray(amplitudes)[idx])
            unwrapped,residual=unwrap_guided(observed,predicted)
            increment=np.angle(np.exp(1j*np.diff(residual)))
            dt=np.diff(t)
            resultant=abs(np.mean(np.exp(1j*increment)))
            g=h.create_group(str(int(station)))
            for key,value in dict(time_s=t,phase_observed_rad=observed,phase_predicted_rad=predicted,phase_unwrapped_rad=unwrapped,residual_phase_rad=residual,
                residual_increment_rad=increment,pair_dt_s=dt,pair_path_rate_correction_m_s=-d['wavelength']*increment/(2*np.pi*dt),phase_thermal_std_rad=np.asarray(phase_std)[idx]).items():g[key]=value
            g.attrs['increment_resultant']=resultant
            print(folder.name, int(station), 'pulses',len(idx),'increment resultant',round(resultant,3),'phase increment RMS rad',round(float(np.sqrt(np.mean(increment**2))),3),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--events',nargs='+',default=['0012','0110','0162'])
    args=p.parse_args()
    for event in args.events:
        folders=list(args.root.glob('tri_'+event+'_*'))
        if len(folders)!=1:raise ValueError(f'Expected one event for {event}')
        run(folders[0],args.library,args.output)
