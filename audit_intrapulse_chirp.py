"""Test omitted intra-pulse chirp curvature against measured complex echoes."""
import argparse
from pathlib import Path

import h5py
from mpi4py import MPI
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from fit_shrinking_voltage import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    a=p.parse_args();comm=MPI.COMM_WORLD
    jobs=[]
    for folder in sorted(a.audit.glob('tri_*')):
        with h5py.File(folder/'pulse/pulse_audit.h5') as h:
            station=h['station'][:]
            # Strongest echoes are diagnosed without treating them as extra data.
            energy=h['independent_waveform_statistics'][:,0]
            for s in range(3):
                indices=np.flatnonzero(station==s)
                jobs.extend((folder.name,int(i)) for i in indices[np.argsort(energy[indices])[-3:]])
    results=[]
    for event,index in jobs[comm.rank::comm.size]:
        folder=a.source/event
        with h5py.File(folder/'input.h5') as h:
            t=h['sample_time_s'][index];fs=float(h['fs'][()]);bw=float(h['bandwidth'][()]);duration=float(h['duration'][()])
            wavelength=float(h['wavelength'][()]);raw=h['real'][index]+1j*h['imag'][index]
            filt=h['whitener_real'][index]+1j*h['whitener_imag'][index];noise=float(h['noise_power'][index])
            station=int(h['station'][index]);time_s=float(h['time_s'][index])
        with h5py.File(a.audit/event/'pulse/pulse_audit.h5') as h:
            path,beat=h['independent_pulse_fit'][index,:2]
            oldfd=h['trajectory_observables'][index,1]
        tau=t-path/299792458.;rate=bw/duration;fd=beat-rate*path/299792458.
        envelope=np.clip((tau+.5/fs)*fs,0,1)*np.clip((duration-tau+.5/fs)*fs,0,1)
        template=envelope*np.exp(2j*np.pi*(bw/2*tau-rate/2*tau*tau+fd*t))
        half=duration/2;x=(tau-half)/half
        white=lfilter(filt,[1.],raw);valid=np.arange(len(t))>=len(filt)-1
        def evaluate(q,details=False):
            unit=template*np.exp(1j*(q[0]*x+q[1]*x*x))
            filtered=lfilter(filt,[1.],unit)
            amp=np.vdot(filtered[valid],white[valid])/np.vdot(filtered[valid],filtered[valid])
            residual=white-amp*filtered
            value=float(np.sum(abs(residual[valid])**2)/noise)
            return (value,amp,unit,residual) if details else value
        zero=evaluate([0.,0.])
        delay_rate=-oldfd*wavelength/299792458.
        expected_rate=rate*(2*delay_rate-delay_rate**2)
        expected_q=np.pi*expected_rate*half**2
        fit=minimize(evaluate,[0.,expected_q],method='Powell',options=dict(xtol=1e-9,ftol=1e-10,maxiter=200))
        value,amp,unit,residual=evaluate(fit.x,True)
        result=dict(event=event,index=index,station=station,time_s=time_s,
                    expected_curvature_rad=expected_q,fitted_curvature_rad=float(fit.x[1]),
                    fitted_chirp_rate_hz_s=float(fit.x[1]/(np.pi*half**2)),
                    expected_chirp_rate_hz_s=expected_rate,
                    before_energy=zero,after_energy=value,energy_ratio=value/zero,
                    optimizer_success=bool(fit.success))
        target=a.audit/event/'chirp'
        target.mkdir(exist_ok=True)
        original_amp=evaluate([0.,0.],True)[1]
        signal=abs(template)>.9
        # Dechirp only the physical in-pulse samples; never unwrap noise-only rows.
        z=raw[signal]*np.conj(template[signal])/original_amp
        save(target/f'pulse_{index:04d}.h5',
             dict(time_us=(tau[signal]-half)*1e6,dechirped_real=z.real,dechirped_imag=z.imag,
                  phase_rad=np.unwrap(np.angle(z)),amplitude=abs(z),
                  fitted_phase_rad=fit.x[0]*x[signal]+fit.x[1]*x[signal]**2,
                  expected_phase_rad=expected_q*x[signal]**2),result)
        print('CHIRP',result,flush=True);results.append(result)
    gathered=comm.gather(results,root=0)
    if comm.rank==0:
        rows=sum(gathered,[])
        with h5py.File(a.audit/'chirp_summary.h5','w') as h:
            for key in rows[0]:
                vals=[r[key] for r in rows]
                h[key]=np.asarray(vals,dtype=h5py.string_dtype() if key=='event' else None)
            h.attrs.update(diagnostic_only=True,production_products_unchanged=True,
                model='Free residual phase slope and quadratic phase, fixed free-pulse delay; complex amplitudes optimized analytically')


if __name__=='__main__':main()
