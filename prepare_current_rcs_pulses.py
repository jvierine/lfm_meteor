"""RCS panels (a,b) from the accepted direct physical trajectories only.

Reuse measured SNR by exact UTC pulse ID, but recompute range, beam gain,
height, speed and RCS. Mass-comparison products are deliberately not touched.
"""
import h5py
import numpy as np
from prepare_current_kinematics import OUTPUT, CATALOG, MANIFEST, check_export, sha
from estimate_sanya_rcs_from_snr import sanya_gain_for_positions, rcs_from_sanya_snr, gfit
from noise_model import MeasuredSystemNoise


def main():
    noise_path = OUTPUT.parent/'system_noise.h5'
    noise = MeasuredSystemNoise(str(noise_path))
    arrays = {k:[] for k in ('event_id','time_ns','alt_km','speed_km_s','rcs_m2','rcs_dbsm',
                            'sanya_gain_dbi','sanya_relative_gain_db','sanya_snr_db',
                            'sanya_range_km','sanya_tsys_k')}
    audit = {}
    with h5py.File(check_export()) as h:
        ids = h['event_id'].asstr()[:]
        for event in ids:
            g = h['events/'+event]
            folder = CATALOG/event
            if sha(folder/'fit.h5') != g.attrs['fit_sha256']:
                raise ValueError(f'{event}: changed fit')
            retained = g['retained_time_ns'][:][g['retained_station'][:]==0]
            pulse_times = g['time_ns'][:]
            index = {int(t):i for i,t in enumerate(pulse_times)}
            selected = np.asarray([index[int(t)] for t in retained],dtype=int)
            xyz = g['position_ecef_m'][:][selected]
            speed = g['speed_km_s'][:][selected]
            height = g['height_km'][:][selected]
            beam = folder/'beam_rcs.h5'
            with h5py.File(beam) as b:
                snr_index = {int(t):i for i,t in enumerate(b['time_ns'][:])}
                snr = b['snr_db'][:][[snr_index[int(t)] for t in retained]]
            relative, dbi, linear = sanya_gain_for_positions(xyz)
            distance = np.linalg.norm(xyz-gfit.LINK_TX_POSITIONS_M[0],axis=1)
            tsys = noise.tsys_k('Sanya',retained)
            rcs = rcs_from_sanya_snr(snr,distance,linear,tsys)
            keep = (relative>=-3)&np.isfinite(rcs)&(rcs>0)&np.isfinite(tsys)
            values = dict(event_id=np.asarray([event]*len(retained)),time_ns=retained,
                          alt_km=height,speed_km_s=speed,rcs_m2=rcs,rcs_dbsm=10*np.log10(rcs),
                          sanya_gain_dbi=dbi,sanya_relative_gain_db=relative,sanya_snr_db=snr,
                          sanya_range_km=distance/1000,sanya_tsys_k=tsys)
            for key,value in values.items():
                arrays[key].extend(value[keep])
            audit[event] = dict(fit_sha256=g.attrs['fit_sha256'],beam_measurement_sha256=sha(beam),
                                retained_sanya_pulses=len(retained),in_beam_pulses=int(np.sum(keep)))
    target = OUTPUT.parent/'rcs_pulses.h5'
    with h5py.File(target,'w') as h:
        h.attrs.update(source_program=__file__,source_sha256=sha(OUTPUT),manifest_sha256=sha(MANIFEST),
                       system_noise_sha256=sha(noise_path),model='direct shrinking radius',
                       selection='retained Sanya pulses of all 232 accepted events; one-way gain >= -3 dB; no mass cut',
                       calibration='measured noise temperature, 1/pulse-duration bandwidth, gain and range recomputed from physical states')
        h['accepted_event_id'] = np.asarray(ids,dtype=h5py.string_dtype())
        p = h.create_group('pulses')
        for key, values in arrays.items():
            p[key] = np.asarray(values,dtype=h5py.string_dtype() if key=='event_id' else np.int64 if key=='time_ns' else float)
        for event,attrs in audit.items():
            h.create_group('source_audit/'+event).attrs.update(attrs)
    print(target,len(arrays['time_ns']),np.quantile(arrays['rcs_dbsm'],[.25,.5,.75]),flush=True)


if __name__=='__main__':
    main()
