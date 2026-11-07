"""Current direct-fit height/beam arrays and recovered monostatic measurements."""
import csv
import hashlib
import io
import subprocess

import h5py
import numpy as np
from prepare_current_kinematics import OUTPUT, AREA, MANIFEST, check_export, sha
import plot_height_histogram as science

MONO = OUTPUT.parent/'monostatic_measurements.h5'


def recover_monostatic():
    source = '/home/j/src/lfm_meteor/results/all_detection_time_delay_velocity.csv'
    result = subprocess.run(['ssh','-o','BatchMode=yes','revontuli','cat '+source],
                            check=True,capture_output=True,text=True)
    rows = [row for row in csv.DictReader(io.StringIO(result.stdout)) if row['site']=='sanya']
    if not rows:
        raise ValueError('No monostatic measurements recovered')
    with h5py.File(MONO,'w') as h:
        h.attrs.update(source='revontuli:'+source,source_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                       selection='Sanya detections with event polynomial range RMS <=100 m; fixed-beam heights',
                       source_program=__file__,trajectory_model='monostatic measured range; not WJ')
        h['event_id'] = np.asarray([r['event_id'] for r in rows],dtype=h5py.string_dtype())
        h['time_ns'] = np.asarray([int(r['time_ns']) for r in rows],dtype=np.int64)
        for name in ('height_km','range_km','radial_velocity_km_s','range_poly_fit_rms_m','snr_peak_db'):
            h[name] = np.asarray([float(r[name]) for r in rows])
        if np.max(h['range_poly_fit_rms_m'][:]) > 100.:
            raise ValueError('Unexpected monostatic range-track selection')


def main():
    check_export()
    if not MONO.exists():
        recover_monostatic()
    with h5py.File(MONO) as h:
        mono = h['height_km'][:]
        velocity = h['radial_velocity_km_s'][:]
        names = h['event_id'].asstr()[:]
        event_ids = sorted(set(names))
        mono_medians = np.asarray([np.median(mono[names==n]) for n in event_ids])
        keep = np.isfinite(mono)&np.isfinite(velocity)&(((mono>=80)&(mono<=120))|(velocity<-10))
        mono = mono[keep]
    tri,_ = science.collect_heights()
    heights = np.linspace(80.,120.,401)
    profiles = science.receiver_gain_profiles(heights)
    beam = science.beam_panel_data(OUTPUT)
    with h5py.File(OUTPUT) as h:
        ids = h['event_id'].asstr()[:]
        tri_medians = np.asarray([np.median(h['events/'+n+'/height_km'][:]) for n in ids])
    with h5py.File(AREA) as h:
        cutoff = h.attrs['cutoff_relative_gain_db']
    target = OUTPUT.parent/'height.h5'
    with h5py.File(target,'w') as h:
        h.attrs.update(source=str(OUTPUT),source_sha256=sha(OUTPUT),manifest_sha256=sha(MANIFEST),
                       monostatic_source=str(MONO),monostatic_sha256=sha(MONO),
                       containment_level_db=cutoff,area_sha256=sha(AREA),source_program=__file__)
        h['event_id'] = np.asarray(ids,dtype=h5py.string_dtype())
        h['tristatic_height_km'] = tri
        h['monostatic_height_km'] = mono
        h['gain_height_km'] = heights
        for site, gain in profiles.items():
            h['gain_'+site] = gain
        for key in ('east_edges','north_edges','gain_db','n_events','n_positions'):
            if np.isscalar(beam[key]):
                h.attrs[key] = beam[key]
            else:
                h[key] = beam[key]
        h['counts'] = beam['counts'].filled(np.nan)
        h['gain_grid_deg'] = beam['east_grid'][0]
    mono_layer = int(np.sum((mono_medians>=80)&(mono_medians<=120)))
    mono_common = int(np.sum((mono_medians>=92)&(mono_medians<95)))
    tri_common = int(np.sum((tri_medians>=92)&(tri_medians<95)))
    factor = mono_layer*tri_common/mono_common/len(ids)
    target = OUTPUT.parent/'altitude_correction.h5'
    with h5py.File(target,'w') as h:
        h.attrs.update(monostatic_source=str(MONO),monostatic_sha256=sha(MONO),
                       kinematic_source=str(OUTPUT),kinematic_sha256=sha(OUTPUT),
                       source_program=__file__,method='event median heights; common interval [92,95) km; layer [80,120] km',
                       monostatic_events=len(event_ids),monostatic_layer_events=mono_layer,
                       monostatic_common_events=mono_common,tristatic_events=len(ids),
                       tristatic_common_events=tri_common,altitude_sampling_factor=factor,
                       expected_tristatic_events=mono_layer*tri_common/mono_common,
                       limitation='Heuristic constant event recovery versus altitude; not a calibrated mass-selection correction')
        h['event_id'] = np.asarray(ids,dtype=h5py.string_dtype())
        h['tristatic_event_median_height_km'] = tri_medians
        h['monostatic_event_id'] = np.asarray(event_ids,dtype=h5py.string_dtype())
        h['monostatic_event_median_height_km'] = mono_medians
    print('HEIGHT',len(tri),len(mono),'ALTITUDE',len(event_ids),mono_layer,mono_common,tri_common,factor,flush=True)


if __name__=='__main__':
    main()
