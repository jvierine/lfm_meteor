"""Build manuscript products from a single direct physical posterior catalogue."""
import argparse
import hashlib
import importlib
import os
from pathlib import Path
import shutil
import time

import h5py
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(root):
    import prepare_current_kinematics as k
    k.CATALOG = root/'catalog'
    k.MANIFEST = root/'products/catalogue.h5'
    k.FLAGS = root/'flags_snapshot'
    k.OUTPUT = root/'products/kinematics.h5'
    k.AREA = root/'products/beam_area.h5'
    return k


def manifest(root):
    target = root/'products/catalogue.h5'
    target.parent.mkdir(exist_ok=True)
    ids, omitted = [], []
    for folder in sorted((root/'catalog').glob('tri_*')):
        flag = root/'flags_snapshot'/(folder.name+'.h5')
        bad, revision = False, 0
        if flag.exists():
            with h5py.File(flag) as h:
                bad, revision = bool(h.attrs.get('bad',False)), int(h.attrs['version'])
        if bad:
            omitted.append((folder.name,'review-rejected trajectory'))
            continue
        with h5py.File(folder/'plot.h5') as h:
            assert h.attrs['fit_flag_revision'] == revision
        ids.append(folder.name)
    with h5py.File(target,'w') as h:
        h['event_id'] = np.asarray(ids,dtype=h5py.string_dtype())
        h['excluded_event_id'] = np.asarray([x[0] for x in omitted],dtype=h5py.string_dtype())
        h['exclusion_reason'] = np.asarray([x[1] for x in omitted],dtype=h5py.string_dtype())
        h.attrs.update(model='direct shrinking radius', mass_quality_cut=False,
                       created_unix=time.time(), uncertainty_method='independent defensive importance posterior')
        for event in ids:
            folder=root/'catalog'/event
            with h5py.File(folder/'plot.h5') as p:
                h.create_group('source_audit/'+event).attrs.update(
                    fit_sha256=sha(folder/'fit.h5'), input_sha256=sha(folder/'input.h5'),
                    current_plot=str(folder/'plot.h5'), flag_revision=p.attrs['fit_flag_revision'])
    return ids


def kinematics(root, library, ancillary):
    ids = manifest(root)
    k = configure(root)
    os.environ['TRISTATIC_LIBRARY'] = str(library)
    k.main()
    for source, name in [(ancillary/'monostatic_measurements.h5','monostatic_measurements.h5'),
                         (ancillary/'system_noise.h5','system_noise.h5')]:
        shutil.copy2(source,root/'products'/name)
    import plot_height_histogram as height_science
    height_science.INPUT_CATALOG_DIR = k.OUTPUT
    height = importlib.import_module('prepare_current_height_data')
    height.main()
    radiants = importlib.import_module('prepare_native_radiant_data')
    radiants.main()
    pulses = importlib.import_module('prepare_current_rcs_pulses')
    pulses.main()
    area_uncertainty = importlib.import_module('estimate_collecting_area_uncertainty')
    area_uncertainty.SOURCE = k.AREA
    area_uncertainty.OUTPUT = root/'products/area_uncertainty.h5'
    area_uncertainty.main()
    print('KINEMATICS',len(ids),flush=True)


def masses(root):
    with h5py.File(root/'products/catalogue.h5') as h:
        ids = list(h['event_id'].asstr()[:])
    dynamic, rcs, omissions = {}, {}, []
    for event in ids:
        folder = root/'catalog'/event
        with h5py.File(folder/'plot.h5') as p:
            if p.attrs.get('dynamic_mass_included',False):
                with h5py.File(folder/'dynamic_mass_posterior.h5') as d:
                    dynamic[event] = (float(p.attrs['best_initial_mass_kg']),
                        d['mass_kg'][:],p['initial_mass_kg_quantiles'][:],
                        float(p['speed_km_s'][0]),sha(folder/'dynamic_mass_posterior.h5'))
            else:
                omissions.append(('dynamic',event,str(p.attrs['dynamic_mass_exclusion_reason'])))
        try:
            with h5py.File(folder/'rcs_mass.h5') as r:
                flag = root/'flags_snapshot'/(event+'.h5')
                for key,path in [('fit_sha256',folder/'fit.h5'),('beam_sha256',folder/'beam_rcs.h5')]:
                    assert r.attrs[key] == sha(path), 'stale RCS product'
                expected = hashlib.sha256(flag.read_bytes() if flag.exists() else b'').hexdigest()
                assert r.attrs['flag_sha256'] == expected, 'stale RCS flags'
                m,lo,hi = [float(r.attrs[key]) for key in ('mass_kg','lo95_scatter_kg','hi95_scatter_kg')]
                draws = r['mass_scatter_samples_kg'][:]
                assert np.all(np.isfinite(draws)) and np.all(draws>0)
                assert min(m,lo,hi)>0 and np.isfinite([m,lo,hi]).all()
                rcs[event] = (m,draws,np.array([lo,np.median(draws),hi]),
                              float(r.attrs['uncertainty_clamped_fraction']),sha(folder/'rcs_mass.h5'))
        except (OSError,KeyError,ValueError,AssertionError) as exc:
            omissions.append(('rcs',event,str(exc)))
    dyn_ids = [e for e in ids if e in dynamic]
    paired = [e for e in dyn_ids if e in rcs]
    string = h5py.string_dtype()
    output = root/'products'
    with h5py.File(output/'mass_distribution.h5','w') as h:
        h['event_id'] = np.asarray(dyn_ids,dtype=string)
        for key,values in [('initial_mass_kg',[dynamic[e][0] for e in dyn_ids]),
                           ('initial_speed_km_s',[dynamic[e][3] for e in dyn_ids]),
                           ('mass_95_lo_kg',[dynamic[e][2][0] for e in dyn_ids]),
                           ('mass_95_hi_kg',[dynamic[e][2][2] for e in dyn_ids])]:
            h[key] = values
        h['selected'] = np.ones(len(dyn_ids),bool)
        h['rcs_event_id'] = np.asarray(paired,dtype=string)
        for key,values in [('rcs_mass_kg',[rcs[e][0] for e in paired]),
                           ('rcs_speed_km_s',[dynamic[e][3] for e in paired]),
                           ('rcs_lo95_kg',[rcs[e][2][0] for e in paired]),
                           ('rcs_hi95_kg',[rcs[e][2][2] for e in paired])]:
            h[key] = values
        h.attrs.update(density_kg_m3=3000.,bin_width_dex=.48,
                       selection='Verified importance posterior, finite two-sided interval narrower than one decade, no radius-prior boundary contact; reviewed trajectories only',
                       mass_estimator='Best fit and conditional posterior 95 percent interval',
                       rcs_meaning='Observed in-beam mass loss, matched dynamic-selected events')
    with h5py.File(output/'rcs_pulses.h5') as pulses, h5py.File(output/'rcs_figure.h5','w') as h:
        pulses.copy('pulses',h)
        g = h.create_group('mass_comparison')
        g['event_id'] = np.asarray(paired,dtype=string)
        for key,values in [('dynamic_mass_kg',[dynamic[e][0] for e in paired]),
                          ('dynamic_lo95_kg',[dynamic[e][2][0] for e in paired]),
                          ('dynamic_hi95_kg',[dynamic[e][2][2] for e in paired]),
                          ('rcs_mass_kg',[rcs[e][0] for e in paired]),
                          ('delta_mass_lo95_noise_only_kg',[rcs[e][2][0] for e in paired]),
                          ('delta_mass_hi95_noise_only_kg',[rcs[e][2][2] for e in paired])]:
            g[key] = values
        g.attrs['sample_count'] = len(paired)
    with h5py.File(output/'radiants.h5') as radiant,h5py.File(output/'beam_area.h5') as beam, \
         h5py.File(output/'altitude_correction.h5') as altitude:
        np.testing.assert_array_equal(ids,radiant['event_id'].asstr()[:])
        exposures = dict(zip(ids,radiant['family_effective_exposure_sum_h'][:]))
        speeds = dict(zip(ids,radiant['speed_km_s'][:]))
        area = float(beam.attrs['area_horizontal_km2'])
        factor = float(altitude.attrs['altitude_sampling_factor'])
    # Keep the established survey normalization; replace all mass draws and
    # recompute every exposure, speed, area and altitude factor above.
    from refresh_submission_mass_influx import summarize
    rng = np.random.default_rng(20261108)
    usable = {e for e in ids if np.isfinite(exposures[e]) and exposures[e]>0
              and np.isfinite(speeds[e]) and speeds[e]>0}
    with h5py.File(output/'current_mass_influx.h5','w') as h:
        h.attrs.update(n_catalogue=len(ids),collecting_area_km2=area,altitude_sampling_factor=factor,
                       uncertainty='Conditional event mass posterior or RCS RMS draws and independent Poisson counts; fixed selection and weights; physical systematics excluded',
                       created_unix=time.time(), no_mass_imputation=True)
        h['catalogue_event_id'] = np.asarray(ids,dtype=string)
        for kind, rows in [('dynamic',dynamic),('rcs',rcs)]:
            converted = {e:(v[0],v[1], 'independent defensive importance posterior' if kind=='dynamic' else 'event RCS dBsm RMS shared-offset draws',
                            0. if kind=='dynamic' else v[3],v[4]) for e,v in rows.items()}
            for matched in (False,True):
                names = [e for e in ids if e in rows and e in usable and (not matched or e in paired)]
                g = h.create_group(kind+('_matched' if matched else ''))
                summarize(g,names,converted,np.array([exposures[e] for e in names]),
                          np.array([speeds[e] for e in names]),area,rng,factor)
        h['omitted_mass_kind'] = np.asarray([x[0] for x in omissions],dtype=string)
        h['omitted_mass_event'] = np.asarray([x[1] for x in omissions],dtype=string)
        h['mass_omission_reason'] = np.asarray([x[2] for x in omissions],dtype=string)
    # The GUI reads this exact selection, including source hashes, rather than
    # inferring inclusion from whether a plotted interval happens to exist.
    with h5py.File(output/'current_mass_influx.h5') as influx, h5py.File(root/'mass_selection.h5','w') as h:
        h.attrs.update(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                       source_sha256=sha(output/'current_mass_influx.h5'))
        for kind in ('dynamic','rcs'):
            g = h.create_group(kind)
            g['event_id'] = influx[kind+'/event_id'][:]
            g['source_sha256'] = influx[kind+'/mass_source_sha256'][:]
        for key, index in [('omitted_kind',0),('omitted_event',1),('omission_reason',2)]:
            h[key] = np.asarray([x[index] for x in omissions],dtype=string)
    print('MASS',len(dynamic),'RCS',len(rcs),'PAIRED',len(paired),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--ancillary',type=Path,required=True)
    p.add_argument('--stage',choices=['kinematics','mass','all'],default='all')
    a = p.parse_args()
    if a.stage in ('kinematics','all'):
        kinematics(a.root,a.library,a.ancillary)
    if a.stage in ('mass','all'):
        masses(a.root)


if __name__ == '__main__':
    main()
