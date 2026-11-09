"""Current 232-ID influx from refreshed non-WJ area and radiant exposures.

No historical area/exposures or altitude normalization are read. The existing
mass collector is used only for its event-quality and saved-draw checks.
"""
from pathlib import Path
from datetime import datetime, timezone
import h5py
import numpy as np
import estimate_current_mass_influx as masses
from radiant_symmetry import daily_weight_from_summed_hours

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'results/current_kinematics_20261107'
OUT=ROOT/'results/submission_mass_remote_20261107/current_mass_influx.h5'
SNAP=OUT.parent/'snapshot'
ORBITS=ROOT/'results/tristatic_shrinking_orbits_current_20261107/orbits.h5'


def summarize(g,names,rows,exposure,speed,area,rng,altitude_factor):
    base=4*np.pi*(6371+95)**2/area*daily_weight_from_summed_hours(exposure)
    velocity=(72/speed)**3
    mass=np.array([rows[n][0] for n in names])
    draws=np.array([rng.choice(rows[n][1],20000) for n in names])
    counts=rng.poisson(1,draws.shape)
    g.attrs.update(n_events=len(names),sum_mass_kg=mass.sum(),mass_min_kg=mass.min(),mass_max_kg=mass.max(),
                   n_events_with_clamped_rcs_draws=sum(rows[n][3]>0 for n in names))
    for key,val in [('event_id',names),('mass_source_sha256',[rows[n][4] for n in names]),
                    ('uncertainty_method',[rows[n][2] for n in names])]:
        g[key]=np.asarray(val,dtype=h5py.string_dtype())
    for key,val in [('mass_kg',mass),('effective_hours',exposure),('speed_km_s',speed),('velocity_factor',velocity)]:
        g[key]=val
    for n in names:
        g.create_dataset('input_mass_draws/'+n,data=rows[n][1],compression='gzip')
    for name,weight in [('area_exposure',base),('plus_velocity_heuristic',base*velocity),
                        ('plus_altitude_heuristic',base*velocity*altitude_factor)]:
        h=g.create_group(name)
        contribution=mass*weight/1000
        sampled=draws*weight[:,None]/1000
        combined=(sampled*counts).sum(axis=0)
        top=np.argsort(contribution)[-2:]
        keep=np.ones(len(names),bool);keep[top]=False
        h.attrs.update(rate_ton_day=contribution.sum(),top_two_fraction=contribution[top].sum()/contribution.sum(),
                       omit_two_rate_ton_day=contribution[keep].sum(),weighted_poisson_sigma_ton_day=np.linalg.norm(contribution))
        for key,val in [('weight_per_day',weight),('contribution_ton_day',contribution),
                        ('mass_only_draws_ton_day',sampled.sum(axis=0)),('combined_draws_ton_day',combined),
                        ('mass_only_interval95_ton_day',np.quantile(sampled.sum(axis=0),[.025,.975])),
                        ('combined_interval95_ton_day',np.quantile(combined,[.025,.975])),
                        ('omit_two_interval95_ton_day',np.quantile((sampled[keep]*counts[keep]).sum(axis=0),[.025,.975]))]:
            h[key]=val
        h['top_two_event_id']=np.asarray(np.array(names)[top],dtype=h5py.string_dtype())
        print(g.name,name,dict(h.attrs),'95%',h['combined_interval95_ton_day'][:])


def main():
    masses.ROOT=SNAP
    masses.CAT=SNAP/'results/tristatic_cached_shrinking_20261029'
    masses.FLAGS=SNAP/'results/voltage_review_flags'
    with h5py.File(OUT.parent/'remote_hash_manifest.h5') as h:
        assert h.attrs['all_match']
        for path,digest in zip(h['relative_path'].asstr()[:],h['local_sha256'].asstr()[:]):
            assert masses.digest(SNAP/path)==digest,path
    rows,omitted=masses.collect()
    paths=[INPUT/'kinematics.h5',INPUT/'beam_area.h5',INPUT/'radiants.h5',ORBITS,
           INPUT/'altitude_correction.h5',INPUT/'monostatic_measurements.h5',OUT.parent/'remote_hash_manifest.h5']
    hashes={str(p):masses.digest(p) for p in paths}
    with h5py.File(paths[0]) as kin,h5py.File(paths[1]) as beam,h5py.File(paths[2]) as radiant,h5py.File(ORBITS) as orbit:
        names=radiant['event_id'].asstr()[:]
        assert len(names)==len(set(names))==232
        for h in [kin,beam,orbit]:
            np.testing.assert_array_equal(names,h['event_id'].asstr()[:])
        for h in [kin,beam,radiant]:
            assert h.attrs['manifest_sha256']==hashes[str(ORBITS)]
            assert h.attrs['model']=='direct shrinking radius'
        assert beam.attrs['source_sha256']==radiant.attrs['source_sha256']==hashes[str(paths[0])]
        assert radiant.attrs['area_sha256']==hashes[str(paths[1])]
        assert not radiant.attrs['altitude_correction_included']
        assert not radiant.attrs['velocity_correction_included']
        with h5py.File(INPUT/'altitude_correction.h5') as altitude:
            assert altitude.attrs['kinematic_sha256']==hashes[str(paths[0])]
            assert altitude.attrs['monostatic_sha256']==hashes[str(INPUT/'monostatic_measurements.h5')]
            np.testing.assert_array_equal(names,altitude['event_id'].asstr()[:])
            altitude_factor=float(altitude.attrs['altitude_sampling_factor'])
            expected=float(altitude.attrs['monostatic_layer_events'])*float(altitude.attrs['tristatic_common_events'])/float(altitude.attrs['monostatic_common_events'])/len(names)
            np.testing.assert_allclose(altitude_factor,expected,rtol=1e-12)
        exposure=radiant['family_effective_exposure_sum_h'][:]
        np.testing.assert_allclose(exposure,radiant['family_effective_exposure_h'][:].sum(axis=1))
        speed=radiant['speed_km_s'][:]
        area=float(beam.attrs['area_horizontal_km2'])
        assert area>0 and np.isfinite(area)
        good=np.isfinite(exposure)&(exposure>0)&np.isfinite(speed)&(speed>0)
        with h5py.File(OUT,'w') as h:
            h.attrs.update(created_utc=datetime.now(timezone.utc).isoformat(),source_program=Path(__file__).read_text(),
                mass_collector_source=(ROOT/'estimate_current_mass_influx.py').read_text(),
                collecting_area_km2=area,n_catalogue=232,altitude_correction_applied=True,
                altitude_sampling_factor=altitude_factor,
                altitude_correction_status='Separate final stage from fresh monostatic/current232 heights; constant recovery versus altitude heuristic, not calibrated mass selection',
                normalization='Earth area at95km; four-quadrant fold; 4*24/sum effective exposure hours; separate heuristic(72/v)^3',
                uncertainty='20000 replicates; saved event mass draws and independent Poisson(1) counts; fixed selection and weights; not calibrated global confidence intervals',
                caveat='Dynamic first-detection mass and RCS observed in-beam mass loss differ; no pre-detection mass reconstruction or imputation',
                seed=20261030,observation_start_utc=radiant.attrs['observation_start_utc'],
                observation_stop_utc=radiant.attrs['observation_stop_utc'],observed_hours=radiant.attrs['observed_hours'])
            provenance=h.create_group('provenance')
            provenance.attrs.update(hashes)
            h['catalogue_event_id']=np.asarray(names,dtype=h5py.string_dtype())
            h['usable_weight']=good
            rng=np.random.default_rng(20261030)
            for kind in rows:
                for matched in [False,True]:
                    keep=np.array([n in rows[kind] and (not matched or all(n in r for r in rows.values())) for n in names])&good
                    g=h.create_group(kind+('_matched' if matched else ''))
                    summarize(g,names[keep].tolist(),rows[kind],exposure[keep],speed[keep],area,rng,altitude_factor)
                missing=sorted(set(rows[kind])-set(names[good]))
                h[kind+'_omitted_from_weighted_subset']=np.asarray(missing,dtype=h5py.string_dtype())
            for i,key in enumerate(['omitted_mass_kind','omitted_mass_event','mass_omission_reason']):
                h[key]=np.asarray([x[i] for x in omitted],dtype=h5py.string_dtype())
    for path,digest in hashes.items():
        assert masses.digest(Path(path))==digest,'Input changed during calculation: '+path
    print(OUT)


if __name__=='__main__':
    main()
