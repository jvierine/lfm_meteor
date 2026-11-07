"""Publish remote-source mass selection and panel c without touching panels a/b."""
from pathlib import Path
from collections import Counter
import os
import shutil
import subprocess
import h5py
import numpy as np
from astropy.constants import L_sun, GM_sun, c
from finalize_submission_mass_figures import pr_years
import estimate_current_mass_influx as masses

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/submission_mass_remote_20261107'
SNAP=OUT/'snapshot'
CAT=SNAP/'results/tristatic_cached_shrinking_20261029'
FLAGS=SNAP/'results/voltage_review_flags'
ORBITS=ROOT/'results/tristatic_shrinking_orbits_current_20261107/orbits.h5'


def main():
    with h5py.File(OUT/'remote_hash_manifest.h5') as manifest:
        assert manifest.attrs['all_match']
        for p,d in zip(manifest['relative_path'].asstr()[:],manifest['local_sha256'].asstr()[:]):
            assert masses.digest(SNAP/p)==d,p
        catalogue=manifest['accepted_event_id'].asstr()[:]
    masses.ROOT=SNAP;masses.CAT=CAT;masses.FLAGS=FLAGS
    rows,omitted=masses.collect()
    names=sorted(rows['dynamic']);paired=sorted(set(names)&set(rows['rcs']))
    print('counts',len(names),len(rows['rcs']),len(paired),flush=True)
    mass=[];lo=[];hi=[];speed=[];methods=[];hashes=[]
    event_rms=[];event_speed=[];event_speed_hw=[];residuals=[];nobs=[]
    verified_bootstrap=0
    for n in catalogue:
        with h5py.File(CAT/n/'plot.h5') as p:
            assert 'shrinking' in p.attrs['trajectory_model'].lower()
            method=str(p.attrs.get('uncertainty_method','conditional'))
            if 'bootstrap' in method:
                with h5py.File(CAT/n/'block_bootstrap_states.h5') as b,h5py.File(CAT/n/'dynamic_mass_posterior.h5') as d:
                    np.testing.assert_allclose(4*np.pi/3*3000*10**(3*b['params'][:,6]),d['mass_kg'][:],rtol=1e-12)
                    assert method==d.attrs['uncertainty_method']==b.attrs['uncertainty_method']
                    np.testing.assert_allclose(np.quantile(d['mass_kg'][:],[.025,.5,.975]),p['initial_mass_kg_quantiles'][:],rtol=1e-12)
                verified_bootstrap+=1
            elif method=='conditional' and (CAT/n/'dynamic_mass_posterior.h5').exists():
                with h5py.File(CAT/n/'dynamic_mass_posterior.h5') as d:
                    assert d.attrs.get('model')=='direct shrinking radius',n
            flag=FLAGS/(n+'.h5');rev=0;excluded=set()
            if flag.exists():
                with h5py.File(flag) as f:
                    rev=int(f.attrs['version']);assert not f.attrs.get('bad',False)
                    excluded=set(f['excluded'].asstr()[:])
            assert int(p.attrs['fit_flag_revision'])==rev
            residual=p['modeled_path_m'][:]-p['seed_apparent_path_m'][:]
            keep=np.array([f'{int(s)}:{int(t)}' not in excluded for s,t in zip(p['station'][:],p['time_ns'][:])])&np.isfinite(residual)
            residuals.extend(residual[keep]);nobs.append(keep.sum());event_rms.append(np.sqrt(np.mean(residual[keep]**2)))
            _,indices=np.unique(p['speed_time_s'][:],return_index=True)
            event_speed.append(p['speed_km_s'][indices[0]])
            event_speed_hw.append(np.median(500*(p['speed_upper95_km_s'][:]-p['speed_lower95_km_s'][:])[indices]))
            if n in rows['dynamic']:
                # Catalogue and selected lists are both sorted by the same ID.
                mass.append(rows['dynamic'][n][0]);lo.append(p['initial_mass_kg_quantiles'][0]);hi.append(p['initial_mass_kg_quantiles'][2]);speed.append(p['speed_km_s'][indices[0]])
                methods.append(method);hashes.append(masses.digest(CAT/n/'plot.h5'))
    assert list(catalogue)==sorted(catalogue)
    print('bootstrap mass draws verified',verified_bootstrap,flush=True)
    with h5py.File(OUT/'joint_fft_mass_distribution.h5','w') as f:
        f.attrs.update(bin_width_dex=.48,density_kg_m3=3000.,quality_selected=True,
            selection='Current remote review/uncertainty-audit/draw-consistency cuts; finite positive95% mass width<1dex; no radius boundary; direct shrinking-radius only',
            remote_manifest_sha256=masses.digest(OUT/'remote_hash_manifest.h5'),source_program=str(Path(__file__).resolve()))
        for key,val in [('event_id',names),('rcs_event_id',paired),('uncertainty_method',methods),('plot_sha256',hashes)]:
            f[key]=np.asarray(val,dtype=h5py.string_dtype())
        for key,val in [('initial_mass_kg',mass),('mass_95_lo_kg',lo),('mass_95_hi_kg',hi),('initial_speed_km_s',speed),('selected',np.ones(len(names),bool)),
                        ('rcs_mass_kg',[rows['rcs'][n][0] for n in paired]),('rcs_speed_km_s',[speed[names.index(n)] for n in paired])]:
            f[key]=val
        for key,attr in [('rcs_lo95_kg','lo95_scatter_kg'),('rcs_hi95_kg','hi95_scatter_kg')]:
            values=[]
            for n in paired:
                with h5py.File(CAT/n/'rcs_mass.h5') as r:values.append(r.attrs[attr])
            f[key]=values
    with h5py.File(OUT/'joint_fft_mass_distribution.h5') as f,h5py.File(OUT/'rcs_mass_comparison.h5','w') as h:
        g=h.create_group('mass_comparison');g['event_id']=f['rcs_event_id'][:]
        indices=[names.index(n) for n in paired]
        for key,src in [('dynamic_mass_kg','initial_mass_kg'),('dynamic_lo95_kg','mass_95_lo_kg'),('dynamic_hi95_kg','mass_95_hi_kg')]:g[key]=f[src][:][indices]
        for key,src in [('rcs_mass_kg','rcs_mass_kg'),('delta_mass_lo95_noise_only_kg','rcs_lo95_kg'),('delta_mass_hi95_noise_only_kg','rcs_hi95_kg')]:g[key]=f[src][:]
        g.attrs.update(sample_count=len(paired),selection_source=str(OUT/'joint_fft_mass_distribution.h5'),selection_sha256=masses.digest(OUT/'joint_fft_mass_distribution.h5'))
    with h5py.File(ORBITS) as o,h5py.File(OUT/'mass_size_pr_precision.h5','w') as h:
        np.testing.assert_array_equal(catalogue,o['event_id'].asstr()[:])
        indices=[list(catalogue).index(n) for n in names]
        a=o['a_au'][:][indices];e=o['e'][:][indices]
        assert np.all(o['valid'][:][indices]&(a>0)&(e>=0)&(e<1))
        radius=(3*np.array(mass)/(4*np.pi*3000))**(1/3)
        tau=pr_years(radius[:,None],a,e)
        beta=3*L_sun.value/(16*np.pi*GM_sun.value*c.value*3000*radius)
        for key,val in [('event_id',names),('catalogue_event_id',catalogue),('paired_event_id',paired)]:h[key]=np.asarray(val,dtype=h5py.string_dtype())
        for key,val in [('diameter_um',2e6*radius),('beta',beta),('pr_years_orbit_draws_fixed_mass',tau),('pr_years_event_median',np.median(tau,axis=1)),
                        ('event_initial_speed_km_s',event_speed),('event_apparent_path_rms_m',event_rms),('event_speed_halfwidth_m_s',event_speed_hw),('event_measurement_count',nobs)]:h[key]=val
        for key,val in [('diameter_um',2e6*radius),('beta',beta),('pr_years',np.median(tau,axis=1))]:
            h[key+'_min_q25_median_q75_max']=np.quantile(val,[0,.25,.5,.75,1]);print(key,h[key+'_min_q25_median_q75_max'][:],flush=True)
        h['speed_min_median_max_km_s']=np.quantile(event_speed,[0,.5,1])
        h['event_rms_quartiles_m']=np.quantile(event_rms,[.25,.5,.75])
        h['event_speed_halfwidth_quartiles_m_s']=np.quantile(event_speed_hw,[.25,.5,.75])
        h.attrs.update(orbit_source=str(ORBITS),orbit_sha256=masses.digest(ORBITS),orbit_epoch=o.attrs['orbit_epoch'],
                       remote_manifest_sha256=masses.digest(OUT/'remote_hash_manifest.h5'),source_program=Path(__file__).read_text(),
                       n_dynamic=len(names),n_rcs=len(rows['rcs']),n_paired=len(paired),n_catalogue=232,n_measurements=len(residuals),
                       pooled_apparent_path_rms_m=np.sqrt(np.mean(np.array(residuals)**2)),
                       precision_definition='Apparent chirp-coupled path residual, not3D accuracy. Speed half-widths are reported95% intervals, not validated total error.',
                       pr_definition='Median over100 bound orbit realizations per event at fixed best-fit first-detection radius, then equal-event quantiles; not joint mass-orbit uncertainty')
        print('precision',h.attrs['pooled_apparent_path_rms_m'],h['speed_min_median_max_km_s'][:],h['event_rms_quartiles_m'][:],h['event_speed_halfwidth_quartiles_m_s'][:],flush=True)
    binary=ROOT.parent/'rustplotlib/target/release/native-paper-figures'
    subprocess.run([str(binary),'mass',str(OUT/'joint_fft_mass_distribution.h5'),str(OUT/'joint_fft_mass_distribution')],check=True,env=dict(os.environ,DYLD_LIBRARY_PATH='/opt/anaconda3/lib'))
    paper=ROOT.parent/'sanya_tristatic_paper/figures/rustplotlib'
    for ext in ['pdf','png','svg','include.tex']:shutil.copy2(OUT/('joint_fft_mass_distribution.'+ext),paper/('joint_fft_mass_distribution.'+ext))
    print('panel c released',masses.digest(OUT/'rcs_mass_comparison.h5'),flush=True)


if __name__=='__main__':main()
