"""Finalize the shared 118/101 mass figures and read-only audited-orbit PR join.

No exposure weights, collecting-area products, or WJ mass fallback are used.
Source provenance is confined to metadata and comments, never figure labels.
"""
from pathlib import Path
from datetime import datetime, timezone
import os
import shutil
import subprocess
import h5py
import numpy as np
from scipy.special import hyp2f1
from astropy.constants import L_sun, GM_sun, c, au
from astropy.units import yr
import estimate_current_mass_influx as masses

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT/'results/submission_mass_audit_20261107'
OUT = ROOT/'results/submission_mass_current_20261107'
ORBITS = ROOT/'results/tristatic_shrinking_orbits_audited_20261107/orbits.h5'
PAPER = ROOT.parent/'sanya_tristatic_paper/figures/rustplotlib'
BIN = ROOT.parent/'rustplotlib/target/release/native-paper-figures'
YEAR_S = yr.to('s')


def pr_years(radius_m, a_au, eccentricity):
    """Analytic evaluation of the manuscript's secular eccentric-orbit integral.

    Substituting y=e*x gives integral=(5/8)*e**(8/5)*
    hyp2f1(4/5,3/2;9/5;e**2), including the finite circular limit.
    Fitzpatrick, Celestial Mechanics, Sec. 10.6, Eqs. 10.183 and 10.185:
    https://farside.ph.utexas.edu/teaching/celestial/Celestial/node95.html
    """
    r, a, e = np.broadcast_arrays(radius_m, a_au, eccentricity)
    if not np.all(np.isfinite(r)&np.isfinite(a)&np.isfinite(e)&(r>0)&(a>0)&(e>=0)&(e<1)):
        raise ValueError('PR lifetime requires finite positive radius and bound elliptic orbit')
    bc = 4/3*3000*r
    circular = np.pi*bc*c.value**2*(a*au.value)**2/L_sun.value
    return circular*(1-e**2)**2*hyp2f1(.8,1.5,1.8,e**2)/YEAR_S


def main():
    OUT.mkdir(exist_ok=True)
    rows, _ = masses.collect()
    with h5py.File(AUDIT/'figure7.h5') as f, h5py.File(ORBITS) as o:
        names = f['event_id'].asstr()[:]
        paired = f['rcs_event_id'].asstr()[:]
        orbit_names = o['event_id'].asstr()[:]
        assert len(set(orbit_names)) == len(orbit_names) == 232
        assert len(names) == 118 and len(paired) == 101
        assert set(names) <= set(orbit_names)
        assert set(names) == set(rows['dynamic']) & set(orbit_names)
        assert set(paired) == set(names) & set(rows['rcs'])
        for i,n in enumerate(names):
            np.testing.assert_equal(f['initial_mass_kg'][i], rows['dynamic'][n][0])
            with h5py.File(masses.CAT/n/'plot.h5') as p:
                assert 'shrinking' in p.attrs['trajectory_model'].lower()
                np.testing.assert_array_equal(f['mass_95_lo_kg'][i],p['initial_mass_kg_quantiles'][0])
                np.testing.assert_array_equal(f['mass_95_hi_kg'][i],p['initial_mass_kg_quantiles'][2])
        for i,n in enumerate(paired):
            with h5py.File(masses.CAT/n/'rcs_mass.h5') as r:
                for key, attr in [('rcs_mass_kg','mass_kg'),('rcs_lo95_kg','lo95_scatter_kg'),('rcs_hi95_kg','hi95_scatter_kg')]:
                    np.testing.assert_equal(f[key][i],r.attrs[attr])
        idx = {n:i for i,n in enumerate(orbit_names)}
        indices = [idx[n] for n in names]
        a = o['a_au'][:][indices]; e = o['e'][:][indices]
        valid = o['valid'][:][indices]
        bound = valid & np.isfinite(a)&np.isfinite(e)&(a>0)&(e>=0)&(e<1)
        assert np.all(bound), 'Report new unbound/missing orbit realizations before changing PR selection'
        mass = f['initial_mass_kg'][:]
        radius = (3*mass/(4*np.pi*3000))**(1/3)
        tau = pr_years(radius[:,None],a,e)
        event_tau = np.median(tau,axis=1)
        with h5py.File(OUT/'mass_size_pr.h5','w') as h:
            h.attrs.update(created_utc=datetime.now(timezone.utc).isoformat(),
                source_program=Path(__file__).read_text(), orbit_source=str(ORBITS),
                orbit_sha256=masses.digest(ORBITS), mass_source=str(AUDIT/'figure7.h5'),
                mass_sha256=masses.digest(AUDIT/'figure7.h5'), orbit_epoch=o.attrs['orbit_epoch'],
                density_kg_m3=3000., radiation_pressure_efficiency=1.,
                n_orbit_catalogue=232, n_dynamic=118, n_paired=101,
                n_independent_rcs_in_catalogue=len(set(rows['rcs'])&set(orbit_names)),
                estimator='Per-event median PR lifetime over 100 saved bound orbit realizations at fixed best-fit first-detection mass; equal event weights',
                caveat='Illustrative gravity-orbit secular PR timescale, not a residence-time prediction or joint mass-orbit uncertainty; first-detection radius is not pre-atmospheric radius',
                solar_luminosity_W=L_sun.value, solar_GM_m3_s2=GM_sun.value,
                au_m=au.value, c_m_s=c.value, year_s=YEAR_S,
                formula_source='https://farside.ph.utexas.edu/teaching/celestial/Celestial/node95.html')
            for key, values in [('event_id',names),('paired_event_id',paired),('catalogue_event_id',orbit_names)]:
                h[key]=np.asarray(values,dtype=h5py.string_dtype())
            for key,values in [('mass_kg',mass),('radius_m',radius),('diameter_um',2e6*radius),
                               ('a_au',a),('e',e),('valid_bound',bound),
                               ('pr_years_orbit_draws_fixed_mass',tau),('pr_years_event_median',event_tau)]:
                h[key]=values
            beta=3*L_sun.value/(16*np.pi*GM_sun.value*c.value*3000*radius)
            h['beta']=beta
            for key,values in [('diameter_um',2e6*radius),('beta',beta),('pr_years',event_tau)]:
                h[key+'_min_q25_median_q75_max']=np.quantile(values,[0,.25,.5,.75,1])
            h['pr_years_at_componentwise_median_orbit']=pr_years(radius,np.median(a,axis=1),np.median(e,axis=1))
            print('diameter:',h['diameter_um_min_q25_median_q75_max'][:])
            print('beta:',h['beta_min_q25_median_q75_max'][:])
            print('PR years:',h['pr_years_min_q25_median_q75_max'][:])
    for old,new in [('figure7.h5','joint_fft_mass_distribution.h5'),('figure8.h5','sanya_rcs_from_snr_article.h5')]:
        target=OUT/new
        shutil.copy2(AUDIT/old,target)
        with h5py.File(target,'a') as h:
            h.attrs.update(shared_catalogue=str(ORBITS),shared_catalogue_sha256=masses.digest(ORBITS),
                n_shared_catalogue=232,quality_selected=True,
                selection='Current direct shrinking-radius fits; review accepted and revision matched; uncertainty convergence and bootstrap-audit checks passed; finite positive 95% mass interval <1 dex; estimate and interval off radius-prior boundaries; saved draws consistent with interval; no WJ fallback',
                source_program=str(Path(__file__).resolve()))
    with h5py.File(OUT/'joint_fft_mass_distribution.h5') as f,h5py.File(OUT/'sanya_rcs_from_snr_article.h5') as h:
        group=h['mass_comparison']; names=f['event_id'].asstr()[:].tolist()
        paired=f['rcs_event_id'].asstr()[:]
        np.testing.assert_array_equal(paired,group['event_id'].asstr()[:])
        indices=[names.index(n) for n in paired]
        for key,src in [('dynamic_mass_kg','initial_mass_kg'),('dynamic_lo95_kg','mass_95_lo_kg'),('dynamic_hi95_kg','mass_95_hi_kg')]:
            np.testing.assert_array_equal(group[key][:],f[src][:][indices])
        for key,src in [('rcs_mass_kg','rcs_mass_kg'),('delta_mass_lo95_noise_only_kg','rcs_lo95_kg'),('delta_mass_hi95_noise_only_kg','rcs_hi95_kg')]:
            np.testing.assert_array_equal(group[key][:],f[src][:])
    for mode,name in [('mass','joint_fft_mass_distribution'),('rcs','sanya_rcs_from_snr_article')]:
        subprocess.run([str(BIN),mode,str(OUT/(name+'.h5')),str(OUT/name)],check=True,
                       env=dict(os.environ,DYLD_LIBRARY_PATH='/opt/anaconda3/lib'))
        for ext in ['pdf','png','svg','include.tex']:
            destination=PAPER/(name+'.'+ext)
            backup=OUT/'previous_paper_figures'/destination.name
            backup.parent.mkdir(exist_ok=True)
            if destination.exists() and not backup.exists():
                shutil.copy2(destination,backup)
            shutil.copy2(OUT/(name+'.'+ext),destination)
    with h5py.File(OUT/'artifact_manifest.h5','w') as h:
        paths=sorted(OUT.glob('*.h5'))+sorted(OUT.glob('*.pdf'))+sorted(OUT.glob('*.png'))+sorted(OUT.glob('*.svg'))+sorted(OUT.glob('*.include.tex'))
        paths=[p for p in paths if p.name!='artifact_manifest.h5']
        paths += [PAPER/(name+'.'+ext) for name in ['joint_fft_mass_distribution','sanya_rcs_from_snr_article'] for ext in ['pdf','png','svg','include.tex']]
        h['path']=np.asarray([str(p) for p in paths],dtype=h5py.string_dtype())
        h['sha256']=np.asarray([masses.digest(p) for p in paths],dtype=h5py.string_dtype())
        h.attrs.update(orbit_input_sha256=masses.digest(ORBITS),source_program_sha256=masses.digest(Path(__file__)),
                       native_renderer_sha256=masses.digest(BIN),created_utc=datetime.now(timezone.utc).isoformat())


if __name__=='__main__':
    main()
