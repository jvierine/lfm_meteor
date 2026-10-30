"""Measured-subset influx, with independent saved-mass and Poisson resampling."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import subprocess
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
CAT = ROOT / 'results/tristatic_cached_shrinking_20261029'
FLAGS = ROOT / 'results/review_flags_scatter_snapshot'
OUT = ROOT / 'results/mass_influx_dynamic_rcs_20261030.h5'


def digest(path):
    return hashlib.sha256(path.read_bytes() if path.exists() else b'').hexdigest()


def collect():
    rows = {'dynamic': {}, 'rcs': {}}
    omitted = []
    for folder in sorted(CAT.glob('tri_*')):
        flag = FLAGS / (folder.name + '.h5')
        revision = 0
        if flag.exists():
            with h5py.File(flag) as h:
                revision = int(h.attrs['version'])
        for kind in rows:
            try:
                with h5py.File(folder / 'plot.h5') as p:
                    if int(p.attrs.get('fit_flag_revision', -1)) != revision:
                        raise ValueError('flags awaiting refit')
                    if kind == 'dynamic':
                        audit = ROOT/'results/suspicious_mass_bootstrap_20261029'/folder.name/'status.h5'
                        if audit.exists():
                            with h5py.File(audit) as check:
                                reason = str(check.attrs.get('reason', ''))
                                published = bool(check.attrs.get('published', False))
                                audit_finished = float(check.attrs.get('finished_unix', np.inf))
                            if 'failed' in reason or 'budget exhausted' in reason:
                                raise ValueError('bootstrap audit unresolved; old thermal interval is not cleared')
                            method = str(p.attrs.get('uncertainty_method', ''))
                            newer_subset_refit = (
                                'random-deletion' in method
                                and float(p.attrs.get('fit_unix_time', 0)) > audit_finished
                                and bool(p.attrs.get('uncertainty_converged', False))
                                and not bool(p.attrs.get('uncertainty_prior_boundary', True))
                            )
                            if published and 'bootstrap' not in method and not newer_subset_refit:
                                raise ValueError('published bootstrap missing from local snapshot')
                        if not p.attrs.get('uncertainty_converged', False):
                            raise ValueError('unresolved uncertainty')
                        lo, _, hi = p['initial_mass_kg_quantiles'][:]
                        rlo, _, rhi = p['initial_radius_m_quantiles'][:]
                        radius = float(p.attrs['best_initial_radius_m'])
                        if radius >= .01*.999 or radius <= 1e-7*1.001 or rhi >= .01*.99 or rlo <= 1e-7*1.01:
                            raise ValueError('radius prior boundary')
                        if not (0 < lo < hi < 10*lo):
                            raise ValueError('95 percent mass interval spans at least a decade')
                        mass = float(p.attrs['best_initial_mass_kg'])
                        source = folder / 'dynamic_mass_posterior.h5'
                        with h5py.File(source) as h:
                            draws = h['mass_kg'][:]
                            method = str(h.attrs.get('uncertainty_method', 'conditional marginal posterior'))
                        clamp = 0.
                    else:
                        source = folder / 'rcs_mass.h5'
                        with h5py.File(source) as h:
                            for key, path in [('fit_sha256', folder/'fit.h5'), ('beam_sha256', folder/'beam_rcs.h5'), ('flag_sha256', flag)]:
                                if h.attrs.get(key) != digest(path):
                                    raise ValueError('stale RCS input hash')
                            mass = float(h.attrs['mass_kg'])
                            lo, hi = h.attrs['lo95_scatter_kg'], h.attrs['hi95_scatter_kg']
                            draws = h['mass_scatter_samples_kg'][:]
                            method = str(h.attrs['uncertainty'])
                            clamp = float(h.attrs['uncertainty_clamped_fraction'])
                if not np.isfinite(mass) or mass <= 0 or not np.all(np.isfinite(draws)) or np.any(draws <= 0):
                    raise ValueError('missing positive mass or uncertainty draws')
                if not np.allclose(np.quantile(draws, [.025, .975]), [lo, hi], rtol=.03, atol=0):
                    raise ValueError('saved samples disagree with displayed interval')
                rows[kind][folder.name] = (mass, draws, method, clamp, digest(source))
            except (OSError, KeyError, ValueError) as exc:
                omitted.append((kind, folder.name, str(exc)))
    return rows, omitted


def summarize(group, names, rows, exposure, speed, area, rng):
    nrep = 20000
    base = 4*np.pi*(6371+95)**2/area * 4 * 24/exposure
    weight = base * (1258/234) * (72/speed)**3
    mass = np.array([rows[n][0] for n in names])
    contribution = mass*weight/1000
    largest = np.argsort(contribution)[-2:]
    sampled = np.array([rng.choice(rows[n][1], nrep) for n in names])
    counts = rng.poisson(1, sampled.shape)
    draws = (sampled*weight[:, None]/1000).sum(axis=0)
    combined = (sampled*counts*weight[:, None]/1000).sum(axis=0)
    keep = np.ones(len(names), bool)
    keep[largest] = False
    group.attrs.update(n_events=len(names), sum_mass_kg=mass.sum(),
        mass_min_kg=mass.min(), mass_max_kg=mass.max(),
        baseline_ton_day=np.sum(mass*base)/1000,
        corrected_ton_day=contribution.sum(),
        amplification=contribution.sum()/(np.sum(mass*base)/1000),
        weighted_poisson_sigma_ton_day=np.linalg.norm(contribution),
        omit_two_largest_contributions_ton_day=contribution[keep].sum(),
        top_two_fraction=contribution[largest].sum()/contribution.sum(),
        n_events_with_clamped_rcs_draws=sum(rows[n][3] > 0 for n in names))
    for key, value in [('mass_only_interval95_ton_day', np.quantile(draws,[.025,.975])),
                       ('combined_interval95_ton_day', np.quantile(combined,[.025,.975])),
                       ('omit_two_interval95_ton_day', np.quantile((sampled[keep]*counts[keep]*weight[keep,None]/1000).sum(axis=0),[.025,.975])),
                       ('combined_draws_ton_day', combined), ('mass_only_draws_ton_day', draws),
                       ('mass_kg',mass),('weight_per_day',weight),('baseline_weight_per_day',base),
                       ('speed_km_s',speed),('effective_hours',exposure),('contribution_ton_day',contribution)]:
        group[key] = value
    for key, value in [('event_id',names), ('top_two_event_id',np.array(names)[largest]),
                       ('uncertainty_method',[rows[n][2] for n in names]),
                       ('source_sha256',[rows[n][4] for n in names])]:
        group[key] = np.asarray(value,dtype=h5py.string_dtype())
    for name in names:
        group.create_dataset('input_mass_draws/'+name,data=rows[name][1],compression='gzip')
    print(group.name,dict(group.attrs), '95%',group['combined_interval95_ton_day'][:],flush=True)


def main():
    rows, omitted = collect()
    flux = ROOT/'results/sky_averaged_flux_empirical_centroid_area_v20260624c.h5'
    beam = ROOT/'results/sanya_gain_containment_area_v20260703b.h5'
    with h5py.File(flux) as h:
        names = h['event_id'].asstr()[:]
        exposure = h['family_effective_exposure_sum_h'][:]
        speed = h['speed_km_s'][:]
    with h5py.File(beam) as h:
        area = float(h.attrs['area_horizontal_km2'])
    rng = np.random.default_rng(20261030)
    with h5py.File(OUT,'w') as h:
        h.attrs.update(created_utc=datetime.now(timezone.utc).isoformat(),
            git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_program=Path(__file__).read_text(), flux_sha256=digest(flux), beam_sha256=digest(beam),
            collecting_area_km2=area, n_weighted_catalogue=len(names), imputed_masses=False,
            normalization='Archived geocentric speeds, folded radiant exposures, area; Falt=1258/234; heuristic (72/v)^3',
            uncertainty='Independent event saved mass draws and Poisson(1) counts, 20000 replicates; includes random-deletion robustness draws, not a calibrated confidence interval; fixed selection and weights; no calibration or physical-model systematics',
            caveat='RCS is in-beam mass loss; pilot FDTD lookup and user low-speed override; dynamic mass is at first detection. Neither estimates unobserved pre-detection loss.')
        for kind in rows:
            for matched in (False,True):
                selected = np.array([n in rows[kind] and (not matched or all(n in r for r in rows.values())) for n in names])
                selected &= np.isfinite(exposure)&(exposure>0)&np.isfinite(speed)&(speed>0)
                summarize(h.create_group(kind+('_matched' if matched else '')),names[selected].tolist(),rows[kind],exposure[selected],speed[selected],area,rng)
            for name in rows[kind]:
                if name not in names:
                    omitted.append((kind,name,'no archived exposure weight'))
        for j,key in enumerate(['omitted_kind','omitted_event','omission_reason']):
            h[key]=np.asarray([r[j] for r in omitted],dtype=h5py.string_dtype())
    with h5py.File(OUT) as source, h5py.File(OUT.with_name(OUT.stem+'_summary.h5'),'w') as summary:
        summary.attrs.update(source.attrs)
        summary.attrs['full_snapshot_sha256'] = digest(OUT)
        for name in source:
            obj = source[name]
            if isinstance(obj, h5py.Group):
                target = summary.create_group(name)
                target.attrs.update(obj.attrs)
                for key in obj:
                    if key != 'input_mass_draws':
                        obj.copy(key,target)
            else:
                source.copy(name,summary)
    print(OUT)


if __name__ == '__main__':
    main()
