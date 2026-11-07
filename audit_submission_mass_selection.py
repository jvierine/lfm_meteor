"""Audit frozen and live mass selections without changing publication inputs.

Uses the existing collector's review/uncertainty checks, with no WJ fallback.
All outputs are isolated; no orbit products are read or written.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import subprocess
import sys
import h5py
import numpy as np
from astropy.constants import L_sun, GM_sun, c
import estimate_current_mass_influx as influx

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/submission_mass_audit_20261107'
OLD = ROOT / 'results/current_dynamic_mass_distribution/figure6.h5'
PANELS = ROOT / 'results/current_review_mass_comparison/rcs_panels.h5'
ARCHIVE = ROOT / 'results/mass_influx_dynamic_rcs_including0142_20261030.h5'
FLUX = ROOT / 'results/sky_averaged_flux_empirical_centroid_area_v20260624c.h5'
BEAM = ROOT / 'results/sanya_gain_containment_area_v20260703b.h5'


def strings(group, key, values):
    group[key] = np.asarray(list(values), dtype=h5py.string_dtype())


def sizes(group, names, mass):
    diameter = (6*np.asarray(mass)/(np.pi*3000))**(1/3)*1e6
    beta = 3*L_sun.value/(16*np.pi*GM_sun.value*c.value*3000*(diameter*1e-6/2))
    strings(group, 'event_id', names)
    group['mass_kg'] = mass
    group['diameter_um'] = diameter
    group['beta'] = beta
    group['diameter_min_q25_median_q75_max_um'] = np.quantile(diameter, [0,.25,.5,.75,1])
    group['beta_min_q25_median_q75_max'] = np.quantile(beta, [0,.25,.5,.75,1])
    group.attrs.update(density_kg_m3=3000., radiation_pressure_efficiency=1.)
    print(group.name, len(mass), group['diameter_min_q25_median_q75_max_um'][:],
          group['beta_min_q25_median_q75_max'][:])


def main():
    OUT.mkdir(exist_ok=True)
    rows, omitted = influx.collect()
    print('Live accepted', {k:len(v) for k,v in rows.items()})
    print('Omissions', Counter((k,r) for k,n,r in omitted))
    with h5py.File(OLD) as old, h5py.File(PANELS) as panels, h5py.File(ARCHIVE) as archived, h5py.File(OUT/'audit.h5','w') as out:
        out.attrs.update(created_utc=datetime.now(timezone.utc).isoformat(),
                         source_program=Path(__file__).read_text(),
                         collector_source=(ROOT/'estimate_current_mass_influx.py').read_text(),
                         status='Candidate audit; publication inputs unchanged; no WJ fallback')
        provenance = out.create_group('provenance')
        for path in [OLD, PANELS, ARCHIVE, FLUX, BEAM]:
            provenance.attrs[str(path)] = influx.digest(path)
        old_names = old['event_id'].asstr()[:]
        selected = old['selected'][:]
        old_selected = set(old_names[selected])
        paired = old['rcs_event_id'].asstr()[:]
        panel = panels['mass_comparison']
        np.testing.assert_array_equal(paired, panel['event_id'].asstr()[:])
        idx = {n:i for i,n in enumerate(old_names)}
        for key, src in [('dynamic_mass_kg','initial_mass_kg'),('dynamic_lo95_kg','mass_95_lo_kg'),('dynamic_hi95_kg','mass_95_hi_kg')]:
            np.testing.assert_array_equal(panel[key][:],old[src][:][[idx[n] for n in paired]])
        for key, src in [('rcs_mass_kg','rcs_mass_kg'),('delta_mass_lo95_noise_only_kg','rcs_lo95_kg'),('delta_mass_hi95_noise_only_kg','rcs_hi95_kg')]:
            np.testing.assert_array_equal(panel[key][:],old[src][:])
        out.attrs['frozen_figures_exact_pair_and_interval_match'] = True
        sizes(out.create_group('frozen_figure_sizes'),old_names[selected],old['initial_mass_kg'][:][selected])
        names = sorted(rows['dynamic'])
        sizes(out.create_group('current_sizes'),names,[rows['dynamic'][n][0] for n in names])
        strings(out,'removed_from_frozen_dynamic',sorted(old_selected-set(names)))
        strings(out,'added_to_frozen_dynamic',sorted(set(names)-old_selected))
        reasons = {(k,n):r for k,n,r in omitted}
        strings(out,'removed_dynamic_reason',[reasons.get(('dynamic',n),'unknown') for n in sorted(old_selected-set(names))])
        print('Frozen dynamic removed',len(old_selected-set(names)),Counter(reasons.get(('dynamic',n),'unknown') for n in old_selected-set(names)))
        print('Frozen dynamic added',sorted(set(names)-old_selected))
        for kind in rows:
            past = archived[kind]['event_id'].asstr()[:]
            past_index = {n:i for i,n in enumerate(past)}
            strings(out,kind+'_archive_removed',sorted(set(past)-set(rows[kind])))
            strings(out,kind+'_archive_added_before_exposure',sorted(set(rows[kind])-set(past)))
            changed = [n for n in past if n in rows[kind] and archived[kind]['source_sha256'].asstr()[past_index[n]] != rows[kind][n][4]]
            strings(out,kind+'_archive_source_changed',changed)
            print(kind,'archive removed', sorted(set(past)-set(rows[kind])), 'changed sources',len(changed))
        for j,key in enumerate(['omitted_kind','omitted_event','omission_reason']):
            strings(out,key,[r[j] for r in omitted])
        current_rcs = sorted(set(names)&set(rows['rcs']))
        with h5py.File(OUT/'figure7.h5','w') as fig:
            fig.attrs.update(density_kg_m3=3000., bin_width_dex=.48,
                             selection='Existing influx collector validation; finite 95% interval <1 dex; no radius boundary; current review; cleared audit; saved draws consistent',
                             source_program=str(Path(__file__).resolve()),
                             source_catalogue=str(influx.CAT))
            strings(fig,'event_id',names)
            fig['selected'] = np.ones(len(names),dtype=bool)
            fig['initial_mass_kg'] = [rows['dynamic'][n][0] for n in names]
            lows, highs, speeds, methods = [], [], [], []
            for n in names:
                with h5py.File(influx.CAT/n/'plot.h5') as p:
                    lo,_,hi = p['initial_mass_kg_quantiles'][:]
                    lows.append(lo); highs.append(hi); speeds.append(p['speed_km_s'][0])
                    methods.append(str(p.attrs.get('uncertainty_method',rows['dynamic'][n][2])))
            fig['mass_95_lo_kg'] = lows; fig['mass_95_hi_kg'] = highs
            fig['initial_speed_km_s'] = speeds
            strings(fig,'uncertainty_method',methods)
            strings(fig,'rcs_event_id',current_rcs)
            fig['rcs_mass_kg'] = [rows['rcs'][n][0] for n in current_rcs]
            fig['rcs_speed_km_s'] = [speeds[names.index(n)] for n in current_rcs]
            for key, attr in [('rcs_lo95_kg','lo95_scatter_kg'),('rcs_hi95_kg','hi95_scatter_kg')]:
                values = []
                for n in current_rcs:
                    with h5py.File(influx.CAT/n/'rcs_mass.h5') as r:
                        values.append(r.attrs[attr])
                fig[key] = values
        with h5py.File(OUT/'figure7.h5') as fig, h5py.File(OUT/'figure8.h5','w') as target:
            panels.copy('pulses',target)
            p = target.create_group('mass_comparison')
            strings(p,'event_id',current_rcs)
            indices = [names.index(n) for n in current_rcs]
            for key, src in [('dynamic_mass_kg','initial_mass_kg'),('dynamic_lo95_kg','mass_95_lo_kg'),('dynamic_hi95_kg','mass_95_hi_kg')]:
                p[key] = fig[src][:][indices]
            for key, src in [('rcs_mass_kg','rcs_mass_kg'),('delta_mass_lo95_noise_only_kg','rcs_lo95_kg'),('delta_mass_hi95_noise_only_kg','rcs_hi95_kg')]:
                p[key] = fig[src][:]
            p.attrs['sample_count'] = len(current_rcs)
        print('Candidate figure dynamic/RCS',len(names),len(current_rcs))
        with h5py.File(FLUX) as f, h5py.File(BEAM) as b:
            weighted_names = f['event_id'].asstr()[:]
            exposure = f['family_effective_exposure_sum_h'][:]
            speed = f['speed_km_s'][:]
            area = float(b.attrs['area_horizontal_km2'])
        rng = np.random.default_rng(20261030)
        for kind in rows:
            for matched in (False,True):
                keep = np.array([n in rows[kind] and (not matched or all(n in r for r in rows.values())) for n in weighted_names])
                keep &= np.isfinite(exposure)&(exposure>0)&np.isfinite(speed)&(speed>0)
                group = out.create_group(kind+('_matched' if matched else ''))
                influx.summarize(group,weighted_names[keep].tolist(),rows[kind],exposure[keep],speed[keep],area,rng)
            strings(out,kind+'_no_valid_exposure',sorted(set(rows[kind])-set(weighted_names[np.isfinite(exposure)&(exposure>0)&np.isfinite(speed)&(speed>0)])))
        out.attrs['collecting_area_km2'] = area
        legacy = ROOT/'results/joint_fft_mass_distribution_profile_fixedmask_capped_logr61_20260709.h5'
        with h5py.File(legacy) as h:
            keep = h['selected'][:]
            sizes(out.create_group('legacy_110_sizes'),h['event_id'].asstr()[:][keep],h['initial_mass_kg'][:][keep])
        provenance.attrs[str(legacy)] = influx.digest(legacy)
    precision = OUT/'precision'
    subprocess.run([sys.executable, str(ROOT/'collect_delay_residuals.py'),
                    str(influx.CAT), str(influx.FLAGS), str(precision)],check=True)
    subprocess.run([sys.executable, str(ROOT/'summarize_trajectory_precision.py'),
                    str(influx.CAT),str(precision/'plot.h5'),
                    str(precision/'trajectory_precision.h5')],check=True)
    with h5py.File(precision/'plot.h5') as selection, h5py.File(precision/'catalogue_statistics.h5','w') as h:
        names = selection['events'].asstr()[:]
        speeds, residual_rms, hashes, flag_hashes, counts = [], [], [], [], []
        for n in names:
            source = influx.CAT/n/'plot.h5'
            flag = influx.FLAGS/(n+'.h5')
            excluded = set()
            if flag.exists():
                with h5py.File(flag) as f:
                    excluded = set(f['excluded'].asstr()[:])
            with h5py.File(source) as p:
                speed_time = p['speed_time_s'][:]
                speeds.append(p['speed_km_s'][np.argmin(speed_time)])
                residual = p['modeled_path_m'][:]-p['seed_apparent_path_m'][:]
                keep = np.array([f'{int(s)}:{int(t)}' not in excluded for s,t in zip(p['station'][:],p['time_ns'][:])])
                keep &= np.isfinite(residual)
                residual_rms.append(np.sqrt(np.mean(residual[keep]**2)))
                counts.append(keep.sum())
            hashes.append(influx.digest(source)); flag_hashes.append(influx.digest(flag))
        strings(h,'event_id',names)
        strings(h,'plot_sha256',hashes); strings(h,'flag_sha256',flag_hashes)
        h['initial_speed_km_s'] = speeds
        h['event_apparent_path_rms_m'] = residual_rms
        h['event_measurement_count'] = counts
        h['speed_min_median_max_km_s'] = np.quantile(speeds,[0,.5,1])
        h['event_rms_quartiles_m'] = np.quantile(residual_rms,[.25,.5,.75])
        h.attrs.update(source_program=str(Path(__file__).resolve()),
                       selection_sha256=influx.digest(precision/'plot.h5'),
                       scope='Local saved direct shrinking-radius fits with current local snapshot flags; no mass-quality cut; not validated 3D accuracy')
        assert sum(counts) == len(selection['station'])
        print('Catalogue speeds',h['speed_min_median_max_km_s'][:], 'event RMS',h['event_rms_quartiles_m'][:])
    print(OUT)


if __name__ == '__main__':
    main()
