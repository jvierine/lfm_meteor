"""Audit stored mass intervals against three-scale block-bootstrap diagnostics."""
from pathlib import Path
from datetime import datetime, timezone
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent/'results'
CAT = ROOT/'tristatic_cached_shrinking_20261029'
BOOT = ROOT/'suspicious_mass_bootstrap_20261029'
OUT = ROOT/'high_dynamic_mass_audit_20261030.h5'


def main():
    with h5py.File(ROOT/'mass_influx_dynamic_rcs_20261030.h5') as f:
        g=f['dynamic']; ids=g['event_id'].asstr()[:]
        contributions=dict(zip(ids,g['contribution_ton_day'][:]))
    with h5py.File(OUT,'w') as out:
        out.attrs.update(created_utc=datetime.now(timezone.utc).isoformat(),
            source_program=Path(__file__).read_text(),
            interpretation='Diagnostic only: failed publication guards do not validate old thermal intervals; boundary values are prior limits, not measured upper bounds')
        for p in sorted(CAT.glob('tri_*')):
            with h5py.File(p/'plot.h5') as h:
                mass=float(h.attrs['best_initial_mass_kg'])
                if mass<7e-8 and contributions.get(p.name,0)<4:continue
                g=out.create_group(p.name)
                g.attrs.update(best_mass_kg=mass,contribution_ton_day=contributions.get(p.name,np.nan),
                    displayed_uncertainty_method=h.attrs.get('uncertainty_method','thermal conditional posterior'),
                    fitted_speed_drop_m_s=np.ptp(h['speed_km_s'][:])*1000)
                g['displayed_mass_quantiles_kg']=h['initial_mass_kg_quantiles'][:]
                station=h['station'][:];times=h['time_s'][:]
                for s in np.unique(station):
                    idx=np.flatnonzero(station==s);idx=idx[np.argsort(times[idx])]
                    beat=h['residual_beat_hz'][:][idx];std=h['conditional_beat_std_hz'][:][idx]
                    sg=g.create_group('station_'+str(s))
                    sg.attrs.update(n=len(idx),beat_rms_hz=np.sqrt(np.mean(beat**2)),
                        standardized_beat_rms=np.sqrt(np.mean((beat/std)**2)),
                        lag_one_correlation=np.corrcoef(beat[:-1],beat[1:])[0,1])
            b=BOOT/p.name
            if (b/'status.h5').exists():
                with h5py.File(b/'status.h5') as h:g.attrs['bootstrap_publication_status']=str(h.attrs['reason'])
            if (b/'central_refit.h5').exists():
                with h5py.File(b/'central_refit.h5') as h:
                    g.attrs['central_log_likelihood_gain']=h.attrs['logp_gain']
                    g['central_params']=h['params'][:]
            if (b/'bootstrap.h5').exists():
                with h5py.File(b/'bootstrap.h5') as h:
                    for width in np.unique(h['block_seconds'][:]):
                        mask=h['block_seconds'][:]==width
                        q=h['params'][:][mask]; massdraw=h['mass_kg'][:][mask]
                        sg=g.create_group('block_'+str(width))
                        sg.attrs.update(block_seconds=width,all_optimizer_converged=bool(h['converged'][:][mask].all()),
                            boundary_fraction=float(np.mean((q[:,6]<=-6.99)|(q[:,6]>=-2.01))))
                        sg['mass_quantiles_kg']=np.quantile(massdraw,[.025,.5,.975])
            print(p.name,dict(g.attrs),g['displayed_mass_quantiles_kg'][:])
    print(OUT)


if __name__=='__main__':main()
