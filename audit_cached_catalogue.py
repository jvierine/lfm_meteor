"""Audit event-level physical fits, intervals, flags and convergence without rewriting products."""
import argparse
from pathlib import Path
import time
import h5py
import numpy as np
from fit_shrinking_voltage import save

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();events=sorted(x.parent.name for x in a.source.glob('tri_*/input.h5'))
    rows=[]
    for event in events:
        folder=a.root/event;row=dict(has_fit=False,has_interval=False,converged=False,physical=False,flags_match=False,
                                   rhat=np.nan,bulk_ess=np.nan,tail_ess=np.nan,mass_best=np.nan,mass_lo=np.nan,mass_hi=np.nan,
                                   upper_radius_prior_proximity=False,fit_unix_time=np.nan)
        if (folder/'plot.h5').exists():
            with h5py.File(folder/'plot.h5') as h:
                row.update(has_fit=True,physical=h.attrs.get('trajectory_model')=='Direct shrinking-radius voltage fit',
                           fit_unix_time=h.attrs.get('fit_unix_time',np.nan),mass_best=h.attrs.get('best_initial_mass_kg',np.nan),
                           converged=bool(h.attrs.get('uncertainty_converged',False)),
                           rhat=h.attrs.get('posterior_max_rhat',np.nan),bulk_ess=h.attrs.get('posterior_min_bulk_ess',np.nan),tail_ess=h.attrs.get('posterior_min_tail_ess',np.nan))
                if 'initial_mass_kg_quantiles' in h:
                    mass=h['initial_mass_kg_quantiles'][:];radius=h['initial_radius_m_quantiles'][:]
                    row.update(has_interval=bool(np.all(np.isfinite(mass))),mass_lo=mass[0],mass_hi=mass[-1],
                               upper_radius_prior_proximity=bool(radius[-1]>=.005))
                flag=a.root/'flags_snapshot'/(event+'.h5')
                revision=0
                if flag.exists():
                    with h5py.File(flag) as f:revision=int(f.attrs.get('version',0))
                row['flags_match']=int(h.attrs.get('fit_flag_revision',-1))==revision
        rows.append(row)
    fields={key:np.array([row[key] for row in rows]) for key in rows[0]}
    fields['event_id']=np.asarray(events,dtype=h5py.string_dtype('utf-8'))
    counts={key:int(fields[key].sum()) for key in ('has_fit','has_interval','converged','physical','flags_match','upper_radius_prior_proximity')}
    save(a.root/'catalogue_audit.h5',fields,dict(input_events=len(events),audited_unix=time.time(),conditional_only=True,**counts))
    print(dict(input_events=len(events),**counts),flush=True)
    print('Unresolved:',[event for event,row in zip(events,rows) if not row['converged']],flush=True)

if __name__=='__main__':main()
