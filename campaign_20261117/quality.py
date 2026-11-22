"""Hard path-residual quality gate on observations used by the dynamic fit."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from analyze import save

LIMIT_M=40.


def path_quality(residual,mask):
    values=np.asarray(residual)[np.asarray(mask,dtype=bool)]
    maximum=float(np.max(abs(values))) if len(values) and np.isfinite(values).all() else np.inf
    bad=int(np.count_nonzero(~np.isfinite(values)|(abs(values)>=LIMIT_M)))
    return bool(len(values) and maximum<LIMIT_M),maximum,bad


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/gpfs/pub/data_other/juha/20261117_tristatic'))
    a=p.parse_args()
    with h5py.File(a.root/'association_selection.h5') as h:
        names=[n.decode() for n,k in zip(h['event_id'][:],h['accepted'][:]) if k]
    rows=[]
    for name in names:
        path=a.root/'events'/name/'physical.h5'
        if not path.exists():continue
        with h5py.File(path,'r+') as h:
            good,maximum,bad=path_quality(h['path_residual_m'][:],h['dynamic_mass_fit_mask'][:])
            interval=bool(h.attrs.get('mass_interval_resolved_profile',h.attrs.get('mass_resolved_profile',False)))
            h.attrs.update(mass_interval_resolved_profile=interval,
                path_quality_pass=good,path_quality_limit_m=LIMIT_M,
                max_used_path_residual_m=maximum,bad_used_path_measurements=bad,
                mass_resolved_profile=interval and good,
                mass_resolved_local_curvature=interval and good)
            if not good:h.attrs['dynamic_mass_fit_status']='rejected: fitted path residual reaches or exceeds 40 m; review required'
        rows.append((name,good,maximum,bad,interval and good))
    save(a.root/'path_quality.h5',dict(event_id=np.array([r[0] for r in rows],dtype='S32'),
        passed=np.array([r[1] for r in rows]),max_used_residual_m=np.array([r[2] for r in rows]),
        bad_measurements=np.array([r[3] for r in rows]),accepted_mass=np.array([r[4] for r in rows])),
        {'criterion':'every used absolute apparent-path residual must be strictly below 40 m; excluded tails do not enter gate'})
    print('QUALITY events',len(rows),'passed',sum(r[1] for r in rows),
          'flagged',sum(not r[1] for r in rows),'accepted_masses',sum(r[4] for r in rows),flush=True)


if __name__=='__main__':main()
