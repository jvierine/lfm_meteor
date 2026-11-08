"""Verify identity, uncertainty policy and current masks before GUI publication."""
import argparse
import hashlib
from pathlib import Path
import time

import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--flags',type=Path,required=True)
    a=p.parse_args()
    failures=[]
    verified=selected=0
    folders=sorted((a.root/'catalog').glob('tri_*'))
    for folder in folders:
        try:
            live=a.flags/(folder.name+'.h5')
            snapshot=a.root/'flags_snapshot'/live.name
            assert (live.read_bytes() if live.exists() else b'') == (snapshot.read_bytes() if snapshot.exists() else b''), 'flags changed since sampling'
            with h5py.File(folder/'input.h5') as inp,h5py.File(folder/'plot.h5') as plot:
                assert int(inp.attrs['manual_flag_revision'])==int(plot.attrs['fit_flag_revision'])
                np.testing.assert_array_equal(inp['time_ns'][:],plot['time_ns'][:])
                np.testing.assert_array_equal(inp['station'][:],plot['station'][:])
                ids={f'{int(s)}:{int(t)}' for s,t in zip(inp['station'][:],inp['time_ns'][:])}
                if live.exists():
                    with h5py.File(live) as flag:
                        assert not ids.intersection(flag['excluded'].asstr()[:]), 'flagged point used in fit'
                good=bool(plot.attrs['uncertainty_converged'])
                assert 'importance' in plot.attrs['uncertainty_method']
                if good:
                    verified+=1
                    for lo,hi in [('path_lower95_m','path_upper95_m'),('speed_lower95_km_s','speed_upper95_km_s')]:
                        assert len(plot[lo])==len(plot['time_s'])
                        assert np.isfinite(plot[lo][:]).all() and np.all(plot[lo][:]<=plot[hi][:])
                    if plot.attrs['dynamic_mass_included']:
                        lo,_,hi=plot['initial_mass_kg_quantiles'][:]
                        assert 0<lo<hi<10*lo and not plot.attrs['uncertainty_prior_boundary']
                        selected+=1
                else:
                    assert not plot.attrs['dynamic_mass_included']
                    assert 'initial_mass_kg_quantiles' not in plot
                assert (folder/'physical_states.h5').exists()
                assert (folder/'path_measurement_uncertainty.h5').exists()
                with h5py.File(folder/'rcs_mass.h5') as r:
                    assert r.attrs['fit_sha256']==hashlib.sha256((folder/'fit.h5').read_bytes()).hexdigest()
        except Exception as exc:
            failures.append((folder.name,repr(exc)))
    with h5py.File(a.root/'verification.h5','w') as h:
        h.attrs.update(events=len(folders),posterior_verified=verified,
                       preliminary_mass_selected=selected,passed=not failures,
                       checked_unix=time.time(),physical_systematics_validated=False)
        for i,key in enumerate(('error_event','error_reason')):
            h[key]=np.asarray([x[i] for x in failures],dtype=h5py.string_dtype())
    print('VERIFIED',len(folders),'posterior',verified,'pre-review selected',selected,'errors',failures,flush=True)
    if failures:
        raise SystemExit(1)


if __name__=='__main__':
    main()
