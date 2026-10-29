"""Measured inversion locations in the altitude slices used for interpolation."""
from pathlib import Path
import argparse
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/rcs_mass_lookup_audit_20261029/measurement_overlay.h5'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lookup',type=Path,default=ROOT/'results/syisr_fdtd_lookup_20261027/lookup.h5');args=parser.parse_args()
    rows=[]
    with h5py.File(args.lookup) as src,h5py.File(OUT,'w') as dst:
        for key in ('altitude_km','velocity_km_s','source_n0_rM2_m_inv','rcs_dbsm'):src.copy(key,dst)
        heights=src['altitude_km'][:]
        for path in sorted((ROOT/'results/tristatic_voltage_all_uncertainty_20261029').glob('tri_*/rcs_mass.h5')):
            with h5py.File(path) as h:
                if not np.array_equal(h['lookup/rcs_dbsm'][:],src['rcs_dbsm'][:],equal_nan=True):continue
                for g in h.get('intervals',{}).values():
                    times=g['time_s'][:];roots=g['log10_source'][:];weights=g['integration_weights_s'][:]
                    for index in g['anchor_indices'][:]:
                        t=h['measurements/time_s'][index]
                        k=int(np.argmin(abs(times-t)))
                        if weights[k]<=0 or not np.isfinite(roots[k]):continue
                        a=float(g['altitude_km'][k]);v=float(g['speed_km_s'][k])
                        lower=int(np.clip(np.searchsorted(heights,a,side='right')-1,0,len(heights)-2))
                        fraction=(a-heights[lower])/(heights[lower+1]-heights[lower])
                        for panel,w in ((lower,1-fraction),(lower+1,fraction)):
                            if w>1e-10:rows.append((path.parent.name,index,panel,a,v,roots[k],w))
        g=dst.create_group('measurements')
        for j,key in enumerate(('event_id','measurement_index','altitude_panel','altitude_km','speed_km_s','log10_source','altitude_weight')):
            g[key]=np.asarray([r[j] for r in rows],dtype=h5py.string_dtype() if j==0 else None)
        dst.attrs.update(source_program=Path(__file__).read_text(),
            lookup_file=str(args.lookup),lookup_correction=src.attrs.get('lookup_correction','none'),
            selection='Measured anchors with unique lookup inversion contributing to mass integration; no interpolated pseudo-measurements',
            panel_assignment='Both bracketing altitude slices, weighted by linear altitude interpolation; repeated markers are not independent measurements')
    print('Unique measurements',len(set((r[0],r[1]) for r in rows)),'events',len(set(r[0] for r in rows)),'panel markers',len(rows))

if __name__=='__main__':main()
