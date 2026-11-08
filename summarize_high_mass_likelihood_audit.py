"""Archive compact evidence for physical-mass model adequacy, not new intervals."""
import argparse
from pathlib import Path
import time

import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    a=p.parse_args()
    with h5py.File(a.audit/'summary.h5','w') as out:
        out.attrs.update(created_unix=time.time(),production_products_unchanged=True,
            interpretation='Conditional posterior precision does not include waveform/trajectory discrepancy. Subset point estimates are sensitivity diagnostics, not confidence intervals.',
            curvature_warning='Small-step observed Hessians were step-sensitive; sandwich values are exploratory and are not published error bars.')
        for folder in sorted(a.audit.glob('tri_*')):
            g=out.create_group(folder.name)
            with h5py.File(a.source/folder.name/'plot.h5') as h:
                for k in ['speed_time_s','speed_km_s','speed_lower95_km_s','speed_upper95_km_s',
                          'initial_mass_kg_quantiles','time_s','station']:
                    g[k]=h[k][:]
                g.attrs['best_mass_kg']=h.attrs['best_initial_mass_kg']
            with h5py.File(folder/'pulse/pulse_audit.h5') as h:
                for k,v in h.attrs.items():g.attrs[k]=v
                for k in ['trajectory_waveform_statistics','independent_waveform_statistics']:
                    g[k]=h[k][:]
                h0,h1=h['curvature_step_comparison'][:]
                g.attrs['relative_hessian_step_change']=np.linalg.norm(h1-h0)/np.linalg.norm(h0)
            masses=[];repeat=[]
            for j in range(6):
                with h5py.File(folder/f'omit_chunk_{j}'/'fit.h5') as h:
                    masses.append(h.attrs['mass_kg']);repeat.append(h.attrs['repeat_gain'])
            g['omit_sixth_mass_kg']=masses;g['omit_sixth_repeat_gain']=repeat
            g.attrs.update(subset_mass_min_kg=min(masses),subset_mass_max_kg=max(masses))
            print(folder.name,'full/free reduced chi2',g.attrs['full_reduced_chi2'],g.attrs['independent_pulse_reduced_chi2'],
                  'subset masses',min(masses),max(masses),flush=True)
    print(a.audit/'summary.h5',flush=True)


if __name__=='__main__':main()
