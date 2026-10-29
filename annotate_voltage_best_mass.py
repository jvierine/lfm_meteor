"""Attach best-trajectory estimates without replacing posterior quantiles."""
import sys
import os
import shutil
from pathlib import Path
import h5py
from export_voltage_plot import initial_dynamic_mass

for root in map(Path,sys.argv[1:]):
    folders=[root] if (root/'plot.h5').exists() else sorted(root.glob('tri_*'))
    for folder in folders:
        plot=folder/'plot.h5'
        if not plot.exists() or not (folder/'complete.h5').exists():continue
        with h5py.File(plot) as h:
            if 'initial_mass_kg_quantiles' not in h or 'best_initial_mass_kg' in h.attrs:continue
        with h5py.File(folder/'input.h5') as h:data={k:h[k][()] for k in h}
        with h5py.File(folder/'fit.h5') as h:q=h['params'][:]
        try:r,m,_=initial_dynamic_mass(q[None,:],data)
        except ValueError as e:
            print(folder.name,str(e),flush=True);continue
        temp=folder/'plot.best.partial'
        shutil.copy2(plot,temp)
        with h5py.File(temp,'a') as h:
            h.attrs['best_initial_mass_kg']=float(m[0]);h.attrs['best_initial_radius_m']=float(r[0])
        os.replace(temp,plot)
        print(folder.name,flush=True)
