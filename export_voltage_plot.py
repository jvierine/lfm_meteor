"""Export compact HDF5 plot inputs; all rendering is native rustplotlib."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from run_native_voltage_catalog import apparent_paths


def export(folder):
    folder=Path(folder)
    with h5py.File(folder/'input.h5') as h:
        data={k:h[k][()] for k in h}
    with h5py.File(folder/'fit.h5') as h:
        q=h['params'][:]
        fields={k:h[k][()] for k in ('time_s','station','seed_apparent_path_m',
                'residual_beat_hz','conditional_beat_std_hz')}
        passed=bool(h.attrs['beat_guard_pass'])
    fields['modeled_path_m']=apparent_paths(q,data)
    fields['time_ns']=data['time_ns']
    t=np.linspace(data['time_s'].min(),data['time_s'].max(),200)
    fields['speed_time_s']=t
    fields['speed_km_s']=(np.linalg.norm(data['center'][3:]+1000*q[3:6])-10**q[6]*np.exp(10**q[7]*t))/1000
    with h5py.File(folder/'plot.h5','w') as h:
        for k,v in fields.items(): h[k]=v
        h.attrs['beat_guard_pass']=passed
        h.attrs['event_id']=folder.name
        h.attrs['source_program']=__file__


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args()
    root=Path(args.root)
    folders=[root] if (root/'fit.h5').exists() else sorted(x.parent for x in root.glob('*/fit.h5'))
    for folder in folders:
        export(folder)
    print(f'Exported {len(folders)} plot bundles')
