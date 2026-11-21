"""Render diagnostic refits without changing the production event products."""
import argparse
import concurrent.futures
from pathlib import Path
import subprocess
import h5py
import numpy as np
from astropy.coordinates import EarthLocation
from astropy import units as u
from analyze import Native, C, GAMMA
from prepare_figures import create_event_input


def one(task):
    path,root,library,renderer=task
    folder=path.parent
    event=root/'events'/folder.parent.name
    with h5py.File(path) as h:q=h['params'][:];mask=h['retained_mask'][:]
    with h5py.File(folder/'input.h5') as h:d={k:h[k][()] for k in h}
    with h5py.File(event/'fit.h5') as h:
        fit={k:h[k][()] for k in ('time_s','station','apparent_path_m','matched_filter_path_m',
             'measured_gate_beat_hz','residual_beat_hz','conditional_beat_std_hz','snr_db',
             'speed_mps','model_apparent_path_m','position_ecef_m')}
    fit={k:v[mask] for k,v in fit.items()}
    model=Native(library,folder/'input.h5',len(d['time_s']))
    states=model.call('voltage_states',q,(len(d['time_s']),4))
    obs=model.call('voltage_observables',q,(len(d['time_s']),2))
    velocity=d['center'][3:]+1000*q[3:6]
    gcrs=d['center'][:3]+1000*q[:3]+states[:,:1]*velocity/np.linalg.norm(velocity)
    frames=d['frames']
    position=frames[:,:3]+np.einsum('nij,nj->ni',frames[:,6:15].reshape(-1,3,3),gcrs)
    height=EarthLocation.from_geocentric(position[:,0]*u.m,position[:,1]*u.m,position[:,2]*u.m).height.to_value(u.km)
    physical=dict(model_apparent_path_m=obs[:,0]+C*obs[:,1]/GAMMA,
        physical_path_m=obs[:,0],physical_doppler_hz=obs[:,1],speed_mps=states[:,1],position_ecef_m=position)
    create_event_input(dict(folder=folder,fit=fit,physical=physical,rcs={},height=height))
    model.close()
    subprocess.run([str(renderer),'event',str(folder/'event_plot.h5'),str(folder/'event')],check=True)
    return str(folder)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/gpfs/pub/data_other/juha/20261117_tristatic'))
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--renderer',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    paths=sorted((a.root/'extreme_review').glob('tri_*/*/review.h5'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(one,[(p,a.root,a.library.resolve(),a.renderer.resolve()) for p in paths]))


if __name__=='__main__':main()
