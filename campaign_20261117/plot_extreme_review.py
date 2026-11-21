"""Render diagnostic refits without changing the production event products."""
import argparse
import concurrent.futures
from pathlib import Path
import subprocess
import html
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
    write_gallery(a.root/'extreme_review')


def write_gallery(root):
    sections=[]
    for event in sorted(root.glob('tri_*')):
        figures=[]
        for variant in ('mainlobe','full','core','multi2','clean586'):
            path=event/variant/'review.h5'
            if not path.exists():continue
            with h5py.File(path) as h:
                rms=np.sqrt(np.mean(h['path_residual_m'][:]**2))
                mass=float(h['initial_mass_kg'][()])
                interval=h['radius_interval_m'][:]
                status='finite conditional interval' if np.isfinite(interval).all() else 'mass interval unresolved'
            relative=event.name+'/'+variant+'/event'
            figures.append('<figure><figcaption>%s: RMS %.2f m; trial mass %.3g kg; %s</figcaption><a href="%s.pdf"><img loading="lazy" src="%s.png"></a></figure>'%(variant,rms,mass,status,relative,relative))
        sections.append('<section><h2>%s</h2>%s</section>'%(html.escape(event.name),''.join(figures)))
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Extreme-event fit review</title><style>body{font:16px sans-serif;margin:24px;color:#222}figure{margin:12px 0 28px}img{width:100%;max-width:1100px}figcaption{margin:8px 0}section{border-top:1px solid #bbb}</style><h1>Extreme-event fit review</h1><p>Diagnostic trials, not catalogue replacements. Mainlobe: three-station core within 10 dB of peak Sanya one-way gain. Full: all retained pulses. Core: common three-station time interval. Multi2: contemporaneous detections at two or more receivers. Clean586: multi2 after 35 ms. Trial masses are not accepted estimates; finite conditional intervals alone do not establish robustness.</p>'+''.join(sections))


if __name__=='__main__':main()
