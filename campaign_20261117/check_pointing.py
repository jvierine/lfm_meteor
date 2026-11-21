"""Check beam registration on disjoint events with timing held fixed."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from analyze import STATIONS, ENU, save, DELAY_CORRECTION_US
from derived import angular_offsets, beam_frame, beam_power, directions_from_offsets


def register(counts, x, y, region):
    xx, yy = np.meshgrid((x[1:]+x[:-1])/2, (y[1:]+y[:-1])/2)
    target = np.log1p(gaussian_filter(counts, 1.))
    radius = np.maximum(abs(xx), abs(yy))
    mask = (radius < 2.3) & ((radius > .8) if region == 'sidelobes' else True)
    axis = beam_frame()[0]

    def residual(q, fixed=False):
        dx, dy, angle, exponent, amp, bx, by, floor = q
        if fixed:
            dx = dy = angle = 0.
        a = np.deg2rad(angle)
        u = (xx-dx)*np.cos(a)+(yy-dy)*np.sin(a)
        v = -(xx-dx)*np.sin(a)+(yy-dy)*np.cos(a)
        power = beam_power(directions_from_offsets(u, v), axis)
        model = np.exp(amp+bx*xx+by*yy)*power**exponent+np.exp(floor)
        return (np.log1p(gaussian_filter(model, 1.))-target)[mask]

    low = [-.7,-.7,-15,.1,0,-1,-1,-10]
    high = [.7,.7,15,2,12,1,1,4]
    fits = [least_squares(residual, [0,dy,0,.7,np.log(max(counts.max(),2)),0,0,-2],
                         bounds=(low,high), max_nfev=130) for dy in (0.,.4)]
    fit = min(fits,key=lambda f:f.cost)
    fixed = least_squares(lambda q: residual(np.r_[0.,0.,0.,q],True), fit.x[3:],
                          bounds=(low[3:],high[3:]),max_nfev=130)
    return fit.x, fit.cost, fixed.cost


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/gpfs/pub/data_other/juha/20261117_tristatic'))
    a=p.parse_args()
    with h5py.File(a.root/'association_selection.h5') as h:
        names=[n.decode() for n,k in zip(h['event_id'][:],h['accepted'][:]) if k]
    events=[]
    for name in names:
        with h5py.File(a.root/'events'/name/'fit.h5') as h:
            assert abs(h.attrs['delay_correction_us']-DELAY_CORRECTION_US)<1e-10
            local=(h['position_ecef_m'][:]-STATIONS[0])@ENU.T
            ex,ny=angular_offsets(local/np.linalg.norm(local,axis=1)[:,None])
            events.append((int(np.min(h['time_ns'][:])),ex,ny))
    events.sort(key=lambda e:e[0])
    n=len(events);grid=np.linspace(-2.6,2.6,105)
    groups={'all':range(n),'early':range(n//2),'late':range(n//2,n),
            'even':range(0,n,2),'odd':range(1,n,2)}
    fields={}
    for name,indices in groups.items():
        selected=[events[i] for i in indices]
        x=np.concatenate([e[1] for e in selected]);y=np.concatenate([e[2] for e in selected])
        count,xe,ye=np.histogram2d(x,y,bins=(grid,grid));count=count.T
        fields[name+'_counts']=count
        for region in ('all','sidelobes'):
            q,cost,fixed=register(count,xe,ye,region)
            key=name+'_'+region
            fields[key+'_parameters']=q
            fields[key+'_costs']=np.array([cost,fixed])
            print(key,'events',len(selected),'east_north_rotation',q[:3],
                  'cost_ratio',cost/fixed,flush=True)
    fields['edges']=grid
    save(a.root/'figure_data/pointing_check.h5',fields,
         {'delay_correction_us':DELAY_CORRECTION_US,
          'interpretation':'count-pattern registration; disjoint-event checks, not independent hardware pointing measurement'})


if __name__=='__main__':
    main()
