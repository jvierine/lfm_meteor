from pathlib import Path
import h5py,numpy as np
r=Path('/gpfs/pub/data_other/juha/20261121_tristatic');out=r/'phase_squint_prediction';out.mkdir(exist_ok=True)
with h5py.File(out/'headers.h5','w') as o:
 for s,part in [('sanya','sanya/early'),('sanya_late','sanya/late'),('danzhou','danzhou'),('wenchang','wenchang')]:
  p=next((r/part/'event_cuts').glob('event_*.h5'))
  with h5py.File(p) as h:
   g=o.create_group(s);g.attrs['source_file']=str(p)
   for k in h['pulses']:
    if any(t in k for t in ['azimuth','elevation','freq','utc','time']):
     v=h['pulses'][k][:];g[k]=np.unique(v);print(s,k,np.unique(v)[:3])
   for k,v in h.attrs.items():g.attrs[k]=v
 o.attrs['fixed_phase_status']='assumed by user; steering tables/controller unavailable'
