"""Read-only fit products with separately versioned, atomic HDF5 review flags."""
import argparse
import hashlib
import json
import os
import re
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import h5py
import numpy as np

LOCK=threading.RLock()
EVENT=re.compile(r'^tri_\d+_\d+$')


class Store:
    def __init__(self, root, flags, refits=None, latest=None):
        self.root=Path(root);self.flags=Path(flags);self.flags.mkdir(parents=True,exist_ok=True)
        self.refits=Path(refits) if refits else None
        self.latest=Path(latest) if latest else None

    def path(self,event):
        if not EVENT.fullmatch(event): raise ValueError('Invalid event identifier')
        if self.latest and (self.latest/event/'plot.h5').exists():return self.latest/event
        if self.refits and (self.refits/event/'plot.h5').exists():return self.refits/event
        return self.root/event

    def annotation(self,event):
        path=self.flags/(event+'.h5')
        if not path.exists(): return dict(version=0,excluded=[],bad=False,note='')
        with h5py.File(path) as h:
            return dict(version=int(h.attrs['version']),excluded=h['excluded'].asstr()[:].tolist(),
                        bad=bool(h.attrs['bad']),note=str(h.attrs['note']))

    def data(self,event):
        with h5py.File(self.path(event)/'plot.h5') as h:
            data={k:h[k][:].tolist() for k in h}
            data['beat_pass']=bool(h.attrs['beat_guard_pass'])
            uncertainty_status=str(h.attrs.get('uncertainty_status','Model uncertainty unavailable'))
            fit_metadata={k:h.attrs[k].item() if hasattr(h.attrs[k],'item') else h.attrs[k] for k in ('fit_unix_time','fit_timestamp_source','fit_flag_revision','fit_excluded_count','mass_status','best_initial_mass_kg','best_initial_radius_m') if k in h.attrs}
        data['time_ns']=[str(t) for t in data['time_ns']]
        data['ids']=[f'{s}:{t}' for s,t in zip(data['station'],data['time_ns'])]
        if len(set(data['ids']))!=len(data['ids']): raise ValueError('Duplicate station/pulse identity')
        # This fingerprint is stable when the same data are re-exported.
        core={k:v for k,v in data.items() if k not in ('path_lower95_m','path_upper95_m','speed_lower95_km_s','speed_upper95_km_s','initial_radius_m_quantiles','initial_mass_kg_quantiles')}
        data['fingerprint']=hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest()
        uncertainty=self.path(event)/'path_measurement_uncertainty.h5'
        if uncertainty.exists():
            with h5py.File(uncertainty) as h:
                if np.array_equal(h['time_ns'][:].astype(str),data['time_ns']) and np.array_equal(h['station'][:],data['station']):
                    for k in ('path_measurement_lower95_m','path_measurement_upper95_m'):data[k]=h[k][:].tolist()
        data['uncertainty_status']=uncertainty_status
        data.update(fit_metadata)
        data['event']=event;data['annotation']=self.annotation(event)
        data['omitted_measurements']=[]
        ghost=self.path(event)/'excluded_measurements.h5'
        if ghost.exists():
            with h5py.File(ghost) as h:
                for i,(s,t) in enumerate(zip(h['station'][:],h['time_ns'][:])):
                    data['omitted_measurements'].append(dict(id=f'{s}:{t}',station=int(s),time_ns=str(t),**{k:float(h[k][i]) if np.isfinite(h[k][i]) else None for k in ('time_s','path_m','modeled_path_m','residual_beat_hz')}))
        elif self.path(event)!=self.root/event and (self.root/event/'plot.h5').exists():
            with h5py.File(self.root/event/'plot.h5') as h:
                present=set(data['ids'])
                for i,(s,t) in enumerate(zip(h['station'][:],h['time_ns'][:])):
                    ident=f'{s}:{t}'
                    if ident not in present:
                        data['omitted_measurements'].append(dict(id=ident,station=int(s),time_ns=str(t),time_s=float(h['time_s'][i]),path_m=float(h['seed_apparent_path_m'][i])))
        beam=self.path(event)/'beam_rcs.h5'
        if beam.exists():
            with h5py.File(beam) as h:
                data['beam']={k:[float(x) if np.isfinite(x) else None for x in h[k][:]] for k in ('time_s','snr_db','relative_gain_db','rcs_dbsm','east_m','north_m')}
                data['beam']['time_ns']=[str(t) for t in h['time_ns'][:]]
                data['beam']['plane_range_m']=float(h.attrs['plane_range_m'])
                data['beam']['extent_m']=float(h.attrs['extent_m'])
                data['beam']['contours']=[dict(level=float(g.attrs['gain_db']),lines=[v[:].tolist() for v in g.values()]) for g in h['contours'].values()]
        mass=self.path(event)/'rcs_mass.h5'
        if mass.exists():
            with h5py.File(mass) as h:
                flag=self.flags/(event+'.h5')
                inputs={'fit_sha256':self.path(event)/'fit.h5','beam_sha256':beam,'flag_sha256':flag}
                if any(h.attrs.get(k)!=hashlib.sha256(p.read_bytes() if p.exists() else b'').hexdigest() for k,p in inputs.items()):return data
                if 'lo95_scatter_kg' not in h.attrs:return data
                data['rcs_mass']={k:float(h.attrs[k]) if np.isfinite(h.attrs[k]) else None for k in ('mass_kg','lo95_scatter_kg','hi95_scatter_kg','rcs_rms_db','scatter_valid_fraction','covered_duration_s','requested_duration_s','n_measured','n_missing_in_beam')}
                data['rcs_mass']['uncertainty_clamped_fraction']=float(h.attrs.get('uncertainty_clamped_fraction',0.))
                data['rcs_mass']['curves']=[dict(time_s=g['time_s'][:].tolist(),rcs_dbsm=g['rcs_dbsm'][:].tolist()) for g in h.get('intervals',{}).values()]
        return data

    def catalog(self):
        result=[]
        folders={f.name:f for f in self.root.glob('tri_*')}
        if self.refits:folders.update({f.name:f for f in self.refits.glob('tri_*')})
        if self.latest:folders.update({f.name:f for f in self.latest.glob('tri_*')})
        for folder in sorted(folders.values(),key=lambda f:f.name):
            if not EVENT.fullmatch(folder.name):continue
            plot=self.path(folder.name)/'plot.h5'
            if not plot.exists():continue
            try:
                with h5py.File(plot) as h:
                    item=dict(event=folder.name,n=len(h['time_s']),beat_pass=bool(h.attrs['beat_guard_pass']))
                a=self.annotation(folder.name)
                item.update(flagged=len(a['excluded']),bad=a['bad'],version=a['version'])
                result.append(item)
            except OSError:continue
        return result

    def save(self,event,payload):
        with LOCK:
            data=self.data(event);old=data['annotation']
            if payload.get('version')!=old['version']: raise RuntimeError('Review changed elsewhere; reload before saving')
            if payload.get('fingerprint')!=data['fingerprint']:raise RuntimeError('Fit data changed; reload before saving')
            excluded=payload.get('excluded',[])
            original_path=self.root/event/'plot.h5'
            if not original_path.exists():original_path=self.path(event)/'plot.h5'
            with h5py.File(original_path) as original:
                all_station=original['station'][:];all_time=original['time_ns'][:]
            all_ids=[f'{s}:{t}' for s,t in zip(all_station,all_time)]
            if not isinstance(excluded,list) or any(not isinstance(i,str) or i not in all_ids for i in excluded):raise ValueError('Unknown measurement identity')
            if type(payload.get('bad')) is not bool:raise ValueError('Bad state must be boolean')
            note=payload.get('note','')
            if not isinstance(note,str) or len(note)>4000:raise ValueError('Invalid review note')
            target=self.flags/(event+'.h5');tmp=target.with_suffix('.h5.partial')
            revision=dict(version=old['version']+1,excluded=sorted(set(excluded)),bad=payload['bad'],note=note)
            with h5py.File(tmp,'w') as h:
                if target.exists():
                    with h5py.File(target) as source:
                        if 'history' in source: source.copy('history',h)
                history=h.require_group('history');entry=history.create_group(str(revision['version']))
                entry.attrs.update(unix_time=time.time(),bad=revision['bad'],note=note,fingerprint=data['fingerprint'])
                entry['excluded']=np.array(revision['excluded'],dtype=h5py.string_dtype())
                h.attrs.update(event=event,version=revision['version'],bad=revision['bad'],note=note,
                               fingerprint=data['fingerprint'],format='station:UTC-unix-nanoseconds manual exclusions')
                h['excluded']=np.array(revision['excluded'],dtype=h5py.string_dtype())
                h['station']=np.asarray(all_station,dtype=np.int64)
                h['time_ns']=np.asarray(all_time,dtype=np.int64)
                h['manual_reject_mask']=np.array([i in revision['excluded'] for i in all_ids],dtype=bool)
                h.flush()
            with open(tmp,'rb') as f:os.fsync(f.fileno())
            os.replace(tmp,target)
            return revision


def handler(store,static,wasm):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(static),**kwargs)
        def end_headers(self):
            self.send_header('Cache-Control','no-store')
            super().end_headers()
        def reply(self,status,obj):
            body=json.dumps(obj,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=urlsplit(self.path).path
            if path.startswith('/api/'):
                try:
                    with LOCK:
                        if path=='/api/events':return self.reply(200,store.catalog())
                        if path.startswith('/api/event/'):return self.reply(200,store.data(path.rsplit('/',1)[1]))
                    return self.reply(404,{'error':'Not found'})
                except (OSError,ValueError,KeyError) as e:return self.reply(404,{'error':str(e)})
            if path.startswith('/pkg/'):
                name=path.removeprefix('/pkg/')
                if name not in ('rustplotlib_web.js','rustplotlib_web_bg.wasm','navigation.js'):return self.send_error(404)
                content=(wasm/name).read_bytes();self.send_response(200)
                self.send_header('Content-Type','application/wasm' if name.endswith('.wasm') else 'text/javascript')
                self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content);return
            if path not in ('/','/index.html','/app.js','/beam.js','/style.css'):return self.send_error(404)
            return super().do_GET()
        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and origin != 'http://'+self.headers.get('Host',''):
                return self.reply(403,{'error':'Cross-origin writes are not allowed'})
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                return self.reply(415,{'error':'Expected application/json'})
            path=urlsplit(self.path).path
            if not path.startswith('/api/flags/'):return self.reply(404,{'error':'Not found'})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=65536:raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(n))
                if not isinstance(payload,dict):raise ValueError('Expected object')
                return self.reply(200,store.save(path.rsplit('/',1)[1],payload))
            except RuntimeError as e:return self.reply(409,{'error':str(e)})
            except (ValueError,KeyError,OSError) as e:return self.reply(400,{'error':str(e)})
        def log_message(self,format,*args):
            # Request bodies and access tokens are never logged.
            super().log_message(format,*args)
    return Handler


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--flags',required=True)
    p.add_argument('--refits')
    p.add_argument('--latest')
    p.add_argument('--wasm',required=True)
    p.add_argument('--port',type=int,default=8767);p.add_argument('--host',default='127.0.0.1');a=p.parse_args()
    if a.host not in ('127.0.0.1','localhost'):
        p.error('The token-free reviewer must bind to localhost')
    server=ThreadingHTTPServer((a.host,a.port),handler(Store(a.data,a.flags,a.refits,a.latest),Path(__file__).parent,Path(a.wasm)))
    print(f'Listening on {a.host}:{a.port}',flush=True);server.serve_forever()
