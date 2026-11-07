"""Read-only remote mass snapshot and hash audit, independent of propagation."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import io
import json
import shlex
import subprocess
import tarfile
import h5py
import numpy as np
import estimate_current_mass_influx as collector

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/submission_mass_remote_20261107'
REMOTE='/home/juha/src/lfm_meteor'
CAT='tristatic_cached_shrinking_20261029'
ORBIT=ROOT/'results/tristatic_shrinking_orbits_audited_20261107/orbits.h5'


def ssh(command,**kwargs):
    result=subprocess.run(['ssh','-o','BatchMode=yes','avaruus',command],capture_output=True,**kwargs)
    if result.returncode:
        raise RuntimeError(str(result.stderr))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    with h5py.File(ORBIT) as h:
        events=h['event_id'].asstr()[:].tolist()
    assert len(set(events))==232
    code=f'''from pathlib import Path
import hashlib,json
root=Path({REMOTE!r}); events={events!r}; records=[]; missing=[]
for e in events:
 folder=root/'results'/{CAT!r}/e
 files=[folder/n for n in ['plot.h5','dynamic_mass_posterior.h5','fit.h5','input.h5','beam_rcs.h5','rcs_mass.h5','block_bootstrap_states.h5']]
 files += [root/'results/voltage_review_flags'/(e+'.h5'),root/'results/suspicious_mass_bootstrap_20261029'/e/'status.h5']
 for p in files:
  rel=str(p.relative_to(root))
  if p.exists(): records.append([e,rel,hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size])
  else: missing.append([e,rel])
print(json.dumps(dict(records=records,missing=missing)))
'''
    manifest=json.loads(ssh('/usr/bin/python3 -c '+shlex.quote(code),text=True).stdout)
    records=manifest['records']; paths=[x[1] for x in records]
    print('Fetching',len(paths),'files;',sum(x[3] for x in records)/1e6,'MB; missing',len(manifest['missing']),flush=True)
    archive=ssh('tar -C '+shlex.quote(REMOTE)+' --null -T - -czf -',input=('\0'.join(paths)+'\0').encode()).stdout
    snapshot=OUT/'snapshot'
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
        for member in tar.getmembers():
            rel=Path(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            target=snapshot/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    after=ssh('xargs -0 sha256sum --',input='\0'.join(REMOTE+'/'+p for p in paths)+'\0',text=True).stdout
    post={line.split('  ',1)[1]:line.split('  ',1)[0] for line in after.splitlines()}
    matches=[]
    for _,path,digest,_ in records:
        matches.append(collector.digest(snapshot/path)==digest==post[REMOTE+'/'+path])
    with h5py.File(OUT/'remote_hash_manifest.h5','w') as h:
        h.attrs.update(host='avaruus',remote_root=REMOTE,created_utc=datetime.now(timezone.utc).isoformat(),
                       source_program=Path(__file__).read_text(),all_match=bool(all(matches)),
                       shared_selection_sha256=collector.digest(ORBIT))
        for i,key in enumerate(['event_id','relative_path','remote_before_sha256']):
            h[key]=np.asarray([x[i] for x in records],dtype=h5py.string_dtype())
        h['remote_after_sha256']=np.asarray([post[REMOTE+'/'+p] for p in paths],dtype=h5py.string_dtype())
        h['local_sha256']=np.asarray([collector.digest(snapshot/p) for p in paths],dtype=h5py.string_dtype())
        h['matches']=matches
        h['accepted_event_id']=np.asarray(events,dtype=h5py.string_dtype())
        h['missing_relative_path']=np.asarray([x[1] for x in manifest['missing']],dtype=h5py.string_dtype())
    assert all(matches),'Remote changed during fetch: do not publish'
    # Point the established quality collector at the isolated remote snapshot.
    collector.ROOT=snapshot
    collector.CAT=snapshot/'results'/CAT
    collector.FLAGS=snapshot/'results/voltage_review_flags'
    rows,omitted=collector.collect()
    print('REMOTE selected', {k:len(v) for k,v in rows.items()},'paired',len(set(rows['dynamic'])&set(rows['rcs'])),flush=True)
    print('Omissions',Counter((k,r) for k,e,r in omitted),flush=True)
    with h5py.File(OUT/'selection.h5','w') as h:
        h.attrs.update(source_program=Path(__file__).read_text(),collector_source=(ROOT/'estimate_current_mass_influx.py').read_text(),
                       remote_manifest_sha256=collector.digest(OUT/'remote_hash_manifest.h5'),n_catalogue=232,
                       selection='Existing collector cuts on current remote plot/draws/review/audit/RCS hashes; no WJ fallback')
        h['catalogue_event_id']=np.asarray(events,dtype=h5py.string_dtype())
        for kind in rows:
            g=h.create_group(kind); names=sorted(rows[kind])
            g['event_id']=np.asarray(names,dtype=h5py.string_dtype())
            g['mass_kg']=[rows[kind][n][0] for n in names]
            g['source_sha256']=np.asarray([rows[kind][n][4] for n in names],dtype=h5py.string_dtype())
            for n in names:
                g.create_dataset('input_mass_draws/'+n,data=rows[kind][n][1],compression='gzip')
        for i,key in enumerate(['omitted_kind','omitted_event','omission_reason']):
            h[key]=np.asarray([r[i] for r in omitted],dtype=h5py.string_dtype())
        d=(6*h['dynamic/mass_kg'][:]/(np.pi*3000))**(1/3)*1e6
        h['diameter_min_q25_median_q75_max_um']=np.quantile(d,[0,.25,.5,.75,1])
        print('Diameters',h['diameter_min_q25_median_q75_max_um'][:],flush=True)
    # Compare every remotely selected plot with Sartre's staged plot, if available.
    staged=ROOT/'results/tristatic_shrinking_orbits_current_20261107/current_sources'
    comparison=[]
    for e in events:
        p=staged/e/'plot.h5'
        if p.exists(): comparison.append((e,collector.digest(p)==collector.digest(collector.CAT/e/'plot.h5')))
    print('Sartre staged plot matches',sum(x[1] for x in comparison),'/',len(comparison),flush=True)


if __name__=='__main__':
    main()
