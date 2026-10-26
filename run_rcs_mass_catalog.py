"""Bounded parallel event jobs with independent HDF5 and log outputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
from pathlib import Path
import subprocess
import sys
import time
import h5py


def event_job(task):
    event,inputs,rcs,output=task
    output=Path(output)
    command=[sys.executable,str(Path(__file__).with_name('fit_tristatic_rcs_mass.py')),
        '--event-id',event,'--input-dir',inputs,'--rcs-input',rcs,
        '--output',str(output/(event+'.h5'))]
    start=time.time()
    with (output/(event+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    return event,result.returncode,time.time()-start


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',required=True,type=Path)
    p.add_argument('--rcs-input',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--workers',type=int,default=100)
    a=p.parse_args()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key]='1'
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with h5py.File(a.input_dir/'rcs_mass_inversion_inputs.h5') as h:events=sorted(h.keys())
    tasks=[(e,str(a.input_dir.resolve()),str(a.rcs_input.resolve()),str(a.output_dir.resolve())) for e in events]
    print(f'Starting {len(tasks)} events; workers={a.workers}',flush=True)
    failed=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(event_job,t) for t in tasks]
        for done,future in enumerate(as_completed(futures),1):
            event,code,seconds=future.result()
            failed+=int(code!=0)
            print(f'{done}/{len(events)} {event} exit={code} seconds={seconds:.1f}',flush=True)
    print(f'Completed {len(events)} jobs; process failures={failed}',flush=True)
    if failed:sys.exit(1)


if __name__=='__main__':main()
