"""Wait for prior writers, then launch and record a bounded MPI retry pass."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from fit_shrinking_voltage import save

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--library',required=True)
    p.add_argument('--wait-pid',type=int,action='append',default=[])
    a=p.parse_args();root=a.root;base=root.parent;code=Path(__file__).resolve().parent
    for pid in a.wait_pid:
        while Path(f'/proc/{pid}').exists():time.sleep(5)
    command=['/home/juha/miniforge3/bin/mpirun','-np','128','--bind-to','core',
             '/home/juha/miniforge3/bin/python','-u',str(code/'fit_cached_shrinking_voltage.py'),
             '--source',str(base/'tristatic_voltage_correlated_20261029'),
             '--initial',str(base/'tristatic_voltage_refit_flags_20261029'),
             '--flags',str(root/'flags_snapshot'),'--output',str(root),
             '--seed',str(base/'tristatic_shrinking_voltage_20261029'),'--library',a.library,
             '--budget','300','--budget-generation','reduced-scaled-v1']
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with open(root/'central_retry.log','ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=code,stdout=log,stderr=log,env=env)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
        save(root/'central_launch.h5',{},dict(pid=process.pid,started_unix=time.time(),workers=128,code_commit=commit,budget_generation='reduced-scaled-v1'))
        result=process.wait()
    save(root/'central_finished.h5',{},dict(exit_code=result,finished_unix=time.time(),event_success_requires_individual_checks=True))

if __name__=='__main__':main()
