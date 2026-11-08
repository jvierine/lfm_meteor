"""Detached, bounded pilot and simulation checks; never installs paper products."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import h5py


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--catalogue', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--cores', type=int, default=128)
    a = p.parse_args()
    code = Path(__file__).resolve().parent
    library = code/'tristatic_native/target/release/libtristatic_native.so'
    a.root.mkdir(parents=True, exist_ok=True)
    began = time.time()
    def status(stage, passed=False, error=''):
        target = a.root/'status.h5.partial'
        with h5py.File(target,'w') as h:
            h.attrs.update(stage=stage,completed=passed,error=error,started_unix=began,
                updated_unix=time.time(),canonical_products_unchanged=True,
                source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip())
        target.replace(a.root/'status.h5')
    def run(stage, command):
        status(stage)
        with open(a.root/(stage+'.log'),'ab') as log:
            subprocess.run(command,cwd=code,stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,check=True,timeout=600)
    prefix = ['/usr/bin/mpirun','-np',str(a.cores),'--bind-to','core',sys.executable,'-u',
              str(code/'posterior_pulse_bootstrap.py'),'--library',str(library)]
    try:
        events = sorted(x.name for x in a.audit.glob('tri_*'))
        run('real_pilot',prefix+['--source',str(a.catalogue/'inputs'),
            '--posterior',str(a.catalogue/'posterior'),'--output',str(a.pilot),
            '--draws','512','--optimize-seconds','2.','--resume','--events',*events])
        for experiment in ['noise','scatter']:
            inputs = a.root/(experiment+'_inputs')
            events = sorted(x.name for x in inputs.glob('sim_*'))
            if len(events)!=24:
                raise ValueError(f'{experiment}: expected 24 prepared simulations, got {len(events)}')
            output = a.root/(experiment+'_posterior')
            run(experiment,prefix+['--source',str(inputs),'--output',str(output),
                '--draws','192','--optimize-seconds','1.5','--events',*events])
            run(experiment+'_coverage',[sys.executable,str(code/'validate_pulse_posterior.py'),
                '--inputs',str(inputs),'--posterior',str(output),
                '--output',str(a.root/(experiment+'_coverage.h5'))])
        status('finished',True)
    except Exception as exc:
        status('failed',False,str(exc))
        raise


if __name__ == '__main__':
    main()
