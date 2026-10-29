"""Bounded output-only refresh bridge between the compute and review hosts."""
import subprocess
import time
from pathlib import Path

ROOT='tristatic_voltage_reviewed_20261029_01'
local=Path(__file__).resolve().parent/'results'/ROOT
local.mkdir(parents=True,exist_ok=True)
for _ in range(240):
    try:
        subprocess.run(['rsync','-a','--include=*/','--include=plot.h5','--include=complete.h5',
            '--include=path_measurement_uncertainty.h5','--include=excluded_measurements.h5','--include=beam_rcs.h5','--include=error.log','--exclude=*',
            f'revontuli:/home/j/src/lfm_meteor/results/{ROOT}/',str(local)+'/'],check=True,timeout=90)
        subprocess.run(['rsync','-a',str(local)+'/',
            f'avaruus:/home/juha/src/lfm_meteor/results/{ROOT}/'],check=True,timeout=90)
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'synced',flush=True)
    except (subprocess.SubprocessError,OSError) as e:
        print(str(e),flush=True)
    time.sleep(30)
