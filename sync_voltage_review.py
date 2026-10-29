"""Bounded output-only refresh bridge between the compute and review hosts."""
import subprocess
import time
from pathlib import Path

ROOTS=('tristatic_voltage_reviewed_20261029_01','tristatic_voltage_all_uncertainty_20261029')
for _ in range(1440):
    for root in ROOTS:
        local=Path(__file__).resolve().parent/'results'/root
        local.mkdir(parents=True,exist_ok=True)
        try:
            migrated=root=='tristatic_voltage_all_uncertainty_20261029'
            source=f'avaruus:/home/juha/src/lfm_meteor/results/{root}/' if migrated else f'revontuli:/home/j/src/lfm_meteor/results/{root}/'
            subprocess.run(['rsync','-a','--include=*/','--include=plot.h5','--include=complete.h5',
                '--include=path_measurement_uncertainty.h5','--include=excluded_measurements.h5','--include=beam_rcs.h5','--include=error.log','--exclude=*',
                source,str(local)+'/'],check=True,timeout=90)
            if not migrated:
                subprocess.run(['rsync','-a',str(local)+'/',
                    f'avaruus:/home/juha/src/lfm_meteor/results/{root}/'],check=True,timeout=90)
            print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),root,'synced',flush=True)
        except (subprocess.SubprocessError,OSError) as e:
            print(str(e),flush=True)
    time.sleep(30)
