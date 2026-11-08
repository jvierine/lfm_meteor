"""Recover numerical Figure 2 inputs and prepare native Rustplotlib data."""
import csv
import io
import subprocess

import h5py
import numpy as np

import prepare_native_paper_data as preparation


def main():
    root = preparation.ROOT / 'results/single_column_20261108'
    root.mkdir(exist_ok=True)
    source = '/home/j/src/lfm_meteor/results/all_detection_time_delay_velocity.csv'
    text = subprocess.check_output(['ssh', 'revontuli', 'cat', source], text=True)
    rows = list(csv.DictReader(io.StringIO(text)))
    with h5py.File(root / 'observations.h5', 'w') as h:
        h.attrs['source'] = 'revontuli:' + source
        for key in ['event_id', 'site']:
            h.create_dataset(key, data=[r[key] for r in rows], dtype=h5py.string_dtype())
        h['time_ns'] = np.array([int(r['time_ns']) for r in rows], dtype=np.int64)
        h['delay_us'] = np.array([float(r['delay_us']) for r in rows])
    preparation.OUT = root
    preparation.detection(root / 'observations.h5', root / 'noise.h5')


if __name__ == '__main__':
    main()
