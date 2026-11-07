"""Read-only remote SHA256 audit of the original Avaruus orbit source files."""
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from assemble_current_orbit_products import ROOT, CANONICAL, OUTPUT, REFRESHED, sha

REMOTE = Path('/home/juha/src/lfm_meteor')


def main():
    checks = []
    with h5py.File(OUTPUT/'orbits.h5') as product:
        product_sha = sha(OUTPUT/'orbits.h5')
        for event in product['event_id'].asstr()[:]:
            audit = product['source_audit'][event]
            snapshot = Path(audit.attrs['snapshot'])
            folder = CANONICAL/event
            with h5py.File(snapshot) as h:
                for kind, path, expected in [
                    ('input', REMOTE/'results'/CANONICAL.name/event/'input.h5', h.attrs['input_sha256']),
                    ('fit', REMOTE/'results'/CANONICAL.name/event/'fit.h5', h.attrs['fit_sha256']),
                    ('plot', REMOTE/'results'/CANONICAL.name/event/'plot.h5', audit.attrs['current_plot_sha256']),
                ]:
                    checks.append((event, kind, str(path), str(expected)))
                if not snapshot.is_relative_to(REFRESHED):
                    checks.append((event, 'snapshot', str(REMOTE/snapshot.relative_to(ROOT)), sha(snapshot)))
                    checks.append((event, 'propagated', str(REMOTE/Path(audit.attrs['propagated_source']).relative_to(ROOT)), audit.attrs['propagated_sha256']))
                for path, expected in zip(json.loads(h.attrs['source_files']), json.loads(h.attrs['source_sha256'])):
                    checks.append((event, 'uncertainty_source', path, expected))
    paths = sorted({row[2] for row in checks})
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'avaruus',
                             'xargs -0 sha256sum --'],
                            input='\0'.join(paths)+'\0', text=True, capture_output=True)
    actual = {}
    for line in result.stdout.splitlines():
        digest, path = line.split('  ', 1)
        actual[path] = digest
    with h5py.File(OUTPUT/'remote_source_audit.h5', 'w') as h:
        h.attrs.update(host='avaruus', checked_unix=time.time(), orbit_sha256=product_sha,
                       command='ssh avaruus xargs -0 sha256sum --',
                       remote_returncode=result.returncode, remote_stderr=result.stderr,
                       interpretation='Input/fit/currently selected full-source hashes against propagation snapshot; remote plot against staged current plot; reused remote snapshots/events checked; 40 newly propagated local ensembles checked by replay; no remote writes')
        dtype = h5py.string_dtype()
        for index, name in enumerate(('event_id', 'kind', 'remote_path', 'expected_sha256')):
            h.create_dataset(name, data=[row[index] for row in checks], dtype=dtype)
        h.create_dataset('actual_sha256', data=[actual.get(row[2], '') for row in checks], dtype=dtype)
        matches = np.array([actual.get(row[2]) == row[3] for row in checks])
        h['matches'] = matches
        h.attrs['all_match'] = bool(matches.all())
        for kind in sorted({row[1] for row in checks}):
            selected = np.array([row[1] == kind for row in checks])
            print(kind, 'matched', sum(matches & selected), '/', sum(selected), flush=True)
        for row, match in zip(checks, matches):
            if not match:
                print('MISMATCH', row[0], row[1], row[2], actual.get(row[2], 'MISSING'), flush=True)
        print('record', OUTPUT/'remote_source_audit.h5', flush=True)


if __name__ == '__main__':
    main()
