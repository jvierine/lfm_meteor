"""Replay saved orbit states and validate the audited assembly without refitting."""
import time
import h5py
import numpy as np

from assemble_current_orbit_products import OUTPUT, FIELDS, REJECTED, REPAIRED, CURRENT_SOURCES, sha
import hashlib
from propagate_tristatic_orbits_1000d_classification import propagate_event
from pathlib import Path


def main():
    started = time.monotonic()
    with h5py.File(OUTPUT/'orbits.h5') as h, h5py.File(OUTPUT/'verification.h5', 'w') as out:
        out.attrs['orbit_sha256'] = sha(OUTPUT/'orbits.h5')
        ids = list(h['event_id'].asstr()[:])
        assert len(ids) == len(set(ids)) == 232
        assert REJECTED not in ids and REPAIRED in ids
        assert h['valid'][:].all()
        with h5py.File(OUTPUT/'refresh_manifest.h5') as manifest:
            assert ids == list(manifest['accepted_event_id'].asstr()[:])
            refreshed = set(manifest['refreshed_event_id'].asstr()[:])
        np.testing.assert_equal(h['bound_sample_count'][:] + h['hyperbolic_sample_count'][:], 100)
        for index, event in enumerate(ids):
            path = Path(h['source_audit'][event].attrs['snapshot'])
            if event in refreshed:
                with h5py.File(CURRENT_SOURCES/event/'block_bootstrap_states.h5') as b, h5py.File(path) as s:
                    draws = b['params'][:]
                    seed = int(hashlib.sha256(event.encode()).hexdigest()[:8], 16)
                    chosen = draws[np.random.default_rng(seed).choice(len(draws), 100, replace=False)]
                    np.testing.assert_equal(chosen, s['sample_params'][:])
            ev = propagate_event(path, 1000)
            g = out.create_group(event)
            for key in FIELDS:
                actual, expected = ev[key], h[key][index]
                if key == 'valid':
                    np.testing.assert_equal(actual, expected)
                else:
                    # Independent reruns of IAS15 can differ in the final few digits.
                    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7, equal_nan=True)
                    finite = np.isfinite(actual) & np.isfinite(expected)
                    if finite.any():
                        g.attrs[key+'_max_abs_delta'] = np.max(abs(actual[finite]-expected[finite]))
            if (index+1) % 25 == 0:
                print('replayed', index+1, '/', len(ids), flush=True)
        out.attrs.update(verified_events=len(ids), elapsed_seconds=time.monotonic()-started,
                         verification='Replay each saved state ensemble for 1000 days; compare every saved orbital field')
        print('PASS',len(ids),'ensembles',time.monotonic()-started,'seconds',flush=True)


if __name__ == '__main__':
    main()
