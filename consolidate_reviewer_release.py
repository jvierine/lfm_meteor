"""Consolidate frozen supporting data without modifying the original release."""
import argparse
import hashlib
import io
from pathlib import Path, PurePosixPath
import tarfile

import h5py
import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replication', type=Path, required=True)
    args = parser.parse_args()
    args.release = args.release.resolve()
    args.output = args.output.resolve()
    args.replication = args.replication.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    records, omissions, known, destinations = [], [], {}, set()

    def add(out, name, data, origin):
        p = PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts
        sha = digest(data)
        key = (out.name, name)
        assert key not in destinations, key
        destinations.add(key)
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o644
        out.addfile(info, io.BytesIO(data))
        records.append((out.name, name, sha, len(data), origin))
        known.setdefault(sha, name)

    def copy_archive(out, source):
        with tarfile.open(args.release/source) as old:
            for m in old:
                if not m.isfile():
                    continue
                if Path(m.name).name in ('PARENT_NASH_HANDOFF.txt', 'AUDIT.txt'):
                    omissions.append((source, m.name, 'internal handoff/audit narrative'))
                    continue
                if '__pycache__' in PurePosixPath(m.name).parts or m.name.endswith('.pyc'):
                    omissions.append((source, m.name, 'generated Python cache'))
                    continue
                add(out, m.name, old.extractfile(m).read(), source+':'+m.name)
        print('COPIED', source, flush=True)

    with tarfile.open(args.output/'sanya-raw-data.tar.gz', 'w:gz', compresslevel=1) as out:
        for source in ('cuts-core/raw-cuts.tar.gz', 'cuts-core/detections.tar.gz'):
            copy_archive(out, source)
    with tarfile.open(args.output/'sanya-results.tar.gz', 'w:gz', compresslevel=1) as out:
        for source in ('catalog-core/fits.tar.gz', 'catalog-core/lookup.tar.gz',
                       'uncertainty-current/uncertainty.tar.gz', 'orbits-refreshed/orbits.tar.gz',
                       'l1-complete-archive/l1-velocity.tar.gz', 'yield-current/yield.tar.gz',
                       'mass-derived-final/mass-derived.tar.gz', 'kinematics-final/kinematics.tar.gz'):
            copy_archive(out, source)
        # Preserve unique selection/uncertainty inputs, but not duplicate event trees.
        source = 'mass-evidence-final/mass-evidence.tar.gz'
        unique = 0
        with tarfile.open(args.release/source) as old:
            for m in old:
                if not m.isfile():
                    continue
                data = old.extractfile(m).read()
                sha = digest(data)
                if sha in known:
                    omissions.append((source, m.name, 'byte-identical to '+known[sha]))
                elif m.name.endswith('.h5'):
                    name = m.name.removeprefix('mass-evidence/snapshot/')
                    if name == m.name:
                        name = 'selection-inputs/'+Path(m.name).name
                    else:
                        name = 'selection-inputs/'+name
                    add(out, name, data, source+':'+m.name)
                    unique += 1
                else:
                    omissions.append((source, m.name, 'internal non-scientific evidence'))
        print('UNIQUE SUPPORT INPUTS', unique, flush=True)
        for kind in ('reproduced', 'reproduced-raw-cuts'):
            p = args.replication/kind/'replication.h5'
            add(out, 'replication/'+kind+'.h5', p.read_bytes(), str(p))
    with tarfile.open(args.output/'sanya-code.tar.gz', 'w:gz', compresslevel=1) as out:
        copy_archive(out, 'code-committed/code-final.tar.gz')
        p = Path(__file__).parent/'reproduction/replicate_event.py'
        add(out, 'source/reproduction/replicate_event.py', p.read_bytes(), str(p))
        p = Path(__file__)
        add(out, 'source/lfm_meteor/consolidate_reviewer_release.py', p.read_bytes(), str(p))
        inventory = args.output/'inventory.h5'
        with h5py.File(inventory, 'w') as h:
            s = h5py.string_dtype('utf-8')
            for i, key in enumerate(('archive', 'path', 'sha256', 'bytes', 'origin')):
                values = [r[i] for r in records]
                h[key] = values if key == 'bytes' else np.asarray(values, dtype=s)
            for i, key in enumerate(('old_archive', 'old_path', 'reason')):
                h.create_dataset('omissions/'+key, data=np.asarray([r[i] for r in omissions], dtype=s))
            h.attrs['policy'] = 'Original scientific bytes retained; duplicate evidence omitted by SHA256; unique evidence HDF5 retained as selection inputs'
        add(out, 'source/reproduction/inventory.h5', inventory.read_bytes(), 'consolidated member manifest')
        inventory.unlink()

    expected = {(r[0], r[1]): r[2] for r in records}
    for archive in sorted(args.output.glob('*.tar.gz')):
        with tarfile.open(archive) as h:
            for m in h:
                if m.isfile():
                    assert digest(h.extractfile(m).read()) == expected.pop((str(archive), m.name)), m.name
        print('VERIFIED', archive.name, archive.stat().st_size, flush=True)
    assert not expected
    raw_count = sum(r[1].startswith('L0/') and r[1].endswith('.h5') for r in records)
    uncertainty = sum(r[1].startswith('uncertainty/') for r in records)
    assert raw_count == 5215, raw_count
    assert uncertainty == 649, uncertainty
    readme = Path(__file__).parent/'reproduction/README.txt'
    target = args.output/'README.txt'
    target.write_bytes(readme.read_bytes())
    with h5py.File(args.output/'checksums.h5', 'w') as h:
        paths = sorted(args.output.iterdir())
        paths = [p for p in paths if p.name != 'checksums.h5']
        for key, values in [('filename', [p.name for p in paths]),
                            ('sha256', [file_hash(p) for p in paths]),
                            ('md5', [file_hash(p, 'md5') for p in paths])]:
            h[key] = np.asarray(values, dtype=h5py.string_dtype())
        h['bytes'] = [p.stat().st_size for p in paths]
        h.attrs.update(record_id='24037714', raw_cut_count=raw_count,
                       uncertainty_source_count=uncertainty, scientific_member_hashes_verified=True,
                       intended_state='Unpublished draft; public release only after paper acceptance')
    print('COMPLETE', len(records), 'members;', len(omissions), 'omissions; five downloads', flush=True)


if __name__ == '__main__':
    main()
