"""Require actual shared radar pulses, not merely nearby event intervals."""
import argparse
import hashlib
import concurrent.futures
from pathlib import Path
import h5py
import numpy as np

TOLERANCE_NS = 100000
MIN_SHARED = 6
MIN_SPAN_NS = 25000000
MIN_FRACTION = .5


def measure_overlap(times):
    times = [np.unique(np.asarray(t, dtype=np.int64)) for t in times]
    counts = np.array([len(t) for t in times], dtype=np.int64)
    shared = []
    used = [set(), set()]
    if np.all(counts):
        for t in times[0]:
            match = []
            for remote in times[1:]:
                j = int(np.searchsorted(remote, t))
                candidates = [k for k in (j-1, j) if 0 <= k < len(remote)]
                match.append(min(candidates, key=lambda k: abs(int(remote[k])-int(t))))
            stamps = [int(t)] + [int(times[s+1][j]) for s,j in enumerate(match)]
            if max(stamps)-min(stamps) <= TOLERANCE_NS and all(j not in used[s] for s,j in enumerate(match)):
                shared.append(int(t))
                for s,j in enumerate(match): used[s].add(j)
    count = len(shared)
    span = shared[-1]-shared[0] if count else 0
    fraction = count / int(counts.max()) if counts.max() else 0.
    accepted = count >= MIN_SHARED and span >= MIN_SPAN_NS and fraction >= MIN_FRACTION
    return accepted, count, span, fraction, counts


def audit_one(folder):
    with h5py.File(folder/"fit.h5") as h:
        t, station = h["time_ns"][:], h["station"][:]
    return folder.name, measure_overlap([t[station == s] for s in range(3)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("/gpfs/pub/data_other/juha/20261117_tristatic"))
    p.add_argument("--workers",type=int,default=1)
    args=p.parse_args()
    folders=sorted(f.parent for f in (args.root/"events").glob("tri_*/fit.h5"))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        result=list(pool.map(audit_one,folders))
    from analyze import save
    save(args.root/"association_selection.h5",dict(
        event_id=np.array([name for name, _ in result],dtype="S32"),
        accepted=np.array([r[0] for _,r in result],dtype=bool),
        shared_pulses=np.array([r[1] for _,r in result]),
        shared_span_ms=np.array([r[2]/1e6 for _,r in result]),
        minimum_station_fraction=np.array([r[3] for _,r in result]),
        station_pulses=np.array([r[4] for _,r in result])),
        dict(tolerance_ns=TOLERANCE_NS,min_shared_pulses=MIN_SHARED,
             min_shared_span_ns=MIN_SPAN_NS,min_station_fraction=MIN_FRACTION,
             associations_sha256=hashlib.sha256((args.root/"associations.h5").read_bytes()).hexdigest()))
    print("OVERLAP",len(result),"candidates",sum(r[0] for _,r in result),"retained",flush=True)


if __name__=="__main__": main()
