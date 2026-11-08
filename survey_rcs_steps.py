"""Detect abrupt RCS decreases rather than cumulative smooth fades."""
from pathlib import Path
import hashlib
import json
import math

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/rcs_drop_survey_20261108_median/survey.h5"
OUT = ROOT / "results/rcs_drop_survey_20261108_abrupt"


def abrupt_drop(t, rcs, valid):
    """Consecutive-pulse fall confirmed by a second low pulse; recovery is allowed."""
    best = (np.nan, -1, -1)
    eligible = False
    for i in range(len(rcs) - 2):
        if not np.all(valid[i:i+3]) or np.any(np.diff(t[i:i+3]) > 7.55):
            continue
        eligible = True
        # Both post-transition samples must remain below the pre-transition level.
        score = float(rcs[i] - max(rcs[i+1], rcs[i+2]))
        if not np.isfinite(best[0]) or score > best[0]:
            best = (score, i, i+1)
    return eligible, best


def self_test():
    t = np.arange(6) * 5.
    valid = np.ones(6, bool)
    assert abrupt_drop(t, np.array([0., 0., -6., -6., 0., 0.]), valid)[1][0] == 6.
    assert abrupt_drop(t, np.array([0., 0., -6., 0., 0., 0.]), valid)[1][0] == 0.
    assert abrupt_drop(t, -np.arange(6, dtype=float), valid)[1][0] == 1.
    assert not abrupt_drop(t, np.zeros(6), np.zeros(6, bool))[0]
    assert not abrupt_drop(np.array([0., 10., 15.]), np.array([0., -6., -6.]), np.ones(3, bool))[0]


def main():
    self_test()
    OUT.mkdir(exist_ok=True)
    rows = []
    with h5py.File(SOURCE) as source, h5py.File(OUT / "survey.h5", "w") as out:
        ids = source["event_id"].asstr()[:]
        source.copy("event_id", out)
        source.copy("events", out)
        out.attrs.update(
            source_program=Path(__file__).read_text(), source_hdf5=str(SOURCE),
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            primary_threshold_db=5., primary_window_ms=5., primary_snr_min_db=20.,
            primary_gain_min_db=-10., catalogue_events=len(ids),
            detector="abrupt", persistence_samples=2,
            definition="At least 5 dB decrease between consecutive nominal 5 ms pulses, confirmed by the next pulse remaining at least 5 dB below the pre-drop level. All three samples retained, SNR >=20 dB, one-way relative gain >=-10 dB. Gaps longer than 7.55 ms excluded. Later recovery permitted.",
            interpretation="Observed abrupt RCS decreases; not an inferred fragmentation frequency",
        )
        for snr in (14., 20., 25.):
            for gain in (-10., -3.):
                eligible, scores, high, low = [], [], [], []
                for event in ids:
                    g = source["events/" + event]
                    rcs = g["rcs_dbsm"][:]
                    valid = (g["retained"][:] & np.isfinite(rcs)
                             & (g["snr_db"][:] >= snr) & (g["relative_gain_db"][:] >= gain))
                    ok, (score, i, j) = abrupt_drop(g["time_ms"][:], rcs, valid)
                    eligible.append(ok); scores.append(score); high.append(i); low.append(j)
                case = f"snr{snr:g}_gain{abs(gain):g}_window5"
                g = out.create_group("cases/" + case)
                for key, data in (("eligible", eligible), ("drop_db", scores), ("high_index", high), ("low_index", low)):
                    g[key] = data
                for threshold in (4., 5., 6., 8., 10.):
                    count = int(np.sum(np.asarray(eligible) & (np.asarray(scores) >= threshold)))
                    rows.append(dict(case=case, threshold_db=threshold, eligible=int(sum(eligible)), drops=count))
        out.attrs["summary_json"] = json.dumps(rows)
        primary = out["cases/snr20_gain10_window5"]
        eligible, scores = primary["eligible"][:], primary["drop_db"][:]
        expected = {12: True, 43: True, 110: True, 128: False, 188: False, 211: False}
        for number, label in expected.items():
            i = next(i for i, event in enumerate(ids) if event.startswith(f"tri_{number:04d}_"))
            result = bool(eligible[i] and scores[i] >= 5.)
            assert result == label, (number, scores[i], label)
            print(f"Check {number:04d}: {scores[i]:.3f} dB, drop={result}")
        n = int(eligible.sum())
        count = int(np.sum(eligible & (scores >= 5.)))
    for row in rows:
        print(json.dumps(row))
    print(f"FINAL: {count}/{n} = {100*count/n:.2f}%")
    sections = []
    for name, title, total in (("drop", "Abrupt drops", count), ("below_threshold", "No qualifying abrupt drop", n-count)):
        imgs = "".join(f'<a href="{name}_{i:02}.pdf"><img loading="lazy" src="{name}_{i:02}.png" alt="{title}, sheet {i}"></a>'
                       for i in range(1, math.ceil(total/9)+1))
        sections.append(f"<h2>{title}: {total}</h2>{imgs}")
    (OUT / "index.html").write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Abrupt Sanya RCS drops</title><style>body{{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#222}}img{{display:block;width:100%;margin:25px 0 45px}}p{{max-width:900px;line-height:1.5}}</style>
<h1>Abrupt RCS drops: {count} of {n} assessable events ({100*count/n:.1f}%)</h1>
<p>At least 5 dB between consecutive 5 ms pulses, confirmed by a second low pulse.
Subsequent recovery is allowed. All three samples must have SNR at least 20 dB,
be retained in the trajectory fit, and be within 10 dB of the one-way beam peak.
The other {len(ids)-n} catalogue events have insufficient qualifying coverage.</p>
<p>Red: abrupt-drop events. Blue: no qualifying abrupt drop. Gray: other in-beam samples.
Black segments mark the tested transition and black dots the pre-drop and two confirming samples.
The panel label gives the event number and largest confirmed single-pulse decrease.</p>
<p>Events 0012, 0043, and 0110 pass; smooth declines 0128, 0188, and 0211 do not.
This is an observational classification, not a physical-cause classification.</p>{''.join(sections)}</html>''')


if __name__ == "__main__":
    main()
