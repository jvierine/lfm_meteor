"""Count rapid, persistent RCS decreases in the frozen 232-event sample."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "results/submission_mass_remote_20261107"


def strongest_drop(time_ms, rcs, valid, window_ms, persistence=3):
    """Compare non-overlapping three-pulse medians without bridging gaps."""
    best = (np.nan, -1, -1)
    eligible = False
    for i in range(len(rcs) - 2 * persistence + 1):
        for j in range(i + persistence, len(rcs) - persistence + 1):
            separation = np.median(time_ms[j:j+persistence]) - np.median(time_ms[i:i+persistence])
            if separation > window_ms + 0.05:
                break
            stop = j + persistence
            if not np.all(valid[i:stop]):
                continue
            if np.any(np.diff(time_ms[i:stop]) > 7.55):
                continue
            eligible = True
            drop = float(np.median(rcs[i:i+persistence]) - np.median(rcs[j:stop]))
            if not np.isfinite(best[0]) or drop > best[0]:
                best = (drop, i, j)
    return eligible, best


def self_test():
    t = np.arange(6) * 5.0
    r = np.array([0., 0., 0., -12., -13., -13.])
    ok, (d, i, j) = strongest_drop(t, r, np.ones(6, bool), 15)
    assert ok and d == 13 and (i, j) == (0, 3)
    spike = np.array([0., 0., 0., -12., 0., 0.])
    assert strongest_drop(t, spike, np.ones(6, bool), 20)[1][0] == 0
    assert not strongest_drop(t, r, np.zeros(6, bool), 20)[0]
    assert not strongest_drop(np.array([0., 5., 10., 20., 25., 30.]), r, np.ones(6, bool), 30)[0]
    assert strongest_drop(t, -r, np.ones(6, bool), 20)[1][0] <= 0
    print("Five drop-detector checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/rcs_drop_survey_20261108_median")
    args = parser.parse_args()
    self_test()
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(RELEASE / "remote_hash_manifest.h5") as f:
        ids = f["accepted_event_id"].asstr()[:]
    assert len(ids) == 232 and len(set(ids)) == 232
    base = RELEASE / "snapshot/results/tristatic_cached_shrinking_20261029"
    events = []
    for event in ids:
        folder = base / event
        with h5py.File(folder / "beam_rcs.h5") as f:
            data = {k: f[k][:] for k in ("time_ns", "rcs_dbsm", "snr_db", "relative_gain_db")}
        with h5py.File(folder / "plot.h5") as f:
            accepted = f["time_ns"][:][f["station"][:] == 0]
        data["retained"] = np.isin(data["time_ns"], accepted)
        data["time_ms"] = (data["time_ns"] - data["time_ns"][0]) / 1e6
        assert np.all(np.diff(data["time_ns"]) > 0), event
        data["event_id"] = event
        data["beam_sha256"] = hashlib.sha256((folder / "beam_rcs.h5").read_bytes()).hexdigest()
        events.append(data)
    summaries = []
    output = args.output / "survey.h5"
    with h5py.File(output, "w") as out:
        out.attrs.update(
            source_program=Path(__file__).read_text(), catalogue_events=len(ids),
            source_release=str(RELEASE), primary_threshold_db=6., primary_window_ms=40.,
            primary_snr_min_db=20., primary_gain_min_db=-10., persistence_samples=3,
            definition="A decrease between non-overlapping three-pulse median RCS levels, with median times separated by no more than the window; all intervening samples meet gain/SNR cuts; no gaps longer than 7.55 ms",
            interpretation="Observed rapid RCS decreases, not a fragmentation or differential-ablation classification",
            threshold_rationale="6 dB is a factor-four RCS change; three-pulse median levels suppress isolated extrema; 40 ms covers broader observed declines; sensitivity to thresholds and window lengths is retained",
        )
        out.create_dataset("event_id", data=ids, dtype=h5py.string_dtype())
        eg = out.create_group("events")
        for data in events:
            g = eg.create_group(data["event_id"])
            for key in ("time_ms", "rcs_dbsm", "snr_db", "relative_gain_db", "retained"):
                g[key] = data[key]
            g.attrs["beam_sha256"] = data["beam_sha256"]
        for snr in (14., 20., 25.):
            for gain in (-10., -3.):
                for window in (15., 20., 30., 40., 50.):
                    scores, eligible, high, low = [], [], [], []
                    for data in events:
                        valid = (data["retained"] & np.isfinite(data["rcs_dbsm"])
                                 & (data["snr_db"] >= snr) & (data["relative_gain_db"] >= gain))
                        ok, (d, i, j) = strongest_drop(data["time_ms"], data["rcs_dbsm"], valid, window)
                        eligible.append(ok); scores.append(d); high.append(i); low.append(j)
                    name = f"snr{snr:g}_gain{abs(gain):g}_window{window:g}"
                    g = out.create_group("cases/" + name)
                    g.attrs.update(snr_min_db=snr, gain_min_db=gain, window_ms=window)
                    for key, values in (("eligible", eligible), ("drop_db", scores), ("high_index", high), ("low_index", low)):
                        g[key] = values
                    eligible = np.asarray(eligible)
                    for threshold in (4., 6., 8., 10., 15., 20.):
                        flagged = eligible & (np.asarray(scores) >= threshold)
                        summaries.append(dict(case=name, threshold_db=threshold,
                                              eligible=int(eligible.sum()), drops=int(flagged.sum()),
                                              fraction=float(flagged.sum() / max(eligible.sum(), 1))))
        out.attrs["summary_json"] = json.dumps(summaries)
    for row in summaries:
        if row["case"].startswith("snr20_gain10") or (row["case"].endswith("window30") and row["threshold_db"] == 6.):
            print(json.dumps(row))
    with h5py.File(output) as f:
        g = f["cases/snr20_gain10_window40"]
        for number in (12, 110):
            k = next(i for i, event in enumerate(ids) if event.startswith(f"tri_{number:04d}_"))
            print(ids[k], "drop", g["drop_db"][k], "indices", g["high_index"][k], g["low_index"][k])
        scores = g["drop_db"][:]
        eligible = g["eligible"][:]
        n = int(eligible.sum())
        count = int(np.sum(eligible & (scores >= 6)))
    sections = []
    for prefix, title, total in (("drop", "Drops of at least 6 dB", count),
                                 ("below_threshold", "Below the 6 dB threshold", n-count)):
        figures = "".join(f'<a href="{prefix}_{i:02}.pdf"><img loading="lazy" src="{prefix}_{i:02}.png" alt="{title}, page {i}"></a>'
                          for i in range(1, math.ceil(total/9)+1))
        sections.append(f"<h2>{title} ({total})</h2>{figures}")
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Sanya RCS decreases</title><style>
body{{font:16px system-ui;margin:28px auto;max-width:1100px;padding:0 20px;color:#222;background:white}}
img{{display:block;width:100%;height:auto;margin:20px 0 44px}}p{{max-width:850px;line-height:1.5}}
</style><h1>Sanya RCS decreases</h1>
<p>{count} of {n} assessable events ({100*count/n:.1f}%). A drop is a decrease of at least 6 dB between
two non-overlapping three-pulse median RCS levels, with median times no more than 40 ms apart.
All intervening samples must be retained, have SNR at least 20 dB and one-way beam gain within 10 dB
of its peak. Gaps longer than 7.55 ms are excluded. The remaining {len(ids)-n} catalogue events
have insufficient qualifying coverage for this test.</p>
<p>Red: qualifying drop events. Blue: events below the threshold. Gray: other in-beam measurements.
The black dashed segment joins the two median levels giving the largest decrease.
Panel labels give the event number and this decrease. Click a sheet for its PDF.</p>
<p>This is a threshold-based description of observed RCS changes, not a classification of their physical cause.</p>
{''.join(sections)}</html>'''
    (args.output / "index.html").write_text(html)
    print(output)


if __name__ == "__main__":
    main()
