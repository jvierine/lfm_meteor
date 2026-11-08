"""Compare speed, orbital class and measured masses by abrupt-RCS-drop status."""
from pathlib import Path
import hashlib
import json

import h5py
import numpy as np

from propagate_tristatic_orbits_1000d_classification import jopek_williams_criteria

ROOT = Path(__file__).resolve().parent
SURVEY = ROOT / "results/rcs_drop_survey_20261108_abrupt/survey.h5"
SPEEDS = ROOT / "results/current_kinematics_20261107/radiants.h5"
ORBITS = ROOT / "results/tristatic_shrinking_orbits_current_20261107/orbits.h5"
MASSES = ROOT / "results/submission_mass_remote_20261107/selection.h5"


def by_id(path, id_key, value_key):
    with h5py.File(path) as f:
        return dict(zip(f[id_key].asstr()[:], f[value_key][:]))


def main():
    with h5py.File(SURVEY) as f:
        ids = f["event_id"].asstr()[:]
        g = f["cases/snr20_gain10_window5"]
        eligible = g["eligible"][:]
        drops = g["drop_db"][:] >= 5.
    velocity = by_id(SPEEDS, "event_id", "speed_km_s")
    dynamic = by_id(MASSES, "dynamic/event_id", "dynamic/mass_kg")
    rcs = by_id(MASSES, "rcs/event_id", "rcs/mass_kg")
    with h5py.File(ORBITS) as f:
        oid = f["event_id"].asstr()[:]
        a, e, inc, period = (f[k][:] for k in ("a_au", "e", "i_deg", "period_years"))
        valid = f["valid"][:] & (e < 1.)
        criteria = jopek_williams_criteria(a, e, inc, period)
        keys = list(criteria)
        probabilities = np.array([np.sum(c & valid, axis=1)/np.sum(valid, axis=1)
                                  for c in criteria.values()]).T
        expected = f["classification_summary/cometary_event_mean_fraction"][:5]
        np.testing.assert_allclose(probabilities.mean(axis=0), expected, atol=1e-12)
        combined = probabilities.mean(axis=1)
        np.testing.assert_allclose(combined.mean(), f["classification_summary/cometary_event_mean_fraction"][-1])
    orbital = {event: probabilities[i] for i, event in enumerate(oid)}
    output = SURVEY.parent / "population_comparison.h5"
    summaries = {}
    with h5py.File(output, "w") as out:
        out.attrs.update(source_program=Path(__file__).read_text(),
                         classification="Equal event weights, bound orbit samples, arithmetic mean of five Jopek-Williams criteria",
                         sample="Only the 193 events assessable by the abrupt-drop screen; masses use independently quality-selected available estimates")
        for name, path in (("survey", SURVEY), ("speed", SPEEDS), ("orbit", ORBITS), ("mass", MASSES)):
            out.attrs[name+"_source"] = str(path)
            out.attrs[name+"_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        out["criterion_key"] = np.array(keys, dtype=h5py.string_dtype())
        for name, mask in (("drop", eligible & drops), ("no_drop", eligible & ~drops)):
            selected = ids[mask]
            g = out.create_group(name)
            g["event_id"] = np.array(selected, dtype=h5py.string_dtype())
            speed = np.array([velocity[event] for event in selected])
            class_prob = np.array([orbital[event] for event in selected])
            g["initial_speed_km_s"] = speed
            g["cometary_probability_by_criterion"] = class_prob
            summary = dict(events=len(selected), mean_initial_speed_km_s=float(speed.mean()),
                           median_initial_speed_km_s=float(np.median(speed)),
                           cometary_fraction=float(class_prob.mean()),
                           asteroidal_fraction=float(1-class_prob.mean()),
                           cometary_fraction_by_criterion=class_prob.mean(axis=0).tolist())
            for kind, lookup in (("dynamic", dynamic), ("rcs", rcs)):
                mass_ids = [event for event in selected if event in lookup]
                values = np.array([lookup[event] for event in mass_ids])
                assert np.isfinite(values).all() and (values > 0).all()
                mg = g.create_group(kind)
                mg["event_id"] = np.array(mass_ids, dtype=h5py.string_dtype())
                mg["mass_kg"] = values
                summary[kind] = dict(events=len(values), mean_mass_kg=float(values.mean()),
                                     median_mass_kg=float(np.median(values)),
                                     largest_fraction=float(values.max()/values.sum()))
                for k, v in summary[kind].items():
                    mg.attrs[k] = v
            for k in ("events", "mean_initial_speed_km_s", "median_initial_speed_km_s", "cometary_fraction", "asteroidal_fraction"):
                g.attrs[k] = summary[k]
            summaries[name] = summary
        out.attrs["summary_json"] = json.dumps(summaries)
    print(json.dumps(summaries, indent=2))
    print(output)


if __name__ == "__main__":
    main()
