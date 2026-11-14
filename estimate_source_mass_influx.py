"""Partition the existing influx calculation by saved orbital membership.

No new scattering calculations or trajectory fits are performed. Membership is
the mean cometary classification across five criteria and saved orbital draws;
it is not a calibrated probability of a physical parent-body origin.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np


def partition(contributions, membership):
    """Keep unavailable classifications separate and conserve total mass rate."""
    values = np.asarray(contributions, dtype=float)
    membership = np.asarray(membership, dtype=float)
    if values.shape[-1] != len(membership):
        raise ValueError("Membership must align with the event axis")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Contributions must be finite and nonnegative")
    known = np.isfinite(membership)
    if np.any((membership[known] < 0) | (membership[known] > 1)):
        raise ValueError("Membership must lie between zero and one")
    result = np.stack([
        np.sum(values[..., known] * membership[known], axis=-1),
        np.sum(values[..., known] * (1 - membership[known]), axis=-1),
        np.sum(values[..., ~known], axis=-1),
    ], axis=-1)
    np.testing.assert_allclose(result.sum(axis=-1), values.sum(axis=-1))
    return result


def calculate(ids, mass, bounds, weights, speed, catalogue, membership, exponents):
    ids = list(ids)
    if len(set(ids)) != len(ids) or not set(ids).issubset(catalogue):
        raise ValueError("Measured events must be unique catalogue members")
    missing = sorted(set(catalogue) - set(ids))
    all_ids = ids + missing
    votes = np.array([membership.get(event, np.nan) for event in all_ids])
    rates = []
    envelopes = []
    measured = []
    for exponent in exponents:
        w = weights * (72 / speed) ** (exponent - 3)
        c = mass * w
        c_bounds = bounds * w[:, None]
        full = np.r_[c, np.full(len(missing), np.median(c))]
        full_bounds = np.concatenate([
            c_bounds, np.tile(np.median(c_bounds, axis=0), (len(missing), 1))
        ])
        rates.append(partition(full, votes))
        envelopes.append(partition(full_bounds.T, votes))
        measured.append(partition(c, votes[:len(ids)]))
    return all_ids, votes, np.array(rates), np.array(envelopes), np.array(measured)


def main():
    base = Path(__file__).resolve().parent
    root = base / "results/tristatic_calibrated_likelihood_20261109"
    orbits = base / "results/tristatic_seven_parameter_catalogue_20261108/orbits/orbits.h5"
    output = root / "products/source_mass_influx.h5"
    sources = [orbits, root / "products/mass_influx.h5",
               root / "products/mass_distribution.h5", root / "current_mass_influx.h5",
               root / "products/imputed_mass_influx.h5"]
    with h5py.File(orbits) as h:
        membership = dict(zip(h["event_id"].asstr()[:], h["cometary_probability_bound"][:]))
    exponents = np.linspace(3, 3.5, 101)
    with h5py.File(sources[1]) as dyn, h5py.File(sources[2]) as distribution, \
            h5py.File(sources[3]) as old, h5py.File(sources[4]) as canonical, \
            h5py.File(output, "w") as out:
        catalogue = canonical["catalogue_event_id"].asstr()[:].tolist()
        out.attrs.update(
            membership="Mean of five Jopek-Williams criteria over saved bound orbital draws",
            missing_mass="Global median weighted contribution at each exponent; same as total influx",
            missing_orbit="Unclassified; never reassigned to a source",
            fraction_range="Nominal-mass sensitivity over p=3..3.5, not a confidence interval",
            envelope="Individual mass-bound endpoint sensitivity; not a global confidence interval",
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        out["p"] = exponents
        out["population"] = np.array(["cometary", "asteroidal", "unclassified"], dtype=h5py.string_dtype())
        provenance = out.create_group("provenance")
        for i, path in enumerate(sources):
            provenance.attrs[str(i)] = str(path)
            provenance.attrs[f"{i}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rcs = old["rcs"]
        np.testing.assert_array_equal(dyn["event_id"].asstr()[:], distribution["event_id"].asstr()[:])
        inputs = {
            "dynamic": (dyn["event_id"].asstr()[:], dyn["mass_kg"][:],
                        np.column_stack([dyn["mass_lo_kg"][:], dyn["mass_hi_kg"][:]]),
                        dyn["beam_sampling/weight_ton_day_per_kg"][:], distribution["initial_speed_km_s"][:]),
            "rcs": (rcs["event_id"].asstr()[:], rcs["mass_kg"][:],
                    np.array([np.quantile(rcs["input_mass_draws/" + e][:], [.025, .975])
                              for e in rcs["event_id"].asstr()[:]]),
                    rcs["plus_altitude_heuristic/weight_per_day"][:] / 1000, rcs["speed_km_s"][:]),
        }
        for name, args in inputs.items():
            ids, votes, rates, envelopes, measured = calculate(
                *args, catalogue, membership, exponents)
            group = out.create_group(name)
            group["event_id"] = np.array(ids, dtype=h5py.string_dtype())
            group["cometary_membership"] = votes
            group["rate_ton_day"] = rates
            group["measured_subset_rate_ton_day"] = measured
            group["mass_bound_envelopes_ton_day"] = envelopes
            fractions = rates[:, :2] / rates[:, :2].sum(axis=1, keepdims=True)
            group["classified_mass_fractions"] = fractions
            group["classified_mass_fraction_ranges"] = np.stack([fractions.min(axis=0), fractions.max(axis=0)])
            group["unclassified_fraction_of_total"] = rates[:, 2] / rates.sum(axis=1)
            target = canonical[("rcs/" if name == "rcs" else "") + "beam_sampling/exponent_sensitivity/range_ton_day"][:]
            np.testing.assert_allclose([envelopes[0, 0].sum(), envelopes[-1, 1].sum()], target)
            print(name, "p endpoints rates [cometary, asteroidal, unclassified]:", rates[[0, -1]])
            print("classified mass fractions:", group["classified_mass_fraction_ranges"][:])
            print("unclassified fraction:", group["unclassified_fraction_of_total"][:][[0, -1]])
    print(output)


if __name__ == "__main__":
    main()
