"""Record finite-simulation coverage checks, not a total physical-uncertainty claim."""
import argparse
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy.stats import binomtest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "posterior", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    names, coverage, biases, effective, accepted = [], [], [], [], []
    for folder in sorted(a.input.glob("sim_*")):
        names.append(folder.name)
        with h5py.File(folder/"truth.h5") as h:
            truth = h["params"][:]
        with h5py.File(a.posterior/folder.name/"posterior.h5") as h:
            q = h["parameter_quantiles"][:]
            coverage.append((q[0] <= truth) & (truth <= q[2]))
            biases.append((h["posterior_mean"][:]-truth)/np.sqrt(np.diag(h["posterior_covariance"][:])))
            effective.append(h.attrs["importance_ess"])
            accepted.append(bool(h.attrs["convergence_verified"]))
    coverage, biases = np.asarray(coverage), np.asarray(biases)
    mass = int(np.sum(coverage[:, 6]))
    test = binomtest(mass, len(names), .95, alternative="less")
    interval = binomtest(mass, len(names)).proportion_ci(method="exact")
    with h5py.File(a.output, "w") as h:
        h.attrs.update(simulations=len(names), verified_posteriors=sum(accepted),
            scope="Noise-only simulations at one measured geometry and physical truth; not total physical uncertainty",
            mass_covered=mass, mass_coverage_fraction=mass/len(names),
            nominal_coverage=.95, mass_undercoverage_pvalue=test.pvalue,
            mass_coverage_binomial_lower95=interval.low, mass_coverage_binomial_upper95=interval.high,
            source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            conditional_recovery_check_passed=all(accepted) and test.pvalue >= .05,
            source_program=Path(__file__).read_text())
        h["case_id"] = np.asarray(names, dtype=h5py.string_dtype())
        h["parameter_covered"] = coverage
        h["parameter_coverage_counts"] = np.sum(coverage, axis=0)
        h["standardized_posterior_mean_error"] = biases
        h["mean_standardized_error"] = np.mean(biases, axis=0)
        h["importance_ess"] = effective
        h["posterior_verified"] = accepted
    print(f"Verified {sum(accepted)}/{len(names)}; mass coverage {mass}/{len(names)}; binomial p={test.pvalue:.3f}")


if __name__ == "__main__":
    main()
