"""Fit the Figure 8c paired masses in log space, including both-axis errors."""
from pathlib import Path
import argparse
import hashlib
import json

import h5py
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/current_kinematics_20261107/sanya_rcs_from_snr_article.h5"
CANONICAL = ROOT / "results/submission_mass_remote_20261107/joint_fft_mass_distribution.h5"
PIVOT = 1e-9


def fit(x, y, sx, sy, initial=None):
    """Forward relation y=a+b*x, Gaussian log errors and intrinsic y scatter."""
    if initial is None:
        b, a = np.polyfit(x, y, 1)
        initial = [a, b, np.log(max(np.std(y-a-b*x), .02))]

    def objective(q):
        a, b, log_scatter = q
        intrinsic = np.exp(2*log_scatter)
        variance = sy**2 + b*b*sx**2 + intrinsic
        residual = y-a-b*x
        term = 1/variance-residual**2/variance**2
        value = .5*np.sum(np.log(variance)+residual**2/variance)
        gradient = [-np.sum(residual/variance),
                    np.sum(-residual*x/variance+b*sx**2*term),
                    intrinsic*np.sum(term)]
        return value, np.array(gradient)

    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(-5., 5.), (-3., 3.), (-8., 2.)],
                      options=dict(maxiter=1000, ftol=1e-12, gtol=1e-7))
    if not result.success and np.linalg.norm(result.jac) > 1e-3:
        raise RuntimeError(result.message)
    return result.x


def self_test():
    rng = np.random.default_rng(1729)
    truth = rng.uniform(-2, 2, 2000)
    sx = np.full(len(truth), .08)
    sy = np.full(len(truth), .06)
    x = truth+rng.normal(size=len(truth))*sx
    y = -.35+.65*truth+rng.normal(size=len(truth))*np.hypot(sy, .2)
    q = fit(x, y, sx, sy)
    np.testing.assert_allclose(q[:2], [-.35, .65], atol=.025)
    assert abs(np.exp(q[2])-.2) < .025
    shifted = fit(x+1.3, y, sx, sy)
    np.testing.assert_allclose(shifted[1:], q[1:], atol=1e-5)
    np.testing.assert_allclose(shifted[0]+1.3*shifted[1], q[0], atol=1e-5)
    print("Synthetic slope, scatter and pivot-invariance checks passed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--output", type=Path, default=ROOT / "results/rcs_mass_powerlaw_20261108")
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--mass-distribution", type=Path, default=CANONICAL)
    args = p.parse_args()
    self_test()
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source) as source, h5py.File(args.mass_distribution) as canonical:
        group = source["mass_comparison"]
        ids = group["event_id"].asstr()[:]
        assert len(ids) >= 4 and len(set(ids)) == len(ids)
        np.testing.assert_array_equal(ids, canonical["rcs_event_id"].asstr()[:])
        index = dict(zip(canonical["event_id"].asstr()[:], canonical["initial_mass_kg"][:]))
        mx, my = group["dynamic_mass_kg"][:], group["rcs_mass_kg"][:]
        np.testing.assert_array_equal(mx, [index[event] for event in ids])
        np.testing.assert_array_equal(my, canonical["rcs_mass_kg"][:])
        x, y = np.log10(mx/PIVOT), np.log10(my/PIVOT)
        sigma = lambda lo, hi: (np.log10(group[hi][:])-np.log10(group[lo][:]))/3.919927969080108
        sx = sigma("dynamic_lo95_kg", "dynamic_hi95_kg")
        sy = sigma("delta_mass_lo95_noise_only_kg", "delta_mass_hi95_noise_only_kg")
        assert np.all(np.isfinite(np.r_[x, y, sx, sy])) and np.all(np.r_[sx, sy] > 0)
        best = fit(x, y, sx, sy)
        rng = np.random.default_rng(20261108)
        draws = np.empty((args.draws, 3))
        for i in range(args.draws):
            selected = rng.integers(0, len(x), len(x))
            draws[i] = fit(x[selected], y[selected], sx[selected], sy[selected], best)
        grid = np.linspace(x.min(), x.max(), 201)
        lines = PIVOT*10**(draws[:, :1]+draws[:, 1:2]*grid)
        low, high = np.quantile(lines, [.025, .975], axis=0)
        summary = dict(events=len(ids), pivot_mass_kg=PIVOT,
                       normalization_kg=float(PIVOT*10**best[0]), slope=float(best[1]),
                       intrinsic_scatter_dex=float(np.exp(best[2])),
                       normalization_95_kg=(PIVOT*10**np.quantile(draws[:, 0], [.025, .975])).tolist(),
                       slope_95=np.quantile(draws[:, 1], [.025, .975]).tolist())
        target = args.output / "figure8.h5"
        with h5py.File(target, "w") as out:
            source.copy("pulses", out)
            source.copy("mass_comparison", out)
            g = out.create_group("mass_powerlaw")
            for key, value in summary.items():
                g.attrs[key] = value
            g.attrs.update(source_program=Path(__file__).read_text(),
                           data_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                           source_data=str(args.source), bootstrap_events=args.draws,
                           model="log10(m_rcs/pivot)=alpha+beta*log10(m_dyn/pivot), intrinsic vertical Gaussian scatter",
                           likelihood="variance=sigma_y^2+beta^2*sigma_x^2+scatter^2; includes log variance term",
                           error_model="Logarithmic 95% interval widths divided by 3.919927969; independent x/y measurement errors",
                           band="Pointwise 95% uncertainty of fitted mean log relation; excludes intrinsic population scatter and shared calibration systematics")
            for key, values in (("event_id", np.asarray(ids, dtype=h5py.string_dtype())),
                                ("log_dynamic_sigma", sx), ("log_rcs_sigma", sy),
                                ("best_parameters", best), ("parameter_samples", draws),
                                ("dynamic_mass_kg", PIVOT*10**grid),
                                ("rcs_mass_kg", PIVOT*10**(best[0]+best[1]*grid)),
                                ("rcs_lower95_kg", low), ("rcs_upper95_kg", high)):
                g[key] = values
            for name in ("pulses", "mass_comparison"):
                for key, values in source[name].items():
                    if isinstance(values, h5py.Dataset):
                        np.testing.assert_array_equal(values[()], out[name+"/"+key][()])
    print(json.dumps(summary, indent=2))
    print(target)


if __name__ == "__main__":
    main()
