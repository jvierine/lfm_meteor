"""Fit an effective Sanya pointing to a binned main lobe, not its sidelobes."""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

from derived import SANYA_AZ_EL, beam_power, directions_from_offsets


def fit(count, east, north, radius):
    use = (east**2 + north**2 <= radius**2) & np.isfinite(count)
    directions = directions_from_offsets(east[use], north[use])
    k = count[use]

    def loss(q):
        axis = directions_from_offsets(q[0], q[1])
        lam = np.exp(q[2]) * beam_power(directions, axis)**np.exp(q[3]) + np.exp(q[4])
        return np.sum(lam - xlogy(k, lam))

    peak = float(k.max())
    bounds = [(-.55,.55),(-.55,.55),(-5,15),(-2,2),(-12,8)]
    fits = [minimize(loss, [0,y,np.log(peak),-.6,-2], bounds=bounds,
                     method="L-BFGS-B") for y in (0.,.3)]
    result = min((r for r in fits if r.success), key=lambda r: r.fun)
    return result.x[:2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    windows = np.array([.65,.75,.85,.95,1.05])
    for i,path in enumerate(a.inputs):
        with h5py.File(path) as f:
            x,y = (f[k][:] for k in ("east_edges","north_edges"))
            count = f["counts"][:].reshape(len(y)-1,len(x)-1)
            count = np.nan_to_num(count,nan=0.)
            east,north = np.meshgrid((x[:-1]+x[1:])/2,(y[:-1]+y[1:])/2)
            offsets = np.array([fit(count,east,north,r) for r in windows])
            axes = directions_from_offsets(*offsets.T)
            angles = np.rad2deg(np.column_stack([np.arctan2(axes[:,0],axes[:,1]),
                                                 np.arcsin(axes[:,2])]))
            g = f["gain_grid_deg"][:]
            xx,yy = np.meshgrid(g,g)
            gain = beam_power(directions_from_offsets(xx,yy),axes[2])
            with h5py.File(a.output/f"beam-{i}.h5","w") as out:
                for k in ("east_edges","north_edges","counts","gain_grid_deg"):
                    f.copy(k,out)
                out["gain_430_check_db"] = 10*np.log10(np.maximum(gain,1e-30))
                out["fit_radius_deg"] = windows
                out["fitted_offset_deg"] = offsets
                out["fitted_az_el_deg"] = angles
                out.attrs["nominal_az_el_deg"] = SANYA_AZ_EL
                out.attrs["interpretation"] = "Effective count-pattern fit; fixed trajectories; window sensitivity not a confidence interval"
        print(path, "offsets", offsets.round(3), "az/el",angles[2].round(3))


if __name__ == "__main__":
    main()
