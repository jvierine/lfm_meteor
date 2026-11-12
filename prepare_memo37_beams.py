"""Physical beam slices for the proposed two-altitude SYISR experiment."""
from pathlib import Path
import hashlib
import h5py
import jcoord
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/memo37_beams.h5'
SITES = [(18.3492, 109.6222, 50.), (19.5281, 109.1322, 99.9),
         (19.5982, 110.7908, 24.9)]


def basis(lat, lon):
    a, b = np.deg2rad([lat, lon])
    return np.array([[-np.sin(b), -np.sin(a)*np.cos(b), np.cos(a)*np.cos(b)],
                     [np.cos(b), -np.sin(a)*np.sin(b), np.cos(a)*np.sin(b)],
                     [0., np.cos(a), np.sin(a)]])


def direction(az, el):
    a, e = np.deg2rad([az, el])
    return np.array([np.cos(e)*np.sin(a), np.cos(e)*np.cos(a), np.sin(e)])


def angles(v):
    return np.rad2deg(np.arctan2(v[0], v[1])) % 360, np.rad2deg(np.arctan2(v[2], np.hypot(v[0], v[1])))


def altitude(p):
    return jcoord.ecef2geodetic(*p)[2] / 1000


def main():
    positions = [np.asarray(jcoord.geodetic2ecef(*s)) for s in SITES]
    bases = [basis(*s[:2]) for s in SITES]
    tx = positions[0]
    axis = bases[0] @ direction(15., 75.)
    heights = np.linspace(70., 130., 1201)
    def target(h):
        r = brentq(lambda r: altitude(tx + r*axis)-h, 0., 160000.)
        return tx + r*axis
    column = np.array([target(h) for h in heights])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT, 'w') as f:
        f.attrs.update(source_program=Path(__file__).read_text(),
                       model='Uniform rectangular aperture, full 32x24 m; half 16x24 m; WGS84 heights.',
                       carrier_hz=430e6, lower_km=95., upper_km=105.,
                       split='Contiguous halves along the panel tilt-plane dimension; illustrative, not as-built partition.')
        f['altitude_km'] = heights
        for site, boresight in zip(range(1,3), [(158.3,70.),(221.9,70.)]):
            g = f.create_group(['danzhou','wenchang'][site-1])
            origin, rot = positions[site], bases[site]
            center = rot.T @ (target(95.)-origin)
            az0, _ = angles(center)
            horizontal = direction(az0, 0.)
            def project(xyz):
                enu = (xyz-origin) @ rot
                return np.c_[enu @ horizontal / 1000, [altitude(p) for p in xyz]]
            g['column'] = project(column)
            g['tx_axis'] = project(np.array([tx, target(130.)]))
            rtx = np.linspace(0., 140000., 220)
            txedges = [project(tx+rtx[:,None]*(bases[0]@direction(15.,e))) for e in (74.55,75.45)]
            g['tx_polygon'] = np.vstack([txedges[0], txedges[1][::-1], txedges[0][:1]])
            normal = direction(*boresight)
            tilt = np.array([0.,0.,1.])-normal[2]*normal
            tilt /= np.linalg.norm(tilt)
            cross = np.cross(normal, tilt)
            dirs = (column-origin) @ rot
            dirs /= np.linalg.norm(dirs,axis=1)[:,None]
            def power(d, p, length):
                return np.sinc(length/(299792458/430e6)*((d-p)@tilt))**2 * np.sinc(24/(299792458/430e6)*((d-p)@cross))**2
            reference = 43.+25*np.log10(direction(*angles(center))@normal)
            for name, h, length, loss in [('full95',95.,32.,0.),('lower',95.,16.,-3.0103),('upper',105.,16.,-3.0103)]:
                b = g.create_group(name)
                p = rot.T @ (target(h)-origin)
                p /= np.linalg.norm(p)
                az, el = angles(p)
                peak = 43.+loss+25*np.log10(p@normal)
                gain = peak-reference+10*np.log10(np.maximum(power(dirs,p,length),1e-12))
                b['gain_relative_full95_db'] = gain
                def edge(sign):
                    d = brentq(lambda d: power(direction(az,el+sign*d),p,length)-.5, 0., 8.)
                    return d
                minus, plus = edge(-1), edge(1)
                rr = np.linspace(0., 260000., 300)
                lo = project(origin+rr[:,None]*(rot@direction(az,el-minus)))
                hi = project(origin+rr[:,None]*(rot@direction(az,el+plus)))
                b['polygon'] = np.vstack([lo,hi[::-1],lo[:1]])
                b['axis'] = project(np.array([origin,target(h)]))
                b['target'] = project(np.array([target(h)]))
                selected = abs(heights-h)<12
                good = (gain >= peak-reference-3.0103) & selected
                b.attrs.update(azimuth_deg=az,elevation_deg=el,vertical_hpbw_deg=minus+plus,
                               peak_gain_dbi=peak,half_power_low_km=heights[good].min(),
                               half_power_high_km=heights[good].max())
                print(g.name,name,dict(b.attrs))
        f.attrs['sha256_source'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(OUT)


if __name__ == '__main__':
    main()
