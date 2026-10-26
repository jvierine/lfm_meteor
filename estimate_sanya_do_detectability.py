"""D-O density thresholds for the Sanya monostatic link; no plotting backend.

Scientific output is HDF5. Render with the native memo-do Rust builder.
The critical-contour equivalent sphere is an explicit approximation, not FDTD.
"""
from pathlib import Path
import sys

import h5py
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import c, k, epsilon_0, m_e, e, atomic_mass
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erf
from scipy.stats import ncx2
from pymsis import msis
from pymsis.utils import get_f107_ap

from noise_model import MEDIAN_TSYS_K

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/sanya_do_detectability"


def density(z, r, order=64):
    """n_e/n_star, D-O (2017b) Eqs. 45--48, z positive in the wake.

    Gauss-Legendre quadrature with xi=c+(1-c)*sin(t)^2 removes
    the endpoint singularity. Checked against sabmod's adaptive quadrature.
    """
    z, r = np.broadcast_arrays(np.asarray(z, float), np.asarray(r, float))
    radius = np.hypot(z, r)
    radius = np.where(radius == 0, np.nan, radius)
    ct = z / radius
    ca = abs(ct)
    q = radius ** (2 / 3)
    f1 = np.sqrt(2*np.pi/3)/radius * erf(np.sqrt(1.5)*(radius*ca)**(1/3))
    f1 -= np.exp(-1.5*q*ca**(2/3)) * (
        (4-np.pi)*ca/(2*np.sqrt(1+(4-np.pi)**2*q*ca**(2/3)/(2*np.pi)))
        + 2*ca**(1/3)/q)
    f2 = np.sqrt(2*np.pi/3)/radius*erf(np.sqrt(1.5*q)) - (1+2/q)*np.exp(-1.5*q)
    nodes, weights = leggauss(order)
    f3 = np.zeros_like(radius)
    for t, w in zip((nodes+1)*np.pi/4, weights*np.pi/4):
        xi = ca + (1-ca)*np.sin(t)**2
        kernel = np.sqrt(1+2*q*xi**(2/3)/np.pi)*np.exp(-1.5*q*xi**(2/3))
        arg = np.divide(ca*np.sqrt((1-ca)*np.cos(t)**2*(1+xi)),
                        xi*np.sqrt(1-ca*ca), out=np.zeros_like(ca), where=ca<1)
        angle = np.arcsin(np.clip(arg, 0, 1))
        root = np.sqrt(np.sin(t)**2*(xi+ca)/(np.cos(t)**2*(1+xi)))
        f3 += w*kernel*(root+ca*angle)*2*(1-ca)*np.sin(t)*np.cos(t)
    return (ca*f1 + ct*f2 + f3)/radius


def maximum_along_wake(r):
    """Maximum density on a cylindrical radius; scan and refine the peak."""
    z = np.r_[0., -np.geomspace(max(r*1e-3, 1e-9), max(100., 100*r), 100),
              np.geomspace(max(r*1e-3, 1e-9), max(100., 100*r), 200)]
    z.sort()
    values = density(z, r)
    j = int(np.nanargmax(values))
    assert 0 < j < len(z)-1, "Contour peak not bracketed"
    result = minimize_scalar(lambda zz: -float(density(zz, r)),
                             bounds=(z[j-1], z[j+1]), method="bounded",
                             options={"xatol": max(1e-12, r*1e-8)})
    return -result.fun, result.x


def sphere_rcs(radius, wavelength):
    geometric = np.pi*radius**2
    rayleigh = 9*geometric*(2*np.pi*radius/wavelength)**4
    return 1/(1/geometric + 1/rayleigh)


def main():
    OUTPUT.mkdir(exist_ok=True)
    velocities = np.array([10., 30., 50., 72.])
    altitudes = np.array([80., 95., 105., 120.])
    dates = np.array(["2024-04-22T18:00:00"], dtype="datetime64[s]")
    f107, f107a, ap = get_f107_ap(dates)
    atmosphere = np.squeeze(msis.run(
        dates, [109.683719], [18.567821], altitudes,
        f107, f107a, ap, version=2.1))
    # Thermal species only: exclude MSIS anomalous oxygen and temperature.
    neutral = np.nansum(atmosphere[:, [1,2,3,4,5,6,7,9]], axis=1, dtype=np.float64)
    assert np.all(np.isfinite(neutral)) and np.all(neutral > 0)
    wavelength = c/430e6
    critical = epsilon_0*m_e*(2*np.pi*430e6)**2/e**2
    gain = 10**4.6*np.cos(np.deg2rad(15))**2.5
    temperature = MEDIAN_TSYS_K["Sanya"]
    pulse = 200e-6
    threshold = 10**(14/10)
    # Detector power contains both coherent signal and unit-mean complex noise.
    snr90 = brentq(lambda snr: ncx2.sf(2*threshold, 2, 2*snr)-.9, 0, 200)
    ranges = -(6371000+50)*np.sin(np.deg2rad(75)) + np.sqrt(
        (6371000+altitudes*1e3)**2-(6371000+50)**2*np.cos(np.deg2rad(75))**2)
    sigma_min = threshold*k*temperature/pulse*(4*np.pi)**3*ranges**4/(4.7e6*gain**2*wavelength**2)
    radius_min = np.array([brentq(lambda a: sphere_rcs(a,wavelength)-s,1e-8,10) for s in sigma_min])
    ablated_mass = 28.085*atomic_mass
    thermal_speed = np.sqrt(k*2000/ablated_mass)
    collision_sigma = 5.61e-19*velocities**(-.8)
    ambient_path = 1/(neutral[:,None]*collision_sigma[None,:])
    lam = thermal_speed/(neutral[:, None]*collision_sigma[None,:]*velocities[None,:]*1e3)
    shape = (len(altitudes), len(velocities))
    nstar = np.zeros(shape)
    nstar90 = np.zeros(shape)
    peak_z = np.zeros(shape)
    # Resolve the small Sanya contour while retaining the much wider VHF contour.
    z_edges = np.unique(np.r_[np.linspace(-.2,1.2,282),np.linspace(-.02,.03,102)])
    r_edges = np.unique(np.r_[np.linspace(-.5,.5,202),np.linspace(-.025,.025,102)])
    zz, rr = np.meshgrid((z_edges[:-1]+z_edges[1:])/2,(r_edges[:-1]+r_edges[1:])/2)
    maps = np.empty((*shape,len(r_edges)-1,len(z_edges)-1))
    for i in range(len(altitudes)):
        radius90 = brentq(lambda a: sphere_rcs(a,wavelength)-sigma_min[i]*snr90/threshold,1e-8,10)
        for j in range(len(velocities)):
            maximum, peak = maximum_along_wake(radius_min[i]/lam[i,j])
            nstar[i,j] = critical/maximum
            nstar90[i,j] = critical/maximum_along_wake(radius90/lam[i,j])[0]
            peak_z[i,j] = peak*lam[i,j]
            maps[i,j] = np.log10(np.maximum(nstar[i,j]*density(zz/lam[i,j],abs(rr)/lam[i,j])/critical,1e-8))
    # Independent adaptive quadrature and doubled-order checks.
    sys.path.insert(0, str(ROOT.parent / "sabmod"))
    from dimant_oppenheim import electron_density
    check_points = [(1.,1.),(-1.,1.),(.01,.01),(5.,0.),(0.,.1),
                    (.001,.003),(-.01,.003),(0.,4.),(4.,4.)]
    errors=[]
    for z,r in check_points:
        expected=electron_density(z,r)
        errors.append(abs(float(density(z,r,128))/expected-1))
        np.testing.assert_allclose(density(z,r),expected,rtol=2e-4)
    assert max(errors)<2e-4, errors
    assert np.all(np.isfinite(nstar)) and np.all(nstar>0) and np.all(nstar90>nstar)
    assert np.allclose([sphere_rcs(a,wavelength) for a in radius_min],sigma_min,rtol=1e-7)
    # Independent slide-deck implementation of both physical collision lengths.
    sys.path.insert(0, str(ROOT.parent / "manims"))
    from meteor_rcs.prepare_assets import collision_lengths
    for i, number in enumerate(neutral):
        reference_air, reference_do = collision_lengths(velocities*1000,2000,number)
        np.testing.assert_allclose(ambient_path[i],reference_air,rtol=1e-12)
        np.testing.assert_allclose(lam[i],reference_do,rtol=1e-12)
    np.testing.assert_allclose(lam/ambient_path,
        np.broadcast_to(thermal_speed/(velocities[None,:]*1000),shape),rtol=1e-12)
    # Fourfold range loss changes the required Rayleigh radius as R^(2/3).
    np.testing.assert_allclose(radius_min/radius_min[0],(ranges/ranges[0])**(2/3),rtol=1e-4)
    with h5py.File(OUTPUT/"detectability.h5","w") as h:
        for name, value in dict(velocity_km_s=velocities,altitude_km=altitudes,
                atmosphere_msis=atmosphere,neutral_number_density_m3=neutral,
                f107=f107,f107a=f107a,ap=ap,collision_cross_section_m2=collision_sigma,
                ambient_collision_length_m=ambient_path,
                range_m=ranges,lambda_T_m=lam,n_star_min_m3=nstar,n_star_pd90_m3=nstar90,
                sigma_min_m2=sigma_min,effective_radius_min_m=radius_min,
                peak_contour_z_m=peak_z,z_edges_m=z_edges,r_edges_m=r_edges,
                log10_density_over_critical=maps,quadrature_relative_errors=errors).items():
            h[name]=value
        h.attrs.update(dict(source_program=__file__,frequency_hz=430e6,tx_power_w=4.7e6,
            gain_linear=gain,system_temperature_k=temperature,pulse_s=pulse,
            effective_noise_bandwidth_hz=1/pulse,threshold_db=14.,critical_density_m3=critical,
            pd90_signal_snr=snr90,thermal_speed_m_s=thermal_speed,
            reference_time_utc="2024-04-22T18:00:00",atmosphere_model="NRLMSIS 2.1",
            scattering="head-on critical-contour projected-area equivalent PEC sphere; harmonic Rayleigh/geometric interpolation",
            normalization="ne = n_star * D(z/lambda_T, r/lambda_T); includes 8*pi in n_star",
            ablated_mass_kg=ablated_mass,pymsis_version=__import__('pymsis').__version__,
            neutral_species_sum="N2+O2+O+He+H+Ar+N+NO; anomalous O excluded; unavailable minor species omitted",
            assumptions="2000 K ablated Si neutrals; mass 28.085 u; Bronshten cross section at bulk speed U; on-axis beam; no extra losses",
            temperature_source="noise_model.MEDIAN_TSYS_K; raw noise file currently on unavailable external volume"))
    print("ncrit",critical,"Tsys",temperature,"gain dBi",10*np.log10(gain),flush=True)
    print("sigma_min",sigma_min,"nstar range",nstar.min(),nstar.max(),flush=True)
    print("n_star_min (altitude rows, speed columns):\n",nstar,flush=True)
    rows=[]
    for i,alt in enumerate(altitudes):
        rows.append(f"{alt:.0f} & " + " & ".join(f"{np.log10(v):.2f}" for v in nstar[i])+r" \\")
    # A generated LaTeX table is a document output, not a tabular data product.
    table=ROOT.parent/"sanya_tristatic_paper/tables/memo30_detectability.tex"
    content="% Generated by estimate_sanya_do_detectability.py\n"
    content+=r"\begin{tabular}{r*{"+str(len(velocities))+r"}{r}}"+"\n"+r"\toprule"+"\n"
    content+="Altitude (km) & "+" & ".join(f"{v:g}" for v in velocities)+r" \\"+"\n"+r"\midrule"+"\n"
    content+="\n".join(rows)+"\n"+r"\bottomrule\end{tabular}"+"\n"
    # Keep source edits in apply_patch; the caller installs this generated table.
    with h5py.File(OUTPUT/"detectability.h5","a") as h:
        h["table_latex"]=content


if __name__ == "__main__":
    main()
