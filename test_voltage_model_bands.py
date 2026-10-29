import numpy as np


def test_initial_dynamic_mass_recovers_free_molecular_sphere(monkeypatch):
    import export_voltage_plot as export
    import fit_all_ballistic_snr_weighted as base
    rho_air, material, radius, speed = 1e-6, 3000., 1e-4, 30000.
    distance = 6471000.
    gravity = base.EARTH_MU_M3_S2/distance**2
    decel = 3*rho_air*speed**2/(4*material*radius)-gravity
    a = 100.
    q = np.zeros((12,8)); q[:,6]=np.log10(a); q[:,7]=np.log10(decel/a)
    frame = np.zeros(42);frame[6:15]=np.eye(3).ravel();frame[33:42]=np.eye(3).ravel()
    data=dict(center=np.array([distance,0.,0.,-speed-a,0.,0.]),time_s=np.array([0.]),
              time_ns=np.array([1713795876414348602]),frames=frame[None,:])
    monkeypatch.setattr(base,'density_interpolator',lambda *args:(lambda h:np.full_like(h,rho_air),{}))
    r,m,_=export.initial_dynamic_mass(q,data)
    np.testing.assert_allclose(r,radius,rtol=1e-12)
    np.testing.assert_allclose(m,4*np.pi/3*material*radius**3,rtol=1e-12)
