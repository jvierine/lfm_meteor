"""Fourfold radiant symmetry and its exposure normalization."""
import numpy as np


def symmetry_members(longitude_deg, latitude_deg):
    longitude, latitude = np.broadcast_arrays(longitude_deg, latitude_deg)
    reflected = (540. - longitude) % 360.
    return (np.stack((longitude % 360., reflected, longitude % 360., reflected), axis=-1),
            np.stack((latitude, latitude, -latitude, -latitude), axis=-1))


def daily_weight_from_summed_hours(hours):
    hours = np.asarray(hours, dtype=float)
    if np.any(~np.isfinite(hours)) or np.any(hours <= 0):
        raise ValueError('A symmetry family needs positive summed exposure')
    # Four equal intrinsic populations, observed with their summed exposure.
    return 4. * 24. / hours
