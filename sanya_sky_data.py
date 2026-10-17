"""GSM2008 scientific sampling without plotting imports.

Uses the same public Haslam-locked PCA data and log-frequency PCHIP as
PyGDSM GlobalSkyModel (gsm08.py). Astropy HEALPix supplies the same RING pixel
lookup as healpy; no map-rendering API or Matplotlib dependency is involved.
"""

import h5py
import numpy as np
from astropy.utils.data import download_file
from astropy_healpix import HEALPix
from scipy.interpolate import PchipInterpolator


class GSM2008:
    def __init__(self, frequency_mhz=430.0):
        self.frequency_mhz = float(frequency_mhz)
        if not 10.0 <= self.frequency_mhz <= 94000.0:
            raise ValueError("GSM2008 frequency outside 10--94000 MHz")
        self.source = download_file(
            "https://apps.datacentral.org.au/pygdsm/data/gsm_components.h5", cache=True
        )
        with h5py.File(self.source) as h:
            table = h["components"][()]
            components = h["component_maps_408locked"][()]
        log_frequency = np.log(table[:, 0])
        target = np.log(self.frequency_mhz)
        scaling = np.exp(PchipInterpolator(log_frequency, np.log(table[:, 1]))(target))
        coefficients = np.array(
            [PchipInterpolator(log_frequency, table[:, i])(target) for i in range(2, 5)]
        )
        self.map = np.einsum(
            "cf,pc,f->fp", coefficients[:, None], components, np.atleast_1d(scaling)
        )[0]
        self.healpix = HEALPix(nside=512, order="ring", frame="galactic")

    def get_sky_temperature(self, coords, freqs=None, include_cmb=False):
        if freqs is not None and float(freqs) != self.frequency_mhz:
            raise ValueError("Frequency differs from prepared scientific sky map")
        galactic = coords.galactic
        pixels = self.healpix.lonlat_to_healpix(galactic.l, galactic.b)
        return self.map[pixels] + (2.725 if include_cmb else 0.0)
