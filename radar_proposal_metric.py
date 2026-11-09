"""Numerical proposal scaling only; never modifies a radar likelihood or prior."""
import numpy as np


def cap_covariance(covariance, limits):
    covariance = np.asarray(covariance, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if (covariance.shape != (len(limits),len(limits))
            or not np.isfinite(covariance).all() or not np.all(limits>0)
            or not np.all(np.diag(covariance)>0)):
        raise ValueError('Invalid proposal covariance or limits')
    scale = np.minimum(1.,limits/np.sqrt(np.diag(covariance)))
    return covariance*scale[:,None]*scale[None,:]
