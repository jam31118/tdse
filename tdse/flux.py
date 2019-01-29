import numpy as np

from .analytic import Gaussian1D

def probability_flux_Gaussian_1D(x_arr, t, k0_x):
    _psi_arr = Gaussian1D(x_arr, t, k0_x)
    _psi_norm_square_arr = (_psi_arr * _psi_arr.conjugate()).real
    _prob_flux = _psi_norm_square_arr * (2/(1+4*t*t)) * (0.5*k0_x + 2*t*x_arr)
    return _prob_flux

