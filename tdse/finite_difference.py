"""Finite Difference Approximation for derivatives"""

from numbers import Real
import numpy as np


def get_first_deriv_tri_diagonals(N_x0, delta_x0, dtype=complex):
    for arg in [N_x0, delta_x0]: assert isinstance(arg, Real)
    coef = 1.0 / (2.0 * delta_x0)
    diag = np.full(N_x0, fill_value=0.0 * coef, dtype=dtype)
    off_diag_upper_1 = np.full(N_x0-1, fill_value=1.0 * coef, dtype=dtype)
    off_diag_lower_1 = np.full(N_x0-1, fill_value=-1.0 * coef, dtype=dtype)
    return diag, off_diag_lower_1, off_diag_upper_1

def get_second_deriv_tri_diagonals(N_x0, delta_x0, dtype=complex):
    for arg in [N_x0, delta_x0]: assert isinstance(arg, Real)
    coef = 1.0 / (delta_x0 * delta_x0)
    diag = np.full(N_x0, fill_value=-2.0 * coef, dtype=dtype)
    off_diag = np.full(N_x0-1, fill_value=1.0 * coef, dtype=dtype)
    return diag, off_diag, off_diag
