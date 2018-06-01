"""Finite Difference Approximation for derivatives"""

from numbers import Real

import numpy as np

from .matrix import mat_vec_mul_tridiag


def get_right_side_first_deriv_2nd_order_weight(delta_x):
    return np.array([-3.0, 4.0, -1.0], dtype=float) / (2.0 * delta_x)

def get_left_side_first_deriv_2nd_order_weight(delta_x):
    return np.array([1.0, -4.0, 3.0], dtype=float) / (2.0 * delta_x)


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


def get_first_deriv(fx, delta_x, out=None, zero_boundary=True):

    assert fx.ndim == 1
    N_x = fx.size
    
    tridiags = get_first_deriv_tri_diagonals(N_x, delta_x, dtype=float)

    ## Determine output array
    dfx_num = None
    if out is None: dfx_num = np.empty_like(fx)
    elif isinstance(out, np.ndarray): dfx_num = out
    assert dfx_num is not None

    ## Evaluate derivative including boundary
    dfx_num[:] = mat_vec_mul_tridiag(*tridiags, fx)  # it include boundary effect at both ends
    if not zero_boundary:
        dfx_num[0] = fx[:3].dot(get_right_side_first_deriv_2nd_order_weight(delta_x))
        dfx_num[-1] = fx[-3:].dot(get_left_side_first_deriv_2nd_order_weight(delta_x))
    
    return dfx_num


def get_m_values_of_big_o(fx, dfx_ana, delta_x, zero_boundary=True, fd_approximation_order=2):
    if fd_approximation_order != 2: raise NotImplementedError()
    dfx_num = get_first_deriv(fx, delta_x, zero_boundary=zero_boundary)
    m_array = np.abs(dfx_ana - dfx_num) / (delta_x ** fd_approximation_order)
    return m_array

