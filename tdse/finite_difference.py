"""Finite Difference Approximation for derivatives"""

from numbers import Real

import numpy as np

from .matrix import mat_vec_mul_tridiag



def it_seems_increasing_equidistanced_arr(x_arr, thres_order=-14):
    _x_arr_diff = np.diff(x_arr)
    _is_increasing = np.all(_x_arr_diff > 0)
    _seems_equidistanced = _x_arr_diff.std() < pow(10, thres_order)
    return _is_increasing and _seems_equidistanced




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


def fd_diff_1d(f0_ana, kernel_center_unflipped, kernel_left_unflipped, kernel_right_unflipped):
    """
    Apply finite-difference approximation kernel on given array to calculate derivative corresponding to the given kernels.
    """

#    #### Prepare finite difference convolution kernels
#
#    # centered kernel
#    kernel_unflipped = kernel_center_unflipped
#    kernel = np.flip(kernel_unflipped, axis=0)
#
#    # one-sided kernel
#    kernel_left_unflipped = kernel_left_unflipped
#    kernel_left = np.flip(kernel_left_unflipped, axis=0)
#    kernel_right = kernel_left_unflipped
#

    kernels_unflipped = [
            kernel_center_unflipped, 
            kernel_left_unflipped, 
            kernel_right_unflipped
    ]

    for arg_array in kernels_unflipped + [f0_ana]:
        assert isinstance(arg_array, np.ndarray)
        assert arg_array.ndim == 1

    kernel_center, kernel_left, kernel_right = (
        np.flip(kernel_unflipped, axis=0) for kernel_unflipped 
        in kernels_unflipped
    )

    kernel_center_length = kernel_center.size
    kernel_left_length = kernel_left.size
    kernel_right_length = kernel_right.size

    assert (kernel_center_length % 2) == 1
    assert f0_ana.size >= max(kernel_center_length, kernel_left_length, kernel_right_length)

    kernel_center_half_width = kernel_center_length // 2  # excluding center point
    valid_center_slice = slice(kernel_center_half_width, -kernel_center_half_width)

    f1_num = np.empty_like(f0_ana)

    # fill in center parts
    f1_num[valid_center_slice] = np.convolve(f0_ana, kernel_center, mode='valid')

    # fill in edges (boundaries)
    assert kernel_left_length == kernel_right_length
    one_sided_slice_size = kernel_center_half_width + kernel_left_length - 1
    f1_num[:kernel_center_half_width] = np.convolve(f0_ana[:one_sided_slice_size], kernel_left, mode='valid')
    f1_num[-kernel_center_half_width-1:] = np.convolve(f0_ana[-one_sided_slice_size-1:], kernel_right, mode='valid')
    
    return f1_num


def deriv_1st_fd_2nd_1d(f_x, delta_x):
    """"""
    deriv_order = 1
    fd_order = 2
    kernel_center_unflipped_1_2 = np.array([-1/2, 0, 1/2]) / delta_x ** deriv_order
    kernel_left_unflipped_1_2 = np.array([-3/2, 2, -1/2]) / delta_x ** deriv_order
    kernel_right_unflipped_1_2 = (-1)**deriv_order * np.flip(kernel_left_unflipped_1_2, axis=0)
    deriv = fd_diff_1d(f_x, kernel_center_unflipped_1_2, kernel_left_unflipped_1_2, kernel_right_unflipped_1_2)
    return deriv


def deriv_2nd_fd_2nd_1d(f_x, delta_x):
    """"""
    deriv_order = 2
    fd_order = 2
    kernel_center_unflipped_2_2 = np.array([1, -2, 1]) / delta_x ** deriv_order
    kernel_left_unflipped_2_2 = np.array([2, -5, 4, -1]) / delta_x ** deriv_order
    kernel_right_unflipped_2_2 = (-1)**deriv_order * np.flip(kernel_left_unflipped_2_2, axis=0)
    deriv = fd_diff_1d(f_x, kernel_center_unflipped_2_2, kernel_left_unflipped_2_2, kernel_right_unflipped_2_2)
    return deriv


def get_error_over_delta_x_power_for_1st_deriv_and_2nd_order_accuracy(fx, delta_x):
    
    ## Evaluate 3rd derivative with 4th order accuracy
    deriv_order = 3
    fd_order = 4
    kernel_unflipped = np.array([1/8, -1, 13/8, 0, -13/8, 1, -1/8]) / delta_x ** deriv_order
    kernel_left_unflipped = np.array([-49/8, 29, -461/8, 62, -307/8, 13, -15/8]) / delta_x ** deriv_order
    kernel_right_unflipped = (-1)**deriv_order * np.flip(kernel_left_unflipped, axis=0)

    f3_num = fd_diff_1d(fx, kernel_unflipped, kernel_left_unflipped, kernel_right_unflipped)

    
    ## Evaluate 5th derivative with 2nd order accuracy
    deriv_order = 5
    kernel_unflipped = np.array([-1/2,2,-5/2,0,5/2,-2,1/2]) / delta_x ** deriv_order
    kernel_left_unflipped = np.array([-7/2,20,-95/2,60,-85/2,16,-5/2]) / delta_x ** deriv_order # 2nd order accuracy
    # kernel_left_unflipped = np.array([-46,295,-810,1235,-1130,621,-190,25]) / 6 / delta_x ** deriv_order  # 3rd order accuracy
    kernel_right_unflipped = (-1)**deriv_order * np.flip(kernel_left_unflipped, axis=0)

    f5_num = fd_diff_1d(fx, kernel_unflipped, kernel_left_unflipped, kernel_right_unflipped)

    
    ## Build approximated expression
    m_arr_ana_estimated = np.abs(f3_num / (1*2*3) + f5_num / (1*2*3*4*5) * delta_x**2)
    
    return m_arr_ana_estimated



from scipy.special import factorial
from numpy.linalg import solve

def eval_deriv_on_equidistanced_grid(x_p_arr, x_arr, sf_arr, orders, N_s):
    """
    # Argument
    `x_arr`: equi-distanced array
    """
    
    ## Define variables
    _N_p, _N_x = x_p_arr.size, x_arr.size
    _delta_x = x_arr[1] - x_arr[0]
    
    ## Construct an array of indices of first stencil for each particle position (`x_p`)
    _i_p_arr = np.asarray((x_p_arr - x_arr[0]) // _delta_x, dtype=int)
    _i_s0_p_arr_unshift = _i_p_arr - int((N_s//2) - 1)
    _i_s0 = _i_s0_p_arr_unshift # aliasing
    _i_s0_p_arr = _i_s0 - (_i_s0 < 0) * _i_s0 - (_i_s0 + N_s - _N_x > 0) * (_i_s0 + N_s - _N_x)

    ## Construct an array of power matrices
    _pow_mat = np.empty((_N_p,N_s,N_s), dtype=float)
    _pow_mat[:,0,:] = 1.0
    for _s_idx in range(N_s):
        _pow_mat[:,1,_s_idx] = (x_arr[_i_s0_p_arr+_s_idx] - x_p_arr)
    for _s_pow_idx in range(2,N_s):
        _pow_mat[:,_s_pow_idx,:] = _pow_mat[:,_s_pow_idx-1,:] * _pow_mat[:,1,:]

    ## Evaluate FD derivatives
    _deriv_sf_arr = np.zeros((len(orders), _N_p), dtype=sf_arr.dtype)
    for _deriv_order, _deriv_sf in zip(orders, _deriv_sf_arr):
        # Construct matrix on right for finite-difference linear system
        _b_mat = np.zeros((_N_p,N_s), dtype=float)
        _b_mat[:,_deriv_order] = factorial(_deriv_order, exact=True)
        # Evaluate coefficient matrix
        _coef_mat = solve(_pow_mat, _b_mat)
        # Get derivative of state function by linear combination
        for _s in range(N_s):
            _deriv_sf += sf_arr[_i_s0_p_arr+_s] * _coef_mat[:,_s]
    
    return _deriv_sf_arr

