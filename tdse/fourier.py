import numpy as np

from tdse.integral import numerical_integral_trapezoidal as int_trapz


def fourier_forward(f_x_arr, x_arr, k_arr):
    _f_k_arr = np.empty_like(k_arr, dtype=complex)

    _integrand = np.empty_like(_f_k_arr, dtype=complex)
    for _k_idx, _k in enumerate(k_arr):
        _integrand[:] = f_x_arr * np.exp(-1.0j * _k * x_arr)
        _f_k_arr[_k_idx] = 1.0 / np.sqrt(2.0*np.pi) * int_trapz(x_arr, _integrand)
    
    return _f_k_arr


def get_k_arr_at_Nyquist_limit(x_arr):
    _delta_x = None
    if np.diff(x_arr).std() < 1e-13: _delta_x = x_arr[1] - x_arr[0]
    else: raise Exception("'x_arr' should be equidistanced")
    assert _delta_x is not None
    
    _N = x_arr.size
    _k_max = np.pi / _delta_x
    _delta_k = 2.0 * np.pi / (_delta_x * (_N-1))
    _k_arr = np.linspace(-_k_max, _k_max, _N)
    assert np.isclose(_k_arr[1] - _k_arr[0], _delta_k, atol=1e-13, rtol=0)  # check consistency
    
    return _k_arr

