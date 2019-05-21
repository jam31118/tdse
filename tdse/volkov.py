import numpy as np

from .integral import numerical_integral_trapezoidal as int_trapz

def eval_volkov_phase_k_arr(k_arr, t_idx, t_arr, A_t_func):
    _volkov_phase_k_arr = np.empty_like(k_arr, dtype=float)
    if t_idx == 0:
        _volkov_phase_k_arr[:] = 0
        return _volkov_phase_k_arr
    _t_arr_slice = t_arr[:(t_idx+1)]
    _A_t_arr_slice = A_t_func(_t_arr_slice)
    for k_idx in range(k_arr.size):
        k = k_arr[k_idx]
        _volkov_phase_k_arr[k_idx] = int_trapz(_t_arr_slice, 0.5 * (k*k + 2*k*_A_t_arr_slice))
    return _volkov_phase_k_arr


from .integral import numerical_integral_trapezoidal as int_trapz
from .fourier import get_k_arr_at_Nyquist_limit
from .fourier import fourier_forward

def free_space_volkov(x_arr, sf_x_t0_arr, t_idx, t_arr, A_t_func):
    
    assert x_arr.size == sf_x_t0_arr.size
    
    _sf_x_t_arr = np.empty_like(sf_x_t0_arr)
    _k_arr = get_k_arr_at_Nyquist_limit(x_arr)
    _sf_k_t0_arr = fourier_forward(sf_x_t0_arr, x_arr, _k_arr)
    _volkov_phase_k_arr = eval_volkov_phase_k_arr(_k_arr, t_idx, t_arr, A_t_func)

    _integrand = np.empty_like(_k_arr, dtype=complex)
    for _x_idx, _x in enumerate(x_arr):
        _x = x_arr[_x_idx]
        _integrand = _sf_k_t0_arr * np.exp(1.0j * (_k_arr * _x - _volkov_phase_k_arr))
        _sf_x_t_arr[_x_idx] = 1.0 / np.sqrt(2.0*np.pi) * int_trapz(_k_arr, _integrand)
    
    return _sf_x_t_arr


