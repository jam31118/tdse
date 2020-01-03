import numpy as np

from tdse.integral import numerical_integral_trapezoidal as int_trapz

from numpy.fft import fft, ifft

def transform_x_to_k_space_fft(psi_x_arr, delta_x):
    """ Transforms the given state function in x-space (position) 
    to that of k-space (wave vector)
    
    'psi_x_arr': state function in x-space (position)
    'delta_x': grid spacing of x-array (equidistant)
    """
    _N = psi_x_arr.size
    _minus_1_power_n = 1 - 2*(np.arange(_N, dtype=int) % 2)  # i.e. (-1)^n
    _coef_arr = delta_x * 1.0 / np.sqrt(2*pi) * (-1.0j)**_N * _minus_1_power_n
    _psi_k_arr = _coef_arr * fft(_minus_1_power_n * psi_x_arr, _N)
    return _psi_k_arr

def transform_k_to_x_space_ifft(psi_k_arr, delta_k):
    """ Transforms the given state function in k-space (wave vector) 
    that of x-space (position)
    
    'psi_k_arr': state function in k-space (wave vector)
        The wave vector represents momentum in orthodox quantum mechanics.
    'delta_k': grid spacing of k-array (equidistant)
    """
    _N = psi_k_arr.size
    _minus_1_power_n = 1 - 2*(np.arange(_N, dtype=int) % 2) # i.e. (-1)^n
    _coef_arr = delta_k / np.sqrt(2*pi) * (1.0j)**_N * _minus_1_power_n * _N
    _psi_x_arr = _coef_arr * ifft(_minus_1_power_n * psi_k_arr, _N)
    return _psi_x_arr



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




import numpy as np
from numpy import pi

def nyquist_condition_satisfied(x_arr, k_arr):
    _dx_arr, _dk_arr = np.diff(x_arr), np.diff(k_arr)
    assert np.all(_dx_arr > 0) and np.all(_dk_arr > 0)
    _x_max, _k_max = max(np.abs(x_arr[[0,-1]])), max(np.abs(k_arr[[0,-1]]))
    _dx_min, _dk_min = _dx_arr.min(), _dk_arr.min()
    return (_k_max < 2*pi/_dx_min) and (_x_max < 2*pi/_dk_min)

def construct_k_arr(x_arr):
    
    _dx_arr = np.diff(x_arr)
    assert _dx_arr.std() < 1e-14 and np.all(_dx_arr > 0)
    _dx = _dx_arr[0]
    
    _k_max = 0.5 * 2*pi/_dx

    _x_max = max(np.abs(x_arr[[0,-1]]))
    _N_k = int(_k_max/pi*_x_max + 1 + 1)

    _k_arr = np.linspace(-_k_max, _k_max, _N_k)
    
    assert nyquist_condition_satisfied(x_arr, _k_arr)
    
    return _k_arr

from numpy import exp, pi

def ft(fx_arr, x_arr, k_arr=None):
    
    _dx_arr = np.diff(x_arr)
    assert np.all(_dx_arr > 0)
    _k_arr = k_arr
    if _k_arr is None: _k_arr = construct_k_arr(x_arr)
    else: assert isinstance(_k_arr, np.ndarray) \
        and nyquist_condition_satisfied(x_arr, _k_arr)
    _k_mesh, _x_mesh = np.meshgrid(_k_arr, x_arr, indexing='ij')

    _integrand = exp(-1.0j*_k_mesh *_x_mesh) * fx_arr
    _fk_arr = pow(2*pi,-0.5)*(
        0.5*(_integrand[:,:-1]+_integrand[:,1:])*_dx_arr).sum(axis=-1)
    
    if k_arr is None:
        return _fk_arr, _k_arr
    else: return _fk_arr


