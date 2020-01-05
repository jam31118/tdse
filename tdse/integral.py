import numpy as np

## Numerical integration routine
def numerical_integral_trapezoidal(x_arr, y_arr):
    _delta_x_arr = np.diff(x_arr)
    _mean_height_arr = 0.5 * (y_arr[:-1] + y_arr[1:])
    _sum = np.dot(_delta_x_arr, _mean_height_arr)
    return _sum


## Norm evaluation routine
def eval_norm_trapezoid(x_arr, psi_arr):
    _abs_square_arr = (psi_arr * psi_arr.conjugate()).real
    _norm = numerical_integral_trapezoidal(x_arr, _abs_square_arr)
    return _norm


## Normalization routine
def normalize_trapezoid(x_arr, psi_arr):
    _norm = eval_norm_trapezoid(x_arr, psi_arr)
    psi_arr[:] *= 1.0 / np.sqrt(_norm)


## Integrate on a regular grid with support of masked array
def integrate_on_reg_grid(fx, *dxargs):
    """
    Integrate the given N-D array on a regular grid
    It supports masked array.
    
    Concerning the masked array, 
    this routine doens't use trapzoidal integration
    
    # Example
    
    >>> f = np.arange(12).reshape(3,4)
    >>> dx, dy = 1, 2
    >>> integrate_on_reg_grid(f,dx,dy)
    36
    """
    
    _dx_arr = np.array(dxargs, copy=False)
    assert (_dx_arr.ndim <= 1) and np.all(_dx_arr > 0)
    _coord_dim = _dx_arr.size
    _fx = np.array(fx, copy=False)
    assert _fx.ndim == _coord_dim
    
    _da = np.prod(_dx_arr)
    _mask_excluding_last_elements = np.index_exp[:-1] * _coord_dim
    _sum = fx[_mask_excluding_last_elements].sum() * _da
    if isinstance(fx, np.ma.masked_array) \
	and isinstance(_sum, np.ma.core.MaskedConstant):
        _sum = 0.0
        
    return _sum


