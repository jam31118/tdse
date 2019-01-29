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


