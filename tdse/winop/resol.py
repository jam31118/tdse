import numpy as np

## Enlarging spatial grid and other arrays accordingly
def enlarge_x_arr(N_plus, x_arr):
    _N_winop = x_arr.size + 2*N_plus
    _x_arr_winop = np.empty((_N_winop,), dtype=float)
    _delta_x = x_arr[1] - x_arr[0]
    _x_arr_winop[:N_plus] = x_arr[0] + np.arange(-N_plus,0) * _delta_x
    _x_arr_winop[N_plus:_N_winop-N_plus] = x_arr
    _x_arr_winop[_N_winop-N_plus:] = x_arr[-1] + np.arange(1,N_plus+1)*_delta_x
    return _x_arr_winop

def enlarge_arr(N_plus, arr):
    _N_winop = arr.size + 2*N_plus
    _arr_enlarged = np.zeros((_N_winop,), dtype=arr.dtype)
    _arr_enlarged[N_plus:_N_winop-N_plus] = arr
    return _arr_enlarged


## Estimating energy spacing due to finite spatial range
def eval_delta_E(E, x_arr):
    _L = x_arr[-1] - x_arr[0]
    _n = np.sqrt(2) * _L / np.pi * np.sqrt(E)
    _delta_E = np.pi**2 * 0.5 / _L**2 * (_n*2 + 1)
    return _delta_E

