import numpy as np

from ..integral import eval_norm_trapezoid
from ..evol import get_D2_tridiag, get_M2_tridiag, mul_tridiag_and_diag
from ..tridiag import tridiag_forward, tridiag_backward


def eval_energy_spectrum_for_1D_hamil(
        sf_arr, x_arr, V_x_arr, E_arr, winop_n, gamma, use_only_real_pot=True, 
        eval_momentum=False, m=1.0, zero_thres=1e-13):
    """
    Evaluate energy spectrum

    Argument
    ----------
    x_arr : (N,) numpy.ndarray
        Equidistanced grid in position space

    E_arr : (M,) numpy.ndarray
        Monotonically increasing array of energy values
        at which the probability density of energy would be evaluated
    """

    ## Check arguments
    assert np.all(np.diff(E_arr) > 0)
    if eval_momentum:
        _min_abs_x_index = np.argmin(np.abs(x_arr))
        _min_abs_x = x_arr[_min_abs_x_index]
        if abs(_min_abs_x) > zero_thres:
            raise Exception("The given `x_arr` doesn't seem to have a zero " \
                    + "as an element")
        _zero_x_index = _min_abs_x_index
        _nega_x_arr = x_arr[:_zero_x_index+1]
        _posi_x_arr = x_arr[_zero_x_index:]
        assert (_nega_x_arr.size + _posi_x_arr.size) == x_arr.size + 1
        assert _nega_x_arr[-1] == _posi_x_arr[0]
    if eval_momentum:
        _min_abs_E_index = np.argmin(np.abs(E_arr))
        _min_abs_E = E_arr[_min_abs_E_index]
        if abs(_min_abs_E) > zero_thres:
            raise Exception("The E_arr should contain zero")
    
    ## Define some variables
    _N_x = x_arr.size
    _delta_x = x_arr[1] - x_arr[0]
    _V_x_arr = V_x_arr
    if use_only_real_pot: _V_x_arr = V_x_arr.real
    
    ## Allocate memory
    # for output
    spectrum_E_arr = np.empty_like(E_arr, dtype=float)
    # for intermediate result
    right_arr = np.empty_like(x_arr, dtype=complex)

    ## Evaluate constant components
    D2 = get_D2_tridiag(_N_x, _delta_x)
    M2 = get_M2_tridiag(_N_x)
    M2V = mul_tridiag_and_diag(M2, _V_x_arr)

    tridiag_shape = M2.shape
    M2H = np.empty(tridiag_shape, dtype=float)
    left_tridiag = np.empty(tridiag_shape, dtype=complex)
    gamma_M2 = np.empty(tridiag_shape, dtype=complex)
    gamma_M2[:] = gamma * M2 

    norm_const = np.sin(np.pi/pow(2,winop_n)) / (np.pi/pow(2,winop_n))
    M2H[:] = -0.5*D2 + M2V

    if eval_momentum:
        _posi_E_mask = E_arr > 0
        _posi_E_arr = E_arr[_posi_E_mask]
        _num_of_posi_E_val = _posi_E_arr.size
        _num_of_momentum = 2 * _posi_E_arr.size + 1
        _p_arr = np.empty((_num_of_momentum,), dtype=float)
        _posi_p_arr = np.sqrt(2.0*m*_posi_E_arr)
        _p_arr[:_num_of_momentum//2] = - np.flipud(_posi_p_arr)
        _p_arr[_num_of_momentum//2 + 1:] = _posi_p_arr
        _p_arr[_num_of_momentum//2] = 0.0
        # [NOTE] The probability density of momentum when E = 0 (thus, p = 0) 
        # .. is always zero since |dE/dp| = |p|/m = 0
        _spectrum_p_arr = np.empty_like(_p_arr, dtype=float)
        _spectrum_p_arr[_num_of_momentum//2] = 0.0

    ## Evaluate energy-dependent values
    for E_idx, E0 in enumerate(E_arr):
        _sf_arr = sf_arr.copy()
        for _m in range(1,pow(2,winop_n-1)+1):
            _phase = np.exp(1.0j * (2.0*_m-1)/pow(2,winop_n) * np.pi)
            left_tridiag[:] = M2H + (_phase * gamma - E0) * M2

            tridiag_forward(gamma_M2, _sf_arr, right_arr)
            tridiag_backward(left_tridiag, _sf_arr, right_arr)

        _norm_on_x = eval_norm_trapezoid(x_arr, _sf_arr)
        spectrum_E_arr[E_idx] = 1.0 / (2.0*gamma) * norm_const \
                * _norm_on_x

        if eval_momentum and _posi_E_mask[E_idx]:

            
#            print("E0 = {}, 2.0*m*E0 = {}, np.sqrt(2.0*m*E0) = {}".format(E0, 2.0*m*E0, np.sqrt(2.0*m*E0)))
            _p0 = np.sqrt(2.0*m*E0)
            assert _p0 > 0
#            print("_p0 = {}".format(_p0))

            _E_posi_idx = E_idx - (E_arr.size - _posi_E_arr.size)
            _p_posi_idx = _num_of_momentum // 2 + 1 + _E_posi_idx
            _p_nega_idx = _num_of_momentum // 2 - 1 - _E_posi_idx

            if (_p0 != _p_arr[_p_posi_idx]) or (-_p0 != _p_arr[_p_nega_idx]):
                raise Exception("_p0 = {}, _p_arr[_p_posi_idx] = {}, _p_arr[_p_nega_idx] = {}".format(_p0, _p_arr[_p_posi_idx], _p_arr[_p_nega_idx]))

            _norm_on_nega_x = eval_norm_trapezoid(
                    _nega_x_arr, _sf_arr[:_zero_x_index+1]) 
            _norm_on_posi_x = eval_norm_trapezoid(
                    _posi_x_arr, _sf_arr[_zero_x_index:])

            if abs(_norm_on_x - (_norm_on_nega_x + _norm_on_posi_x)) > zero_thres:
                raise Exception("norm_on_x = {}, _norm_on_nega_x = {}, _norm_on_posi_x = {}, _norm_on_nega_x + _norm_on_posi_x = {}".format(
                    _norm_on_x, _norm_on_nega_x, _norm_on_posi_x, _norm_on_nega_x + _norm_on_posi_x))

            _spectrum_p_arr[_p_nega_idx] = 1.0 / (2.0 * gamma) * abs(_p0) \
                    * _norm_on_nega_x
            _spectrum_p_arr[_p_posi_idx] = 1.0 / (2.0 * gamma) * abs(_p0) \
                    * _norm_on_posi_x
    
    if eval_momentum:
        return spectrum_E_arr, _spectrum_p_arr, _p_arr
    else: return spectrum_E_arr

