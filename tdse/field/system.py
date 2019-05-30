from numbers import Integral, Real

import numpy as np

from ..tridiag import tridiag_forward, tridiag_backward, get_tridiag_shape
from ..integral import normalize_trapezoid, numerical_integral_trapezoidal
from ..integral import eval_norm_trapezoid
from ..evol import get_M2_tridiag,get_D2_tridiag,get_M1_tridiag,get_D1_tridiag
from ..evol import mul_tridiag_and_diag


def construct_spatial_array(delta_x, R_in):
    """Construct spatial array"""
    N_x = 2 * int(R_in // delta_x) + 1
    N_x_width = N_x // 2
    R = delta_x * N_x_width
    x_arr = np.linspace(-R, R, N_x)

    # check consistency
    assert np.isclose(x_arr[1] - x_arr[0], delta_x, atol=1e-13, rtol=0)
    assert R == (R_in - (R_in % delta_x))  # check consistency

    return x_arr



class System(object): pass


class LinearPolariVelocityGaugeSystem(System):
    
    _imag_prop_diff_thres = 1e-13
    _max_imag_prop_time_steps = 5000
    imag_pot_ampl = 100.0
    
    def __init__(self, 
                 R_in, delta_x, delta_t_real, 
                 t0, imag_pot_width, V_x_func, 
                 A_t_func, t_max_in, sf_arr):
        """
        # Notation
        - `N_x`: the number of spatial grid points
        
        # Arguments
        - `sf_arr`: array of shape (N_x,) or None
            if None: `self.sf_arr` is initialized randomly
        """
        
        ## Check input arguments
        _real_arg_tuple = (
                R_in, delta_x, delta_t_real, t0, imag_pot_width, t_max_in)
        for _real_arg in _real_arg_tuple: assert isinstance(_real_arg, Real)
        for _callable_arg in (V_x_func, A_t_func): 
            assert callable(_callable_arg)
        assert delta_x > 0 and R_in > delta_x and delta_t_real > 0 \
                and imag_pot_width >= 0
        assert isinstance(sf_arr, np.ndarray) or sf_arr is None
        
        ## Construct spatial array
        self.delta_x = delta_x
        self.N_inner_width = int(R_in // self.delta_x)
        self.N_absorb_width = int(imag_pot_width // self.delta_x)
        self.N_inner_total = 1 + 2 * self.N_inner_width
        self.N_absorb_total = 2 * self.N_absorb_width
        self.N_total_width = self.N_inner_width + self.N_absorb_width
        self.N_total = 1 + 2 * self.N_total_width
        self.R_inner = self.delta_x * self.N_inner_width
        self.R_absorb = self.delta_x * self.N_absorb_width
        self.R_total = self.R_inner + self.R_absorb
        assert np.abs(self.R_total - self.N_total_width * self.delta_x) < 1e-13
        self.x_arr = np.linspace(-self.R_total, self.R_total, self.N_total)
        
        # Aliasing
        self.N_x = self.N_total
        
        # Generate masks for spatial coordinate array
        _inner_x_mask_arr = np.full_like(self.x_arr, False, dtype=bool)
        sl = np.index_exp[self.N_absorb_width:self.N_total-self.N_absorb_width] 
        _inner_x_mask_arr[sl] = True
        self.inner_x_mask_arr = _inner_x_mask_arr
        
        ## Construct temporal array
        self.delta_t_real = delta_t_real
        self.t0 = t0
        self.N_timestep = int( (t_max_in - t0 + delta_t_real) // delta_t_real )
        self.t_max = self.t0 + self.delta_t_real * self.N_timestep
        self.N_timepoint = self.N_timestep + 1
        self.t_arr = np.linspace(self.t0, self.t_max, self.N_timepoint)
        assert np.isclose(self.t_arr[1] - self.t_arr[0], self.delta_t_real, 
                atol=1e-14, rtol=0)  # check consistency
        
        ## Assign to members
        self.A_t_func = A_t_func
        self.V_x_func = V_x_func
        
        ## Define some variables
        self._tridiag_shape = get_tridiag_shape(self.N_x)
        
        ## Set the initial state function array
        self.sf_arr = np.empty_like(self.x_arr, dtype=complex)
        if sf_arr is None: self.sf_arr[:] = np.random.rand(self.N_x) - 0.5
        else: self.sf_arr[:] = sf_arr
        assert self.sf_arr is not None
        normalize_trapezoid(self.x_arr, self.sf_arr)
        self.sf_arr_0 = self.sf_arr.copy()
        
        ## Construct the spatial potential array
        self.V_x_arr = np.empty_like(self.x_arr, dtype=complex)
        self.V_x_arr[:] = self.V_x_func(self.x_arr)
        
        # Add imaginary potential for norm absorption
        _absorb_x_mask_arr = ~self.inner_x_mask_arr
        _absorb_x_arr = self.x_arr[_absorb_x_mask_arr]
        self.V_x_arr[_absorb_x_mask_arr] = -1.0j * self.imag_pot_ampl \
            * ((_absorb_x_arr - np.sign(_absorb_x_arr) * self.R_inner) / imag_pot_width)**16
        
        ## Evaluate static parts of matrices
        self._M2 = get_M2_tridiag(self.N_x)
        self._D2 = get_D2_tridiag(self.N_x, self.delta_x)
        self._M1 = get_M1_tridiag(self.N_x)
        self._D1 = get_D1_tridiag(self.N_x, self.delta_x)
        self._M2V = mul_tridiag_and_diag(self._M2, self.V_x_arr, dtype=self.V_x_arr.dtype)
        
        self._M2H0 = -0.5 * self._D2 + self._M2V
        
        ## Allocate memory 
        # for time-evolution operator
        self._UA = np.empty(self._tridiag_shape, dtype=complex)
        self._UA_conj = np.empty(self._tridiag_shape, dtype=complex)
        self._U0_half = np.empty(self._tridiag_shape, dtype=complex)
        self._U0_half_conj = np.empty(self._tridiag_shape, dtype=complex)
        self._U0 = np.empty(self._tridiag_shape, dtype=complex)
        self._U0_conj = np.empty(self._tridiag_shape, dtype=complex)


    def eval_energy_expectation_value(self):
        _H0_sf_arr = np.empty_like(self.sf_arr, dtype=complex)
        _M2H0_sf_arr = np.empty_like(self.sf_arr, dtype=complex)
        tridiag_forward(np.asarray(self._M2H0, dtype=complex), self.sf_arr, _M2H0_sf_arr)
        tridiag_backward(self._M2, _H0_sf_arr, _M2H0_sf_arr)
        _energy_exp_val = numerical_integral_trapezoidal(sys.x_arr, self.sf_arr.conj() * _H0_sf_arr)
        return _energy_exp_val
    
    def initialize_sf_arr(self):
        self.sf_arr[:] = self.sf_arr_0
     
    
    def propagate_field_free(self, num_timestep=None, imag_prop=False):
    
        if num_timestep is None:
            if imag_prop: num_timestep = self._max_imag_prop_time_steps
            else: num_timestep = 1
        
        _delta_t = None
        if imag_prop:
            _delta_t = -1.0j * self.delta_t_real
        else: _delta_t = self.delta_t_real
        assert _delta_t is not None

        ## Construct unitary time evolution operators
        self._U0[:] = self._M2 - 1.0j * _delta_t * 0.5 * self._M2H0
        self._U0_conj[:] = self._M2 + 1.0j * _delta_t * 0.5 * self._M2H0
        
        ## Backup the state function for comparison with adjacent timesteps
        _sf_prev_arr = None
        
        ## Prepare intermediate array
        _sf_arr_mid = np.empty_like(self.x_arr, dtype=complex)
        
        ## Iteration for propagation
        for _t_idx in range(num_timestep):
            
            if imag_prop: _sf_prev_arr = self.sf_arr.copy()
                
            tridiag_forward(self._U0, self.sf_arr, _sf_arr_mid)
            tridiag_backward(self._U0_conj, self.sf_arr, _sf_arr_mid)
            
            if imag_prop: normalize_trapezoid(self.x_arr, self.sf_arr)
            if imag_prop:
                _inner_prod = numerical_integral_trapezoidal(self.x_arr, _sf_prev_arr.conj()*self.sf_arr)
                _inner_diff = 1 - np.square(np.abs(_inner_prod))
                print("[{:03d}] difference between previous and current state function: {:.5e}".format(_t_idx, _inner_diff))
                if _inner_diff < self._imag_prop_diff_thres: break
        
    
    def go_to_ground_state(self):
        self.propagate_field_free(num_timestep=self._max_imag_prop_time_steps, imag_prop=True)

    
    def propagate_field_present(self, start_time_index, num_time_step):
        
        ## Allocation
        _sf_arr_mid = np.empty_like(self.sf_arr, dtype=complex)
        
        ## Construct unitary time evolution operators
        self._U0_half[:] = self._M2 - 1.0j * self.delta_t_real * 0.25 * self._M2H0
        self._U0_half_conj[:] = self._M2 + 1.0j * self.delta_t_real * 0.25 * self._M2H0
        self._U0[:] = self._M2 - 1.0j * self.delta_t_real * 0.5 * self._M2H0
        self._U0_conj[:] = self._M2 + 1.0j * self.delta_t_real * 0.5 * self._M2H0
        
        ## iteration for propagation
        _time_index = start_time_index
        _middle_time = self.t0 + self.delta_t_real * (_time_index + 0.5)
        _A_t = self.A_t_func(_middle_time)
        
        self._UA[:] = self._M1 - self.delta_t_real * 0.5 * _A_t * self._D1
        self._UA_conj[:] = self._M1 + self.delta_t_real * 0.5 * _A_t * self._D1

        tridiag_forward(self._U0_half, self.sf_arr, _sf_arr_mid)
        tridiag_backward(self._U0_half_conj, self.sf_arr, _sf_arr_mid)
        
        tridiag_forward(self._UA, self.sf_arr, _sf_arr_mid)
        tridiag_backward(self._UA_conj, self.sf_arr, _sf_arr_mid)
        
        for _time_index in range(start_time_index+1, start_time_index+num_time_step):
            
            _middle_time = self.t0 + self.delta_t_real * (_time_index + 0.5)
            _A_t = self.A_t_func(_middle_time)
            
            self._UA[:] = self._M1 - self.delta_t_real * 0.5 * _A_t * self._D1
            self._UA_conj[:] = self._M1 + self.delta_t_real * 0.5 * _A_t * self._D1

            tridiag_forward(self._U0, self.sf_arr, _sf_arr_mid)
            tridiag_backward(self._U0_conj, self.sf_arr, _sf_arr_mid)
            
            tridiag_forward(self._UA, self.sf_arr, _sf_arr_mid)
            tridiag_backward(self._UA_conj, self.sf_arr, _sf_arr_mid)
            
        tridiag_forward(self._U0_half, self.sf_arr, _sf_arr_mid)
        tridiag_backward(self._U0_half_conj, self.sf_arr, _sf_arr_mid)
