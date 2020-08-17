"""Module for wavefunction propagator"""

from numbers import Integral

from numpy import asarray

from ._base import Wavefunction, Propagator, _eval_f_and_derivs_by_FD


class Wavefunction_Uniform_1D_Box(Wavefunction):
    """
    An object for specifying the wavefunction in a one-dimensional box
    with uniform (i.e. constant grid spacing) grid
    """
    dim = 1

    def __init__(self, N, dx, x0=0.0):
        """
        Initialize the wavefunction
        
        Parameters
        ----------
        Nr : int
            The number of grid points,
            excluding the both ends where the wavefunction values are zero.
        dr : float
            The spacing for spatial grid. It should be a postive real number.
        """
        # Check and set arguments as members
        if not isinstance(N, Integral) or not (N > 0):
            _msg = "`N` should be a positive integer. Given: {}"
            raise ValueError(_msg.format(N))
        self.N = N
        
        if not (float(dx) > 0):
            _msg = "`dx` should be a positive real number. Given: {}"
            raise ValueError(_msg.format(dx))
        self.dx = float(dx)

        if not isinstance(x0, Real):
            _msg = "`x0` should be a real number. Given: {}"
            raise ValueError(_msg.format(x0))
        self.x0 = x0

        self.shape = (self.N,)
        
        self.x_tot = self.x0 + self.dx * np.arange(1+self.N+1)
#        self.x = self.x0 + self.dx * np.arange(1, self.N+1)
        self.x = self.x_tot[1:-1]
        self.xmax = self.x[-1] + self.dx
        

    @staticmethod
    def norm_sq(wf, dx):
        _wf = asarray(wf)
        _abs_sq_wf = np.real(np.conj(_wf) * _wf)
        return dx * np.sum(_abs_sq_wf)


    @staticmethod
    def inner(wf1, wf2, dx):
        """Evaluate an inner product of given wavefunctions"""
        _wf1, _wf2 = (asarray(_wf) for _wf in (wf1, wf2))
        assert _wf1.shape == _wf2.shape and _wf1.ndim > 0
        assert dx > 0
        return dx * np.sum(np.conj(_wf1) * _wf2, axis=-1)


    def eval_wf_with_wf_deriv_at_x(self, x, wf, with_fd_xlim=False):
        """
        Evaluate wavefunction and its derivative at given coordinate `x`

        Parameters
        ----------
        x : float
            A coordinate at which the wavefunction and its derivative
            is evaluated
        wf : (Nx,) array-like
            A wavefunction defined on a spatial grid with unifrom spacings
            The both ends (i.e. wf[0] and wf[-1]) should be zero as 
            the boundary condition of a 'box'.
        """
        _x = float(x)
        if (_x < self.x0) or (_x >= self.xmax): 
            _msg = "Given coordinate `x`(={}) is out of bound [{},{})"
            raise ValueError(_msg.format(x, self.x0, self.xmax))
        _wf = asarray(wf)
        if (_wf[0] != 0.0) or (_wf[-1] != 0.0):
            _msg = "Both ends of the wavefunction should be zero. Given: {},{}"
            raise ValueError(_msg.format(_wf[0], _wf[-1]))
        _eval_fdf_result = _eval_f_and_derivs_by_FD(_x, _wf, self.dx, 
                _r0=self.x0, _with_fd_rlim=with_fd_xlim)

        if with_fd_xlim: 
            _wf_derivs, _fd_xlim = _eval_fdf_result
            return _wf_derivs[0], _wf_derivs[1], _fd_xlim
        else:
            _wf_derivs = _eval_fdf_result
            return _wf_derivs[0], _wf_derivs[1]



from numbers import Integral, Real

import numpy as np

from tdse.evol import get_D2_tridiag, get_M2_tridiag, mul_tridiag_and_diag
from tdse.tridiag import tridiag_forward, tridiag_backward


class Propagator_on_1D_Box(Propagator):
    """A Propagator for a time-independent Hamiltonian"""

    wf_class = Wavefunction_Uniform_1D_Box
    
    def __init__(self, N, dx, Vx, x0=0.0, hbar=1.0, mass=1.0):

        self.wf = self.wf_class(N, dx, x0=x0)
        for _attr in ("N","dx"):
            setattr(self, _attr, getattr(self.wf, _attr))

        if np.all(Vx == 0.0): self.Vx = np.zeros((self.N,), dtype=np.float)
        elif callable(Vx): self.Vx = asarray([Vx(_x) for _x in self.wf.x])
        else:
            _Vx = asarray(Vx)
            if _Vx.shape != self.wf.shape:
                _msg = "`Vx` should be of shape ({},). Given shape: {}"
                raise ValueError(_msg.format(self.N, _Vx.shape))
            self.Vx = _Vx
        
        if not (hbar > 0.0) or not (mass > 0.0):
            _msg = ("hbar and m (mass) should be positive\n"
                    "Given hbar = {}, m = {}")
            raise TypeError(_msg.format(hbar, m))
        self.hbar, self.mass = hbar, mass
        
        # Construct tridiagonals for propagation
        self.M2 = get_M2_tridiag(self.N)
        self.D2 = get_D2_tridiag(self.N, self.dx)
        _M2V = mul_tridiag_and_diag(self.M2, self.Vx)
        self.M2H = -0.5*self.hbar**2/self.mass * self.D2 + _M2V
        
        
    def propagate(self, sf_arr, dt, Nt=1):
        """Propagate the given state function by the given time interval"""
        assert isinstance(sf_arr, np.ndarray) and sf_arr.dtype == np.complex
        _sf_at_mid_time = np.empty_like(sf_arr)
        _U = self.M2 - 0.5j*dt*self.M2H
        _U_adj = self.M2 + 0.5j*dt*self.M2H
        for _ in range(Nt):
            tridiag_forward(_U, sf_arr, _sf_at_mid_time)
            tridiag_backward(_U_adj, sf_arr, _sf_at_mid_time)

    def propagate_to_ground_state(self, wf=None, dt=None, max_Nt=5000, 
                                  Nt_per_iter=10, norm_thres=1e-13, 
                                  wfs_to_substract=()):
        # Determine the timestep
        _dt = dt
        if dt is None: _dt = self.dx / 4.

        # Check `wfs_to-substract`
#        assert type(wfs_to_substract) in (list, tuple, np.ndarray)
    
        # Determine the wavefucntion
        if wf is None: 
            _wf = np.empty(self.wf.shape, dtype=np.complex)
            _wf[:] = np.random.rand(*_wf.shape)
        else: _wf = np.asarray(wf)

        # Propagate to the ground state
        _normalizer_args = (self.dx,)
        super().propagate_to_ground_state(_wf, _dt, max_Nt, _normalizer_args, 
                Nt_per_iter, norm_thres, wfs_to_substract)

        if wf is None: return _wf
    

    def evaluate_energy_expectation_value(self, wf):
        _wf = asarray(wf)
        assert _wf.shape == (self.N,)
        _temp1 = np.empty_like(_wf)
        tridiag_forward(self.M2H.astype(np.complex), _wf, _temp1)
        _temp2 = np.empty_like(_wf)
        tridiag_backward(self.M2, _temp2, _temp1)
        _energy_expect_val = self.wf_class.inner(_wf, _temp2, self.dx)
        assert abs(_energy_expect_val.imag) < 1e-15
        return _energy_expect_val.real



# Aliasing
Time_Indep_Hamil_Propagator = Propagator_on_1D_Box 



from numbers import Real, Integral

from tdse.propagator.box1d import Propagator_on_1D_Box
from tdse.evol import get_D1_tridiag, get_M1_tridiag
from tdse.tridiag import tridiag_forward, tridiag_backward

class Propagator_on_1D_Box_with_field(Propagator_on_1D_Box):
    def __init__(self, N, dx, Vx, At, q=-1.0, x0=0.0, hbar=1.0, mass=1.0):
        """
        Initalize
        
        Parameters
        ----------
        At : callable of a single real number t (time)
            vector potential as a function of time
        q : float
            charge of a particle described by the wavefunction
        """
        
        if not callable(At):
            _msg = "`At` should be a callable. Given: {}"
            raise ValueError(_msg.format(At))
        self.A = At
        
        if not isinstance(float(q), Real):
            _msg = "`q` should be real number. Given: {}"
            raise ValueError(_msg.format(q))
        self.q = float(q)
        
        # Process arguments that is common with parent propagator
        # and construct matrices that is common with field-absent case
        super().__init__(N, dx, Vx, x0=x0, hbar=hbar, mass=mass)
        
        # Construct matrices for field-present case
        self.M1 = get_M1_tridiag(self.N)
        _D1 = get_D1_tridiag(self.N, self.dx)
        self.M1HA_over_ihbar_At = (self.q / self.mass) * _D1
        
        
    def propagate_with_field(self, wf, dt, t_start, Nt=1):
        """Propagate the wavefunction in the presence of the field
        
        Notes
        -----
        When the field strength is zero,
        consider using `propagate()` method for calculation performance
        
        Returns
        -------
        t_final : float
            time after the propagation ends
        """
        if not isinstance(wf, np.ndarray) or wf.dtype != np.complex:
            raise ValueError("The wavefunction should be a complex type numpy array")
        _wf = wf
        
        if not isinstance(float(t_start), Real):
            raise ValueError("`t_start` should be a real number. Given: {}".format(t_start))
        _t_start = float(t_start)
        
        if not isinstance(float(dt), Real):
            raise ValueError("`dt` should be a real number. Given: {}".format(dt))
        _dt = float(dt)
        
        if not isinstance(Nt, Integral) or Nt <= 0:
            raise ValueError("`Nt` should be a positive integer. Given: {}".format(Nt))
        _Nt = Nt
        
        _quarter_dt_M2H0_over_ihbar = (-0.25j * dt / self.hbar) * self.M2H
        _M2U0_forward_quarter = self.M2 + _quarter_dt_M2H0_over_ihbar
        _M2U0_backward_quarter = self.M2 - _quarter_dt_M2H0_over_ihbar
        
        _wf_mid = np.empty_like(_wf, dtype=_wf.dtype)
        _t = _t_start
        for _it in range(_Nt):
            
            tridiag_forward(_M2U0_forward_quarter, _wf, _wf_mid)
            tridiag_backward(_M2U0_backward_quarter, _wf, _wf_mid)
            
            _half_dt_M1HA_over_ihbar = (0.5*dt * self.A(_t+0.5*_dt)) * self.M1HA_over_ihbar_At
            _M1UA_forward_half = (self.M1 + _half_dt_M1HA_over_ihbar).astype(np.complex)
            _M1UA_backward_half = (self.M1 - _half_dt_M1HA_over_ihbar).astype(np.complex)
            tridiag_forward(_M1UA_forward_half, _wf, _wf_mid)
            tridiag_backward(_M1UA_backward_half, _wf, _wf_mid)
            
            tridiag_forward(_M2U0_forward_quarter, _wf, _wf_mid)
            tridiag_backward(_M2U0_backward_quarter, _wf, _wf_mid)
            
            _t += _dt
            
        return _t

