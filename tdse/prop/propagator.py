"""Module for wavefunction propagator"""

from numbers import Integral

import numpy as np

from ..evol import get_D2_tridiag, get_M2_tridiag, mul_tridiag_and_diag
from ..tridiag import tridiag_forward, tridiag_backward

class Time_Indep_Hamil_Propagator(object):
    """A Propagator for a time-independent Hamiltonian"""
    
    def __init__(self, N, dx, Vx, hbar=1.0, m=1.0):
        """Initialize propagator

        The propagator propagates a wavefunction defined on a spatial grid.
        The spatial grid spacing is assumed to be equal.
        
        Parameters
        ----------
        N : int
            the number of spatial grid points
        dx : float
            grid spacing of the spatial grid
        Vx : (N,) array-like
            poential energy values on the spatial grid points
        hbar : float
            the Planck constant over 2pi
        m : float
            the mass of the particle
        """
        
        # Check arguments
        if not isinstance(N, Integral) or not (N > 0):
            _msg = "`N` should be a positive integer. Given: {}"
            raise TypeError(_msg.format(N))
        self.N = N
        
        if not (dx > 0):
            _msg = "`dx` should be positive. Given: {}"
            raise TypeError(_msg.format(dx))
        self.dx = dx
            
        try: _Vx = np.array(Vx, copy=False)
        except:
            _msg = "Failed to convert the given potential array: `Vx`"
            raise Exception(_msg.format(Vx))
        if _Vx.shape != (N,):
            _msg = ("The potential array should have shape (N,)==({},)\n"
                    "Given: {}")
            raise TypeError(_msg.format(N, Vx))
        self.Vx = _Vx
        
        if not (hbar > 0.0) or not (m > 0.0):
            _msg = ("hbar and m (mass) should be positive\n"
                    "Given hbar = {}, m = {}")
            raise TypeError(_msg.format(hbar, m))
        self.hbar, self.m = hbar, m
        
        # Construct tridiagonals for propagation
        self.M2 = get_M2_tridiag(self.N)
        self.D2 = get_D2_tridiag(self.N, self.dx)
        _M2V = mul_tridiag_and_diag(self.M2, self.Vx)
        self.M2H = -0.5*self.hbar**2/self.m * self.D2 + _M2V
        
        
    def propagate(self, sf_arr, dt, Nt=1):
        """Propagate the given state function by the given time interval"""
        assert isinstance(sf_arr, np.ndarray) and sf_arr.dtype == np.complex
        _sf_at_mid_time = np.empty_like(sf_arr)
        _U = self.M2 - 0.5j*dt*self.M2H
        _U_adj = self.M2 + 0.5j*dt*self.M2H
        for _ in range(Nt):
            tridiag_forward(_U, sf_arr, _sf_at_mid_time)
            tridiag_backward(_U_adj, sf_arr, _sf_at_mid_time)
