"""Propagator for wavefunction in a polar box"""

from math import pi

import numpy as np
from numpy import asarray, sum

from .wavefunction import Wavefunction

class Wavefunction_on_Uniform_Grid_Polar_Box_Over_r(Wavefunction):
    """
    A support class for managing specification of wavefunction
    defined on a uniform radial grid in a polar box
    """

    @staticmethod
    def eval_at_real_space(wf, dr, phi):
        _phi = asarray(phi)
        if _phi.ndim == 0 and int(_phi) == _phi:
            _phi = np.linspace(0, 2.*pi, int(_phi))
        elif _phi.ndim == 1: pass
        else: raise ValueError("Unacceptable given `phi`: {}".format(phi))
        
        _wf = asarray(wf)
        if _wf.ndim == 1: _wf.reshape((1,_wf.size))
        elif _wf.ndim == 2: pass
        else: raise ValueError("Unacceptable wf dimension: {}".format(_wf.ndim))
        _Nm, _Nr = _wf.shape
        
        assert (_Nm % 2) == 1
        _max_m = int(_Nm // 2)
        _m_arr = np.arange(-_max_m, _max_m+1, dtype=int)
        _exp_imphi = np.exp(1.j*np.outer(_m_arr, _phi))
        
        _r_arr = dr * np.arange(1, _Nr+1)
        
        _wf_real_space = 1./ _r_arr \
                * np.einsum(_wf, [0, 1], _exp_imphi, [0, 2],[2, 1]) 
        return _wf_real_space


    @staticmethod
    def norm_sq(wf, dr):
        """Evalaute the norm square of the given wavefunction array
        
        Parameters
        ----------
        wf : (..., Nm, Nr) array-like
            a single or an array of wavefunction arrays
            with each wavefunction of shape (Nm, Nr)
        dr : float
            a grid spacing of the radial grid
        """
        _wf = asarray(wf)
        _Nr = _wf.shape[-1]
        _r_arr = dr * np.arange(1,_Nr+1)
        _wf_abs_sq = np.real(wf.conj() * wf)
        return 2.* pi * dr * sum(sum(_wf_abs_sq / _r_arr, axis=-1), axis=-1)




from ..evol import (get_M2_tridiag, get_D2_tridiag, mul_tridiag_and_diag, 
                       get_M1_tridiag, get_D1_tridiag)
from ..tridiag import tridiag_forward, tridiag_backward, get_tridiag_shape

class Propagator_on_Uniform_Grid_Polar_Box_Over_r(object):
    """Propagator object defined on a polar box with uniform grid"""

    wf_class = Wavefunction_on_Uniform_Grid_Polar_Box_Over_r
    
    def __init__(self, Nr, dr, m_max, Vr=0.0, hbar=1.0, mass=1.0):
        """Initialize
        
        Parameters
        ----------
        Nr : int
            the number of radial grid points
        dr : float
            the radial grid spacing
        m_max : int
            maximum azimuthal quantum number 'm'
        Vr : (Nr,) array-like
            radially symmetric potential values
            
        Notes
        -----
        The total wavefunction is expanded as:
        
        .. math::
        
            \\psi(r,\\phi,t) = 
            \\frac{1}{r}\\sum_{m=-m_{max}}^{m_{max}}
            {g_{m}(r,t)e^{im\\phi}}

        """
        
        # Check argumetns
        if Nr != int(Nr) or not (Nr > 0):
            _msg = "`Nr` should be a positive integer. Given: {}"
            raise ValueError(_msg.format(Nr))
        self.Nr = int(Nr)
        
        if not (float(dr) > 0):
            _msg = "`dr` should be positive real number. Given: {}"
            raise ValueError(_msg.format(dr))
        self.dr = float(dr)
        
        if m_max != int(m_max) or m_max < 0:
            _msg = "`m_max` should be nonnegative integer. Given: {}"
            raise ValueError(_msg.format(m_max))
        self.m_max = int(m_max)
        
        self.Nm = 2 * self.m_max + 1
        self.m_iter = range(-self.m_max, self.m_max+1)
        
        if Vr == 0.0: self.Vr = np.zeros((self.Nr,), dtype=np.float)
        else:
            _Vr = asarray(Vr)
            if _Vr.shape != (self.Nr,):
                _msg = "`Vr` should be of shape ({},). Given shape: {}"
                raise ValueError(_msg.format(self.Nr, _Vr.shape))
            self.Vr = _Vr
        
        self.hbar, self.mass = hbar, mass
        
        
        # Evaluate matrices for constructing propagator
        self.r_arr = self.dr * np.arange(1, self.Nr+1)
        self.r_max = self.r_arr[-1] + dr
        self.M2 = get_M2_tridiag(self.Nr)
        
        if self.M2.shape != get_tridiag_shape(self.Nr):
            raise Exception("Unexpected inner inconsistency on tridiag shape")
        
        _M2Hm_shape = (self.Nm,) + get_tridiag_shape(self.Nr)
        self.M2Hm = np.empty(_M2Hm_shape, dtype=self.Vr.dtype) 
        
        _D2 = get_D2_tridiag(self.Nr, self.dr)
        
        _hbar2m = 0.5 * self.hbar**2 / self.mass
        _Kr = - _hbar2m * _D2  # something like radial kinetic energy
        
        _r_sq_arr = np.square(self.r_arr)
        for _im, _m in enumerate(self.m_iter):
            _Vm = self.Vr - _hbar2m * (1-_m*_m) / _r_sq_arr
            _M2Vm = mul_tridiag_and_diag(self.M2, _Vm)
            self.M2Hm[_im,:,:] = _Kr + _M2Vm
        
        _D1 = get_D1_tridiag(self.Nr, self.dr)
        self.M1H1 = _hbar2m * _D1 / self.r_arr
        self.M1 = get_M1_tridiag(self.Nr)
    
            
    def propagate(self, wf, dt, Nt=1):
        """Propagate given wavefunction by a given timestep
        
        Parameters
        ----------
        wf : (Nm, Nr) or (Nm*Nr,) array-like
            array of wavefunction values
            where `Nm` is the number of azimuthal basis
            and `Nr` is the number of radial grid points
        dt : float
            timestep to propagate
        Nt : int
            number of timesteps
        """
        _wf = wf
        _wf_1d = np.ravel(wf)
        if _wf_1d.shape != (self.Nm * self.Nr,):
            _msg = ("Inconsistent wavefunction shape given: {}\n"
                    "It should be possible to flatten the shape into: {}")
            raise ValueError(_msg.format(_wf_1d.shape, (self.Nm * self.Nr,)))
        
        if Nt != int(Nt) or not (Nt > 0):
            _msg = "`Nt` should be a positive integer"
            raise ValueError(_msg.format(Nr))
        _Nt = int(Nt)
        
        _FO = (-0.5j*dt/self.hbar) * self.M2Hm
        _unitary_shape = (3,)+_wf_1d.shape
        _unitary_forward_half = np.swapaxes(
                self.M2+_FO, 0, 1).reshape(_unitary_shape)
        _unitary_backward_half = np.swapaxes(
                self.M2-_FO, 0, 1).reshape(_unitary_shape)
        
        _FO1 = (-0.25j*dt/self.hbar)*self.M1H1
        _uni1_forward_half_half = self.M1 + _FO1
        _uni1_backward_half_half = self.M1 - _FO1
        
        
        
        # Iterate over time
        _wf_1d_half = np.empty_like(_wf_1d, dtype=complex)
        _wf_m_half = np.empty((self.Nr,), dtype=complex)
        
        for _it in range(_Nt):
            
            for _im in range(self.Nm):
                tridiag_forward(_uni1_forward_half_half,_wf[_im],_wf_m_half)
                tridiag_backward(_uni1_backward_half_half,_wf[_im],_wf_m_half)
            
            tridiag_forward(_unitary_forward_half, _wf_1d, _wf_1d_half)
            tridiag_backward(_unitary_backward_half, _wf_1d, _wf_1d_half)
            
            for _im in range(self.Nm):
                tridiag_forward(_uni1_forward_half_half,_wf[_im],_wf_m_half)
                tridiag_backward(_uni1_backward_half_half,_wf[_im],_wf_m_half)
            
            
    def propagate_to_ground_state(self, wf, dt=None, max_Nt=20000, 
                                  Nt_per_iter=10, norm_thres=1e-13):
        """Propagate given wavefunction to the ground state of this system"""
        if dt is None: _dt = self.dr / 4.
        else:
            _dt = float(dt)
            if not (_dt > 0): 
                _msg = "`dt` should be a positive real number. Given: {}"
                raise ValueError(_msg.format(_dt))
        _imag_dt = -1.0j * _dt  # imaginary time for propagating to ground state
        
        self.wf_class.normalize(wf, self.dr)
        _wf_prev = wf.copy()
        
        _max_iter = int(max_Nt / Nt_per_iter) + 1
        for _i in range(_max_iter):
            self.propagate(wf, _imag_dt, Nt=Nt_per_iter)
            self.wf_class.normalize(wf, self.dr)
            _norm = self.wf_class.norm_sq(wf - _wf_prev, self.dr)
            if _norm < norm_thres: break
            _wf_prev = wf.copy()
        if _i >= _max_iter-1: raise Exception("Maximum iteration exceeded")
        else: print("iteration count at end: {}".format(_i))

