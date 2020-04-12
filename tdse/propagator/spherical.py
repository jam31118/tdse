"""Propagator for wavefunction defined in spherical coordinate system"""

from numpy import pi, asarray

from ._base import Wavefunction, _eval_f_and_derivs_by_FD



from numbers import Integral

from scipy.special import lpmn, factorial
import numpy as np
from numpy import cos, sin

def Plm_and_dtheta_Plm_for_single_m(m, lmax, theta):
    """
    Evaluate associate Legendre function of cosine `theta` and its derivative 
    with respect to `theta` for fixed order `m` and maximum degree `lmax`

    Parameters
    ----------
    m : int
        order of the Legendre function
        should satisfy: `|m| <= lmax`
    lmax : int
        maximum degree of the Legendre function
        should satisfy: `|m| <= lmax`
    theta : float
        The associated Legendre function and its derivative is evalauted
        for its cosine value.
        should satisfy: `sin(theta) >= 0`
    """
    # Check parameters
    assert isinstance(m, Integral) and isinstance(lmax, Integral)
    _abs_m = abs(m)
    assert lmax >= _abs_m 
    _sin_theta, _cos_theta = sin(theta), cos(theta)
    if _sin_theta < 0: 
        _msg = "sin(theta) should be nonnegative. Given: {}"
        raise ValueError(_msg.format(_sin_theta))
    
    # Evaluate
    _Nlm = lmax - _abs_m + 1
    _l = _abs_m + np.arange(_Nlm)
    _Plm = np.empty((_Nlm,), dtype=np.float)
    _dtheta_Plm = np.empty((_Nlm,), dtype=np.float)
    
    if _cos_theta == 1:
        _Plm[:] = ( m == 0 )
        _dtheta_Plm = ( m == 1 ) * (-_l*(_l+1)/2.) \
                    + ( m == -1 ) * 0.5
    elif _cos_theta == -1:
        _Plm[:] = ( m == 0 ) 
        _dtheta_Plm[:] = ( m == 1 ) * (-1)**(_l+1) * (_l*(_l+1)/2.) \
                        + ( m == -1 ) * (-1)**_l * 0.5
    else:
        _Plm_all, _dx_Plm_all = lpmn(m, lmax, _cos_theta)
        _Plm, _dx_Plm = _Plm_all[-1,_abs_m:], _dx_Plm_all[-1,_abs_m:]
        _dtheta_Plm = - _sin_theta * _dx_Plm
    return _Plm, _dtheta_Plm



class Wavefunction_on_Spherical_Box_with_single_m(Wavefunction):
    """
    A specification object for wavefunction defined on a spherical box
    with a single azimuthal quantum number `m`
    """
    
    dim = 3

    def __init__(self, Nr, dr, m, lmax):
        """Initialize"""
        
        # Check and set arguments as members
        if not isinstance(Nr, Integral) or not (Nr > 0):
            _msg = "`Nr` should be a positive integer. Given: {}"
            raise ValueError(_msg.format(Nr))
        self.Nr = Nr
        
        if not (float(dr) > 0):
            _msg = "`dr` should be a positive real number. Given: {}"
            raise ValueError(_msg.format(dr))
        self.dr = float(dr)
        
        if not isinstance(m, Integral) or m < 0:
            _msg = "`m` should be a nonnegative integer. Given: {}"
            raise ValueError(_msg.format(m))
        self.m = m
        
        if not isinstance(lmax, Integral) or lmax < m:
            _msg = "`lmax` should be an integer and `>= m`. Given: {}"
            raise ValueError(_msg.format(lmax))
        self.lmax = lmax

        # Set some members
        self.l = np.arange(self.m, self.lmax+1, dtype=int)
        _Nl = self.l.size
        self.Nlm = _Nl
        self.lm = np.array([(l,self.m) for l in self.l], dtype=int)
        
        self.r_arr = self.get_r_arr(self.Nr, self.dr)
        self.r_max = self.r_arr[-1] + dr

        self.shape = (self.Nlm, self.Nr)
        
        # About spherical harmonics evaluation
        self.have_sph_harm_coef = False


    
    @staticmethod
    def get_r_arr(Nr, dr):
        _r_arr = dr * np.arange(1, Nr+1)
        return _r_arr
    
    @classmethod
    def norm_sq(cls, wf, dr):
        """Evalaute the norm square of the given wavefunction array
        
        Parameters
        ----------
        wf : (..., Nlm, Nr) array-like
            a single or an array of wavefunction arrays
            with each wavefunction of shape (Nlm, Nr)
        dr : float
            a grid spacing of the radial grid
        """
        _wf = asarray(wf)
        _Nr = _wf.shape[-1]
        _r_arr = cls.get_r_arr(_Nr, dr)
        _wf_abs_sq = np.real(_wf.conj() * _wf)
        _norm_sq_lm = np.sum(_wf_abs_sq, axis=-1)
        _norm_sq_total = np.sum(_norm_sq_lm, axis=-1)
        _norm_sq_total *= dr
        return _norm_sq_total

    @classmethod
    def get_each_dimension_of_wf_array(cls, wf):
        _wf = asarray(wf)
        if _wf.ndim not in range(1,cls.dim+1): 
            raise ValueError("Unexpected dimension `wf`: {}".format(_wf.ndim))
        _Nlm, _Nr = (1, _wf.size) if _wf.ndim == 1 else _wf.shape
        return _Nlm, _Nr

    def wf2Rlm(self, wf):
        """
        Evaluate `Rlm = 1/r * wf` 
        with proper boundary condition at r = 0 and rmax
        """
        _wf = asarray(wf)
        _Nlm, _Nr = self.get_each_dimension_of_wf_array(_wf)
        if _Nr != self.Nr or _Nlm != self.Nlm:
            _msg = "The `wf` has inconsistent shape:\nExpected={}\nGiven={}"
            raise ValueError(_msg.format((self.Nlm, self.Nr), _wf.shape))
        
        _Rlm_shape = (self.Nlm, 1+self.Nr+1)
        _Rlm = np.empty(_Rlm_shape, dtype=_wf.dtype)
        _Rlm[:,[0,-1]] = 0.0
        _Rlm[:,1:-1] = _wf / self.r_arr
        if self.m == 0:
            _Rlm[0,0] = 2.*_Rlm[0,1] - _Rlm[0,2]
            
        return _Rlm
    
    def eval_wf_with_wf_deriv_at_q(self, q, Rlm):
        """
        Evaluate wavefunction and its partial derivatives at given coordinate

        Parameters
        ----------
        q : (3,) array-like
            coordinate vector : (r, theta, phi)
            - r : radial coordinate
            - theta : polar angle in radian
            - phi : azimuthal angle in radian
        Rlm : (Nlm, Nr) or (Nr,) array-like
            radial function(s) in this wavefunction's expansion
        """
        _q = asarray(q)
        if _q.shape != (self.dim,):
            _msg = "Unexpected shape of the coordinate vector `q`. Given: {}"
            raise ValueError(_msg.format(q))

        _r, _theta, _phi = _q
        _sin_theta, _cos_theta = sin(_theta), cos(_theta)
        if _r >= 0 and _sin_theta >= 0:
            pass
        elif _r >= 0 and _sin_theta < 0:
            _sin_theta, _phi = -_sin_theta, _phi + pi
        elif _r < 0 and _sin_theta >= 0:
            _r, _cos_theta, _phi = -_r, -_cos_theta, _phi + pi
        elif _r < 0 and _sin_theta < 0:
            _r, _cos_theta, _sin_theta = -_r, -_cos_theta, -_sin_theta
        else: raise Exception("Unexpected case for r and sin_theta")

        assert _sin_theta >= 0.0 and _r >= 0.0
        _theta = np.arctan2(_sin_theta, _cos_theta)

        if not (_r < self.r_max):
            _msg = "`r` is out of the box radius (={}). `r`={}"
            raise ValueError(_msg.format(self.r_max, _r))
        
        # Evaluate Rlm and its derivatives
        _Rlm = asarray(Rlm)
        _Nlm, _Nr_total = self.get_each_dimension_of_wf_array(_Rlm)
        if _Nlm != self.Nlm or _Nr_total != 1+self.Nr+1:
            raise ValueError("Inconsistent array shape for `Rlm`")
        _Rlm_derivs = _eval_f_and_derivs_by_FD(_r, _Rlm, self.dr)

        # Evaluate associated Legendre functions
        _Plm, _dtheta_Plm = Plm_and_dtheta_Plm_for_single_m(
                self.m, self.lmax, _theta)
#        print("r,theta,phi == ({},{},{})".format(_r, _theta, _phi))
#        print("_Plm:{}".format(_Plm))
#        print("_dtheta_Plm:{}".format(_dtheta_Plm))
        
        # Evaluate coefficients of spherical harmonics
        if not self.have_sph_harm_coef: self._eval_sph_harm_coef()
        _clm = self.sph_harm_coef_arr  # aliasing
        
        # Evaluate azimuthal part
        _exp_imphi = np.exp(1.j*self.m*_phi)

        # Evalute wavefunction and its partial derivatives
        _clm_Plm = _clm * _Plm
        _wf_q = np.sum(_Rlm_derivs[0] * _clm_Plm) * _exp_imphi
        _dr_wf_q = np.sum(_Rlm_derivs[1] * _clm_Plm) * _exp_imphi
        _dtheta_wf_q = np.sum(_Rlm_derivs[0] * _clm * _dtheta_Plm) * _exp_imphi
        _dphi_wf_q = 1.j * self.m * _wf_q
        _partial_derivs_wf_q = np.array(
                [_dr_wf_q, _dtheta_wf_q, _dphi_wf_q], dtype=np.complex)

        return _wf_q, _partial_derivs_wf_q


    def _eval_sph_harm_coef(self):
        _l, _m = self.l, self.m
        self.sph_harm_coef_arr = np.sqrt(
                (2*_l+1)/(4.*pi)*factorial(_l-_m)/factorial(_l+_m))
        self.have_sph_harm_coef = True 


    def h_q(self, q):
        """
        Evaluate an array of 
        $h_{i} \equiv |\partial{\mathbf{r}}/\parital{q_{i}}|$
        """
        _h = np.array([1.0, q[0], q[0]*sin(q[1])])
        return _h

        


from numbers import Integral

import numpy as np

from ..evol import (get_M2_tridiag, get_D2_tridiag, 
                       mul_tridiag_and_diag)
from ..tridiag import tridiag_forward, tridiag_backward

from ._base import Propagator

class Propagator_on_Spherical_Box_with_single_m(Propagator):
    """
    Propagator designed to propagate a wavefunction 
    defined on a spherical box with single azimuthal quantum number m
    
    The wavefunction is expanded by a set of spherical harmonics.
    The radial functions are discretized on a uniform grid,
    i.e. with a fixed grid spacing."""
    
    wf_class = Wavefunction_on_Spherical_Box_with_single_m
    
    def __init__(self, Nr, dr, m, lmax, Vr=0.0, hbar=1.0, mass=1.0):

        # Construct wavefunction object from parameters
        self.wf = self.wf_class(Nr, dr, m, lmax)

        # Copy parameters
        for _attr in ("Nr","dr","m","lmax","l","Nlm","lm","r_max","r_arr"):
            setattr(self, _attr, getattr(self.wf, _attr))

        self.hbar, self.mass = hbar, mass
        
        if Vr == 0.0: self.Vr = np.zeros((self.Nr,), dtype=np.float)
        else:
            _Vr = asarray(Vr)
            if _Vr.shape != (self.Nr,):
                _msg = "`Vr` should be of shape ({},). Given shape: {}"
                raise ValueError(_msg.format(self.Nr, _Vr.shape))
            self.Vr = _Vr
        
        _D2 = get_D2_tridiag(self.Nr, self.dr)
        self.M2 = get_M2_tridiag(self.Nr)
        
        # Correction for Coulomb potential
#         _D2[1,0] = 
#         self.M2[1,0] = 
        
        _M2Hl_shape = (self.Nlm,) + self.M2.shape
        self.M2Hl = np.empty(_M2Hl_shape, dtype=self.Vr.dtype)
        _hbar_sq_over_2mass = self.hbar**2 / (2.*self.mass)
        _Kr = - _hbar_sq_over_2mass * _D2
        _r_sq = np.square(self.r_arr)
        for _il, _l in enumerate(self.l):
            _Vl = _hbar_sq_over_2mass * _l * (_l+1) / _r_sq + self.Vr
            _M2Vl = mul_tridiag_and_diag(self.M2, _Vl)
            self.M2Hl[_il] = _Kr + _M2Vl
            
    def propagate(self, wf, dt, Nt=1):
        if Nt < 0: raise ValueError(
            "Nt should be a nonnegative integer. Given: {}".format(Nt))
        _FO = (-0.5j*dt/self.hbar) * self.M2Hl
        _Uf_half = self.M2 + _FO # unitary half timestep prop forward
        _Ub_half = self.M2 - _FO # unitary half timestep prop backward
        _wf_lm_mid = np.empty((self.Nr,), dtype=wf.dtype)
        for _it in range(Nt):
            for _ilm in range(self.Nlm):
                tridiag_forward(_Uf_half[_ilm], wf[_ilm], _wf_lm_mid)
                tridiag_backward(_Ub_half[_ilm], wf[_ilm], _wf_lm_mid)
    
    def propagate_to_ground_state(self, wf=None, dt=None, max_Nt=20000,
                                  Nt_per_iter=10, norm_thres=1e-13):
        _dt = dt
        if dt is None: _dt = self.dr / 4.
        
        if wf is None: 
            _wf = np.empty((self.Nlm, self.Nr), dtype=np.complex)
            _wf[:] = np.random.rand(*_wf.shape)
        else: _wf = np.asarray(wf)
            
        _normalizer_args = (self.dr,)
        super().propagate_to_ground_state(_wf, _dt, max_Nt,
            _normalizer_args, Nt_per_iter, norm_thres)
        
        if wf is None: return _wf



