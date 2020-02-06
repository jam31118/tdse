#### Definition
# $$
# \begin{split}
# \beta & \equiv log(I_{0}/I)\\
# I & = I_{0} e^{-\beta}
# \end{split}
# $$



#### Assumption: in vacuum

# In vacuum, where the refractive index $n = 1$, the Rayleigh range becomes:

# $$
# z_{R} = \frac{\pi \omega_{0}^2 n}{\lambda} = \frac{\pi \omega_{0}^2}{\lambda}
# $$



#### Definition

# $$
# \begin{align}
# z_{0} & \equiv z_{R} \sqrt{I_{0}/I - 1} \\
# & = z_{R} \sqrt{e^{\beta}-1}
# \end{align}
# $$



#### Expression

# The level set of a given intensity for the Gaussian beam 
# is a surface in real space which can be obtained by rotating a curve 
# $\rho(z)$ on $z-\rho$ plane in cylindrical coordinates.

# $$
# \rho^2(z) = \frac{\omega_{0}^2}{2}{g_{1}\left(\frac{z}{z_{R}}\right)}
#    \left( \beta - log\left({g_{1}\left(\frac{z}{z_{R}}\right)}\right) \right)
# $$


import numpy as np
from numpy import log, arctan, pi, sqrt, exp


def rho_sq_z_func(z, beta, omega0, zR):
    _u = z / zR
    _g1_u = 1.0 + _u**2
    _rho_sq_z = omega0**2 / 2.0 * _g1_u * (beta - log(_g1_u))
    return _rho_sq_z

def dbeta_rho_sq_func(z, beta, omega0, zR):
    _u = z / zR
    _g1_u = 1.0 + _u**2
    _dbeta_rho_sq = omega0**2 / 2.0 * _g1_u
    return _dbeta_rho_sq

def dz_rho_sq_func(z, beta, omega0, zR):
    _u = z / zR
    _g1_u = 1.0 + _u**2
    _dz_g1 = 1.0 / zR * 2.0 * _u
    _dz_rho_sq = omega0**2 / 2.0 * _dz_g1 * (beta - log(_g1_u) - 1.0)
    return _dz_rho_sq

def integral_rho_sq_func(z, beta, omega0, zR):
    _u = z / zR
    _g1_u = 1.0 + _u**2
    _G1_u = _u * (1.0 + (1.0 / 3.0) * _u**2)
    _G2_u = _G1_u * log(_g1_u) - 2.0 / 9.0 * (6.0*_u + _u**3 - 6.0*arctan(_u))
    _integral_rho_sq = omega0**2 / 2.0 * zR * (_G1_u * beta - _G2_u)
    return _integral_rho_sq




class GaussianBeam(object):
    pass


from scipy.optimize import brentq

class DoubleGaussianBeam(GaussianBeam):
    def __init__(self, omega0_um_vec, lambda_nm_vec):
        """Initialize
        
        Parameters
        ----------
        omega0_um_vec : (2,) array_like
            beam waist of each beam in micrometres
        lambda_nm_vec : (2,) array_like
            wavelength of each beam in nanometres
            
        Notes
        -----
        The default unit of length is micrometers.
        """
        
#### Get Parameters

        _omega0_um_vec, _lambda_nm_vec = (
                np.array(_a, copy=False, dtype=float) 
                for _a in (omega0_um_vec, lambda_nm_vec))
        
        for _a in (_omega0_um_vec, _lambda_nm_vec): assert np.all(_a > 0)
        
        self.omega0_vec = _omega0_um_vec
        self.lambda_vec = 1.0e-3 * _lambda_nm_vec
        
        self.zR_vec = pi * self.omega0_vec ** 2 / self.lambda_vec
        
        
    def _dbeta1_dbeta2_intersection_volume_at_beta(self, beta1, beta2):
        
        _beta1_arr, _beta2_arr = (
                np.array(_a, copy=False, dtype=float) for _a in (beta1, beta2))
        
        assert _beta1_arr.shape == _beta2_arr.shape
        _beta_grid_shape = _beta1_arr.shape
        
        _pdf_beta_grid = np.empty(_beta_grid_shape, dtype=float)
        
        for _ind in np.ndindex(_beta_grid_shape):
            
            _beta_vec = np.array(
                    [_beta1_arr[_ind], _beta2_arr[_ind]], dtype=float)
            
            _z_roots = self.solve_rho1_eq_rho2(*_beta_vec)
            
            _pdf_beta_grid[_ind] = self._pdf_beta_from_rho_z_roots(
                    _z_roots, _beta_vec)
        
        return _pdf_beta_grid
        

    def num_of_roots_of_rho1_eq_rho2_for_beta_array(self, beta1, beta2):
        
        _beta1_arr, _beta2_arr = (
                np.array(_a, copy=False, dtype=float) for _a in (beta1, beta2))
        
        assert _beta1_arr.shape == _beta2_arr.shape
        _beta_grid_shape = _beta1_arr.shape
        
        _num_of_roots = np.empty(_beta_grid_shape, dtype=float)
        
        for _ind in np.ndindex(_beta_grid_shape):
            _beta_vec = np.array([_beta1_arr[_ind], _beta2_arr[_ind]])
            _z_roots = self.solve_rho1_eq_rho2(*_beta_vec)
            _num_of_roots[_ind] = len(_z_roots)

        return _num_of_roots

        
    def prob_density_of_beta(self, beta1, beta2, beta1_max, beta2_max):
        """Evaluate probability density funciton at given beta values"""
        
        _max_volume = self.intersection_volume(beta1_max, beta2_max)
        _normalizing_constant = 1.0 / _max_volume
        
        _prob_density_beta = self._dbeta1_dbeta2_intersection_volume_at_beta(beta1, beta2)
        _prob_density_beta *= _normalizing_constant
        
        return _prob_density_beta
        
            
    def solve_rho1_eq_rho2(self, beta1, beta2):
        
        _beta_vec = np.array([beta1, beta2], dtype=float)
        assert np.all(_beta_vec) > 0.0
        
    # Find roots $\{z^{i}_{r}\}$ of $\rho^2_{1}(z,\beta)=\rho^2_{2}(z,\beta')$

    #### Construct $z$ array on which an integration will be done

    #### [TODO] Evaluate the resolution of roots with respect to grid spacing of $z$-array
        
        _z0_vec = self.zR_vec * sqrt(exp(_beta_vec) - 1)
        _z0_min, _z0_max = np.amin(_z0_vec), np.amax(_z0_vec)
        assert _z0_min > 0 and _z0_min <= _z0_max

        _dz_max = 1e-2 * self.zR_vec.min()
        _z_arr_min, _z_arr_max = 0.0, _z0_min
        _Nz = int((_z_arr_max - _z_arr_min + _dz_max) / _dz_max) + 1
        _z_arr = np.linspace(_z_arr_min, _z_arr_max, _Nz)
        _dz = _z_arr[1] - _z_arr[0]
        assert _dz <= _dz_max

    #### Evaluate $\rho^{2}$ at given $z$-array

        _rho_sq_z_vec = np.array([rho_sq_z_func(_z_arr, _beta, _omega0, _zR) 
                                 for _beta, _omega0, _zR 
                                 in zip(_beta_vec, self.omega0_vec, self.zR_vec)])

    #### Find all intervals where $\rho_{1}^2 = \rho_{2}^2$ has a root 
    #### .. - ideally one root per one interval found

        _rho_sq_z_0, _rho_sq_z_1 = _rho_sq_z_vec
        _v = _rho_sq_z_1 > _rho_sq_z_0
        _vv = np.logical_xor(_v[:-1], _v[1:])
        _interval_indices, = np.where(_vv)

    #### Find roots

        _fargs = (_beta_vec, self.omega0_vec, self.zR_vec)
        _ztol = 2e-12

        _z_roots_list = []
        for _i in _interval_indices:
            try: _root, _root_result = brentq(
                self.rho_sq_diff, _z_arr[_i], _z_arr[_i+1], 
                args=_fargs, xtol=_ztol, full_output=True, disp=True)
            except RuntimeError as e:
                if not _root_result.converged: raise Exception("The root is not converged.")
                else: raise Exception("Unexpected error even though the root is converged")
            except: raise Exception("Unexpected error")
            _z_roots_list.append(_root)
        _z_roots = np.array(_z_roots_list)

#         _N_roots = _z_roots.size
#         N_roots_beta_grid[i_beta1, i_beta2] = _N_roots


    #### check equality of both rho_sq at found roots

        _rho_sq_z_roots_0, _rho_sq_z_roots_1 = (rho_sq_z_func(_z_roots, _beta, _omega0, _zR) 
                                            for _beta, _omega0, _zR 
                                            in zip(_beta_vec, self.omega0_vec, self.zR_vec))

        if not np.all(np.abs(_rho_sq_z_roots_1 - _rho_sq_z_roots_0) < _ztol):
            raise ValueError("rho_sq_z_roots_1: {}, diff: {}".format(
                _rho_sq_z_roots_1, _rho_sq_z_roots_1 - _rho_sq_z_roots_0))
        _rho_sq_z_roots = _rho_sq_z_roots_0
        
        
    #### Return result
    
        return _z_roots

            
    @staticmethod
    def rho_sq_diff(z, beta_vec, omega0_vec, zR_vec):
        _rho_sq_z_0, _rho_sq_z_1 = (rho_sq_z_func(z, beta, omega0, zR) 
                                    for beta, omega0, zR 
                                    in zip(beta_vec, omega0_vec, zR_vec))
        return _rho_sq_z_1 - _rho_sq_z_0

    
    def _pdf_beta_from_rho_z_roots(self, _z_roots, _beta_vec):
        
        _ind_map_from_onetwo_to_min = self.get_min_ind_map(_beta_vec)
        
        _beta_min_vec = _beta_vec[_ind_map_from_onetwo_to_min]
        _omega0_min_vec = self.omega0_vec[_ind_map_from_onetwo_to_min]
        _zR_min_vec = self.zR_vec[_ind_map_from_onetwo_to_min]
        
        _pdf_beta = 0.0
        for _i in range(len(_z_roots)):
            _z_r_i = _z_roots[_i]

            _dbeta_rho_sq_min_i_mod2, _dbeta_rho_sq_min_ip1_mod2 = (
                dbeta_rho_sq_func(_z_r_i, _beta_min_vec[_ind], _omega0_min_vec[_ind], _zR_min_vec[_ind]) 
                for _ind in (_i%2, (_i+1)%2)
            )

            _dz_rho_sq_min_i_mod2, _dz_rho_sq_min_ip1_mod2 = (
                dz_rho_sq_func(_z_r_i, _beta_min_vec[_ind], _omega0_min_vec[_ind], _zR_min_vec[_ind])
                for _ind in (_i%2, (_i+1)%2)
            )
        
            _numer = _dbeta_rho_sq_min_i_mod2 * _dbeta_rho_sq_min_ip1_mod2 
            _denom = _dz_rho_sq_min_ip1_mod2 - _dz_rho_sq_min_i_mod2
            assert _denom != 0
            _pdf_beta_i = - pi * 2.0 * _numer / _denom

            _pdf_beta += _pdf_beta_i

        # [NOTE] `_pdf_beta_i` does not have to be positive (prove this)
        assert _pdf_beta >= 0
        
        return _pdf_beta


    def get_min_ind_map(self, beta_vec):
        """Return [0,1] or [1,0] depending on order."""

        _rho0_sq_vec = self.omega0_vec ** 2 / 2.0 * beta_vec
#         if _rho0_sq_vec[0] == _rho0_sq_vec[1]:
#             raise ValueError("beta values at out of beta-support")
        _first_rho0_is_smaller = _rho0_sq_vec[0] <= _rho0_sq_vec[1]
        _ind_map_from_onetwo_to_min = np.array(
            [1 - _first_rho0_is_smaller, _first_rho0_is_smaller], dtype=int)

        return _ind_map_from_onetwo_to_min
    
    
    def intersection_volume(self, beta1, beta2):
        
        _beta1_arr, _beta2_arr = (np.array(_a, copy=False, dtype=float) for _a in (beta1, beta2))
        
        assert _beta1_arr.shape == _beta2_arr.shape
        _beta_grid_shape = _beta1_arr.shape
        
        _intersec_vol_arr = np.empty(_beta_grid_shape, dtype=float)
        
        for _ind in np.ndindex(_beta_grid_shape):
            _beta_vec = np.array([_beta1_arr[_ind], _beta2_arr[_ind]], dtype=float)
            _intersec_vol_arr[_ind] = self._intersection_volume_at_single_beta(_beta_vec)
            
        return _intersec_vol_arr
    
    
    def _intersection_volume_at_single_beta(self, beta_vec):
        
        _beta_vec = np.array(beta_vec, copy=False, dtype=float)
        assert _beta_vec.shape == (2,)
#         _z_roots = np.array(z_roots, copy=False, dtype=float)
        _z_roots = self.solve_rho1_eq_rho2(*_beta_vec)
        
    #### Prepare min-indice-arrays
        _ind_map_from_onetwo_to_min = self.get_min_ind_map(_beta_vec)
        
        _beta_min_vec = _beta_vec[_ind_map_from_onetwo_to_min]
        _omega0_min_vec = self.omega0_vec[_ind_map_from_onetwo_to_min]
        _zR_min_vec = self.zR_vec[_ind_map_from_onetwo_to_min]
        
        _z0_vec = self.zR_vec * sqrt(exp(_beta_vec) - 1)
        _z0_min_vec = _z0_vec[_ind_map_from_onetwo_to_min]
        
    #### Evaluate volume at end
        _N_roots = len(_z_roots)
        _vol_end = pi * 2.0 * integral_rho_sq_func(_z0_min_vec[_N_roots%2], _beta_min_vec[_N_roots%2], 
                                                   _omega0_min_vec[_N_roots%2], _zR_min_vec[_N_roots%2])
        
    #### Evaluate target volume
    #### .. We start from volume at end and substract part by part to get the final volume
        _vol_intersect = _vol_end
        for _i in range(_N_roots):
            _z_r_i = _z_roots[_i]

            _intergral_rho_sq_min_i_mod2, _intergral_rho_sq_min_ip1_mod2 = (
                integral_rho_sq_func(_z_r_i, _beta_min_vec[_ind], 
                                     _omega0_min_vec[_ind], _zR_min_vec[_ind])
                for _ind in (_i%2, (_i+1)%2)
            )
            _vol_i_lower = pi * 2.0 * _intergral_rho_sq_min_i_mod2
            _vol_i_upper = pi * 2.0 * _intergral_rho_sq_min_ip1_mod2

            _vol_intersect -= _vol_i_upper - _vol_i_lower
        
        return _vol_intersect
    
    
    def cumulative_dist_func_of_beta(self, beta1, beta2, beta1_max, beta2_max):
        """Evaluate probability density funciton at given beta values"""
        
        _max_volume = self.intersection_volume(beta1_max, beta2_max)
        _normalizing_constant = 1.0 / _max_volume
        
        _cumulative_dist_func_arr = self.intersection_volume(beta1, beta2)
        _cumulative_dist_func_arr *= _normalizing_constant
        
        return _cumulative_dist_func_arr
