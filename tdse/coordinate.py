import numpy as np
from numpy import pi, sin, cos


def transform_to_canonical_range_of_spherical_coordinate(rho, theta, phi):
    """Transform given spherical coordinates into canonical range

    ## NOTE: Canonical range of spherical coordinate
    #
    # Definition
    The canonical range of spherical coodrinate depends on the definition.
    In this funtion, the canoncial range is defined by the following:
    - 0 <= rho
    - 0 <= theta <= pi
    - 0 <= phi < 2.0*pi
    #
    # (OBSOLATE) On 'thate' range:
    Since coordinates associated with 'theta = 0' and 'theta = pi' are differnet,
    theta = pi should not be transformed to theta = 0 in order to be distinguished.
    """
    ## Argument check is omitted for accumulative performance

    ## Perform pre-processing
    # (180319, OBSOLATE) Move 'theta' to range of -pi <= thata < pi
    #theta = ((theta + pi) % (2.0*pi)) - pi
    # Move 'theta' to range of 0 <= thata < 2*pi
    theta %= 2.0 * pi

    ## Identify current range
    rho_is_negative = (rho < 0)
    # '(theta < 2.0) * pi' is assured by 'tehta %= 2.0 * pi' statement
    theta_is_between_phi_and_2phi = (pi < theta)
    #theta_is_negative = (theta < 0)

    ## Transform depending on the identified range
    if not rho_is_negative and not theta_is_between_phi_and_2phi:
        pass
    elif rho_is_negative and not theta_is_between_phi_and_2phi:
        rho *= -1.0
        # the range of phi will be shifted to canonical range in the end
        # .. so adding pi here is expected to raise no problem.
        phi += pi
        # (180319, OBSOLATE) [NOTE] if theta was exacly 0.0, then theta would be pi,
        # .. which isn't in the canonical range.
        # .. this effect will be compensated at the end of thie function (post-processing)
        # => (corrected NOTE) since theta is in range of '0 <= theta <= pi'
        # .. '0 <= pi - theta <= pi', thus, there's no need for rearranging
        # .. 'theta' into canonical range since it is already in there.
        theta = pi - theta
    elif not rho_is_negative and theta_is_between_phi_and_2phi:
        #theta *= -1.0  # (180319, OBSOLATE)
        theta = 2.0 * pi - theta
        phi += pi  # the range of phi will be shifted to canonical range in the end
    else: # rho_is_negative and theta_is_between_phi_and_2phi
        rho *= -1.0
        # (180319, OBSOLATE) adding pi into theta doesn't matter in terms of range,
        # .. since theta was negative and was in range -pi <= theta < pi
        # => (corrected NOTE) since 'theta_is_between_phi_and_2phi',
        # .. 'theta - pi' is in range '0 < theta += pi < pi',
        # .. which belongs to the canoncial range.
        #theta += pi
        theta -= pi

    ## Perform post-processing
    # [180317 NOTE] I should consider setting divider of this modulo operation
    # .. slightly smaller than actual divdier (e.g. 2.0*pi) to prevent strange out-of-range-effect
    phi %= 2.0 * pi  #- small_amount
    # [180319, NOTE] there's no need for this. Actually, it should be prohibited
    # .. since 'theta = pi' will be moved to '0' if this statement remains.
    #theta %= pi  #- small_amount

    return rho, theta, phi


def spherical_to_cartesian(rho, theta, phi):
    """Convert spherical coordinate vector to Cartesian coordinate vector"""
    x = rho * sin(theta) * cos(phi)
    y = rho * sin(theta) * sin(phi)
    z = rho * cos(theta)
    return x, y, z


def move_to_canonical_range_spherical_coord_arr(rho_arr, theta_arr, phi_arr):
    """Transform given spherical coordinates in numpy arrays into canonical range

    # NOTE: Canonical range of spherical coordinate
    - Definition
      : The canonical range of spherical coodrinate depends on the definition.
      : In this funtion, the canoncial range is defined by the following:
      - 0 <= rho
      - 0 <= theta <= pi
      - 0 <= phi < 2.0*pi
    """

    ## Check argument
    for arg in (rho_arr, theta_arr, phi_arr): 
        assert isinstance(arg, np.ndarray)
        assert (arg.ndim == 1) and (arg.size == rho_arr.size)
    

    ## Transform radial cooridnate 'rho'
    _rho_mask = rho_arr < 0
    rho_arr[_rho_mask] *= -1.0
    theta_arr[_rho_mask] = pi - theta_arr[_rho_mask]
    phi_arr[_rho_mask] += pi

    
    ## Transform polar angle coordinate 'theta'
    _theta_mask = (theta_arr < -pi) | (theta_arr >= pi)
    # move theta into the canonical range: -pi <= theta < pi
    _theta_mask[_theta_mask] = ((_theta_mask[_theta_mask] + pi) % (2.0*pi)) - pi
    _theta_negative_mask = _theta_mask & (theta_arr < 0)
    theta_arr[_theta_negative_mask] *= -1.0
    phi_arr[_theta_negative_mask] += pi

    
    ## Transform azimuthal angle coordinate 'phi'
    _phi_mask = (phi_arr < 0) | (phi_arr >= 2.0*pi)
    phi_arr[_phi_mask] %= (2.0*pi)



## Averaging under spherical coordinate
def _p_integrand(t, P_func, rho0, phi0):
    
    _rho = np.sqrt(rho0*rho0 + t*t)
    _theta = np.pi / 2.0 - np.arctan2(t,rho0)
    _theta *= 0.9999999
    _phi = phi0
    _coord = (_rho, _theta, _phi)
    
    _Pval = P_func(_coord)
    _dldt_abs = 1.0
    _integrand = _Pval * _dldt_abs
    
    return _integrand

from .integral import numerical_integral_trapezoidal

def P_bar(rho0, phi0, P_func, z0, N_t=100):
    
    _t_arr = np.linspace(-z0, z0, N_t)
    _integrand_arr = _p_integrand(_t_arr, P_func, rho0, phi0)
    _P_bar_val = 1.0 / (2.0 * z0) * numerical_integral_trapezoidal(_t_arr, _integrand_arr)
    
    return _P_bar_val

P_bar_vec = np.vectorize(P_bar, excluded=['P_func', 'z0', 'N_t'])


