"""
Collection of analytical expressions for TDSE (Time Dependent Schr\"{o}dinger Equation)
"""

import numpy as np
from scipy import special
from scipy.optimize import brentq

from ntype import is_real_number, is_integer, is_integer_valued_real


def spherical_jn_zeros(n, nt):
    """Get nt-th zero point of n-th order spherical Bessel function of a first kind.

    ## Paramters ##
    # n, integer, spherical Bessel function's index (staring from 0)
    # nt, integer, zero point's index (starting from 1)
    """
    assert (int(n) == n) and (n >= 0)
    assert (int(nt) == nt) and (nt > 0)

    a = special.jn_zeros(n, nt)[-1]
    b = special.jn_zeros(n+1, nt)[-1]

    sph_jn = lambda x, n: special.spherical_jn(n, x)

    zero_point = brentq(sph_jn, a, b, args=(n))

    return zero_point



def eigenfunction_spherical_box(rho, theta, phi, t, n, l, m, R0):
    """Return eigenfunction of an spherical box

    The radius of the spherical box is specfied as 'R0'.
    The returned eigenfunction is not normalized.

    ## Arguments ##
    # n (integer)
    - index (starting from 1) of radial function
    # l (integer)
    - order (starting from 0) of spherical radial bessel function
    - one of the index of spherical harmonics
    # m (integer)
    - -l <= m <= l
    # R0 (float)
    - R0 > 0
    """

    ## [180310 NOTE] Following checking procedure
    ## .. may be omitted for performance in future implementation
    for arg in [n,l,m]: assert is_integer_valued_real(arg)
    assert (n > 0) and (l >= 0) and (-l <= m) and (m <= l)
    assert R0 > 0

    zero_point = spherical_jn_zeros(l, n)
    energy = 0.5 * (zero_point / R0) ** 2
    time_independent_part = special.spherical_jn(l, zero_point / R0 * rho) * special.sph_harm(m, l, phi, theta)
    time_dependent_part = np.exp( -1.0j * energy * t)
    return time_independent_part * time_dependent_part



def eigenfunction_polar_box(rho, phi, t, m, n, R0):
    """Calculate energy eigenfunction for a particle in a polar box"""

    ## Check input arguments
    for arg in [m,n]: assert is_integer(arg)
    for arg in [rho, phi, t, R0]: assert is_real_number(arg)
    for arg in [n, R0]: assert arg > 0

    zero_point = special.jn_zeros(m, n)[-1]
    time_independ = special.jn(m, rho * zero_point / R0) * np.exp(1.0j * m * phi)
    time_depend = np.exp( - 0.5j * t * (zero_point / R0)**2)

    return time_independ * time_depend
