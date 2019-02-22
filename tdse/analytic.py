"""
Collection of analytical expressions for TDSE,
the Time Dependent Schr\"{o}dinger Equation
"""

from numbers import Real, Integral

import numpy as np
from scipy import special
from scipy.optimize import brentq

from ntype import is_real_number, is_integer, is_integer_valued_real


def spherical_jn_zeros(n, nt):
    """Get n-th zero point of n-th order spherical Bessel function 
    of a first kind.

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
    time_independent_part = \
        special.spherical_jn(l, zero_point / R0 * rho) \
        * special.sph_harm(m, l, phi, theta)
    time_dependent_part = np.exp( -1.0j * energy * t)
    return time_independent_part * time_dependent_part



def eigenfunction_polar_box(rho, phi, t, m, n, R0):
    """Calculate energy eigenfunction for a particle in a polar box"""

    ## Check input arguments
    for arg in [m,n]: assert is_integer(arg)
    for arg in [rho, phi, t, R0]: assert is_real_number(arg)
    for arg in [n, R0]: assert arg > 0

    zero_point = special.jn_zeros(m, n)[-1]
    time_independ = special.jn(m, rho * zero_point / R0) * np.exp(1.0j*m*phi)
    time_depend = np.exp( - 0.5j * t * (zero_point / R0)**2)

    return time_independ * time_depend


def energy_eigenfuncion_for_1d_box(x, n, a, b, with_energy=False):
    """
    Returns an `n`-th energy eigenfunction array for particle 
    in an one-dimensional box.

    If `with_energy` is True, returns a corresponding energy eigenvalue 
    in Hartree unit.
    """

    for scalar_arg in [a, b]: assert isinstance(scalar_arg, Real)
    assert a < b
    L = b - a

    assert (n == int(n)) and (n > 0)

    result = np.sqrt(2.0 / L) * np.sin(n*np.pi/L * (x-a))
    
    if not with_energy:
        return result
    else:
        energy = 0.5 * (n*np.pi/L)**2
        return (result, energy)



from scipy.special import genlaguerre, factorial

def hydrogen_phi_nlm(r, n, l, m, exact_factorial=False):
    '''
    The radial part of the normalized hydrogen wavefunction,
    multiplied by the radial coordinate `r`.
    
    - Considered only the electron mass, 
    instead of the reduced mass concept, which includes proton.
    - The atomic unit is used.
    '''
    
    _a_0_star = 1.0  # the reduced Bohr radius
    
    for quantum_number in (n, l, m):
        assert isinstance(quantum_number, Integral)
    assert isinstance(exact_factorial, bool)
    assert (n >= 1) and (l >= 0) and (m <= l) and (m >= -l)
    
    _n, _l, _m = n, l, m
    
    _r = r
    _rho = 2.0 * _r / (_n*_a_0_star)
    _constant = np.sqrt(
        (2.0/(_n*_a_0_star)) 
        * factorial(_n-_l-1, exact=exact_factorial)
        / (2.0 * _n * factorial(_n+_l, exact=exact_factorial))
    )
    _result = np.exp(-_rho*0.5)
    _result *= np.power(_rho, _l+1)  # `_l+1` due to multiplied `r`
    _result *= genlaguerre(_n-_l-1, 2*_l+1, monic=False)(_rho)
    _result *= _constant
    
    return _result




def Gaussian1D(x,t,k_x):
    return (2/np.pi)**(0.25) \
        * np.sqrt(1.0/(1+2j*t)) \
        * np.exp(-1.0/4 * (k_x**2)) \
        * np.exp(-1.0/(1+2j*t) * (x - 1j*k_x/2)**2)


# def Gaussian2D(x,y,t,k_x,k_y):
#     return (2/np.pi)**(0.5) \
#         * (1.0/(1+2j*t)) \
#         * np.exp(-1.0/4 * (k_x**2 + k_y**2)) \
#         * np.exp(-1.0/(1+2j*t) * ((x - 1j*k_x/2.0)**2 + (y - 1j*k_y/2.0)**2))

def Gaussian2D(x,y,t,k_x,k_y):
    return Gaussian1D(x,t,k_x) * Gaussian1D(y,t,k_y)

def Gaussian3D(x,y,z,t,k_x,k_y,k_z):
    return Gaussian1D(x,t,k_x) * Gaussian1D(y,t,k_y) * Gaussian1D(z,t,k_z)

# def Gaussian3D(x,y,z,t,k_x,k_y,k_z):
#     return (2/np.pi)**(0.75) \
#         * (1.0/(1+2j*t))**(1.5) \
#         * np.exp(-1.0/4 * (k_x**2 + k_y**2 + k_z**2)) \
#         * np.exp(-1.0/(1+2j*t) * ( (x - 1j*k_x/2.0)**2 + (y - 1j*k_y/2.0)**2 + (z - 1j*k_z/2.0)**2 ) )


def gradient_Gaussian1D(x,t,k_x):
    _chain = (-2.0/np.sqrt(1.0+2.0j*t)) * (x-0.5j*k_x)
    return Gaussian1D(x,t,k_x) * _chain

# def gradient_Gaussian2D(x,y,t,k_x,k_y):
#     return (gradient_Gaussian1D(x,t,k_x) * Gaussian1D(y,t,k_y), Gaussian1D(x,t,k_x) * gradient_Gaussian1D(y,t,k_y))

def gradient_Gaussian2D(x,y,z,t,k_x,k_y,k_z):
    _gaussian2D = Gaussian2D(x,y,z,t,k_x,k_y,k_z)
    _chain_base = (-2.0/np.sqrt(1.0+2.0j*t))
    return tuple(_gaussian2D * _chain_base * (x_i - 0.5j*k_x_i) 
            for x_i, k_x_i in zip((x,y),(k_x,k_y)))

def gradient_Gaussian3D(x,y,z,t,k_x,k_y,k_z):
    _gaussian3D = Gaussian3D(x,y,z,t,k_x,k_y,k_z)
    _chain_base = (-2.0/np.sqrt(1.0+2.0j*t))
    return tuple(_gaussian3D * _chain_base * (x_i - 0.5j*k_x_i) 
            for x_i, k_x_i in zip((x,y,z),(k_x,k_y,k_z)))


def waveInBox(x,t,n,L=1):
    E = 0.5*(n*np.pi/L)**2
    return np.sin(n*np.pi/L * x) * np.exp(1j*E*t)


def superposedWaveInBox(x,t,L,n_list=[], coef=[]):
	if len(coef) > 0:
		assert len(n_list) == len(coef)
	else:
		# Equal weight for each mode
		coef = np.ones_like(n_list)
	result = 0
	for index, n in enumerate(n_list):
		result += coef[index] * waveInBox(x,t,n,L)
	return result

