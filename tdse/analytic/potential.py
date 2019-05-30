"""collection of analytical expressions for several potentials"""

import numpy as np


def get_free_space_pot_func():
    _func = lambda x: np.zeros_like(x)
    return _func

def get_harmonic_pot_func(omega):
    _func = lambda x: 0.5 * omega * omega * np.square(x)
    return _func

def get_soft_core_pot_func(q, a):
    """
    # Arguments
    `q`: amplitude
    `a`: let the potential be soft
    """
    _func = lambda x: - q / np.sqrt(np.square(x) + a*a)
    return _func

