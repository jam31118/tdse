"""Collection of potential generating tools"""

import numpy as np

from .grid import Grid_Cartesian_1D

def finite_well_1D(grid, depth=1.0, half_width=1.0, center=0):
    assert type(grid) is Grid_Cartesian_1D
    assert depth >= 0
    assert half_width >= 0
    assert ((center - half_width) < grid.x.max) and ((center + half_width) > grid.x.min)
    x_n = grid.x.array
    static_potential = np.zeros(grid.x.N, dtype=np.double)
    static_potential[(x_n < (center + half_width)) & (x_n > (center - half_width))] = -depth
    return static_potential
