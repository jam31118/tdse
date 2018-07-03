"""Hamiltonian objects"""

from numbers import Real

import numpy as np

from .grid import Grid_Cartesian_1D
from .finite_difference import get_second_deriv_tri_diagonals

def get_kinetic_energy_tri_diagonals(N_x0, delta_x0, dtype=complex):
    diag, off_diag, _ = get_second_deriv_tri_diagonals(N_x0, delta_x0, dtype=dtype)
    coef = -0.5  # in atomic unit
    for arr in [diag, off_diag]: arr *= coef
    return diag, off_diag, off_diag


def construct_static_kinetic_energy_matrix_1d(delta_x, N_x, dtype=complex):
    """Construct a matrix corresponding to static part of kinetic energy operator.
    
    Returned kinetic energy matrix is for an electron and in atomic unit,
    thus Planck constant and electron mass are set to 1.
    """
    matrix_shape = (N_x, N_x)
    coef = - 0.5 / (delta_x * delta_x)
    KE_static = np.zeros(matrix_shape, dtype=dtype)
    np.fill_diagonal(KE_static, - 2.0 * coef)
    np.fill_diagonal(KE_static[:-1,1:], 1.0 * coef)
    np.fill_diagonal(KE_static[1:,:-1], 1.0 * coef)
    return KE_static


class Hamiltonian_1D(object):
    """Manage Hamiltonian matrix"""
    def __init__(self, grid, static_potential=None, dynamic_potential=None):
        """Construct one dimensional Hamiltonian instance
        
        ## Argument(s):
        # grid [ Grid_Cartesian_1D ]
        
        # static_potential [ numpy.ndarray, None (default) ]
        : static part of the potential in position representation
        - if None:
            the static part of the potential is set to zero 
            .. for all possible position grid points in the system.
            .. The position grid is defined in 'grid' object
        
        # dynamic_potential [ callable, None (default) ]
        : callable with signature f(t) -> numpy.ndarray
        : .. where 'f' is the name of the callable
        : .. 't' is a time in atomic unit
        : .. the resulting numpy.ndarray contains values of dynamic potential
        : .. in position representation defined in 'grid' object.
        - if None:
            the dynamic part of the potential is set to zero
            .. for all time and for all position grid points 
            .. defined in 'grid' object
        
        ## Return(s):
        # None
        
        """
        
        ## Check types and assign input arguments into member variables
        #
        # for grid
        #assert type(grid) is Grid1D
        assert type(grid) is Grid_Cartesian_1D
        self.grid = grid
        #
        # for static potential
        if static_potential is None:
            self.static_potential = np.zeros((grid.x.N,), dtype=np.double)
        elif type(static_potential) is np.ndarray:
            self.static_potential = static_potential
        else: raise TypeError("Unsupported type for 'static_potential': %s" % type(static_potential))
        #   
        # for dynamic potential
        if dynamic_potential is None:
            self.dynamic_potential = lambda x: 0
        elif callable(dynamic_potential):
            self.dynamic_potential = dynamic_potential
        else: raise TypeError("'dynamic_potential' should be of type None or callable")
        
        ## Set shape of matrices of operators such as Hamiltonian
        self.matrix_shape = (self.grid.x.N,self.grid.x.N)
        
        ## Construct static part of the hamiltonian matrix
        self._construct_static_part()
    
    
    def _construct_static_part(self):
        """Construct static part of the hamiltonian matrix."""
        ## Construct a matrix corresponding to static part of potential energy operator
        self.PE_static = np.zeros(self.matrix_shape, dtype=np.complex)
        self.PE_static[np.diag_indices_from(self.PE_static)] = self.static_potential
        

        ## Construct a matrix corresponding to static part of kinetic energy operator
        self.KE_static = construct_static_kinetic_energy_matrix_1d(self.grid.x.delta_x, self.grid.x.N)
#        coef = - 0.5 / (self.grid.x.delta * self.grid.x.delta)
#        self.KE_static = np.zeros(self.matrix_shape, dtype=np.complex)
#        np.fill_diagonal(self.KE_static, - 2.0 * coef)
#        np.fill_diagonal(self.KE_static[:-1,1:], 1.0 * coef)
#        np.fill_diagonal(self.KE_static[1:,:-1], 1.0 * coef)
        
        ## Add static parts of matrix for kinetic energy and for potential energy
        self.static_part = self.KE_static + self.PE_static


    ## [180212 NOTE] Later on, all these matrixes' data types shouldn't be 2D array 
    ## .. but just several 1D arrays to become computationally efficient
    def get_KE_dynamic(self, t):
        """Construct and return the dynamic part of the kinetic energy matrix"""
        KE_dynamic = np.zeros(self.matrix_shape, dtype=np.complex)
        return KE_dynamic
    
    def get_PE_dynamic(self, t):
        """Construct and return the dynamic part of the potential energy matrix"""
        pot_x_t = self.dynamic_potential(t)
        PE_dynamic = np.zeros(self.matrix_shape, dtype=np.complex)
        PE_dynamic[np.diag_indices_from(PE_dynamic)] = pot_x_t
        return PE_dynamic
    
    def get_dynamic_part(self,t):
        """Construct and return the dynamic part of Hamiltonian matrix"""
        return self.get_KE_dynamic(t) + self.get_PE_dynamic(t)
    
    def get_potential(self, t):
        return self.static_potential + self.dynamic_potential(t)


