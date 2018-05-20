import numpy as np
from numpy import pi

from ntype import is_integer_valued_real, is_real_number
from .coordinate import transform_to_canonical_range_of_spherical_coordinate


def squeeze_as_possible(meshes, type_converter):
    result = []
    for arr in meshes:
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = type_converter(arr)
            #if is_integer_valued_real(arr):
            #    arr = int(arr)
        result.append(arr)
    return result


def check_xor(a, b):
    # check 'exclusive or' of cut_min and cut_max
    for arg in [a, b]: assert type(arg) is bool
    return (a or b) and (not (a and b))


def cut_range_to_be_dividable_by_delta(min_val, max_val, delta, cut_min, cut_max):
    ## Check input arguments
    #
    # Check real-ness and sign
    for arg in [min_val, max_val, delta]: assert is_real_number(arg)
    assert delta > 0
    #
    # check 'exclusive or' of cut_min and cut_max
    assert check_xor(cut_min, cut_max)

    ## Determine if the range is dividable by 'delta'
    divided = (max_val - min_val) / delta
    num_of_delta = int(divided)
    range_is_dividable_by_delta = is_integer_valued_real(divided)

    ## Cut range if the range (='max_val'-'min_val') is not dividable by 'delta'
    cut_min_val, cut_max_val = None, None
    if range_is_dividable_by_delta:
        cut_min_val, cut_max_val = min_val, max_val
    else:
        if cut_min:
            cut_min_val = max_val - num_of_delta * delta
            cut_max_val = max_val
        elif cut_max:
            cut_min_val = min_val
            cut_max_val = min_val + num_of_delta * delta
        else: raise Exception("Either 'cut_min' or 'cut_max' should be true")

    ## Return
    return cut_min_val, cut_max_val


class Grid(object):
    def __init__(self, ndim):
        assert is_integer_valued_real(ndim)
        self.ndim = int(ndim)


class Grid_1D(Grid):
    def __init__(self, delta, min_val, max_val, zero_at_min, zero_at_max, fix_delta, periodic,
                 cut_min=None, cut_max=None, name=''):

        ## Check assign input arguments into member variables
        #
        # Check real-ness and sign
        for arg in [min_val, max_val, delta]: assert is_real_number(arg)
        assert min_val < max_val
        assert delta > 0
        #
        # Check booleans
        for arg in [zero_at_min, zero_at_min, fix_delta, periodic]: assert type(arg) is bool
        self.zero_at_min, self.zero_at_max = zero_at_min, zero_at_max
        self.fix_delta = fix_delta
        self.periodic = periodic
        if fix_delta is True:
            for arg in [cut_min, cut_max]: assert type(arg) is bool
            assert check_xor(cut_min, cut_max)
        self.cut_min, self.cut_max = cut_min, cut_max
        #
        # Check other parameter
        assert type(name) is str
        self.name = name

        ## Initialize member variables of parent class such as 'ndim'
        ndim = 1
        super().__init__(ndim)

        ## Cut range so that the range (='self.max'-'self.max') becomes a multiple of delta
        self.min, self.max = None, None
        self.delta = None
        if self.fix_delta:
            self.delta = delta
            self.min, self.max = cut_range_to_be_dividable_by_delta(min_val, max_val, self.delta, self.cut_min, self.cut_max)
        else:
            # If 'delta' isn't fixed, min and max value are fixed instead.
            self.min, self.max = min_val, max_val
            # Make the grid slightly denser if range is not dividable by given 'delta'
            divided = (self.max - self.min) / delta
            if is_integer_valued_real(divided):
                num_of_delta_to_fill_range = int(divided)
            else:
                num_of_delta_to_fill_range = int(divided) + 1
            self.delta = (self.max - self.min) / num_of_delta_to_fill_range

        ## Determine N, first value, last value
        if self.periodic: assert (self.zero_at_min is False) and (self.zero_at_max is False)
        self.N = int(np.round((self.max - self.min) / self.delta) + 1)

        self.first_val, self.last_val = None, None
        if self.periodic:
            self.N -= 1
            self.last_val = self.max - self.delta
        else:
            if self.zero_at_min:
                self.N -= 1
                self.first_val = self.min + self.delta
            #else: self.first_val = self.min

            if self.zero_at_max:
                self.N -= 1
                self.last_val = self.max - self.delta
            #else: self.last_val = self.max
        if self.first_val is None: self.first_val = self.min
        if self.last_val is None: self.last_val = self.max

        self.array = self.first_val + np.arange(self.N) * self.delta
        #print("max: ",self.max, "min:",self.min,"N:",self.N)
        #print("first: ",self.first_val,"delta",self.delta)
        # Check consistency
        #print("self.array[-1], self.last_val",self.array[-1], self.last_val )
        assert abs(self.array[-1] - self.last_val) < 1e-6
        #self.array = np.linspace(self.first_val, self.last_val, self.N)


    def __getitem__(self, index_exp):
        return self.array[index_exp]


def process_index_exp(index_exp, ndim):
    expanded_index_exp = ()
    if len(index_exp) < ndim:
        num_of_normal_experssion = 0
        for exp in index_exp:
            if exp not in [Ellipsis, np.newaxis]:
                num_of_normal_experssion += 1
            if exp is np.newaxis:
                raise NotImplementedError("Processing 'np.newaxis' haven't been implemented.")
        num_of_ellipsis = len(index_exp) - num_of_normal_experssion
        if num_of_ellipsis == 0: raise Exception("Invalid index expression")
        if num_of_ellipsis == 1:
            for exp in index_exp:
                if exp is not Ellipsis:
                    expanded_index_exp += (exp,)
                else:
                    for idx in range(ndim - num_of_normal_experssion):
                        expanded_index_exp += np.index_exp[:]
        elif (num_of_ellipsis < 0) or (num_of_ellipsis > (ndim - 1)):
            raise Exception()
        elif num_of_ellipsis > 1:
            raise NotImplementedError()

    elif len(index_exp) == ndim:
        expanded_index_exp = index_exp
    else: raise Exception("The length of index_exp should be %d" % ndim)

    return expanded_index_exp




class Grid_Cartesian_1D(Grid):
    """Grid for 1D Cartesian coordinate system with zero-boundary condition"""
    def __init__(self, delta_x, delta_t, x_min, x_max, t_min, t_max, fix_delta_x=True, fix_delta_t=True):
        ## Check input arguments
        for arg in [fix_delta_x, fix_delta_t]: assert type(arg) is bool
        
        ndim = 2  # spatial 1 + temporal 1
        super().__init__(ndim)
        
        self.x = Grid_1D(delta_x, x_min, x_max, zero_at_min=True, zero_at_max=True, 
                         fix_delta=fix_delta_x, periodic=False, cut_min=False, cut_max=True, name='x')
        self.t = Grid_1D(delta_t, t_min, t_max, zero_at_min=False, zero_at_max=False,
                        fix_delta=fix_delta_t, periodic=False, cut_min=False, cut_max=True, name='t')
        
        self.axes = [self.x, self.t]
        assert len(self.axes) == self.ndim
        
        ## Define metadata
        self.shape = tuple([axis.array.size for axis in self.axes])
        self.names = [axis.name for axis in self.axes]
    
    def _get_cell_index_of_x(self, x):
        """Return index of the unit cell which corresponds to given 'x'
        
        This method assumes zero-boundary conditions for both ends of 'x' axis,
        where values of function (associated to this grid) at x.min and x.max
        are always zero. This boundary condition is represented by
        (self.x.zero_at_min == True) and (self.x.zero_at_max == True)
        
        Each unit cell's range is like the following,
        where '[,]' and '(,)' mean closed and open interval respectively.
        0-th unit cell: [x.min + 0 * x.delta, x.min + 1 * x.delta)
        1-st unit cell: [x.min + 1 * x.delta, x.min + 2 * x.delta)
        2-nd unit cell: ...
        N-th unit cell: [x.min + N * x.delta, x.min + (N+1) * x.delta)
        """
        i = int((x - self.x.min) // self.x.delta)
        if (i < 0) or (i > self.x.N): 
            raise IndexError("Index of 'x'(=%d) is out of range: [%d,%d]" % (i,0,self.x.N))
        else: return i
    
    def _get_cell_index_of_time(self, t):
        """Return corresponding index of given time coordinate ('t')"""
        ## Determine temporal grid point's index
        j = int((t - self.t.min) // self.t.delta)
        if (j < 0) or (j > (self.t.N - 2)):
            raise IndexError("Index of 't'(=%d) is out of range: [%d,%d]" % (j,0,self.t.N - 2))
        else: return j

    def get_index(self, x, t):
        i = self._get_cell_index_of_x(x)
        j = self._get_cell_index_of_time(t)
        return i, j
    
    def __getitem__(self, index_exp):
        ## [180324 NOTE] Consider generalizing and embeding this method to the base class
        index_exp = process_index_exp(index_exp, self.ndim)
        slices = [self.axes[idx][index_exp[idx]] for idx in range(self.ndim)]
        meshes = np.meshgrid(*slices, indexing='ij')
        result = squeeze_as_possible(meshes, float)
        return result
    
    def get_value(self,*args):
        """Dedicated method for getting grid value at given single set of indices
        The number of indices in the set should be same as self.ndim"""
        ## [180324 NOTE] Consider removing this assertion for performance
        assert len(args) == self.ndim
        return tuple(axis[arg] for axis, arg in zip(self.axes, args))





class Grid_Polar(Grid):
    def __init__(self, delta_rho, delta_phi, delta_t, rho_max, t_max,
                 t_min=None, rho_min=None, phi_max=None, phi_min=None,
                 fix_delta_rho=True, fix_delta_phi=False, fix_delta_t=True):

        ## Initialize member variables of parent class such as 'ndim'
        ndim = 3
        super().__init__(ndim)

        if phi_min is None: phi_min = 0
        if phi_max is None: phi_max = 2.0 * np.pi
        if rho_min is None: rho_min = 0.0
        if t_min is None: t_min = 0.0

        self.rho = Grid_1D(delta_rho, rho_min, rho_max, False, True, fix_delta_rho,
                periodic=False, cut_min=False, cut_max=True)
        self.phi = Grid_1D(delta_phi, phi_min, phi_max, False, False, fix_delta_phi,
                periodic=True)
        self.t = Grid_1D(delta_t, t_min, t_max, False, False, fix_delta_t,
                periodic=False, cut_min=False, cut_max=True)

        self.shape = (self.rho.array.size, self.phi.array.size, self.t.array.size)
        self.names = ['rho', 'phi', 't']



    def get_radial_index(self, rho):
        """Return corresponding index of given radial coordinate ('rho')"""
        ## Determine radial grid point's index
        if rho < self.rho.min:
            if rho < 0:
                rho = - rho
                # [180308 NOTE] it was commented out
                # since addition of pi is considered in 'get_azim_index' method
                #phi += np.pi
                if rho < self.rho.min: i = 0
                else: i = int(rho // self.rho.delta)
            else: raise IndexError("'rho' should be either >= 0 or >= 'self.rho.min'")
        elif rho >= self.rho.max: i = self.rho.N  # out of range
        else: i = int(rho // self.rho.delta)

        return i


    def get_azim_index(self, phi, rho):
        """Return corresponding index of given azimuthal angle coordinate ('phi')"""
        if (rho < self.rho.min) and (rho < 0): phi += np.pi
        ## Determine azimuthal grid point's index
        # 'small_amount' is expected to prevent unwanted excess of index 'j',
        # .. which should be smaller than self.phi.N,
        # .. but can be equal to self.phi.N if (phi // self.phi.delta)
        # .. is same or slightly bigger than self.phi.N,
        # .. which can be the case when the 'phi %= (2.0 * np.pi)'
        # .. is not assured to be in the range
        # .. where the self.phi.N multiples of self.phi.delta cannot be in the range.
        small_amount = 1e-10
        phi %= (2.0 * np.pi - small_amount)
        j = int(phi // self.phi.delta)

        return j


    def get_time_index(self, t, small_amount = 1e-10):
        """Return corresponding index of given time coordinate ('t')"""
        ## Determine temporal grid point's index
        if t < self.t.min: raise IndexError("'t' should be either >= 0 or >= self.t.min")
        elif t >= self.t.max: k = self.t.N - 1
        else: k = int(t // self.t.delta)

        return k


    def get_index(self, rho, phi, t):
        i = self.get_radial_index(rho)
        j = self.get_azim_index(phi, rho)
        k = self.get_time_index(t)
        return i, j, k


    def __getitem__(self, index_exp):
        """
        Return coordinate value in (\rho, \phi, time) coordinate system,
        corresponding to given array slice
        """
        #ndim = 3
        #assert len(index_exp) == 3:
        index_exp = process_index_exp(index_exp, self.ndim)

        A, B, C = np.meshgrid(self.rho[index_exp[0]], self.phi[index_exp[1]], self.t[index_exp[2]], indexing='ij')

        result = []
        for arr in [A, B, C]:
            arr = np.squeeze(arr)
            if arr.ndim == 0:
                arr = float(arr)
                if is_integer_valued_real(arr):
                    arr = int(arr)
            result.append(arr)

        return result



class Grid_Spherical(Grid):
    def __init__(self, delta_rho, delta_theta, delta_phi, delta_t, rho_max, t_max,
                theta_max=None, phi_max=None, rho_min=None, theta_min=None, phi_min=None, t_min=None,
                fix_delta_rho=True, fix_delta_theta=False, fix_delta_phi=False, fix_delta_t=True):


        ## Check input arguments
        # Assign default values if the argument wasn't set
        if theta_max is None: theta_max = pi
        if phi_max is None: phi_max = 2.0 * pi
        if rho_min is None: rho_min = 0.0
        if theta_min is None: theta_min = 0.0
        if phi_min is None: phi_min = 0.0
        if t_min is None: t_min = 0.0
        #
        # Check types
        for bool_arg in [fix_delta_rho, fix_delta_theta, fix_delta_phi, fix_delta_t]:
            assert type(bool_arg) is bool
        #
        # Determine whether 'phi' should be periodic or not
        # .. if 'phi_max' is set to 2*pi, then periodicity is assumed
        phi_should_be_periodic = False
        if abs(phi_max - 2.0 * pi) < 1e-10:
            phi_should_be_periodic = True

        ## Initialize member variables of parent class such as 'ndim'
        ndim = 4
        super().__init__(ndim)


        self.rho = Grid_1D(delta_rho, rho_min, rho_max, zero_at_min=False, zero_at_max=True,
                           fix_delta=fix_delta_rho, periodic=False, cut_min=False, cut_max=True, name='rho')
        self.theta = Grid_1D(delta_theta, theta_min, theta_max, zero_at_min=False, zero_at_max=False,
                             fix_delta=fix_delta_theta, periodic=False, name='theta')
        self.phi = Grid_1D(delta_phi, phi_min, phi_max, zero_at_min=False, zero_at_max=False,
                          fix_delta=fix_delta_phi, periodic=phi_should_be_periodic, name='phi')
        self.t = Grid_1D(delta_t, t_min, t_max, zero_at_min=False, zero_at_max=False,
                        fix_delta=fix_delta_t, periodic=False, cut_min=False, cut_max=True, name='t')

        self.axes = [self.rho, self.theta, self.phi, self.t]
        assert len(self.axes) == self.ndim

        ## Define metadata
        self.shape = tuple([axis.array.size for axis in self.axes])
        self.names = [axis.name for axis in self.axes]


    def _get_radial_index(self, rho):
        """Return corresponding index of given radial coordinate ('rho')

        ## Assumptions:
        # The given coordinate should be in canonical range
        """
        ## [180319 NOTE] Consider checking (possibly by assertion) whether the rho is in canonical region
        ## .. like the folowing assertion statement
        #assert rho >= 0  # Check whether 'rho' is in canonical region

        ## Determine radial grid point's index
        i = None
        if rho < self.rho.min: i = 0 # it is problamatic since meaning of the index 0 become degenerated.
        elif rho >= self.rho.max: i = self.rho.N  # out of range
        else: i = int( (rho - self.rho.min) // self.rho.delta )

        return i


    def _get_polar_index(self, theta, small_amount=1e-10):
        """Return corresponding index of given polar coordinate ('theta')

        ## Assumptions:
        # The given coordinate should be in canonical range
        # The 'theta' value starts from 0 (i.e. self.theta.min == 0)
        """
        ## [180319 NOTE] Consider checking (possibly by assertion) whether the rho is in canonical region

        ## [180317 NOTE] This function assume that the 'theta' is in the canonical range
        ## and the canonical range of 'theta' is '0 <= theta < pi'
        j = int(theta // self.theta.delta)

        return j


    def _get_azim_index(self, phi, rho, theta, small_amount = 1e-10):
        """Return corresponding index of given azimuthal angle coordinate ('phi')

        ## Assumptions:
        # The given coordinate should be in canonical range
        # The 'phi' value starts from 0 (i.e. self.phi.min == 0)
        """
        ## [180317 NOTE] This function assume that the 'phi' is in the canonical range
        ## and the canonical range of 'rho' is '0 <= phi < 2.0 * pi'
        #if (rho < self.rho.min) and (rho < 0): phi += np.pi
        #canonical_theta = (theta % (2.0 * np.pi)) - np.pi   # thus, -pi <= theta < pi
        #if not ((0 <= theta) and (theta < np.pi)): phi += np.pi

        ## [180317 NOTE] Fix the following explanation in a more understandable manner.
        ## Determine azimuthal grid point's index
        # 'small_amount' is expected to prevent unwanted excess of index 'j',
        # .. which should be smaller than self.phi.N,
        # .. but can be equal to self.phi.N if (phi // self.phi.delta)
        # .. is same or slightly bigger than self.phi.N,
        # .. which can be the case when the 'phi %= (2.0 * np.pi)'
        # .. is not assured to be in the range
        # .. where the self.phi.N multiples of self.phi.delta cannot be in the range.
        ## [180317 NOTE] commented because this method assumes that the 'phi' is in canonical range
        #phi %= (2.0 * np.pi - small_amount)
        k = int(phi // self.phi.delta)

        return k


    def _get_time_index(self, t, small_amount = 1e-10):
        """Return corresponding index of given time coordinate ('t')"""
        ## Determine temporal grid point's index
        l = None
        if t < self.t.min: raise IndexError("'t' should be either >= 0 or >= self.t.min")
        elif t >= self.t.max: l = self.t.N - 1
        else: l = int(t // self.t.delta)

        return l


    def get_index(self, rho, theta, phi, t):
        """Return corresponding set of grid's indices for each given coordinate value"""
        rho, theta, phi = transform_to_canonical_range_of_spherical_coordinate(rho, theta, phi)
        i = self._get_radial_index(rho)
        j = self._get_polar_index(theta)
        k = self._get_azim_index(phi, rho, theta)
        l = self._get_time_index(t)
        return i, j, k, l


    def __getitem__(self, index_exp):
        #assert len(index_exp) == self.ndim
        index_exp = process_index_exp(index_exp, self.ndim)
        slices = [self.axes[idx][index_exp[idx]] for idx in range(self.ndim)]
        meshes = np.meshgrid(*slices, indexing='ij')

        result = squeeze_as_possible(meshes, float)

        return result


    def get_value(self, indices):
        """Return grid value corresponding to given indices.
        
        ## Development Notes
        # Consider migrating this method to parent class
        # .. since it seems to be easily generalized.
        """
        assert len(indices) == self.ndim
        return tuple(axis[idx] for axis, idx in zip(self.axes, indices))


    def get_partial_indices(self, index_exp):
        """
        
        ## Development Note
        # This method should be optimized further.
        - Let's considering stop using 'arange'
        """
        index_exp = process_index_exp(index_exp, self.ndim)
        indices_per_dimension = [np.arange(N)[exp] for N, exp in zip(self.shape, index_exp)]
        #print("indices_per_dimension",indices_per_dimension)
        partial_indices = np.meshgrid(*indices_per_dimension, indexing='ij')
        #print("partial_indices",partial_indices)
        squeezed_partial_indices = ()
        for partial_indice in partial_indices:
            squeezed_partial_indices += (squeeze_as_possible(partial_indice, int),)
        return squeezed_partial_indices

    def indices_are_in_supported_range(self, *indices):
        all_in_range = True
        for idx, index in enumerate(indices):
            assert is_integer_valued_real(index)
            all_in_range &= (0 < (index + 1)) and (index < self.shape[idx])
        return all_in_range





class Grid_1D_In_Box(Grid):
    """One dimensional spatial grid."""
    def __init__(self, x_min, x_max, delta_x, boundary_condition, cut_min=False, cut_max=False):
        """
        Initialize one dimensional spatial grid.

        Only one of 'cut_min' or 'cut_max' should be True and the other one should be False.

        - If the boundary_condition is 'infinite\_potential\_box', there are infinite potential wall at position $x = x_{0} - \Delta{x}$ and at $x = x_{N_{x}-1} + \Delta{x}$, where $x_{0}$ and $x_{N_x-1}$ are the first (with 0-th index) and the last (with {$N_x-1$}-th index) position grid points, respectively. Thus, the state function $\psi(x,t)$ at both points are zero.
        """
        ## Check input arguments
        #
        # Check numeric variables are of numeric types
        for argument in [x_min, x_max, delta_x]:
            assert float(argument) == argument
        #
        # either cut_max or cut_min should be true and false.
        assert ((cut_min == True) and (cut_max == False)) \
            or ((cut_min == False) and (cut_max == True))
        #
        # boundary_condition should be of string type
        assert type(boundary_condition) is str

        ## Assign input arguments into members variables
        self.delta_x = delta_x
        self.cut_min, self.cut_max = cut_min, cut_max
        self.boundary_condition = boundary_condition

        ## Determine number of spatial grid points (e.g. N_x)
        N_x_estimate = (x_max - x_min) / self.delta_x - 1
        N_x_estimate_is_integer = int(N_x_estimate) == N_x_estimate
        self.N_x = int(N_x_estimate) - 1   # 'int()' have same effect with floor()

        ## Cut one or more edge of spatial range (e.g. [x_min,x_max])
        ## .. in order for the range to be a multiple of spatial grid interval (e.g. delta_x)
        if N_x_estimate_is_integer:
            self.x_min, self.x_max = x_min, x_max
        else:
            if self.cut_max:
                self.x_max = x_min + (self.N_x+1) * delta_x
                self.x_min = x_min
            elif self.cut_min:
                self.x_min = x_max - (self.N_x+1) * delta_x
                self.x_max = x_max

        self.x_0 = self.x_min + self.delta_x

    def get_grid_array(self):
        """
        Construct and return a spatial grid array
        whose elements are coordinates in position space that corresponds to each grid points.

        [Future Note]
        If the spatial grid array is used frequently inside or outside of the Grid object,
        consider storing this spatial grid array as a member variable,
        preventing continuing construction of this saptial grid array.
        """
        ## Calculate Position Grid
        n = np.arange(self.N_x)
        x_n = self.x_0 + n * self.delta_x
        return x_n

    def get_system_size(self):
        """
        Return system's spatial size

        In N-dimensional, Cartesian spatial grid, a tuple with length L is returned,
        whose i-th element is the spatial length of corresponding dimension.
        """
        if self.boundary_condition.lower() == "infinite_potential_box":
            L_x = (self.N_x - 1 + 2) * self.delta_x
        else: raise Exception("Unsupported boundary condition")

        return (L_x,)


class Grid1D(Grid_1D_In_Box):
    def __init__(self, *args, **kwargs):
        print("'Grid1D' is depreciated at 180304. Use either 'Grid_1D_In_Box' or 'Grid_1D'.")
        super().__init__(*args, **kwargs)
