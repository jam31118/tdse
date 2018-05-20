"""Several types of State Function objects are defined."""

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from nunit import au as unit
from ntype import check_type_ndarray
from vis.plot import get_ylim
from vis.layout import get_text_position_and_inner_alignment

from .grid import Grid_Cartesian_1D



#### State function for numerical propagation ####
## it should support some method such as __getitem__ etc.
## to be used in Bohmian calculation etc.

def is_writable_file(file):
    is_writable = False
    if hasattr(file, 'writable'):
        if file.writable(): is_writable = True
    return is_writable

def is_readable_file(file):
    is_readable = False
    if hasattr(file, 'readable'):
        if file.readable(): is_readable = True
    return is_readable

def is_binary_file(file):
    is_binary = False
    if hasattr(file, 'mode'):
        file_mode_lowercase = file.mode.lower()
        is_binary = 'b' in file_mode_lowercase
    return is_binary

def is_writable_binary_file(file):
    #file_mode_lowercase = file.mode.lower()
    #is_binary = 'b' in file_mode_lowercase
    is_writable_and_binary = is_binary_file(file) and is_writable_file(file)
    return is_writable_and_binary


def process_fig_and_ax(fig, ax):
    ## Check type of given `fig`
    if fig is not None:
        if not isinstance(fig, Figure):
            err_mesg = "Argument `fig` should be of type {0}, but what I've got was: {1}"
            raise TypeError(err_mesg.format(type(Figure), type(fig)))
    
    ## Check type of Axes and check associated Figure object
    ## If `ax` is not given, it is generated or get from Figure's current Axes, if any.
    if ax is None:
        if fig is None: fig = plt.figure()
        ax = fig.gca()  # get current axes of given figure
    elif isinstance(ax, Axes):
        if fig is None: fig = ax.figure
        else: assert ax.figure is fig
    else:
        err_mesg = "Argument `ax` should be of type {0}, but what I've got was: {1}"
        raise TypeError(err_mesg.format(type(Axes), type(ax)))
    
    ## Cehck and return processed Figure and Axes object
    for obj in [fig, ax]: assert obj is not None
    return fig, ax


text_fontdict_template = {'backgroundcolor':(0,0,0,0.4), 
                 'fontsize':'x-large', 'family':'monospace', 'color':'white', 'weight':'bold'}

si_time_unit_to_factor = {
    's':1e1, 'ms':1e3, 'us':1e6, 'ns':1e9, 'ps':1e12, 'fs':1e15, 'as':1e18
}


rcParams['mathtext.fontset'] = 'stix'

class State_function_in_Box_1D(object):
    """
    Handle state function of a system with particle(s)
    
    [NOTE] wavefunction is also known as wave function
    """
    def __init__(self, grid, psi_x_t0, t_0, normalize=True, save_copy=True):
        """
        'normalize': boolean
            True : default, normalize the input initial state function 'psi_x_t0' 
                when it is assigned to the member variable of self.
            False : don't normalize 'psi_x_t0' 
                when it is assigned to the member variable of self.
        """
        ## Check input arguments and assign them into member varables if needed
        # For flags
        assert type(normalize) is bool
        assert type(save_copy) is bool
        #
        # For 'grid'
        assert type(grid) is Grid_Cartesian_1D
        self.grid = grid
        #
        # For 'psi_x_t0'
        self._check_valid_state_function_array(psi_x_t0)
        if save_copy:
            self.psi_x_t = psi_x_t0.astype(np.complex, copy=True)
        else:
            self.psi_x_t = psi_x_t0.astype(np.complex, copy=False)
        #
        # For initial time
        try: t_0 = float(t_0)
        except: raise TypeError("'t_0' should be of type 'float'")
        self.time = t_0
        
        ## Normalize the state function if specified to do so.
        if normalize:
            self.normalize()
    
    def _check_valid_state_function_array(self, sf_array):
        """Check whether an input array for a state function is valid.
        
        Check the type, shape, dimension of the input array 'sf_array'
        to be compliant with the 'grid' of self.
        
        [NOTE] 'sf_array' abbreviates 'state function array'
        """
        assert type(sf_array) is np.ndarray
        assert sf_array.ndim == 1
        assert sf_array.shape[0] == self.grid.x.N
        assert check_type_ndarray(sf_array, 'float') or check_type_ndarray(sf_array, 'complex')
        
            
    def get_norm(self):
        """Calculate and return norm of the current state function"""
        norm = ((self.psi_x_t * self.psi_x_t.conj()).sum() * self.grid.x.delta).real
        return norm
    
    
    def normalize(self):
        """
        Normalize the current state function
        
        [NOTE] It returns nothing.
        """
        norm = self.get_norm()
        normalization_constant = 1.0 / np.sqrt(norm)
        self.psi_x_t *= normalization_constant
        
        
    def get_squared(self):
        """Return norm square of the state function in position representation."""
        return (self.psi_x_t * self.psi_x_t.conj()).real
    
    def _calculate_squared(self, arr):
        self._check_valid_state_function_array(arr)
        return (arr * arr.conj()).real
    
    
    def save(self, file):
        """
        Save state function to disk.
        
        [To Do]
        - Support saving into a binary file.
        """
        ## Check prerequisite
        #assert is_writable_binary_file(file)
        assert is_writable_file(file)
        is_binary = is_binary_file(file)
        
        if is_binary: self.psi_x_t.tofile(file)
        else:
            for idx0 in range(self.grid.x.N):
                c = self.psi_x_t[idx0]
                st = '%.18e %.18e\n' % (c.real, c.imag)
                file.write(st)
    
    def load(self, file, index, save_to_self=False):
        assert is_readable_file(file)
        is_binary = is_binary_file(file)
        
        num_of_element_to_read = self.grid.x.N
        num_of_element_to_ignore = index * num_of_element_to_read
        
        loaded_state_function = np.empty((self.grid.x.N,), dtype=np.complex)
        if is_binary:
            file.seek(num_of_element_to_ignore * np.dtype(np.complex).itemsize, 0)
            loaded_state_function[:] = np.fromfile(file, dtype=np.complex, count=num_of_element_to_read)[:]
        else:
            real_imag_2D = np.empty((num_of_element_to_read,2), dtype=float)
            for idx in range(num_of_element_to_ignore):
                file.readline()
            for idx in range(num_of_element_to_read):
                line = file.readline()
                st_list = line.split(' ')
                real_imag_2D[idx,:] = [float(st_list[0]), float(st_list[1])]
            loaded_state_function[:] = np.apply_along_axis(lambda x: complex(*x), axis=1, arr=real_imag_2D)[:]
        
        if save_to_self:
            self.psi_x_t[:] = loaded_state_function[:]
        else:
            return loaded_state_function
    
    def plot(self, which_part='real', ax=None, fig=None, show_time=True, time_unit='fs', 
             other_state=None, other_time=None):
        """Plot current state function.
        
        ## Argument(s):
        # 'show_time'
        : Determine whether the time of the state function to be shown on the plot
        - if True:
            the time is shown on the plot
        - if False:
            the time is not shown on the plot
        - [NOTE] This parameter is turned off when 'other_state' is specified
        .. and the corresponding 'other_time' parameter isn't specified
        .. It is because the time of the 'other_state' is unknown 
        .. and would generally be different from internal time 'self.time'.
        
        # 'other_time'
        - [NOTE] If 'other_time' is not given, this argument will be ignored.
        """
        
        ## Check and pre-process input arguments
        # Process 'which_part'
        assert type(which_part) is str
        assert which_part.lower() in ['real','re','imag','imaginary','im','square','sq']
        which_part = which_part.lower()
        #
        fig, ax = process_fig_and_ax(fig, ax)
        #
        # Process 'other_time'
        if other_time is not None:
            assert is_real_number(other_time)
            if other_state is None:
                other_time = None
                print("'other_time' will be ignored since there's no given 'other_state'")
        #
        # Process 'show_time'
        # - This parameter is turned off when 'other_state' is specified
        # .. and the corresponding 'other_time' parameter isn't specified
        assert type(show_time) is bool
        if (other_state is not None) and (other_time is None):
            show_time = False
        #
        # Process 'time_unit'
        assert type(time_unit) is str
        assert time_unit in si_time_unit_to_factor.keys()
        #
        # Process 'other_state'
        if other_state is not None:
            self._check_valid_state_function_array(other_state)
            if check_type_ndarray(other_state, 'float'):
                other_state = other_state.astype(np.complex)
        
        ## Set appropriate fontsize system (e.g. 'medium', 'large', 'x-large' etc.)
        fig_size_in_inch = fig.get_size_inches()
        fig_size_len_geom_mean = (fig_size_in_inch[0] * fig_size_in_inch[1]) ** 0.5
        rcParams['font.size'] = fig_size_len_geom_mean * 2.0
        
        ## Set an array to be plotted following the 'which_part' argument
        ylabel=''
        y_unit = 'a.u.'
        
        sf_array = None
        if other_state is not None:
            sf_array = other_state
        else: 
            # Set current state function of self as default
            sf_array = self.psi_x_t
        
        if which_part in ['real','re']:
            to_be_plotted = sf_array.real
            ylabel = r'$Re\,\left[\psi\,\left(x,t\right)\right]\,\,/\,\,%s$' % y_unit
        elif which_part in ['imag','imaginary','im']:
            to_be_plotted = sf_array.imag
            ylabel = r'$Im\,\left[\psi\,\left(x,t\right)\right]\,\,/\,\,%s$' % y_unit
        elif which_part in ['square','sq']:
            #to_be_plotted = self.get_squared()
            to_be_plotted = self._calculate_squared(sf_array)
            ylabel = r'$\left|\psi\,\left(x,t\right)\right|^2\,\,/\,\,%s$' % y_unit
        else: raise Exception("Unsupported plotting data type: %s" % which_part)

        # Determine ylim
        try: 
            ylim_abs_max = max(map(abs,get_ylim(to_be_plotted)))
            ylim = [-ylim_abs_max, ylim_abs_max]
        except:
            ylim = [None,None]
        #else: raise Exception("Unexpected error")
                
        if which_part in ['square','sq']:
            ylim[0] = 0
        
        ax_color = (0.4,0,0,1)
        ax_plot_kwargs = {'color' : ax_color}
        ax.tick_params(axis='y', labelsize='x-large', colors=ax_color)
        
        x_n = self.grid.x.array
        ax.plot(x_n, to_be_plotted, **ax_plot_kwargs)
        
        ax.set_ylim(*ylim)
        
        x_unit = 'a.u.'
        ax.set_ylabel(ylabel, fontsize='xx-large', color=ax_color)
        ax.set_xlabel(r'$x\,\,/\,%s$' % (x_unit), fontsize='xx-large')
        
        
        ax.tick_params(axis='x', labelsize='x-large')
        
        
        ## Add text for representing time, if told to do so.
        if show_time:
            # Determine time to show
            time_to_show = None
            if other_time is None:
                time_to_show = self.time
            else: time_to_show = other_time
            
            # Configure text appearence
            text_fontdict = text_fontdict_template
            
            text_xy, text_align_dict = get_text_position_and_inner_alignment(ax, 'nw',scale=0.05)
            
            pos_x, pos_y = text_xy
            text_fontdict = {**text_fontdict, **text_align_dict}
            #text_fontdict['va'] = 'top'
            #text_fontdict['ha'] = 'left'
            #pos_x, pos_y = get_text_position(fig, ax, ha=text_fontdict['ha'], va=text_fontdict['va'])

            # Construct time text string
            time_unit = time_unit.lower()
            unit_factor = si_time_unit_to_factor[time_unit]
            text_content = r'time = %.3f %s' % (time_to_show * unit.au2si['time'] * unit_factor, time_unit)

            # Add text to the axis
            text = ax.text(pos_x, pos_y, text_content, fontdict=text_fontdict, transform=ax.transAxes)
       
        fig.tight_layout()

        # Return plot related objects for futher modulation
        return fig, ax
    
    
    def plot_real(self, **kwargs):
        """Plot real part of the current state function."""
        return self.plot(which_part='real', **kwargs)





