"""System objects"""

from os import path

import numpy as np

from vis.indicator import Progress_Bar
#from unit import au2si
from nunit.au import au2si

import matrix
from .state import State_function_in_Box_1D
from .grid import Grid_Cartesian_1D
from .hamiltonian import Hamiltonian_1D

## 1 atomic unit of time is eqaul to au2as(around 24) attoseond
au2as = au2si['time'] * 1e18

class System_Box_1D(object):
    def __init__(self, grid, psi_x_t0=None, static_potential=None, dynamic_potential=None, 
                 t_0 = 0.0, state_function_filename='', time_filename='', initialize_snapshot_files=True,
                state_function_file_is_binary=True):
        
        ## Check input arguments
        assert type(grid) is Grid_Cartesian_1D
        self.grid = grid
        
        if psi_x_t0 is None:
            psi_x_t0 = np.random.rand(grid.x.N) - 0.5 # random values centerd at zero
        assert type(psi_x_t0) is np.ndarray
        assert psi_x_t0.ndim == 1
        assert psi_x_t0.shape[0] == grid.x.N
        
        #assert type(hamil) is Hamiltonian_1D
        
        # Check if the 't_0' is real (float), with zero imaginary part
        assert float(t_0) == t_0
        self.t_0 = t_0
        self.time = self.t_0
        # Check if the 'timestep' is real (float), with zero imaginary part
#         if timestep is None:
#             # [NOTE] This should be checked again whether it is sensible or not.
#             timestep = grid.x.delta * 0.25
#         assert float(timestep) == timestep
#         self.timestep = float(timestep)
        self.timestep = self.grid.t.delta
        
        self.state = State_function_in_Box_1D(grid, psi_x_t0, self.time, normalize=True, save_copy=True)
        
        self.hamil = Hamiltonian_1D(self.grid, static_potential, dynamic_potential)
        
        ## state function read/write file configuration
        assert type(state_function_file_is_binary) is bool
        self.state_function_file_is_binary = state_function_file_is_binary
        
        if state_function_filename == '':
            file_extension = ''
            if self.state_function_file_is_binary: file_extension = '.bin'
            else: file_extension = '.txt'
            self.state_function_filename = 'state-function' + file_extension
        elif type(state_function_filename) is str:
            self.state_function_filename = state_function_filename
        else: raise TypeError("'state_function_filename' should be of type 'str'")
        
            
        if time_filename == '':
            self.time_filename = 'time.txt'
        elif type(time_filename) is str:
            self.time_filename = time_filename
        else: raise TypeError("'time_filename' should be of type 'str'")
        
        if initialize_snapshot_files:
            self.initialize_snapshot_files()
            
        self.on_air = False  # saving state function doesn't happen as default.
        
    def set_U_half_static(self, timestep):
        self.U_half_static = np.eye(self.grid.x.N) - 1.0j * 0.5 * timestep * (self.hamil.static_part)
    
    def set_U_half_inv_static(self, timestep):
        self.U_half_inv_static = np.eye(self.grid.x.N) + 1.0j * 0.5 * timestep * (self.hamil.static_part).conj()

    def get_U_half(self, t, timestep):
        # [NOTE] The hamiltonian at 'time = self.time + 0.5 * timestep' should be used
        # .. not 'time = self.time' nor 'time = self.time + timestep'
        U_half_dynamic = -1.0j * 0.5 * timestep * (self.hamil.get_dynamic_part(t + 0.5 * timestep))
        return self.U_half_static + U_half_dynamic

    def get_U_half_inv(self, t, timestep):
        # [NOTE] The hamiltonian at 'time = self.time + 0.5 * timestep' should be used
        # .. not 'time = self.time' nor 'time = self.time + timestep'
        U_half_inv_dynamic = +1.0j * 0.5 * timestep * (self.hamil.get_dynamic_part(t + 0.5 * timestep)).conj()
        return self.U_half_inv_static + U_half_inv_dynamic        
    
    def time_is_consistent(self):
        """Check whether the time of system and its state function are idential."""
        time_of_system_and_state_function_are_same = self.time == self.state.time
        all_time_is_same = time_of_system_and_state_function_are_same
        return all_time_is_same
    
    def time_travel_single_step(self, timestep):
        
        U_half = self.get_U_half(self.time, timestep)
        U_half_inv = self.get_U_half_inv(self.time, timestep)
        
        psi_next_half = matrix.mat_vec_mul_tridiag(U_half.diagonal(0),
                                            U_half.diagonal(-1),
                                            U_half.diagonal(1),
                                            self.state.psi_x_t)

        self.state.psi_x_t[:] = matrix.gaussian_elimination_tridiagonal(U_half_inv.diagonal(0),
                                           U_half_inv.diagonal(-1),
                                           U_half_inv.diagonal(1),
                                           psi_next_half)
        
        self.time += timestep
        self.state.time += timestep

    def _get_print_format(self, num_of_timesteps):
        
        ## Configure logging

        # exp_num == (number of digits of num_of_timesteps)
        exp_num = int(np.floor(np.log10(num_of_timesteps)))

        num_digits_shown = exp_num + 1
        print_format = "propagated timesteps: %" + str(num_digits_shown) \
            + "d / %" + str(num_digits_shown) + "d, time = %.2f as"
        
        return print_format
    
    
    def go_to_ground_state(self, timestep=None, diff_threshold = 1e-30, print_period = 500, verbose=True,
                          max_imaginary_timesteps=20000):
        """
        
        [NOTE] The 'diff_threshold' should be changed .. to be more sensible.
        [TODO] Change printing_period into progress_bar mechanism.
        """
        
        assert self.time_is_consistent()
        
        assert int(max_imaginary_timesteps) == max_imaginary_timesteps
        self.max_imaginary_timesteps = max_imaginary_timesteps
        
        if timestep is None: timestep = self.timestep
        
        self.set_U_half_static( - 1.0j * timestep)
        self.set_U_half_inv_static( - 1.0j * timestep)
        
        converged = False
        #idx = 0
        #while True:
        for idx in range(self.max_imaginary_timesteps):
            psi_x_t_previous = self.state.psi_x_t.copy()

            self.time_travel_single_step( - 1.0j * timestep)
            self.state.normalize()

            diff = np.square(np.abs(psi_x_t_previous - self.state.psi_x_t)).sum() * self.grid.x.delta
            if verbose and ((idx % print_period) == (print_period - 1)): 
                print("[%05d] Difference: %.2e" % (idx+1,diff))
            if diff < diff_threshold:
                converged = True
                break
                
            #idx += 1
        
        if verbose and ((idx % print_period) != (print_period - 1)):
            print("[%05d] Difference: %.2e" % (idx+1,diff))
        
        if not converged:
            warning_message = "The state might have failed to converge as ground state."
            warning_message += "Check input arguments or extend 'max_imaginary_timesteps'"
            raise Warning(warning_message)
        
        # Initialize system time after imaginary time propagation
        self.time = self.t_0
        self.state.time = self.t_0
        
        # Check whether all time of all time-dependent objects in this system are same.
        assert self.time_is_consistent()
    
    
    def time_travel(self, num_of_timesteps=None, print_period = 100, save_period = None,
                   initialize_snapshot_files=False, verbose=False, be_quite=False, 
                    progress_bar_update_period = 10):
        
        assert type(initialize_snapshot_files) is bool
        
        # Check whether all time of all time-dependent objects in this system are same.
        assert self.time_is_consistent()
        
        if initialize_snapshot_files:
            self.initialize_snapshot_files()
        
        #if timestep is None: timestep = self.timestep
        timestep = self.timestep
        
        if num_of_timesteps is None: num_of_timesteps = self.grid.t.N - 1
        else:
            assert int(num_of_timesteps) == num_of_timesteps
            assert num_of_timesteps < self.grid.t.N
        
        propagation_time = num_of_timesteps * timestep
        if verbose:
            print("Total propagation time: %.2f as" % (propagation_time * au2as))
            
        if save_period is not None:
            # Check whether 'save_period' is integer, if not None.
            assert int(save_period) == save_period
            self.on_air = True
        
        if self.on_air:
            # Take initial snapshot before time propagation
            self.save_snapshot(verbose=verbose)
        
        self.set_U_half_static(timestep)
        self.set_U_half_inv_static(timestep)
        
        print_format = self._get_print_format(num_of_timesteps)
        
        if not be_quite:
            progress_bar = Progress_Bar(num_of_timesteps)
        for idx in range(num_of_timesteps):
            self.time_travel_single_step(timestep)

            if verbose and (((idx % print_period) == print_period - 1)):
                pass
                #time_au = (idx + 1) * timestep
                #time_as = time_au * au2as
                #print(print_format % (idx+1,num_of_timesteps,time_as))
                
                ## Logging progress with a bar
            
            if self.on_air and ((idx % save_period) == (save_period - 1)):
                self.save_snapshot(verbose=verbose)
                #PE = hamil.PE_static + hamil.get_PE_dynamic(time)
                #PE_x_t_saved.append(np.diag(PE,k=0).copy())
            
            time_to_update_progress_bar = (idx % progress_bar_update_period == (progress_bar_update_period - 1))
            if time_to_update_progress_bar and (not be_quite):
                progress_bar.print(idx, always_print_new_line=verbose)
                    
        if verbose and ((idx % print_period) != (print_period - 1)):
            time_au = (idx + 1) * timestep
            time_as = time_au * au2as
            print(print_format % (num_of_timesteps, num_of_timesteps, time_as))
            
        if self.on_air and ((idx % save_period) != (save_period - 1)):
            self.save_snapshot(verbose=verbose)
        
        # Turn off the mode of saving state function
        self.on_air = False
        
        # Check whether all time of all time-dependent objects in this system are same.
        assert self.time_is_consistent()
    
    
    def save_snapshot(self, verbose=True):
        #assert type(binary_mode) is bool
        if self.state_function_file_is_empty:
            #self.state_function_file_is_binary = binary_mode
            pass
        else:
            # Check consistency
            #assert binary_mode == self.state_function_file_is_binary
            pass
        
        if verbose:
            time_as = self.time * au2as
            print("Saving the state function at time %.2f as . . . " % time_as, end='', flush=True)
        
        ## Save state function
        # Determine file_open_mode
        file_open_mode = 'a'
        if self.state_function_file_is_binary: file_open_mode += 'b'
        with open(self.state_function_filename, file_open_mode) as f:
            self.state.save(f)
        self.num_of_saved_state_function += 1
        self.state_function_file_is_empty = False
        
        ## Save timepoint at which the state function is snap-shotted
        with open(self.time_filename, 'a') as f:
            f.write('%.18e\n' % self.time)
        
        if verbose:
            print('done', flush=True)
    
    def _check_valid_snapshot_index(self, index):
        assert int(index) == index
        assert index < self.num_of_saved_state_function
    
    def load_snapshot(self, index):
        self._check_valid_snapshot_index(index)
        file_open_mode = 'r'
        if self.state_function_file_is_binary: file_open_mode += 'b'
        with open(self.state_function_filename, file_open_mode) as f:
            loaded_state_function = self.state.load(f, index)
        return loaded_state_function
    
    def load_saved_time_array(self):
        return np.loadtxt(self.time_filename, dtype=float)
    
    def load_time_value(self, index):
        self._check_valid_snapshot_index(index)
        ## Pass throught the 'index' of lines 
        ## .. and read only one line from the saved time file
        i_th_time_value = None
        with open(self.time_filename, 'r') as f:
            for idx in range(index): f.readline()
            i_th_time_value = float(f.readline())
        if i_th_time_value is None: raise Exception("Problem during reading time values file")
        return i_th_time_value
    
    def initialize_snapshot_files(self):
        for filename in [self.state_function_filename, self.time_filename]:
            if path.exists(filename):
                with open(filename, 'w'): pass
        self.state_function_file_is_empty = True
        self.num_of_saved_state_function = 0
    
    def get_potential(self):
        return self.hamil.get_potential(self.time)


