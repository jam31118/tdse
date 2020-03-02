"""Base propagator object"""

from numpy import sqrt

class Wavefunction(object):
    """Base class for wavefunction objects"""
    
    @staticmethod
    def norm_sq(wf, *args):
        """Evaluate norm square of the given wavefunction"""
        pass

    @classmethod
    def normalize(cls, wf, *args):
        """Normalize the given wavefunction inplace"""
        _norm_sq = cls.norm_sq(wf, *args)
        wf *= 1. / sqrt(_norm_sq)



class Propagator(object):
    
    wf_class = Wavefunction

    def propagate(self, *args, **kwargs):
        pass

    def propagate_to_ground_state(self, wf, dt, max_Nt, normalizer_args,
                                  Nt_per_iter=10, norm_thres=1e-13):
        """
        Propagate the given wavefunction by the imaginary timestep
        to get the lowest energy possible from the given initial wavefunction
        """

        # Determine timestep
        _dt = float(dt)
        if not (_dt > 0): 
            _msg = "`dt` should be a positive real number. Given: {}"
            raise ValueError(_msg.format(_dt))
        _imag_dt = -1.0j * _dt # imaginary time for propagating to ground state
        
        # Initialize
        self.wf_class.normalize(wf, *normalizer_args)
        _wf_prev = wf.copy()
        
        # Propagate the wavefunction by the imaginary timestep
        # to get the lowest energy possible from the given initial wavefunction
        _max_iter = int(max_Nt / Nt_per_iter) + 1
        for _i in range(_max_iter):
            self.propagate(wf, _imag_dt, Nt=Nt_per_iter)
            self.wf_class.normalize(wf, *normalizer_args)
            _norm = self.wf_class.norm_sq(wf - _wf_prev, *normalizer_args)
            if _norm < norm_thres: break
            _wf_prev = wf.copy()
        if _i >= _max_iter-1: raise Exception("Maximum iteration exceeded")
        else: print("iteration count at end: {}".format(_i))

