"""Base module for wavefunctions objects"""

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

