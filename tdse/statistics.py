"""routine for several statistics"""

import numpy as np

class ReducedSupport(list):
    def __init__(self, x_arr, start_indices, lengths):
        
        assert isinstance(x_arr, np.ndarray)
        
        self.x_arr = x_arr
        self.start_indices = start_indices
        self.lengths = lengths
        _iterable = (x_arr[i:i+l] for i, l in zip(start_indices, lengths))

        super().__init__(_iterable)
    
    @classmethod
    def from_distribution_and_threshold(_class, x_arr, fx_arr, fx_threshold):
        assert isinstance(fx_arr, np.ndarray) and not np.any(fx_arr < 0)
        _rsup_mask = fx_arr > fx_threshold
        _start_indices, _lengths = _class.extract_start_indices_and_lengths_of_true_segment(_rsup_mask)
        return _class(x_arr, _start_indices, _lengths)
       
    @staticmethod
    def extract_start_indices_and_lengths_of_true_segment(arr):
        _kernel = np.array((1,-1))
        _indicator = np.convolve(arr, _kernel, mode='full')
        _indices_true_after_false, = np.where(_indicator > 0)
        _indices_false_after_true, = np.where(_indicator < 0)
        
        _start_indices_of_true_segment = _indices_true_after_false
        _lengths = _indices_false_after_true - _indices_true_after_false
        return _start_indices_of_true_segment, _lengths
    
    def integrate_distribution(self, fx_arr):
        assert isinstance(fx_arr, np.ndarray)
        assert (fx_arr.ndim == 1) and (fx_arr.size == self.x_arr.size)
        _sum = 0.0
        for i, l in zip(self.start_indices, self.lengths):
            _slice = slice(i,i+l)
            _sum += np.trapz(fx_arr[_slice], x=self.x_arr[_slice])
        return _sum


