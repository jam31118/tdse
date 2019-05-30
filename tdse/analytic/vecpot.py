"""Collection of analytical expressions for vector potential"""

import numpy as np


def get_sin_square_pulse_func(E0, omega, num_cycle, phase, start_time=0.0):
    
    assert omega > 0
    
    def _func(t):
        """
        # Note
        - $E0 = A0 * \omega$ from $\vec{E} = -\partial_{t}{\vec{A}}$
        """
        _duration = num_cycle * 2.0 * np.pi / omega
        _max_time = start_time + _duration
        if (t < start_time) or (t > _max_time): return 0.0
        _t = t - start_time  # time in shifted coordinate
        _A0 = E0 / omega
        _envelope = np.square( np.sin( omega/(2.0*num_cycle) * _t) )
        _A_t = _A0 * _envelope * np.sin( omega * _t + phase )
        return _A_t
    
    return _func


