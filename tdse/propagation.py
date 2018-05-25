"""Unitary propagation utility"""

from numbers import Real

import numpy as np

from tdse.matrix import mat_vec_mul_tridiag, gaussian_elimination_tridiagonal


def get_unitary_tridiagonals(hamil_diag, hamil_off_diag, delta_t, direction='forward'):
    if type(direction) is not str: raise TypeError("`direction` argument should be string")
    direction_lowercase = direction.lower()
    sign = None
    if direction_lowercase == 'forward': sign = -1.0
    elif direction_lowercase == 'backward': sign = 1.0
    else: raise ValueError("Unsupported direction: {}".format(forward))
    unitary_half_diag, unitary_half_off_diag \
        = (const + sign * 0.5j * delta_t * diag 
           for const, diag in zip([1.0, 0.0], [hamil_diag, hamil_off_diag])
          )
    tridiags = [unitary_half_diag, unitary_half_off_diag, unitary_half_off_diag]
    return tridiags


def propagate(psi, num_of_timesteps, tridiags, tridiags_inv, delta_x0,
    do_imaginary_propagation=False, orthogonalize=None):
    
    assert isinstance(psi, np.ndarray)
    assert isinstance(delta_x0, Real)
    assert delta_x0 > 0
    
    if orthogonalize is not None:
        assert np.iterable(orthogonalize)
    
    psi_next_half = np.empty_like(psi, dtype=complex)

    for time_index in range(num_of_timesteps):
        ## Propagation: Crank-Nicolson
        psi_next_half[:] = mat_vec_mul_tridiag(*tridiags, psi)
        psi[:] = gaussian_elimination_tridiagonal(*tridiags_inv, psi_next_half)
        
        ## Orthogonalization
        if orthogonalize is not None:
            for psi_to_substract in orthogonalize:
                component = (psi_to_substract.conj() * psi).sum() * delta_x0
                psi -= component * psi_to_substract

        ## Normalization
        if do_imaginary_propagation:
            norm = (psi * psi.conj()).sum() * delta_x0
            psi /= np.sqrt(norm)


