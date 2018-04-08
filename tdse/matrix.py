import numpy as np

def estimate_irreducible_diagnally_dominant_tridiagnoal(alpha, beta, gamma):
    assert len(beta) == len(gamma)
    assert (len(alpha) - 1) == len(beta)
    N = len(alpha)
    
    # part of this sufficient condition: non-zero tridianoal elements
    for arr in [alpha, beta, gamma]:
        assert not (arr == 0).any()
    
    # part for irreducibility
    for idx in range(1,N-1):
        assert (abs(alpha[idx]) >= (abs(beta[idx-1]) + abs(gamma[idx])))
    assert abs(alpha[0]) >= abs(gamma[0])
    assert abs(alpha[N-1]) >= abs(beta[N-2])

def gaussian_elimination_tridiagonal(alpha, beta, gamma, b):
    assert len(beta) == len(gamma)
    assert (len(alpha) - 1) == len(beta)
    N = len(alpha)
    
    ## Calculate delta
    gamma_beta = gamma * beta
    delta = alpha.copy()
    for idx in range(N-1):
        delta[idx+1] -= gamma_beta[idx] / delta[idx]

    ## Calculate c
    beta_over_delta = beta / delta[:-1]
    c = b.copy()
    for idx in range(N-1):
        c[idx+1] -= c[idx] * beta_over_delta[idx]

    ## Calculate v
    gamma_over_delta = gamma / delta[:-1]
    v = c / delta
    for idx in range(N-1,0,-1):
        v[idx-1] -= v[idx] * gamma_over_delta[idx-1]
    
    return v


def mat_vec_mul_tridiag(alpha, beta, gamma, v, N=None):
    """
    Calculate b = A * v, where A is tridiagonal matrix
    """
    ## Process input arguments
    if N is None: N = len(v)
    else: assert N == len(v)
    
    b = np.empty_like(v, dtype=complex)
    b[0] = alpha[0] * v[0] + gamma[0] * v[1]
    b[N-1] = beta[N-2] * v[N-2] + alpha[N-1] * v[N-1]
    for idx in range(1,N-1):
        b[idx] = beta[idx-1] * v[idx-1] + alpha[idx] * v[idx] + gamma[idx] * v[idx+1]
    
    return b


