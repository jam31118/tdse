from .matrix import mat_vec_mul_tridiag, gaussian_elimination_tridiagonal

def get_tridiag_shape(N):
    return (3, N)

## Define macro function for tridiag datatype
def tridiag_forward(tridiag, v, b):
    """
    The diagonal elements should be tridiag[1,:],
    the lower offdiagonal should be tridiag[0,1:],
    the upper offdiagonal should be tridiag[2,:-1]
    """
    b[:] = mat_vec_mul_tridiag(tridiag[1,:], tridiag[0,1:], tridiag[2,:-1], v)



def tridiag_backward_thomas(tridiag, v, b):
    """
    The diagonal elements should be tridiag[1,:],
    the lower offdiagonal should be tridiag[0,1:],
    the upper offdiagonal should be tridiag[2,:-1]
    """
    v[:] = gaussian_elimination_tridiagonal(
            tridiag[1,:], tridiag[0,1:], tridiag[2,:-1], b)



from scipy.linalg import solve_banded
#from numpy import flipud
def tridiag_backward_scipy_solve_banded(tridiag, v, b):
    """
    The diagonal elements should be tridiag[1,:],
    the lower offdiagonal should be tridiag[0,1:],
    the upper offdiagonal should be tridiag[2,:-1]
    """
    _trd = tridiag.copy()
    _trd[0,1:] = tridiag[2,:-1]
    _trd[2,:-1] = tridiag[0,1:]
    v[:] = solve_banded((1,1), _trd, b)


# [NOTE] Warning, then, the `tridiag_backward` is not anymore for general tridiag
#tridiag_backward_symm = tridiag_backward_scipy_solveh_banded
#tridiag_backward = tridiag_backward_thomas
tridiag_backward = tridiag_backward_scipy_solve_banded

#def tridiag_backward(tridiag, v, b):
#    """
#    The diagonal elements should be tridiag[1,:],
#    the lower offdiagonal should be tridiag[0,1:],
#    the upper offdiagonal should be tridiag[2,:-1]
#    """
#    return tridiag_backward_scipy_solve_banded(tridiag, v, b)



