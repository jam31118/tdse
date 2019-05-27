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

def tridiag_backward(tridiag, v, b):
    """
    The diagonal elements should be tridiag[1,:],
    the lower offdiagonal should be tridiag[0,1:],
    the upper offdiagonal should be tridiag[2,:-1]
    """
    v[:] = gaussian_elimination_tridiagonal(tridiag[1,:], tridiag[0,1:], tridiag[2,:-1], b)

