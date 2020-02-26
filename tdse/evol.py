import numpy as np

def get_M2_tridiag(N):
    _tridiag_shape = (3,N)
    _M2 = np.empty(_tridiag_shape, dtype=float)
    _M2[0,1:], _M2[1,:], _M2[2,:-1] = 1.0/12.0, 10.0/12.0, 1.0/12.0
    _M2[0,0], _M2[2,-1] = 0.0, 0.0
    return _M2

def get_D2_tridiag(N, h):
    _tridiag_shape = (3,N)
    _D2 = np.empty(_tridiag_shape, dtype=float)
    _D2[0,1:], _D2[1,:], _D2[2,:-1] = 1.0, -2.0, 1.0
    _D2 *= 1.0 / (h * h)
    _D2[0,0], _D2[2,-1] = 0.0, 0.0
    return _D2

def mul_tridiag_and_diag(T, D, dtype=None):
    """TD:tridiag / D;tridiag"""
    assert T.shape == (3,D.size) and D.shape == (T.shape[1],)
    if dtype is None: dtype = D.dtype
    _TD = np.empty(T.shape, dtype=dtype)
    _TD[0,1:],_TD[1,:],_TD[2,:-1] = T[0,1:]*D[:-1],T[1,:]*D[:],T[2,:-1]*D[1:]
    _TD[0,0], _TD[2,-1] = 0.0, 0.0
    return _TD

def get_M1_tridiag(N):
    _tridiag_shape = (3,N)
    _M1 = np.empty(_tridiag_shape, dtype=float)
    _M1[0,1:], _M1[1,:], _M1[2,:-1] = 1.0/6.0, 2.0/3.0, 1.0/6.0
    _M1[0,0], _M1[2,-1] = 0.0, 0.0
    return _M1

def get_D1_tridiag(N, h):
    _tridiag_shape = (3,N)
    _D1 = np.empty(_tridiag_shape, dtype=float)
    _D1[0,1:], _D1[1,:], _D1[2,:-1] = -1.0, 0.0, 1.0
    _D1 *= 1.0 / (2.0 * h)
    _D1[0,0], _D1[2,-1] = 0.0, 0.0
    return _D1





