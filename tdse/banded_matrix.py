"""Routines for banded matrices"""

import numpy as np
from numpy import asarray
from scipy.linalg import solve_banded

from .matrix import mat_vec_mul_tridiag






# from numpy import asarray

# from tdse.matrix import mat_vec_mul_tridiag

def tridiag_with_corners_mul_vec(td, cu, cl, x):
    """
    Evaluate `b = Ax` where `x` is a vector and `A` is a tridiagonal matrix 
    with possibly non-zero upper-right and lower-left corner elements.
    
    Parameters
    ----------
    td : (3,M) array-like
        The tridiagonal part of the total matrix
        td[0,1:] : upper-diagonal
        td[1,:] : diagonal
        td[2,:-1] : lower-diagonal
    cu, cl : float or complex
        The upper and lower corner elements of the total matrix
    x : (M,) array-like
        A vector in `b = Ax`
    
    Returns
    -------
    b : (M,) array-like
        Multiplication of `A` with `x`, namely, `Ax`
    """
    _td, _x = asarray(td), asarray(x)
    assert _td.dtype == _x.dtype
    assert _td.ndim == 2 and _x.ndim == 1
    assert _td.shape == (3,_x.shape[0])
    
    b = mat_vec_mul_tridiag(_td[1,:], _td[2,:-1], _td[0,1:], _xx)
    b[0] += cu * _xx[-1]
    b[-1] += cl * _xx[0]
    return b





# from numpy import asarray

# from tdse.matrix import mat_vec_mul_tridiag

def tridiag_with_rank1_mul_vec(td, u, v, x):
    """
    Evaluate `b = Ax` where `x` is a vector and `A = td + u v^T` 
    is a tridiagonal matrix plus an outer product of two vectors `u` and `v`

    Parameters
    ----------
    td : (3,M) array-like
        The tridiagonal part of the total matrix
        td[0,1:] : upper-diagonal
        td[1,:] : diagonal
        td[2,:-1] : lower-diagonal
    u, v : (M,) array-like
        Two vectors whose outer product `u v^T`
        is the rank-1 update of the original tridiagonal matrix `td`
    x : (M,) array-like
        A vector to which `A` is multiplied to get `b = Ax`
    
    Returns
    -------
    b : (M,) array-like
        The multiplication of `A` with `x`, namely, `Ax`
    """
    _td, _x = asarray(td), asarray(x)
    assert _td.dtype == _x.dtype
    assert _td.ndim == 2 and _x.ndim == 1
    assert _td.shape == (3,_x.shape[0])
    _Nx = _x.shape[0]
    _u, _v = asarray(u), asarray(v)
    assert _u.shape == (_Nx,) and _v.shape == (_Nx,)

    td_x = mat_vec_mul_tridiag(_td[1,:], _td[2,:-1], _td[0,1:], _x)
    u_v_x = np.dot(_u, np.dot(_v, _x))
    
    b = td_x + u_v_x
    return b







# import numpy as np
# from scipy.linalg import solve_banded

def solve_banded_with_rank1_update(
        l_and_u, A, u, v, b, thres=1e-10, **solve_kwargs):
    """
    Solve Ax=b for a banded matrix with a rank-1 update
    
    Notes
    -----
    The rank-1 update is specified as an outer product of u and v.
    The rank-1 update is implemented through the Sherman-Morrison formula.
    
    Parameters
    ----------
    l_and_u : (int, int)
        The number of non-zero lower and upper off-diagonals
        given in a form of ('l', 'u')
    A : (`l` + `u` + 1, M) array-like
        A banded matrix.
        Its shape should be consistent to the `ab` argument 
        of `scipy.linalg.solve_banded()`.
    u, v : (M,) array-like
        Two vectors whose outer product 
        is the rank-1 update of the original matrix
    b : (M,) or (M,K) array-like
        The right-hand side of the equation to solve, namely, Ax=b
    thres : positive float
        A threshold for the requirement:
        `abs(1 + v_Ainv_u) > thres`
    solve_kwargs : dictionary
        The keyword arguments passed to `scipy.linalg.solve_banded()`
    """
    Ainv_u_b = solve_banded(
            l_and_u, A, np.stack((u,b), axis=-1), **solve_kwargs)
    Ainv_u, Ainv_b = Ainv_u_b[:,0], Ainv_u_b[:,1]

    ##### Note
    # There is no complex conjugate inside `numpy.dot()` 
    # and it is necessary to give the right answer 
    # for the complex-valued matrices and vectors
    #####
    v_Ainv_u = np.dot(v, Ainv_u)
    assert abs(1 + v_Ainv_u) > thres
    v_Ainv_b = np.dot(v, Ainv_b)

    Atot_inv_b = Ainv_b - (v_Ainv_b / (1 + v_Ainv_u)) * Ainv_u

    return Atot_inv_b





# from numpy import asarray

def solve_tridiag_with_corners(
        td, cu, cl, b, gamma=1., thres=1e-10, **solve_kwargs):
    """
    Solve a system of equations represented by 
    a tridiagonal matrix with corners
    
    Parameters
    ----------
    td : (3,M) array-like
        A tridiagonal part of the total matrix
    cu, cl : float or complex
        An upper and lower corner of the total matrix
    b : (M,) or (M,K) array-like
        The right-hand side of the equation, Ax=b
    thres : positive float
        A threshold for the requirement:
        `abs(1 + v_Ainv_u) > thres`
    solve_kwargs : dictionary
        The keyword arguments passed to `scipy.linalg.solve_banded()`
    """

    ########## Check input arguments
    _td, _b = asarray(td), asarray(b)
    assert _td.ndim == 2 and _b.ndim > 0
    assert _td.shape == (3,_b.shape[0])
    M = _td.shape[1]
    assert gamma != 0
    
    ########## Construct the tridiagonal matrix `td` 
    ########## and vectors `u`, `v` for rank-1 update
    tdp = _td.copy()
    tdp[1,[0,-1]] -= asarray([gamma*cl, cu/gamma])

    u = np.zeros((M,), dtype=_td.dtype)
    u[[0,-1]] = [gamma, 1.]
    v = np.zeros((M,), dtype=_td.dtype)
    v[[0,-1]] = [cl, cu/gamma]
    
    ########## Solve equation and return the solution
    l_and_u = (1,1)
    return solve_banded_with_rank1_update(
            l_and_u, tdp, u, v, b, thres=thres, **solve_kwargs)






