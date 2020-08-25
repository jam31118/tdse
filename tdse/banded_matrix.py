"""Routines for banded matrices"""

import numpy as np
from numpy import asarray
from scipy.linalg import solve_banded


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
    v_Ainv_u = np.sum(np.conj(v) * Ainv_u, axis=-1)
    assert abs(1 + v_Ainv_u) > thres
    v_Ainv_b = np.sum(np.conj(v) * Ainv_b, axis=-1)
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
    _td, _b = asarray(td), asarray(b)
    assert _td.ndim == 2 and _b.ndim > 0
    assert _td.shape == (3,_b.shape[0])
    M = _td.shape[1]
    assert gamma != 0
    
    tdp = _td.copy()
    tdp[1,[0,-1]] -= asarray([gamma*cl, cu/gamma])
    u = np.zeros((M,), dtype=np.float)
    u[[0,-1]] = [gamma, 1.]
    v = np.zeros((M,), dtype=np.float)
    v[[0,-1]] = [cl, cu/gamma]
    
    l_and_u = (1,1)
    return solve_banded_with_rank1_update(
            l_and_u, tdp, u, v, b, thres=1e-10, **solve_kwargs)
