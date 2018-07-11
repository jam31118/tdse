"""Collection of matrix multiplication"""

try: from .matrix_c import mat_vec_mul_tridiag
except ImportError: from .matrix_py import mat_vec_mul_tridiag

from .matrix_py import gaussian_elimination_tridiagonal

