"""Collection of matrix multiplication"""

from sys import stderr

try: from .matrix_c import mat_vec_mul_tridiag
except (ImportError, UnicodeDecodeError) as e:
  print("Could not import due to error: {}".format(e), file=stderr)
  print("Falling back to the default, python version of module", file=stderr)
  from .matrix_py import mat_vec_mul_tridiag

from .matrix_py import gaussian_elimination_tridiagonal

