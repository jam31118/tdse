#include <Python.h>
#include <numpy/arrayobject.h>
#include "matrix.h"

static PyObject *matrix_c_mat_vec_mul_tridiag(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
  {"mat_vec_mul_tridiag", matrix_c_mat_vec_mul_tridiag, METH_VARARGS, "Foward matrix multiplication for tridiagonal matrix with single vector"}
};

static PyModuleDef matrix_c = {
  PyModuleDef_HEAD_INIT,
  "matrix_c",
  "C extension for `matrix` operation module",
  -1,
  module_methods
};

PyMODINIT_FUNC PyInit_matrix_c(void) {
  import_array();
  return PyModule_Create(&matrix_c);
}


// Define wrapper for forward matrix multiplication for tridiagonals
static PyObject *matrix_c_mat_vec_mul_tridiag(PyObject *self, PyObject *args) {
  PyObject *alpha_arg = NULL, *beta_arg = NULL, *gamma_arg = NULL, *v_arg = NULL;
  PyObject *alpha_obj = NULL, *beta_obj = NULL, *gamma_obj = NULL, *v_obj = NULL;
  PyObject *out_obj = NULL;

  long N = -1;
  long ndim = 1;
  npy_intp dims_N[ndim]; // dims_N_1[ndim];  // defines shape of arrays

  // Parse Arguments
  if ( ! PyArg_ParseTuple(
        args, "O!O!O!O!", 
        &PyArray_Type, &alpha_arg, 
        &PyArray_Type, &beta_arg, 
        &PyArray_Type, &gamma_arg,
        &PyArray_Type, &v_arg )
      ) 
  {
    PyErr_SetString(PyExc_Exception, "Failed to parse arguments");
    return NULL;
  }

  // Get numpy array object from argument object
  alpha_obj = PyArray_FROM_OTF(alpha_arg, NPY_DOUBLE, NPY_IN_ARRAY);
  if ( (alpha_obj == NULL) && (PyArray_NDIM(alpha_obj) != ndim) ) { goto fail; }
  beta_obj = PyArray_FROM_OTF(beta_arg, NPY_DOUBLE, NPY_IN_ARRAY);
  if ( (beta_obj == NULL) && (PyArray_NDIM(beta_obj) != ndim) ) { goto fail; }
  gamma_obj = PyArray_FROM_OTF(gamma_arg, NPY_DOUBLE, NPY_IN_ARRAY);
  if ( (gamma_obj == NULL) && (PyArray_NDIM(gamma_obj) != ndim) ) { goto fail; }
  v_obj = PyArray_FROM_OTF(v_arg, NPY_DOUBLE, NPY_IN_ARRAY);
  if ( (v_obj == NULL) && (PyArray_NDIM(v_obj) != ndim) ) { goto fail; }

  // Figure out `N`
  npy_intp *alpha_dims, *beta_dims, *gamma_dims, *v_dims;
// long N_alpha, N_beta, N_gamma, N_v;
  alpha_dims = PyArray_DIMS(alpha_obj);
  beta_dims = PyArray_DIMS(beta_obj);
  gamma_dims = PyArray_DIMS(gamma_obj);
  v_dims = PyArray_DIMS(v_obj);

  if (ndim != 1) { 
    PyErr_SetString(PyExc_Exception, "`ndim` should be 1"); 
    return NULL; 
  }
  long dim_index;
  long all_is_okay = 1;
  for (dim_index = 0; dim_index < ndim; dim_index++) {
    all_is_okay &= alpha_dims[dim_index] == beta_dims[dim_index] + 1;
    all_is_okay &= alpha_dims[dim_index] == gamma_dims[dim_index] + 1;
    all_is_okay &= alpha_dims[dim_index] == v_dims[dim_index];
  }
  if ( !all_is_okay ) { PyErr_SetString(PyExc_Exception, "Unexpected dimension for alpha | beta | gamma | v"); return NULL; }
  N = alpha_dims[0];
  dims_N[0] = N;
  
  out_obj = PyArray_SimpleNew(ndim, dims_N, NPY_DOUBLE);

  // Get pointer to each data array
  double *p_alpha = PyArray_DATA(alpha_obj);
  double *p_beta = PyArray_DATA(beta_obj);
  double *p_gamma = PyArray_DATA(gamma_obj);
  double *p_v = PyArray_DATA(v_obj);
  double *p_out = PyArray_DATA(out_obj);

  // Run core C routine
  if ( mat_vec_mul_tridiag(p_alpha, p_beta, p_gamma, p_v, p_out, N) != 0 ) {
    PyErr_SetString(PyExc_Exception, "Failed to run `mat_vec_mul_tridiag`");
    return NULL;
  }

  // Return results
  PyObject *return_tuple = Py_BuildValue("N", out_obj);
  return return_tuple;
  
  // Run if something gets wrong
fail:
  Py_XDECREF(alpha_obj);
  Py_XDECREF(beta_obj);
  Py_XDECREF(gamma_obj);
  Py_XDECREF(v_obj);
  return NULL;

}

