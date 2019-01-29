#include <Python.h>
#include <numpy/arrayobject.h>
#include "matrix.hh"
#include <complex>
#include <iostream>


extern "C" { static PyObject *matrix_c_mat_vec_mul_tridiag(PyObject *self, PyObject *args); }

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
  std::cout << "init!\n";
  import_array();
  return PyModule_Create(&matrix_c);
}


template <class T>
int mat_vec_mul_tridiag_core_template(PyObject *alpha_obj, PyObject *beta_obj, PyObject *gamma_obj, PyObject *v_obj, PyObject *out_obj, long N) {
  // Get pointer to each data array
  T *p_alpha = (T *) PyArray_DATA(alpha_obj);
  T *p_beta = (T *) PyArray_DATA(beta_obj);
  T *p_gamma = (T *) PyArray_DATA(gamma_obj);
  T *p_v = (T *) PyArray_DATA(v_obj);
  T *p_out = (T *) PyArray_DATA(out_obj);

  // Run core C routine
  if ( mat_vec_mul_tridiag_template<T>(p_alpha, p_beta, p_gamma, p_v, p_out, N) != 0 ) {
    PyErr_SetString(PyExc_Exception, "Failed to run `mat_vec_mul_tridiag()`");
    return 1;
  }
  return 0;
}

PyObject *return_tuple;

// Define wrapper for forward matrix multiplication for tridiagonals
static PyObject *matrix_c_mat_vec_mul_tridiag(PyObject *self, PyObject *args) {
  
  std::cout << "at the head of matrix_c_mat_vec_mul_tridiag()\n";

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

  long dim_index;
  long all_is_okay = 1;
  long num_of_in_obj = 4;
  PyObject *in_obj_array[num_of_in_obj];
  long all_has_same_typenum = 1; 
  int typenum;

  // Get numpy array object from argument object
  alpha_obj = PyArray_FROM_OTF(alpha_arg, NPY_NOTYPE, NPY_IN_ARRAY);
  if ( (alpha_obj == NULL) && (PyArray_NDIM(alpha_obj) != ndim) ) { goto fail; }
  beta_obj = PyArray_FROM_OTF(beta_arg, NPY_NOTYPE, NPY_IN_ARRAY);
  if ( (beta_obj == NULL) && (PyArray_NDIM(beta_obj) != ndim) ) { goto fail; }
  gamma_obj = PyArray_FROM_OTF(gamma_arg, NPY_NOTYPE, NPY_IN_ARRAY);
  if ( (gamma_obj == NULL) && (PyArray_NDIM(gamma_obj) != ndim) ) { goto fail; }
  v_obj = PyArray_FROM_OTF(v_arg, NPY_NOTYPE, NPY_IN_ARRAY);
  if ( (v_obj == NULL) && (PyArray_NDIM(v_obj) != ndim) ) { goto fail; }

  std::cout << "generated objects\n";
  in_obj_array[0] = alpha_obj;
  in_obj_array[1] = beta_obj;
  in_obj_array[2] = gamma_obj;
  in_obj_array[3] = v_obj;
//   = { alpha_obj, beta_obj, gamma_obj, v_obj };

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
  for (dim_index = 0; dim_index < ndim; dim_index++) {
    all_is_okay &= alpha_dims[dim_index] == beta_dims[dim_index] + 1;
    all_is_okay &= alpha_dims[dim_index] == gamma_dims[dim_index] + 1;
    all_is_okay &= alpha_dims[dim_index] == v_dims[dim_index];
  }
  if ( !all_is_okay ) { PyErr_SetString(PyExc_Exception, "Unexpected dimension for alpha | beta | gamma | v"); goto fail; }
  N = alpha_dims[0];
  dims_N[0] = N;
  
  // Check whether all input arrays have same typenum
  typenum = PyArray_TYPE(in_obj_array[0]);
  for (int i = 1; i < 4; i++) { all_has_same_typenum &= typenum == PyArray_TYPE(in_obj_array[i]); }
  if ( ! all_has_same_typenum ) { PyErr_SetString(PyExc_Exception, "Inconsistent typenum for input arrays"); goto fail; }
  
  out_obj = PyArray_SimpleNew(ndim, dims_N, typenum);


  // Run the core function
  if (typenum == NPY_DOUBLE) {
    if (mat_vec_mul_tridiag_core_template<double>(alpha_obj, beta_obj, gamma_obj, v_obj, out_obj, N) != 0) { goto fail; }
  } else if (typenum == NPY_COMPLEX128) {
    if (mat_vec_mul_tridiag_core_template< std::complex<double> >(alpha_obj, beta_obj, gamma_obj, v_obj, out_obj, N) != 0) { goto fail; }
  } else {
    PyErr_SetString(PyExc_Exception, "Unexpected typenum");
    goto fail;
  }

//  // Get pointer to each data array
//  double *p_alpha = PyArray_DATA(alpha_obj);
//  double *p_beta = PyArray_DATA(beta_obj);
//  double *p_gamma = PyArray_DATA(gamma_obj);
//  double *p_v = PyArray_DATA(v_obj);
//  double *p_out = PyArray_DATA(out_obj);
//
//  // Run core C routine
//  if ( mat_vec_mul_tridiag(p_alpha, p_beta, p_gamma, p_v, p_out, N) != 0 ) {
//    PyErr_SetString(PyExc_Exception, "Failed to run `mat_vec_mul_tridiag`");
//    return NULL;
//  }

  // Return results
  return_tuple = Py_BuildValue("N", out_obj);
  return return_tuple;
  
  // Run if something gets wrong
fail:
  Py_XDECREF(alpha_obj);
  Py_XDECREF(beta_obj);
  Py_XDECREF(gamma_obj);
  Py_XDECREF(v_obj);
  Py_XDECREF(out_obj);
  return NULL;

}

