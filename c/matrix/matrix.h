#ifndef _MATRIX_H_
#define _MATRIX_H_


#include <stdio.h>
#include <stdlib.h>

int mat_vec_mul_tridiag(double *alpha, double *beta, double *gamma, double *v, double *out, long N);
int gaussian_elimination_tridiagonal(double *alpha, double *beta, double *gamma, double *b, double *out, long N);


#endif // _MATRIX_H_
