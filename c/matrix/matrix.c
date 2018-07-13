#include "matrix.h"

int mat_vec_mul_tridiag(double *alpha, double *beta, double *gamma, double *v, double *out, long N) {
  
  long offset = 1, num_of_elements_in_loop = N - 2;

  double *p_alpha = alpha + offset, 
         *p_beta = beta + offset, 
         *p_gamma = gamma + offset, 
         *p_v = v + offset, 
         *p_out = out + offset;

  double *p_alpha_max = p_alpha + num_of_elements_in_loop;
//         *p_beta_max = p_beta + num_of_elements_in_loop, 
//         *p_gamma_max = p_gamma + num_of_elements_in_loop, 
//         *p_v_max = p_v + num_of_elements_in_loop, 
//         *p_out_max = p_out + num_of_elements_in_loop;
 
  for ( ; p_alpha < p_alpha_max; ++p_alpha, ++p_beta, ++p_gamma, ++p_v, ++p_out ) {
    *p_out = (*(p_beta - 1)) * (*(p_v - 1)) + (*p_alpha) * (*p_v) + (*p_gamma) * (*(p_v + 1)); 
  }

  out[0] = alpha[0] * v[0] + gamma[0] * v[1];
  out[N-1] = beta[N-2] * v[N-2] + alpha[N-1] * v[N-1];

  return 0;

}

int gaussian_elimination_tridiagonal(double *alpha, double *beta, double *gamma, double *b, double *v, long N) {

  // Allocate arrays for intermediate results
  double *delta = (double *) malloc(sizeof(double) * N);
//  double *v = (double *) malloc(sizeof(double) * N);
  
  long i;
  double *p_alpha, *p_beta, *p_gamma, *p_b, *p_delta, *p_v;
//  double alpha_temp;
  double beta_temp;

  // copy the first element as a start
  *delta = *alpha;  
  *v = *b / (*delta);

  // iteration to bigger index
  for ( p_alpha=alpha+1, p_beta=beta, p_gamma=gamma, p_delta=delta, p_b=b+1, p_v=v, i=0; i < N-1;
      ++p_alpha, ++p_beta, ++p_gamma, ++p_delta, ++p_v, ++i, ++p_b ) {
    
    beta_temp = *p_beta;
    *(p_delta+1) = *(p_alpha) - (*p_gamma) * beta_temp / (*p_delta);
    *(p_v+1) = (*p_b - *p_v * beta_temp) / (*(p_delta+1)); 
    // [NOTE] Using `p_v+1` in place of `p_b` is not valid since `p_v+1` hasn't been set yet.

  }

  // iteration to smaller index
  // [NOTE] Check that the address pointed by `p_v`, `p_gamma`, `p_delta` at previous loop can be used.
  for (--p_gamma, --p_delta; i > 0; --p_gamma, --p_delta, --p_v, --i) {
    *(p_v-1) = *(p_v-1) - *p_v * (*p_gamma) / (*p_delta);
    // [NOTE] Using `p_v-1` is valid since it has already been set.
  }

  return 0;
}


