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


