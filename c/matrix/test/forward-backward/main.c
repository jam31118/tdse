#include <stdio.h>
#include "matrix.h"

int main() {

  // Prepare input and output arrays
  long N = 10;
  double alpha[N], beta[N-1], gamma[N-1], v[N], out[N];
  long i;
  for (i=0; i<N-1; i++) {
    alpha[i] = -2;
    beta[i] = 1;
    gamma[i] = 1;
    v[i] = i * i;
  }
  // At this point, i == N - 1 due to `i++` at the end of for loop statement
  if (i != N - 1) { fprintf(stderr, "[ERROR] Unexpected value for index `i`\n"); return 1; }

  alpha[i] = -2;
  v[i] = i * i;

  for (i=0; i<N-1; i++) { 
    fprintf(stdout, "[ LOG ] out[%ld], v[%ld], alpha[%ld], beta[%ld], gamma[%ld] = %f, %f, %f, %f, %f\n", 
        i, i, i, i, i, out[i], v[i], alpha[i], beta[i], gamma[i]);
  }
  fprintf(stdout, "[ LOG ] out[%ld], v[%ld], alpha[%ld] = %f, %f, %f\n", i, i, i, out[i], v[i], alpha[i]);

  // Calculate!
  if ( mat_vec_mul_tridiag(alpha, beta, gamma, v, out, N) != 0 ) {
    fprintf(stderr, "[ERROR] Failed to run forward tridiagnals matrix multiplication\n");
    return 1;
  }

  for (i=0; i<N; i++) { fprintf(stdout, "[ LOG ] out[%ld] = %f\n", i, out[i]); }

  // Gaussian Elimination
  if ( gaussian_elimination_tridiagonal(alpha, beta, gamma, out, v, N) != 0 ) {
    fprintf(stderr, "[ERROR] Failed to run gaussian_elimination_tridiagonal()\n");
    return 1;
  }

  for (i=0; i<N; i++) { fprintf(stdout, "[ LOG ] v[%ld] = %f\n", i, v[i]); }

  return 0;

}
