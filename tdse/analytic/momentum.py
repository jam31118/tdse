"""analytical expression for momentum"""

from numpy import sqrt, exp, pi

def prob_density_func_of_momentum_gaussian_1D(p,t,sigma_x,hbar=1):
    return sqrt(2*sigma_x**2/(pi*hbar)) * exp(-2*sigma_x**2/hbar**2 * p**2)



