from abstract_defs import *
from transformations import *
import numpy as np

I = np.eye(3, dtype=np.cdouble)
J = np.zeros([3, 3, 3], dtype=np.cdouble)
J[0, 0, 0] = 1+1j
J[1, 0, 0] = 1-1j
J[2, 0, 0] = 2j
J[1, 1, 2] = 3-3j
J *= 0.1

J_poly = VectorPolynomial([np.array([0, 0, 0]), I, J])
minus_J_poly = VectorPolynomial([np.array([0, 0, 0]), I, -J])    
polynomial_transform = make_poly_transform(J_poly)
x0_guess = lambda x, y: minus_J_poly(0, x)
inverse_transform = make_inverse_transform(polynomial_transform, x0_guess)

v = np.array([1, 3, 5])
w = v*1.0j
f_v = polynomial_transform(v)
f_inv_f_v = inverse_transform(f_v)
f_w = polynomial_transform(w)
f_inv_f_w = inverse_transform(f_w)
print('v: ', v)
print('v+p(v): ', f_v)
print('f_inv(v+p(v)): ', f_inv_f_v)
print('w: ', w)
print('w+p(w): ', f_w)
print('f_inv(w+p(w)): ', f_inv_f_w)