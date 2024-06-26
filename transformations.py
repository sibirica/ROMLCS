from abstract_defs import *
from typing import Union, Optional
#from scipy.optimize import root
#from mpmath import findroot
from scipy.optimize import fsolve
import numpy as np

def approximate_series(fun: Callable[[float, np.ndarray], np.ndarray], 
                       x: np.ndarray, order: int, t: float=0) -> VectorPolynomial:
    """
    Estimate the Taylor series of function fun at time t, position x, to order k.
    """
    pass

def invert(fun: Callable[np.ndarray, np.ndarray], p: np.ndarray, x0: np.ndarray, debug: bool=False):
    """
    Solve fun(x) = p for x with initial guess x=x0
    """
    if debug:
        print('x0: ', x0)
        print('p: ', p)
        print('unfolded p: ', fold(p))
        print('initial error:', fun(x0)-p)
    n_dim = len(x0)
    unfold = lambda y: np.hstack([y.real, y.imag])
    fold = lambda z: z[:n_dim]+1.0j*z[n_dim:]
    real_fun = lambda x: unfold(fun(fold(x))-p)
    r = fsolve(real_fun, unfold(x0))
    solution = fold(r)
    if debug:
        print("solution: ", r)
        print("final error of real part: ", fun(r)-p)
        print("final total error: ", fun(solution)-p)
    return solution
    #return r.x[:n_dim]#+1.0j*r.x[n_dim:]
    #return findroot(lambda x: fun(x)-p, list(x0), solver='muller')


#def find_root_complex(fun: Callable[np.ndarray, np.ndarray], x0: np.ndarray):

def translate(shift: Union[np.ndarray, Callable[float, np.ndarray]], dynamics: VectorPolynomial, new_name: str="x'", debug: bool=False) ->CoordinateSystem: #coords: CoordinateSystem,
    if isinstance(shift, np.ndarray):
        #base_point = shift
        transform = lambda x, *args, **kwargs: x-shift
        inverse_transform = lambda x, *args, **kwargs: x+shift
        transform_string = f"{new_name} = x - {shift}"
    else:
        #base_point = shift(0)
        transform = lambda x, t, *args, **kwargs: x-shift(t)
        inverse_transform = lambda x, t, *args, **kwargs: x+shift(t)
        transform_string = f"{new_name} = x - s(t)"
        
    if debug:
        print("shift: ", shift)
        
    return CoordinateSystem(name=new_name, transform=transform, inverse_transform=inverse_transform, 
                            dynamics=dynamics, transform_string=transform_string)

def diagonalize(coords: CoordinateSystem, new_name: str='y', debug: bool=False) -> CoordinateSystem: # for now, assume diagonalizable
    # also, we assume cubic polynomial order for now
    A = coords.dynamics.get_degree_k(1)
    B = coords.dynamics.get_degree_k(2)
    C = coords.dynamics.get_degree_k(3)
    vals, P = np.linalg.eig(A)
    P_inv = np.linalg.inv(P)
    A_tilde = np.diag(vals)
    B_tilde = np.einsum('ab, bcd, ce, df->aef', P_inv, B, P, P)
    C_tilde = np.einsum('ab, bcde, cf, dg, eh->afgh', P_inv, C, P, P, P)
    #polys = [Polynomial([0, A_tilde[i, ...], B_tilde[i, ...], C_tilde[i, ...]]) for i in range(len(vals))]
    new_dynamics = VectorPolynomial([np.array([0, 0, 0]), A_tilde, B_tilde, C_tilde])
    
    transform = compose(lambda x, *args, **kwargs: P_inv @ x, coords.transform)
    inverse_transform = compose(coords.inverse_transform, lambda x, *args, **kwargs: P @ x)
    transform_string = f"{coords.transform_string}, {new_name} = {P_inv}*{coords.name}"
    
    if debug:
        print("P: ", P)
        print("P_inv: ", P_inv)
        print("A_tilde: ", A_tilde)
    
    return CoordinateSystem(name=new_name, transform=transform, inverse_transform=inverse_transform, 
                            dynamics=new_dynamics, transform_string=transform_string)

def make_poly_transform(JK_poly: VectorPolynomial) -> Callable[np.ndarray, np.ndarray]: 
    return lambda x, *a, **kw: JK_poly(0, x)


def make_inverse_transform(transform: Callable[np.ndarray, np.ndarray], choose_x0: Callable[np.ndarray, np.ndarray],
    coords: Optional[CoordinateSystem]=None) -> Callable[np.ndarray, np.ndarray]:
    # make the inverse transformation such that it also sets the memory point in the coordinate system for the future
    def inverse(x, *a, initial: bool=False, reset_value: np.ndarray=None, **kw):
        inverse.result = invert(transform, x, choose_x0(x, coords.transform(reset_value) if initial else inverse.result))
        return inverse.result
        #return reset_value if initial else inverse.result # force correct inverse if reset
    return inverse

def normal_form(coords: CoordinateSystem, new_name: str='z', debug: bool=False) -> CoordinateSystem: # for now, assume non-degenerate
    # also assuming cubic polynomial order for now
    A_tilde = coords.dynamics.get_degree_k(1)
    B_tilde = coords.dynamics.get_degree_k(2)
    C_tilde = coords.dynamics.get_degree_k(3)
    I = np.eye(A_tilde.shape[0], dtype=np.cdouble)
    J = np.zeros(B_tilde.shape, dtype=np.cdouble)
    K = np.zeros(C_tilde.shape, dtype=np.cdouble)
    eigs = np.diag(A_tilde)
    
    it = np.nditer(B_tilde, flags=['multi_index'])
    for x in it:
        inds = it.multi_index
        J[inds] = B_tilde[inds]/(eigs[inds[0]]-eigs[inds[1]]-eigs[inds[2]])
    cal_C = C_tilde+2*np.einsum('ijk, klm->ijlm', J, B_tilde)

    it = np.nditer(cal_C, flags=['multi_index'])
    for x in it:
        inds = it.multi_index
        K[inds] = x/(eigs[inds[0]]-eigs[inds[1]]-eigs[inds[2]]-eigs[inds[3]])
    
    JK_poly = VectorPolynomial([np.array([0, 0, 0]), I, J, K])
    minus_JK_poly = VectorPolynomial([np.array([0, 0, 0]), I, -J, -K])
    polynomial_transform = make_poly_transform(JK_poly)
    x0_guess = lambda x, y: minus_JK_poly(0, x) if y is None else y
    inverse_poly_transform = make_inverse_transform(polynomial_transform, x0_guess, coords)
    transform = compose(polynomial_transform, coords.transform)
    inverse_transform = compose(coords.inverse_transform, inverse_poly_transform)
    transform_string = (f"{coords.transform_string}, {new_name} = {coords.name}"
                        + f" + J({coords.name}, {coords.name}) + K({coords.name}, {coords.name}, {coords.name})")
    dynamics = VectorPolynomial([np.array([0, 0, 0]), A_tilde])
    
    if debug:
        print("I: ", I)
        print("J: ", J)
        print("K: ", K)
        
    def integrator(ts, x0):
        sol = dict()
        sol['y'] = np.stack([x0[i]*np.exp(eigs[i]*ts) for i in range(len(eigs))], axis=0)
        #sol_components = []
        #for eig in np.diag(A_tilde):
        #    sol_components = [np.exp(eig*ts)]
        return sol
    
    return CoordinateSystem(name=new_name, transform=transform, inverse_transform=inverse_transform, 
                            dynamics=dynamics, transform_string=transform_string, integrator=integrator)
                            #dynamics=dynamics, transform_string=transform_string, integrator=None)