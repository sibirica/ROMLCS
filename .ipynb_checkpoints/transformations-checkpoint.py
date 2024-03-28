from abstract_defs import *
from typing import Union
from scipy.optimize import root
import numpy as np

def approximate_series(fun: Callable[[float, np.ndarray], np.ndarray], t: float=0, 
                       x: np.ndarray, order: int) -> VectorPolynomial:
    """
    Estimate the Taylor series of function fun at time t, position x, to order k.
    """
    pass

def invert(fun: Callable[np.ndarray, np.ndarray], p: np.ndarray, x0: np.ndarray):
    """
    Solve fun(x) = p for x with initial guess x=x0
    """
    return root(lambda x: fun(x)-p, x0)

def translate(shift: Union[np.ndarray, Callable[float, np.ndarray]], dynamics: VectorPolynomial, new_name: str="x'", debug: bool=False) ->CoordinateSystem: #coords: CoordinateSystem,
    if isinstance(shift, np.ndarray):
        #base_point = shift
        transform = lambda x, *a: x-shift
        inverse_transform = lambda x, *a: x+shift
        transform_string = f"{new_name} = x - {shift}"
    else:
        #base_point = shift(0)
        transform = lambda x, t: x-shift(t)
        inverse_transform = lambda x, t: x+shift(t)
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
    polys = [Polynomial([0, A_tilde[i, ...], B_tilde[i, ...], C_tilde[i, ...]]) for i in range(len(vals))]
    new_dynamics = VectorPolynomial(polys)
    
    transform = compose(lambda x: P_inv*x, coords.transform)
    inverse_transform = compose(coords.inverse_transform, lambda x: P*x)
    transform_string = f"{coords.transform_string}, {new_name} = {P_inv}*{coords.name}"
    
    if debug:
        print("P: ", P)
        print("P_inv: ", P_inv)
        print("A_tilde: ", A_tilde)
    
    return CoordinateSystem(name=new_name, transform=transform, inverse_transform=inverse_transform, 
                            dynamics=dynamics, transform_string=transform_string)

def normal_form(coords: CoordinateSystem, new_name: str='z', debug: bool=False) -> CoordinateSystem: # for now, assume non-degenerate
    
    I = np.eye(coords.dynamics.get_degree_k(1).shape)
    J = np.zeros(coords.dynamics.get_degree_k(2).shape)
    K = np.zeros(coords.dynamics.get_degree_k(3).shape)
    JK_poly = VectorPolynomial(...)
    polynomial_transform = lambda x: JK_poly(0, x)
    transform = compose(polynomial_transform, coords.transform)
    inverse_transform = compose(coords.inverse_transform, lambda x: invert(polynomial_transform, x, 0))
    transform_string = f"{coords.transform_string}, {new_name} = {coords.name}"
                        + f" + J({coords.name}, {coords.name}) + K({coords.name}, {coords.name}, {coords.name})"
    
    return CoordinateSystem(name=new_name, transform=transform, inverse_transform=inverse_transform, 
                            dynamics=dynamics, transform_string=transform_string)