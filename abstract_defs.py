from dataclasses import dataclass, field
from typing import List, Dict, Union, Tuple, Iterable, Generator, Callable, Optional
from collections import Counter
from utils import VP_from_
import functools
import numpy as np

# implementing function composition
def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))

def compose(*fs):
    return functools.reduce(compose2, fs)

@dataclass
class Polynomial(object):
    """
    Multivariate polynomial object.
    :attribute n_vars: number of variables
    :attribute degree: degree of the polynomial
    :attribute data: iterable where kth entry is a k-dimensional numpy array representing coefficients of a k-linear form. 
                     By convention, each array is upper triangular (i.e. entry is 0 unless i<=j<=k).
    """
    
    data: Iterable[np.ndarray] = field(default_factory = lambda : np.array([0])) #(np.array([0]))
    degree: int = field(init=False)
    n_vars: int = field(init=False)
    
    def __post_init__(self):
        self.degree = len(self.data)-1
        self.n_vars = self.data[1].size
        
    def __repr__(self) -> str: # handle 0th entry separately in case we want to use a scalar as shorthand
        return " + \n".join([str(self.data[0])]+[self.to_string(array) for array in self.data[1:]])
    
    def __call__(self, vector: np.ndarray, t: float=None) -> float:
        """
        Apply the polynomial to a numpy vector.
        """
        return self.data[0] + sum([self.apply(array, vector) for array in self.data[1:]])
    
    def to_string(self, array: np.ndarray):
        string = ""
        it = np.nditer(array, flags=['multi_index'])
        for x in it:
            if x != 0:
                string += f"{x} * "
                c = Counter(it.multi_index)
                addition = " * ".join([f"x{key+1}^{c[key]}" if c[key]>1 else f"x{key+1}" for key in c.keys()]) + " + "
                #print(c, addition)
                string += addition
        return string[:-3] # trim off last " + "
    
    def apply(self, array: np.ndarray, vector: np.ndarray) -> float:
        result = 0
        it = np.nditer(array, flags=['multi_index'])
        for x in it:
            if x != 0:
                subtotal = x
                for y in it.multi_index:
                    subtotal = subtotal * vector[y]
                result += subtotal
        return result

@dataclass
class VectorPolynomial(object):
    """
    A vector where each entry is a multivariate polynomial. 
    The number of entries is generally equal to the number of variables.
    :attribute n_vars: number of variables
    :attribute degree: degree of the polynomial
    :attribute data: iterable where each entry is a polynomial
    """
    data: Iterable[Polynomial]
    degree: int = field(init=False)
    n_vars: int = field(init=False)
    n_entries: int = field(init=False)
    
    def __post_init__(self):
        self.n_entries = len(self.data)
        self.n_vars = self.data[0].n_vars
        self.degree = max([poly.degree for poly in self.data])
        
    def __repr__(self) -> str:
        string = ""
        for ind, poly in enumerate(self.data):
            string += f"entry {ind+1}: {poly} \n\n"
        return string
    
    def __call__(self, t: float, vector: np.ndarray) -> np.ndarray:
        """
        Apply the polynomial to a numpy vector.
        """
        
        return np.array([poly(vector) for poly in self.data])
    
    def get_degree_k(self, degree: int) -> np.ndarray:
        """
        Return the rank-(1, k) tensor corresponding to concatenation of kth degrees of each polynomial along first axis
        """
        return np.stack([poly.data[k] for poly in self.data], axis=0)
    
    def apply(self, array, vector):
        result = 0
        it = np.nditer(array, flags=['multi_index'])
        for x in it:
            if x != 0:
                subtotal = x
                for y in it.multi_index:
                    subtotal = subtotal * vector[y]
                result += subtotal
        return result
    
@dataclass
class TimeVaryingPolynomial(object): # implement if we end up using time-varying polynomials in the dynamics 
    pass

@dataclass
class CoordinateSystem(object):
    """
    Object representing a set of coordinates & associated (approximate) dynamics. 
    :attribute name: name of coordinate system
    :attribute old_name: name of old coordinate system
    :attribute base_point: base point near which the coordinates are defined
    :attribute transform: function mapping original coordinates x to current coordinates x' (t-dependent?).
    :attribute inverse_transform: function mapping current coordinates x' to original coordinates x (t-dependent?).
    :attribute dynamics: callable (e.g. multivariate polynomial) representing RHS dynamics (t-dependent?).
    :attribute transform_string: human-interpretable string representation of transform
    :attribute (optional) integrator: function taking (list of t's, x(0)) to dict sol where sol['y'] is list of x at each t
    """
    name: str = "x"
    #old_name: str = "r"
    #base_point: np.ndarray = None
    transform: Callable[[Optional[float], np.ndarray], np.ndarray] = None
    inverse_transform: Callable[[Optional[float], np.ndarray], np.ndarray] = None
    dynamics: VectorPolynomial = None
    #dynamics: Union[Polynomial, TimeVaryingPolynomial]
    transform_string: str = None
    integrator: Callable[np.ndarray, np.ndarray], dict] = None
    
    def __repr__(self) -> str:
        return f"Transformation: {transform_string}" + " \n" + f"{name}_dot = {dynamics}"
    
#@dataclass
#class CallableTransformation(CoordinateTransformation):
#    """
#    Object representing a single step of coordinate transformation, e.g. constant shift, 
#    leading order diagonalization, normal form.
#    :attribute change_dynamics: (optional) function to compute the new RHS dynamics from the old dynamics 
#    """
#    change_dynamics: Callable[Callable[[np.ndarray, Optional[float]], np.ndarray], Callable[[np.ndarray, Optional[float]], np.ndarray]] = None
#
#    def __call__(self, coords: CoordinateTransformation) -> CoordinateTransformation:
#        new_dynamics = self.change_dynamics(coords.dynamics) if change_dynamics is not None else self.dynamics
#        return CoordinateTransformation(transform=compose(self.transform, coords.transform), 
#                                        inverse_transform=compose(coords.inverse_transform, self.inverse_transform),
#                                        dynamics=new_dynamics, 
#                                        transform_string = f"{coords.transform_string}, {self.transform_string}"
#                                       )