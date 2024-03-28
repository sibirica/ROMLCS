from abstract_defs import *
import numpy as np
from scipy import integrate
from typing import Iterable

def conjugate_dynamics(coords: CoordinateTransformation, ts: np.ndarray, x0: np.ndarray, **kwargs)
    """
    Evaluate dynamics via intermediate coordinate system.
    coords: intermediate coordinate system
    ts: list of times at which to return results
    x0: initial value in base coordinate system
    **kwargs: keyword integration arguments for solve_ivp
    """
    y0 = coords.transform(x0, t=0)
    if coords.integrator is not None:
        sol = coords.integrator(ts, y0)
    else:
        sol = integrate.solve_ivp(coords.dynamics, (min(ts), max(ts)), y0, t_eval=ts, **kwargs)
    return [coords.inverse_transform(y, t) for (t, y) in zip(ts, sol['y'])]

def make_VP_from_arrays(: Iterable
    