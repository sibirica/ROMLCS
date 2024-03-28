from abstract_defs import *
import numpy as np
from scipy import integrate
from typing import Iterable

def conjugate_dynamics(coords: CoordinateSystem, ts: np.ndarray, x0: np.ndarray, debug: bool=False, **kwargs) -> np.ndarray:
    """
    Evaluate dynamics via intermediate coordinate system.
    coords: intermediate coordinate system
    ts: list of times at which to return results
    x0: initial value in base coordinate system
    **kwargs: keyword integration arguments for solve_ivp
    
    Outputs 1d array of x(t) in base coordinates
    """
    y0 = coords.transform(x0, 0)
    if debug:
        print('x0: ', x0)
        print('y0: ', y0)
    if coords.integrator is not None:
        sol = coords.integrator(ts, y0)
    else:
        sol = integrate.solve_ivp(coords.dynamics, (min(ts), max(ts)), y0, t_eval=ts, **kwargs)
    ys = sol['y']
    if debug:
        print("sol['y']: ", ys)
        print("ts: ", ts)
        print(coords.inverse_transform(ys[:, 0], ts[0]))
    return np.stack([coords.inverse_transform(ys[:, i], ts[i]) for i in range(ys.shape[1])], axis=-1)
    