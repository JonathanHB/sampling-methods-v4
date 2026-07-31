import numpy as np
from typing import Callable


class collective_variable:

    def __init__(self, 
                cv_funct: Callable[[np.ndarray], np.ndarray], 
                cv_grad_funct: Callable[[np.ndarray], np.ndarray],
                cv_min: np.array,
                cv_max: np.array,
                grid_n: np.array
                ):
        
        self.cv_funct = cv_funct
        self.cv_grad_funct = cv_grad_funct
        self.cv_min = cv_min
        self.cv_max = cv_max
        self.grid_n = grid_n


def CV_coord0(x):
    return x[:, :1]

def grad_CV_coord0(x):
    n = x.shape[0]
    n_dim = x.shape[1]
    n_cv = 1
    out = np.zeros((n, n_cv, n_dim))
    out[:, 0, 0] = 1.0
    return out

cv_coord0_2d_coord_1d_cv = collective_variable(
    cv_funct = CV_coord0,
    cv_grad_funct= grad_CV_coord0,
    cv_min=np.array([-3.0]),
    cv_max=np.array([3.0]),
    grid_n = 81
)