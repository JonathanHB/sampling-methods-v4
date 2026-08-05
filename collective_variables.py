import numpy as np
from typing import Callable


class collective_variable:

    def __init__(self, 
                cv_funct: Callable[[np.ndarray], np.ndarray], 
                cv_grad_funct: Callable[[np.ndarray], np.ndarray],
                cv_min: np.array,
                cv_max: np.array,
                grid_n: np.array,
                n_cv_dim: int
                ):
        
        self.cv_funct = cv_funct
        self.cv_grad_funct = cv_grad_funct
        self.cv_min = cv_min
        self.cv_max = cv_max
        self.grid_n = grid_n
        self.n_cv_dim = n_cv_dim


#1D CV

def CV_coord0(x):
    return x[:, :1] #I think this differs from x[:, 0] by returning a 2d array with of shape (n,1) rather than a 1d array of shape (n)

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
    grid_n = 81,
    n_cv_dim = 1
)


#2D CV

def CV_coord01(x):
    return x[:, :2]

def grad_CV_coord01(x):
    n = x.shape[0]
    n_dim = x.shape[1]
    n_cv2 = 2
    out = np.zeros((n, n_cv2, n_dim))
    out[:, 0, 0] = 1.0
    out[:, 1, 1] = 1.0
    return out

cv_coord01_2d_coord_2d_cv = collective_variable(
    cv_funct = CV_coord01,
    cv_grad_funct= grad_CV_coord01,
    cv_min=np.array([-3.0, -3.0]),
    cv_max=np.array([3.0, 3.0]),
    grid_n = 81,
    n_cv_dim = 2
)

# #macrostate classifiers
# def macrostate_classifier_1dcv_0boundary(cv):
#     return np.where(cv[:,0] > 0, 1, 0)


#coord macrostate classifier
#note that this is normally applied to flattened coordinates in the observable estimator
#so it does not need another colon :
def macrostate_classifier_coord0_eq0(coords):
    return np.where(coords[:,0] > 0.0, 1, 0)