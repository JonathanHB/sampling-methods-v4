import numpy as np
from typing import Callable


class energy_landscape():
    
    def __init__(self, 
                 G: Callable[[np.ndarray], np.ndarray], 
                 coord_min: np.array, 
                 coord_max: np.array, 
                 grid_n: np.array, 
                 n_dim: int,
                 xi: float
                 ):
        
        self.G = G
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.grid_n = grid_n
        self.n_dim = n_dim
        self.xi = xi


def G_2w_2d_diag(x):
    return 0.5*(x[:, 0]**2 + x[:, 1]**2)**2 + (x[:,0]*x[:,1])*10

diagonal_2well_2d_system = energy_landscape(
    G = G_2w_2d_diag,
    coord_min = np.array([-3,-3]),
    coord_max = np.array([3,3]),
    grid_n = 81,
    n_dim = 2,
    xi = 1

)

