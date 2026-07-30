import numpy as np
from typing import Callable


def propagate(G: Callable[[np.ndarray], np.ndarray], 
              kT: float, 
              dt: float,
              init_coords: np.ndarray, 
              init_potentials: np.ndarray,
              steps_per_saved_frame: int,
              n_gaussians: int, 
              frames_per_gaussian: int, 
              CV: Callable[[np.ndarray], np.ndarray],
              grad_CV: Callable[[np.ndarray], np.ndarray],
              sigma: np.ndarray, 
              omega: float, 
              delta_T: float
              ):

    """
    Run a collection of independent parallel metadynamics simulations, each with its own metadynamics potential.
    There are n_parallel_simulations simulations.
    The microscopic coordinate has n_dimensions dimensions.
    The collective variable has n_cv dimensions.
    
    Parameters
    ----------
    G: 2d numpy array : 1d numpy array
        The first dimension of each array is the simulation index. 
        The former array is of shape (n_parallel_simulations, n_dimensions), the latter is of length n_parallel_simulations

    kT: float
        Boltzmann's constant times the temperature

    dt: float
        The simulation timestep

    init_coords: 2d numpy array of floats
        of shape (n_parallel_simulations, n_dimensions). 
        The first axis is the simulation, the second is the dimension. 
        The initial coordinates of each simulation. 

    init_potential_grids: (1+n_cv)d numpy array of floats
        of length n_parallel_walkers along the first dimension. 
        The initial metadynamics potential of each walker.

    steps_per_saved_frame: int
        How many simulation steps to integrate betwen saved frames

    n_gaussians: int
        How many gaussians to deposit over the course of each simulation.

    frames_per_gaussian: int
        How many frames to save between gaussian depositions.

    CV: 2d numpy array : 2d numpy array
        Calculate the CV of each parallel simulation. 
        The first dimension of each array is the simulation index. 
        The former array is of shape (n_parallel_simulations, n_dimensions), the latter is of shape (n_parallel_simulations, n_cv).

    grad_CV: 2d numpy array : 3d numpy array
        Force = -d(G+V)/dx = - dG/dx - dV/dx = - dG/dx - dV/ds*ds/dx. 
        dV/ds is a vector of length n_cv. 
        ds/dx must be a matrix of shape (n_cv, n_dimensions) (assuming left multiplication by a row vector n_cv). 
        Thus the former array is of shape (n_parallel_simulations, n_dimensions), and the latter is of shape (n_parallel_simulations, n_cv, n_dimensions).

    sigma: 1d numpy array of floats
        of length n_cv. 
        The gaussian width in each dimension. 
        The off-diagonal components of a hypothetical covariance matrix describing the gaussian are assumed to be zero. 
        
    omega: float
        Gaussian height prefactor in units of kT. 
        The gaussian is normalized such that its integral in CV space is equal to omega*e**-(V/k*delta_T).

    delta_T: float
        The gaussian height decay constant from well-tempered metadynamics.

        
    Returns
    -------
    trajectories: 3d numpy array of floats
        of shape (n_parallel_simulations, n_gaussians*n_steps_per_gaussian, n_dimensions). 
        The simulation trajectories
        
    potential_grids: (2+n_cv)d numpy array of floats
        The first two dimensions are the parallel walkers and the gaussian index respectively
        Record the metadynamics potential of each walkers after each gaussian deposition

    
    """



    return trajectories, potentials