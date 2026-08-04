import itertools
import numpy as np
from typing import Callable

#TODO: calculate MTD importance weights in here directly
#rather than carting around a giant variable-dimensional array of potentials

#written by claude sonnet 5 at medium effort on 7/29/26
#a version of propagators_function_spec.py was uploaded with the prompt

def propagate(G: Callable[[np.ndarray], np.ndarray],
              kB: float,
              T: float,
              dt: float,
              xi: int,
              init_coords: np.ndarray,
              init_potentials: np.ndarray,
              steps_per_saved_frame: int,
              n_gaussians: int,
              frames_per_gaussian: int,
              CV: Callable[[np.ndarray], np.ndarray],
              grad_CV: Callable[[np.ndarray], np.ndarray],
              sigma: np.ndarray,
              omega: float,
              delta_T: float,
              cv_min: np.ndarray,
              cv_max: np.ndarray,
              ):
    """
    Run a collection of independent parallel metadynamics simulations, each with its own
    metadynamics potential, represented on a grid over CV space (one grid per walker).

    This is a grid-based counterpart to the list-of-gaussians implementation: instead of
    keeping a growing list of every deposited gaussian and summing over all of them at
    every integration step (cost grows with n_gaussians), the bias and its gradient are
    stored on a fixed-size grid per walker. Depositing a gaussian means adding it onto the
    grid once (O(grid size), independent of how many gaussians have been deposited so far);
    evaluating the bias/force at a walker's continuous position is a constant-cost
    multilinear interpolation off that grid. So cost per MD step no longer grows with
    n_gaussians -- only cost per deposition does, and that happens far less often than
    integration steps.

    There are n_parallel_simulations simulations.
    The microscopic coordinate has n_dimensions dimensions.
    The collective variable has n_cv dimensions.

    Parameters
    ----------
    G: 2d numpy array : 1d numpy array
        The first dimension of each array is the simulation index.
        The former array is of shape (n_parallel_simulations, n_dimensions), the latter is of length n_parallel_simulations

    kB: float
        Boltzmann's constant

    T: float
        Temperature

    dt: float
        The simulation timestep

    xi: float
        The friction coefficient. This merely rescales the timestep

    init_coords: 2d numpy array of floats
        of shape (n_parallel_simulations, n_dimensions).
        The first axis is the simulation, the second is the dimension.
        The initial coordinates of each simulation.

    init_potentials: (1+n_cv)d numpy array of floats
        of shape (n_parallel_simulations, N_1, N_2, ..., N_n_cv).
        The initial metadynamics bias, ON A GRID, for each walker: init_potentials[w]
        is the value of walker w's bias potential at every point of an n_cv-dimensional
        grid spanning [cv_min, cv_max] (grid resolution N_1 x ... x N_n_cv is inferred
        from this array's shape). Pass zeros for a fresh start, or a previously-saved
        grid to continue a run.

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

    cv_min: 1d numpy array of floats
        of length n_cv. The lower bound of the CV-space grid along each CV dimension.
        (Not part of the original spec -- required so the grid has a physical extent,
        not just a resolution.)

    cv_max: 1d numpy array of floats
        of length n_cv. The upper bound of the CV-space grid along each CV dimension.


    Returns
    -------
    trajectories: 3d numpy array of floats
        of shape (n_parallel_simulations, n_gaussians*n_steps_per_gaussian, n_dimensions).
        The simulation trajectories
        
    potentials: (2+n_cv)d numpy array of floats
        of shape (n_parallel_simulations, n_gaussians, N_1, ..., N_n_cv).
        The first two dimensions are the parallel walkers and the time (in increments of 1 gaussian deposition) respectively;
        the remaining n_cv dimensions are the bias grid. Records a snapshot of each walker's
        full bias grid after each gaussian deposition.
    
    mtd_weights: 2d numpy array of floats
        of shape (n_parallel_simulations, n_gaussians*n_steps_per_gaussian).
        The MTD importance weights for each walker at each saved frame.
    
    """

    init_coords = np.asarray(init_coords, dtype=float)
    init_potentials = np.asarray(init_potentials, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cv_min = np.asarray(cv_min, dtype=float)
    cv_max = np.asarray(cv_max, dtype=float)

    n_walkers, n_dim = init_coords.shape
    n_cv = sigma.shape[0]
    grid_shape = init_potentials.shape[1:]

    if len(grid_shape) != n_cv:
        raise ValueError(
            f"init_potentials must have {n_cv} grid axes after the walker axis "
            f"(got shape {init_potentials.shape})"
        )
    if any(n < 2 for n in grid_shape):
        raise ValueError("every grid dimension needs at least 2 points")

    # ------------------------------------------------------------------
    # Grid geometry: axes, per-dimension spacing, and the (n_cv+1)-d
    # array of grid-point coordinates used when depositing a gaussian.
    # ------------------------------------------------------------------
    axes = [np.linspace(cv_min[d], cv_max[d], grid_shape[d]) for d in range(n_cv)]
    spacings = [(cv_max[d] - cv_min[d]) / (grid_shape[d] - 1) for d in range(n_cv)]
    grid_coords = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)  # (*grid_shape, n_cv)

    total_frames = n_gaussians * frames_per_gaussian
    trajectories = np.empty((n_walkers, total_frames, n_dim), dtype=float)
    potentials = np.empty((n_walkers, n_gaussians, *grid_shape), dtype=float)
    mtd_weights = np.empty((n_walkers, total_frames), dtype=float)

    V_grid = init_potentials.copy()  # (n_walkers, *grid_shape)

    coords = init_coords.copy()
    rng = np.random.default_rng()
    dt_over_xi = dt / xi
    sqrt_2kT_dt_over_xi = np.sqrt(2.0 * kB * T * dt / xi)

    fd_eps = 1e-6

    def grad_G(x: np.ndarray) -> np.ndarray:
        grad = np.empty_like(x)
        for d in range(n_dim):
            step = np.zeros_like(x)
            step[:, d] = fd_eps
            grad[:, d] = (G(x + step) - G(x - step)) / (2.0 * fd_eps)
        return grad

    def compute_fields(v_grid: np.ndarray) -> np.ndarray:
        """
        Stack V and its n_cv partial derivatives (via np.gradient on the grid)
        into a single array of shape (n_walkers, *grid_shape, 1+n_cv), so both
        can be multilinearly interpolated together in one pass.
        Recomputed once per deposition, not once per MD step.
        """
        if n_cv == 1:
            grads = [np.gradient(v_grid, spacings[0], axis=1)]
        else:
            grads = np.gradient(v_grid, *spacings, axis=tuple(range(1, 1 + n_cv)))
        return np.stack([v_grid] + list(grads), axis=-1)

    def interp(fields: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Vectorized multilinear interpolation of `fields` (n_walkers, *grid_shape, K)
        at continuous CV positions `s` (n_walkers, n_cv). Returns (n_walkers, K).
        Loops only over the 2**n_cv hypercube corners (small for typical n_cv).
        """
        s_clamped = np.clip(s, cv_min, cv_max)
        grid_dims = np.array(grid_shape, dtype=float)
        idx_f = (s_clamped - cv_min) / (cv_max - cv_min) * (grid_dims - 1)
        idx0 = np.floor(idx_f).astype(int)
        idx0 = np.clip(idx0, 0, np.array(grid_shape) - 2)
        frac = idx_f - idx0  # (n_walkers, n_cv)

        walker_idx = np.arange(s.shape[0])
        result = np.zeros((s.shape[0], fields.shape[-1]))

        for corner in itertools.product((0, 1), repeat=n_cv):
            weight = np.ones(s.shape[0])
            index_tuple = [walker_idx]
            for d, c in enumerate(corner):
                weight = weight * (frac[:, d] if c else (1.0 - frac[:, d]))
                index_tuple.append(idx0[:, d] + c)
            result += weight[:, None] * fields[tuple(index_tuple)]

        return result

    def bias_value_and_grad(x: np.ndarray, fields: np.ndarray):
        s = CV(x)             # (n_walkers, n_cv)
        dsdx = grad_CV(x)     # (n_walkers, n_cv, n_dim)

        interpolated = interp(fields, s)   # (n_walkers, 1+n_cv)
        V = interpolated[:, 0]
        dVds = interpolated[:, 1:]         # (n_walkers, n_cv)

        dVdx = np.einsum('wc,wcd->wd', dVds, dsdx) #TODO verify that this implements the correct matrix multiplication [copilot: for the gradient chain rule. It should be (n_walkers, n_dimensions) in the end.]
        return V, dVdx, s

    def deposit(v_grid: np.ndarray, centers: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """
        Add one gaussian per walker directly onto the grid: O(grid size), regardless
        of how many gaussians have been deposited previously.
        """
        # diff: (n_walkers, *grid_shape, n_cv)
        diff = grid_coords[None, ...] - centers.reshape(centers.shape[0], *([1] * n_cv), n_cv)
        exponent = -0.5 * np.sum((diff / sigma) ** 2, axis=-1)  # (n_walkers, *grid_shape)
        contribution = heights.reshape(heights.shape[0], *([1] * n_cv)) * np.exp(exponent)
        return v_grid + contribution

    frame_idx = 0
    fields = compute_fields(V_grid)  # valid until the first deposition

    for g in range(n_gaussians):
        for f in range(frames_per_gaussian):
            for _ in range(steps_per_saved_frame):
                _, dVdx, _ = bias_value_and_grad(coords, fields)
                dGdx = grad_G(coords)
                force = -(dGdx + dVdx)

                noise = rng.standard_normal(size=coords.shape)
                coords = coords + force * dt_over_xi + sqrt_2kT_dt_over_xi * noise

            trajectories[:, frame_idx, :] = coords

            #|-----> added by JHB on 8/3/26 to compute MTD importance weights for each walker at each saved frame
            V_current, _, _ = bias_value_and_grad(coords, fields) #not terribly efficient to recompute this, but it's only done once per saved frame, not every MD step

            Z0 = np.sum(np.exp(V_grid*(1/T + 1/delta_T)/kB), axis=tuple(range(1, V_grid.ndim)))
            Z1 = np.sum(np.exp(V_grid*(1/delta_T)/kB), axis=tuple(range(1, V_grid.ndim)))
            partition_ratio = np.divide(Z1,Z0)

            exp_factor = np.exp(V_current/(kB*T))

            mtd_weights[:, frame_idx] = np.multiply(exp_factor, partition_ratio)
            #<-----| end addition

            frame_idx += 1

        # deposit a new, well-tempered gaussian at the current position
        V_current, _, s_current = bias_value_and_grad(coords, fields)
        new_heights = omega * np.exp(-V_current / (kB * delta_T)) #originally had an extra factor of T via kT

        V_grid = deposit(V_grid, s_current, new_heights)
        fields = compute_fields(V_grid)  # refresh once per deposition, reused until the next one

        potentials[:, g, ...] = V_grid

    return trajectories, potentials, mtd_weights
