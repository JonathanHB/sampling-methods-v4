import numpy as np


#written by copilot on 7/29/26
def free_energy_on_cv_grid(
    free_energy_fn,
    coord_min,
    coord_max,
    n_micro_grid,
    cv_fn,
    cv_min,
    cv_max,
    n_cv_grid,
    kT=1.0,
):
    """
    Estimate the free energy as a function of a collective variable by integrating
    a microscopic free-energy function over the orthogonal coordinates.

    The routine samples a regular grid in CV space, and at each CV value it builds
    a set of microscopic coordinates consistent with that CV value and evaluates the
    microscopic free-energy function there. The resulting values are averaged over the
    orthogonal microscopic coordinates to approximate the marginal free energy.

    Parameters
    ----------
    free_energy_fn : callable
        A function that accepts an array of shape (n_points, n_dimensions) and
        returns the microscopic free energy for each point.
    coord_min : array-like of shape (n_dimensions,)
        Lower bounds for the microscopic coordinate space used to sample the
        orthogonal directions.
    coord_max : array-like of shape (n_dimensions,)
        Upper bounds for the microscopic coordinate space used to sample the
        orthogonal directions.
    n_micro_grid : int or sequence of ints
        Number of microscopic grid points used to probe the orthogonal directions.
        A single integer is interpreted as the same number of points for every
        microscopic dimension.
    cv_fn : callable
        A function that maps microscopic coordinates to CV values of shape
        (n_points, n_cv).
    cv_min : array-like of shape (n_cv,)
        Lower bounds of the CV grid.
    cv_max : array-like of shape (n_cv,)
        Upper bounds of the CV grid.
    n_cv_grid : int or sequence of ints
        Number of grid points along each CV dimension.
    kT : float, optional
        Thermal energy scale used for the Boltzmann weighting of the marginal
        free energy. Defaults to 1.0.

    Returns
    -------
    cv_grid : ndarray
        The sampled CV values on the grid, with shape (n_points, n_cv).
    free_energy_grid : ndarray
        The estimated marginal free energy at each sampled CV value, with shape
        (n_points,).
    """
    coord_min = np.asarray(coord_min, dtype=float)
    coord_max = np.asarray(coord_max, dtype=float)
    if coord_min.shape != coord_max.shape:
        raise ValueError("coord_min and coord_max must have the same shape")

    cv_min = np.asarray(cv_min, dtype=float)
    cv_max = np.asarray(cv_max, dtype=float)
    if cv_min.shape != cv_max.shape:
        raise ValueError("cv_min and cv_max must have the same shape")

    n_dimensions = coord_min.shape[0]
    n_cv = cv_min.shape[0]

    if np.isscalar(n_cv_grid):
        cv_grid_shape = [int(n_cv_grid)] * n_cv
    else:
        cv_grid_shape = [int(g) for g in n_cv_grid]
        if len(cv_grid_shape) != n_cv:
            raise ValueError("n_grid must match the number of CV dimensions")

    if np.isscalar(n_micro_grid):
        micro_grid_shape = [int(n_micro_grid)] * (n_dimensions - n_cv)
    else:
        micro_grid_shape = [int(g) for g in n_micro_grid]
        if len(micro_grid_shape) != n_dimensions - n_cv:
            raise ValueError("n_micro_grid must match the number of orthogonal dimensions")

    if any(n < 2 for n in cv_grid_shape):
        raise ValueError("each CV dimension needs at least 2 grid points")
    if any(n < 2 for n in micro_grid_shape):
        raise ValueError("each orthogonal dimension needs at least 2 grid points")

    cv_axes = [np.linspace(cv_min[d], cv_max[d], cv_grid_shape[d]) for d in range(n_cv)]
    cv_grid_mesh = np.meshgrid(*cv_axes, indexing="ij")
    cv_grid = np.stack(cv_grid_mesh, axis=-1).reshape(-1, n_cv)

    orthogonal_dims = n_dimensions - n_cv
    if orthogonal_dims < 0:
        raise ValueError("the number of CVs cannot exceed the number of dimensions")

    if orthogonal_dims == 0:
        coords = np.asarray(cv_grid, dtype=float)
        return cv_grid, np.asarray(free_energy_fn(coords), dtype=float).reshape(-1)

    micro_axes = [
        np.linspace(coord_min[n_cv + d], coord_max[n_cv + d], micro_grid_shape[d])
        for d in range(orthogonal_dims)
    ]
    micro_grid_mesh = np.meshgrid(*micro_axes, indexing="ij")
    micro_grid = np.stack(micro_grid_mesh, axis=-1).reshape(-1, orthogonal_dims)

    free_energy_grid = []
    for cv_value in cv_grid:
        coords = np.empty((micro_grid.shape[0], n_dimensions), dtype=float)
        coords[:, :n_cv] = cv_value
        coords[:, n_cv:] = micro_grid

        sample_cv = cv_fn(coords)
        if sample_cv.shape != (coords.shape[0], n_cv):
            raise ValueError("cv_fn must return an array with shape (n_points, n_cv)")

        values = np.asarray(free_energy_fn(coords), dtype=float).reshape(-1)
        weights = np.exp(-values / kT)
        log_partition = np.log(np.mean(weights))
        free_energy_grid.append(-kT * log_partition)

    return cv_grid, np.asarray(free_energy_grid, dtype=float)



def macrostate_delta_g(sim_system, CV, macrostate_classifier, kT):

    cv_grid, free_energy_grid = free_energy_on_cv_grid(
        free_energy_fn=sim_system.G,
        coord_min=sim_system.coord_min,
        coord_max=sim_system.coord_max,
        n_micro_grid=sim_system.grid_n,
        cv_fn=CV.cv_funct,
        cv_min=CV.cv_min,
        cv_max=CV.cv_max,
        n_cv_grid=CV.grid_n,
        kT = kT
    )

    Z = np.sum(np.exp(-free_energy_grid/kT))

    mac0_energies = free_energy_grid[np.where(macrostate_classifier(cv_grid)==0)]
    Z0 = np.sum(np.exp(-mac0_energies/kT))

    mac1_energies = free_energy_grid[np.where(macrostate_classifier(cv_grid)==1)]
    Z1 = np.sum(np.exp(-mac1_energies/kT))

    delta_G_01 = -kT*np.log(Z0/Z1)

    return delta_G_01


def macrostate_delta_g_v2(sim_system, CV, macrostate_classifier, kT):

    cv_grid, free_energy_grid = free_energy_on_cv_grid(
        free_energy_fn=sim_system.G,
        coord_min=sim_system.coord_min,
        coord_max=sim_system.coord_max,
        n_micro_grid=sim_system.grid_n,
        cv_fn=CV.cv_funct,
        cv_min=CV.cv_min,
        cv_max=CV.cv_max,
        n_cv_grid=CV.grid_n,
        kT = kT
    )

    Z = np.sum(np.exp(-free_energy_grid/kT))

    macrostate_assignments = macrostate_classifier(cv_grid)

    mac_fe = np.zeros(len(np.unique(macrostate_assignments)))

    for i, mi in enumerate(np.unique(macrostate_assignments)):

        mac_energies = free_energy_grid[np.where(macrostate_assignments==i)]
        Gi = -kT*np.log(np.sum(np.exp(-mac_energies/kT)))
        mac_fe[mi] = Gi

    return mac_fe