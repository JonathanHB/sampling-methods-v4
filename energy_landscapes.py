#energy_landscapes.py
#Jonathan Borowsky
#2/21/25

#classes representing different systems to be simulated
#these provide energy landscapes, forces, diffusion coefficients,
# kinetically appropriate macrostate definitions,
# and convenient fixed equal width bins and starting coordinates

################################################################################################################

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency in lightweight envs
    plt = None


def free_energy_on_cv_grid(
    free_energy_fn,
    cv_fn,
    coord_min,
    coord_max,
    cv_min,
    cv_max,
    n_grid,
    n_micro_grid,
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
    cv_fn : callable
        A function that maps microscopic coordinates to CV values of shape
        (n_points, n_cv).
    coord_min : array-like of shape (n_dimensions,)
        Lower bounds for the microscopic coordinate space used to sample the
        orthogonal directions.
    coord_max : array-like of shape (n_dimensions,)
        Upper bounds for the microscopic coordinate space used to sample the
        orthogonal directions.
    cv_min : array-like of shape (n_cv,)
        Lower bounds of the CV grid.
    cv_max : array-like of shape (n_cv,)
        Upper bounds of the CV grid.
    n_grid : int or sequence of ints
        Number of grid points along each CV dimension.
    n_micro_grid : int or sequence of ints
        Number of microscopic grid points used to probe the orthogonal directions.
        A single integer is interpreted as the same number of points for every
        microscopic dimension.
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

    if np.isscalar(n_grid):
        cv_grid_shape = [int(n_grid)] * n_cv
    else:
        cv_grid_shape = [int(g) for g in n_grid]
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


#superclass of 1d potential functions
#this contains all the functions which are useful for a system object to have 
# and for which the structure of the algorithm does not depend on the potential or other details of the system
class potential_well_1d():

    def __init__(self, potential, macro_class, standard_analysis_range):
        self.potentiall = potential
        self.macro_classs = macro_class #maybe not the best variable name
        self.standard_analysis_rangee = standard_analysis_range #is this actually necessary??

    #determine which ensemble a trajectory currently ensemble e should be in upon moving to coordinate x
    def ensemble_class(self, x, e):  
        ms = self.macro_classs(x)
        if ms != -1:
            return ms
        else:
            return e
        
    def macro_class_parallel(self, x):
        return [self.macro_classs(xi) for xi in x]
    
    #calculate equilibrium populations and energies for a given set of bins
    # by assuming energy is roughly constant across each bin
    # This is a good approximation for most systems; 
    # if your bins are too large for it to hold it's usually a good idea to make them smaller 
    # instead of integrating across them with the method below
    def normalized_pops_energies(self, kT, bincenters):
        #assume equal bin widths
        binwidth = bincenters[1]-bincenters[0]

        pops_nonnorm = [np.exp(-self.potentiall(x)/kT) for x in bincenters]
        z = sum(pops_nonnorm)
        pops_norm = [p/z for p in pops_nonnorm]
    
        #energies_norm = [-kT*np.log(p/(z*binwidth)) for p in pops_nonnorm]
        energies_norm = -kT*np.log(pops_norm)

        return pops_norm, energies_norm

    #compute equilibrium populations of the given bins by integrating across them
    # this should be more accurate than just using the center point of the bin
    # the increase in accuracy provided by this method seems to be entirely unnecessary in practice
    #tolerance is the permitted energy difference between the edges of a sub-bin in kT
    #bin_boundaries are assumed to increase monotonically
    def compute_true_populations(self, bin_boundaries, kT, tolerance = 0.01):

        bin_centers = []
        bin_populations = []

        for i in range(len(bin_boundaries)-1):
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i+1])/2)

            #figure out how many sub-bins the bin must be divided into for the potential across each bin to be roughtly constant
            #This method assumes negligible curvature and will fail for bins with equal edge energies which curve up or down in between
            #a more general approach would be to randomly sample points in each bin and then average or sum somehow
            energy_gap = abs(self.potentiall(bin_boundaries[i+1])-self.potentiall(bin_boundaries[i]))/kT
            n_subbins = max(int(np.ceil(energy_gap/tolerance)), 1)
            
            bin_pop = 0
            subbin_width = (bin_boundaries[i+1] - bin_boundaries[i])/n_subbins
            
            for sbx in np.linspace(bin_boundaries[i]+subbin_width/2, bin_boundaries[i+1]-subbin_width/2, n_subbins):
                bin_pop += np.exp(-self.potentiall(sbx)/kT)*subbin_width
                
            bin_populations.append(bin_pop)

        z = sum(bin_populations)
        bin_populations = [bp/z for bp in bin_populations]
        
        return bin_centers, bin_populations
    
    #for visualization to check that you've written the potential right
    def plot_quantity(self, quantity): 
        x = np.linspace(self.standard_analysis_rangee[0], self.standard_analysis_rangee[1], 100)
        plt.plot(x, [quantity(i) for i in x])

    #return bins for analysis of each energy landscape, 
    # including end bins for anything outside the standard bin range
    def analysis_bins_1d(self, nbins):
        
        step = (self.standard_analysis_rangee[1][0]-self.standard_analysis_rangee[0][0])/nbins
    
        binbounds = np.linspace(self.standard_analysis_rangee[0][0], self.standard_analysis_rangee[1][0], nbins+1)
        bincenters = np.linspace(self.standard_analysis_rangee[0][0]-step/2, self.standard_analysis_rangee[1][0]+step/2, nbins+2)

        return binbounds, bincenters, step
    
    def bin_trj_nd(self, nbins, trj):
        
        steps = [(self.standard_analysis_rangee[1][i]-self.standard_analysis_rangee[0][i])/nbins for i in range(len(self.standard_analysis_rangee[1]))]


#a double well constructed using a quartic and quadratic potential
class unit_double_well(potential_well_1d):
    #MFPT(10 frame save frequency) = ~800 steps

    def potential(self, x):
        return x**4 - x**2
        
    def F(self, x):
        return -4*x**3 + 2*x
    
    def macro_class(self, x):
        thr = 0.7 #1/np.sqrt(2)
        if x[0] <= -thr:
            return 0
        elif x[0] >= thr:
            return 1
        else:
            return -1

    def macrostate_classifier(self, coords):
        """
        Give each datapoint a binary macrostate assignment. 

        Parameters
        ----------
        coordinates: 2d numpy array of floats 
            of shape (n_datapoints, n_dimensions)
            the microscopic coordinates of each frame

        Returns
        -------
        macrostate: int
            1 for points with a negative first coordinate, 0 otherwise.

        """

        macrostates = np.where(coords[:,0]<0, 1, 0)

        return macrostates

            
    def __init__(self):
        self.true_macrostate_dg = 0
        self.diffusion_coefficient = 1
        self.n_macrostates = 2
        self.standard_init_coord = [-1/np.sqrt(2)]
        self.standard_analysis_range = [[-2],[2]]
        self.start_from_index = False
        super().__init__(self.potential, self.macro_class, self.standard_analysis_range)


#a system of several wells of similar energies constructed using a sinusoidal and a quartic potential
class unit_sine_well(potential_well_1d):
    #MFPT(10 frame save frequency) = ~70000 steps

    def potential(self, x):
        return 0.0001*x**4 + np.cos(x)
        
    def F(self, x):
        return 0.0001*-4*x**3 + np.sin(x)
    
    def macro_class(self, x):
        thr = 2*np.pi
        if x[0] < -thr:
            return 0
        elif x[0] > thr:
            return 1
        else:
            return -1
        
    def __init__(self):
        self.diffusion_coefficient = 10 #fmrly 1
        self.n_macrostates = 2
        self.standard_init_coord = [-3*np.pi]
        self.standard_analysis_range = [[-20],[20]]
        self.start_from_index = False
        super().__init__(self.potential, self.macro_class, self.standard_analysis_range)




