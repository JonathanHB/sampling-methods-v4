import matplotlib.pyplot as plt
import numpy as np

#TODO: fill out an entire project worth of function specs and then give them to Claude and see if it can fill them in


def multiplot_observable_convergence(conditions_replicate_time, condition_names, timepoints, time_axis_label, true_value):
    """plot observables over time across multiple replicates and conditions
    
    Parameters
    ----------
    conditions_replicate_time: numpy array of shape (n_conditions, n_replicates, n_timepoints)
        The data to be plotted. Each entry in the array is an observable value. 
        If there is no data the value should be nan.
    
    condition_names: tuple of strings, of length n_conditions
        what to label each subplot

    timepoints: numpy array of times, of length n_timepoints
        The x axis data for each plot
    
    time_axis_label: string
        Time axis label. Used to provide units and distinguish aggregate and molecular time.

    true_values: float
        the true value of the observable

    Returns
    -------
    None
        The purpose of the function is to save a plot
    
    """
    pass


def run_macrostate_dg_molecular_time(simulator_objects, n_replicates, max_we_rounds, macrostate_classifier):
    """plot macrostate free energy estimates over molecular time across multiple replicates and conditions
    
    Parameters
    ----------
    simulator_objects: tuple of simulator objects
        of length n_conditions
    
    condition_names: tuple of strings
        what to label each subplot, of length n_conditions

    n_replicates: int
        how many simulation replicates to run

    macrostate_classifier: numpy array of floats of shape (n_datapoints, n_dimensions) : numpy array of floats of shape (n_datapoints)
        returns 1 for each datapoint which is in the macrostate and 0 for the rest

    Returns
    -------
    conditions_replicate_time: 3d numpy array of floats
        of shape (n_conditions, n_replicates, n_timepoints)
        The data to be plotted. Each entry in the array is an observable value. 
        If there is no data the value should be nan.
    
    timepoints: 1d numpy array of ints
        of length n_timepoints
        The x axis data for each plot

    time_axis_label: string
        Time axis label. Used to provide units and distinguish aggregate and molecular time.

    """

    conditions_replicate_time = np.nan*np.ones([len(simulator_objects), n_replicates, max_we_rounds])

    for si, s in enumerate(simulator_objects):
        for ri in range(n_replicates):
            trj, potentials = s.run(max_we_rounds)
            fe_by_round = importance_sampling_fe_by_we_round(trj, potentials, macrostate_classifier)
            conditions_replicate_time[si,ri] = fe_by_round


    return conditions_replicate_time, max_we_rounds, "WE rounds"


def importance_sampling_fe_by_we_round(trj, discrete_trj, we_weights, potentials, metadata, macrostate_classifier, kB, T, delta_T):
    """
    Calculate the importance sampling estimate of a macrostate free energy difference 
    via the importance sampling estimate of equilibrium populations.

    Parameters
    ----------
    trj: 4d numpy array of floats
        of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, n_coordinates)
        Each element is a coordinate. NaN values denote walkers which did not exist 
        (i.e. not all bins were filled at the start of the round so less than the maximum number of walkers were spawned)

    discrete_trj: 4d numpy array of ints
        of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, n_coordinates)
        Each element is an MTD grid index along one axis. NaN values are distributed as above.
        This is the discrete version of trj

    we_weights: 2d numpy array of floats
        of shape (n_we_rounds, max_n_walkers)
        Each element is the WE weight of a walker. NaNs as above

    potentials: (3+n_CV_dimensions)d numpy array of floats
        of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, [MTD grid dimensions])
        Each element is an MTD potential value at a specific walker, time, and place
        NaN values are distributed as above.
        Because the MTD potential is not usually updated for every saved frame, this contains some redundant information

    metadata: 2d numpy array of ints
        of shape (n_we_rounds, max_n_walkers)
        Each element is 1 if the walker was spawned, 0 otherwise. This should match the NaNs in the above two parameters.
    
    macrostate_classifier: numpy array of floats of shape (n_datapoints, n_dimensions) : numpy array of floats of shape (n_datapoints)
        returns 1 for each datapoint which is in the macrostate and 0 for the rest

    kB: float
        Boltzmann's constant 
        
    T: float
        Temperature
    
    delta_T: float
        The well-tempered metadynamics temperature factor

    Returns
    -------
    delta_G: 1d numpy array of floats
        of shape (n_we_rounds)
        The estimated free energy difference between the macrostates at the end of each WE round.
        Each element is a free energy in units of kT

    """

    delta_G = np.zeros(len(trj))

    trj_flattened_by_we_round = [[] for _ in range(len(trj))]
    importance_weights_flattened_by_we_round = [[] for _ in range(len(trj))]

    #loop over WE rounds
    for we_i, (trj_r, disc_trj_r, we_weights_r, potentials_r, metadata_r) in enumerate(trj, discrete_trj, we_weights, potentials, metadata):

        #loop over walkers
        for trj_rw, disc_trj_rw, we_weight, potentials_rw, metadata_rw in enumerate(trj_r, disc_trj_r, we_weights_r, potentials_r, metadata_r):

            #if the walker exists
            if metadata_rw: 

                trj_flattened_by_we_round[we_i].append(trj_rw)

                potential_along_trj = potentials_rw[disc_trj_rw]
                exp_factor = np.exp(potential_along_trj/(kB*T))

                Z0 = np.sum(np.exp(potentials_rw*(1/T + 1/delta_T)/kB), axis=tuple(range(1, potentials_rw.ndim)))
                Z1 = np.sum(np.exp(potentials_rw*(1/delta_T)/kB), axis=tuple(range(1, potentials_rw.ndim)))
                partition_ratio = np.divide(Z1,Z0)

                mtd_weights_rw = np.multiply(exp_factor, partition_ratio)
                importance_weights_rw = we_weight*mtd_weights_rw

                importance_weights_flattened_by_we_round[we_i].append(importance_weights_rw)

        #combine the data from all walkers from each round
        trj_flattened_by_we_round[we_i] = np.concatenate(trj_flattened_by_we_round[we_i])
        importance_weights_flattened_by_we_round[we_i] = np.concatenate(importance_weights_flattened_by_we_round[we_i])

        #calculate the cumulative deltaG up to this point
        coords_cumulative = np.concatenate(trj_flattened_by_we_round[:we_i])
        importance_weights_cumulative = np.concatenate(importance_weights_flattened_by_we_round[:we_i])
        pop_state_A = importance_sampling_estimator(coords_cumulative, importance_weights_cumulative, macrostate_classifier)

        delta_G[we_i] = -np.log((1-pop_state_A)/pop_state_A)

        # #trim out NANs from nonexistent walkers < maybe useful for parallelization later
        # trj_r = trj_r[metadata_r==1]
        # disc_trj_r = disc_trj_r[metadata_r==1]
        # potentials_r = potentials_r[metadata_r==1]
        # we_weights_r = we_weights_r[metadata_r==1]

    return delta_G


def importance_sampling_estimator(coords, importance_weights, observable):
    """
    Calculate the importance sampling estimate of an observable given the importance weights and the observable function.
    This is basically just a weighted average.

    Parameters
    ----------
    coordinates: 2d numpy array of floats 
        of shape (n_datapoints, n_dimensions)
        the microscopic coordinates of each frame
    
    importance_weights: 1d numpy array of floats 
        of shape (n_datapoints)
        the importance weight of each frame
    
    observable: numpy array of floats of shape (n_datapoints, n_dimensions) : numpy array of floats of shape (n_datapoints)
        Function that converts microscopic coordinates to scalar observables of the same shape as importance_weights
        This operates on the whole array at once so that vectorizable observables can be vectorized.
        Because vectorizability depends on the observable, this has to be done inside the observable method.
        TODO: Vector/tensor observables may be supported in the future; there's no fundamental reason not to; I just have no need for it at the moment.

    Returns
    -------
    observable_estimate: float
        The importance sampling estimate of the observable. See above about vector observables.
    """

    observable_values = observable(coords)

    observable_estimate = np.multiply(observable_values, importance_weights)/np.sum(importance_weights)

    return observable_estimate