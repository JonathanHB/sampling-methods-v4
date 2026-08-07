import numpy as np
import matplotlib.pyplot as plt
import collective_variable_analysis


#TODO write a variant of this that works by actual time stamp instead of WE round
#since WE round spacing may vary
def importance_sampling_fe_by_we_round(trj, mtd_weights, we_weights, macrostate_classifier, CV, sim_system, kT):
    """
    Calculate the importance sampling estimate of a macrostate free energy difference 
    via the importance sampling estimate of equilibrium populations.

    Parameters
    ----------
    trj: list of 3d numpy arrays of floats
        The list is of length n_we_rounds
        Each entry is an array of shape (n_walkers, n_frames_per_we_round, n_dimensions)
        Each element is a coordinate. 

    mtd_weights: list of 2d numpy arrays of floats
        Each array is of shape (n_walkers, n_frames_per_we_round)
        The MTD importance weights for each walker at each saved frame.

    we_weights: list of 1d numpy arrays of floats
        Each array is of shape (n_walkers)
        Each element is the WE weight of a walker
 
    macrostate_classifier: numpy array of floats of shape (n_datapoints, n_dimensions) : numpy array of floats of shape (n_datapoints)
        returns 1 for each datapoint which is in the macrostate and 0 for the rest

    # kB: float
    #     Boltzmann's constant 
    # T: float
    #     Temperature
    # delta_T: float
    #     The well-tempered metadynamics temperature factor

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

    #beware that the variable naming convention here is the opposite of that used in multiplot_observable_convergence()
    # i.e. sub-arrays here have more suffix letters and sub-arrays in multiplot_observable_convergence() have fewer

    #loop over WE rounds
    for we_i, (trj_r, mtd_weights_r, we_weights_r) in enumerate(zip(trj, mtd_weights, we_weights)):

        #flatten weights and add to list
        trj_flattened_by_we_round[we_i] = trj_r.reshape(-1, trj_r.shape[-1])
        importance_weights_flattened_by_we_round[we_i] = np.multiply(mtd_weights_r, we_weights_r[:, np.newaxis]).flatten()

        #calculate the cumulative deltaG up to this point
        coords_cumulative = np.concatenate(trj_flattened_by_we_round[:we_i+1])
        importance_weights_cumulative = np.concatenate(importance_weights_flattened_by_we_round[:we_i+1])

        pop_state_A = importance_sampling_estimator(coords_cumulative, importance_weights_cumulative, macrostate_classifier)

        delta_G[we_i] = -np.log((1-pop_state_A)/pop_state_A)


    plot_landscape(sim_system, CV, kT, trj_flattened_by_we_round, importance_weights_flattened_by_we_round)


    return delta_G



def importance_sampling_fe_by_molecular_timepoint(trj, mtd_weights, we_weights, macrostate_classifier, CV, sim_system, kT, n_timepoints, max_n_frames):
    """
    Calculate the importance sampling estimate of a macrostate free energy difference 
    via the importance sampling estimate of equilibrium populations.

    Parameters
    ----------
    trj: list of 3d numpy arrays of floats
        The list is of length n_we_rounds
        Each entry is an array of shape (n_walkers, n_frames_per_we_round, n_dimensions)
        Each element is a coordinate. 

    mtd_weights: list of 2d numpy arrays of floats
        Each array is of shape (n_walkers, n_frames_per_we_round)
        The MTD importance weights for each walker at each saved frame.

    we_weights: list of 1d numpy arrays of floats
        Each array is of shape (n_walkers)
        Each element is the WE weight of a walker
 
    macrostate_classifier: numpy array of floats of shape (n_datapoints, n_dimensions) : numpy array of floats of shape (n_datapoints)
        returns 1 for each datapoint which is in the macrostate and 0 for the rest

    # kB: float
    #     Boltzmann's constant 
    # T: float
    #     Temperature
    # delta_T: float
    #     The well-tempered metadynamics temperature factor

    Returns
    -------
    delta_G: 1d numpy array of floats
        of shape (n_we_rounds)
        The estimated free energy difference between the macrostates at the end of each WE round.
        Each element is a free energy in units of kT

    """

    n_frames = sum([tr.shape[1] for tr in trj])
    #print(f"molecular time = {n_frames} frames")
    aggregate_frames = sum([tr.shape[0]*tr.shape[1] for tr in trj])
    #print(f"aggregate time = {aggregate_frames} frames")

    delta_G = np.zeros(n_timepoints)

    trj_flattened_by_timepoint = [[] for _ in range(n_timepoints)]
    importance_weights_flattened_by_timepoint = [[] for _ in range(n_timepoints)]

    #beware that the variable naming convention here is the opposite of that used in multiplot_observable_convergence()
    # i.e. sub-arrays here have more suffix letters and sub-arrays in multiplot_observable_convergence() have fewer

    min_frame = 0

    #TODO: track wall clock time as you go and cut off estimation once you hit the limit

    for tp in range(n_timepoints):

        #print(f"timepoint {tp}")
        max_frame = int(round(max_n_frames*(tp+1)/n_timepoints))

        if max_frame>n_frames:
            delta_G[tp] = -np.inf
            continue

        curr_frame = 0
        #loop over WE rounds
        for we_i, (trj_r, mtd_weights_r, we_weights_r) in enumerate(zip(trj, mtd_weights, we_weights)):
            #print(f"we round {we_i}")

            if min_frame < curr_frame+trj_r.shape[1] and max_frame >= curr_frame:
                #if the min_frame is in the current WE round, 
                # take either all frames for the current timepoint range [min_frame, max_frame] 
                # or all frames from min_frame to the end of the round
                # depending on whether max_frame is also in the current WE round
                weround_min_frame = max(min_frame-curr_frame, 0)
                weround_max_frame = min(max_frame-curr_frame, trj_r.shape[1])
                #print(f"RELATIVE FRAME RANGE {weround_min_frame}-{weround_max_frame}")

                trj_flattened_by_timepoint[tp].append(trj_r[:,weround_min_frame:weround_max_frame].reshape(-1, trj_r.shape[-1]))
                importance_weights_flattened_by_timepoint[tp].append(np.multiply(mtd_weights_r[:,weround_min_frame:weround_max_frame], we_weights_r[:, np.newaxis]).flatten())

            curr_frame += trj_r.shape[1]

            if curr_frame > max_frame:
                break

        min_frame = max_frame

        #combine the data from all walkers from each timepoint
        trj_flattened_by_timepoint[tp] = np.concatenate(trj_flattened_by_timepoint[tp])
        importance_weights_flattened_by_timepoint[tp] = np.concatenate(importance_weights_flattened_by_timepoint[tp])

        #calculate the cumulative deltaG up to this point
        coords_cumulative = np.concatenate(trj_flattened_by_timepoint[:tp+1])
        importance_weights_cumulative = np.concatenate(importance_weights_flattened_by_timepoint[:tp+1])

        pop_state_A = importance_sampling_estimator(coords_cumulative, importance_weights_cumulative, macrostate_classifier)

        delta_G[tp] = -np.log((1-pop_state_A)/pop_state_A)


    # print("-----------------------------------")
    i_max = 0
    for i, tfbt in enumerate(trj_flattened_by_timepoint):
        # print(tfbt)
        # print(tfbt.shape)
        i_max=i
        if len(tfbt) == 0:
            break
    # print("-----------------------------------")

    # print(i_max)

    #calculate the cumulative deltaG for all times
    coords_cumulative = np.concatenate(trj_flattened_by_timepoint[:i_max])
    importance_weights_cumulative = np.concatenate(importance_weights_flattened_by_timepoint[:i_max])

    print(f"used {len(importance_weights_cumulative)} datapoints")

    #plot_landscape(sim_system, CV, kT, coords_cumulative, importance_weights_cumulative)


    return delta_G



def importance_sampling_fe_by_aggregate_timepoint(trj, mtd_weights, we_weights, macrostate_classifier, CV, sim_system, kT, n_timepoints, aggregate_time_increment):
    """
    Calculate the importance sampling estimate of a macrostate free energy difference 
    via the importance sampling estimate of equilibrium populations.

    Parameters
    ----------
    trj: list of 3d numpy arrays of floats
        The list is of length n_we_rounds
        Each entry is an array of shape (n_walkers, n_frames_per_we_round, n_dimensions)
        Each element is a coordinate. 

    mtd_weights: list of 2d numpy arrays of floats
        Each array is of shape (n_walkers, n_frames_per_we_round)
        The MTD importance weights for each walker at each saved frame.

    we_weights: list of 1d numpy arrays of floats
        Each array is of shape (n_walkers)
        Each element is the WE weight of a walker
 
    macrostate_classifier: numpy array of floats of shape (n_datapoints, n_dimensions) : numpy array of floats of shape (n_datapoints)
        returns 1 for each datapoint which is in the macrostate and 0 for the rest

    # kB: float
    #     Boltzmann's constant 
    # T: float
    #     Temperature
    # delta_T: float
    #     The well-tempered metadynamics temperature factor

    Returns
    -------
    delta_G: 1d numpy array of floats
        of shape (n_we_rounds)
        The estimated free energy difference between the macrostates at the end of each WE round.
        Each element is a free energy in units of kT

    """

    n_frames = sum([tr.shape[1] for tr in trj])
    #print(f"molecular time = {n_frames} frames")
    aggregate_frames = sum([tr.shape[0]*tr.shape[1] for tr in trj])
    #print(f"aggregate time = {aggregate_frames} frames")
    aggregate_walkers = sum([tr.shape[0] for tr in trj])
    #print(f"aggregate walkers = {aggregate_walkers}")


    delta_G = np.zeros(n_timepoints)


    # agg_trj = np.concatenate(trj)
    # agg_mtd_weights = np.concatenate(mtd_weights)
    # agg_we_weights = np.concatenate(we_weights)

    # trj_flattened = agg_trj.reshape(-1, agg_trj.shape[-1], order='F')
    # importance_weights_flattened = np.multiply(agg_mtd_weights, agg_we_weights[:, np.newaxis]).flatten(order="F")


    trj_flattened_by_we_round = [[] for _ in range(len(trj))]
    importance_weights_flattened_by_we_round = [[] for _ in range(len(trj))]

    #beware that the variable naming convention here is the opposite of that used in multiplot_observable_convergence()
    # i.e. sub-arrays here have more suffix letters and sub-arrays in multiplot_observable_convergence() have fewer

    #loop over WE rounds
    for we_i, (trj_r, mtd_weights_r, we_weights_r) in enumerate(zip(trj, mtd_weights, we_weights)):

        #flatten weights and add to list
        trj_flattened_by_we_round[we_i] = trj_r.reshape(-1, trj_r.shape[-1], order='F')
        importance_weights_flattened_by_we_round[we_i] = np.multiply(mtd_weights_r, we_weights_r[:, np.newaxis]).flatten(order='F')

    trj_flattened = np.concatenate(trj_flattened_by_we_round)
    importance_weights_flattened = np.concatenate(importance_weights_flattened_by_we_round)


    n_datapoints = 0
    for i in range(1,n_timepoints+1):

        # trj_r = agg_trj[0:i*aggregate_time_increment]
        # trj_flattened = trj_r.reshape(-1, trj_r.shape[-1])
        # importance_weights_flattened = np.multiply(mtd_weights[0:i*aggregate_time_increment], we_weights[0:i*aggregate_time_increment, np.newaxis]).flatten()
        # n_datapoints += len(importance_weights_flattened)
        #print(len(trj_flattened), i*aggregate_time_increment)
        pop_state_A = importance_sampling_estimator(trj_flattened[0:i*aggregate_time_increment], importance_weights_flattened[0:i*aggregate_time_increment], macrostate_classifier)

        if len(trj_flattened) >= i*aggregate_time_increment:
            delta_G[i-1] = -np.log((1-pop_state_A)/pop_state_A)
            n_datapoints += aggregate_time_increment
        else:
            delta_G[i-1] = -np.inf
            # if n_datapoints == -1:
            #     n_datapoints = (i-1)*aggregate_time_increment

    print(f"used {n_datapoints} datapoints")

    # trj_r = agg_trj[0:n_timepoints*aggregate_time_increment]
    # trj_flattened = [trj_r.reshape(-1, trj_r.shape[-1])]
    # importance_weights_flattened = [np.multiply(mtd_weights[0:n_timepoints*aggregate_time_increment], we_weights[0:n_timepoints*aggregate_time_increment, np.newaxis]).flatten()]

    #plot_landscape(sim_system, CV, kT, trj_flattened[0:n_timepoints*aggregate_time_increment], importance_weights_flattened[0:n_timepoints*aggregate_time_increment])


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

    observable_estimate = np.dot(observable_values, importance_weights)/np.sum(importance_weights)

    return observable_estimate


def plot_landscape(sim_system, CV, kT, coords_cumulative, importance_weights_cumulative):

    cv_grid, free_energy_grid = collective_variable_analysis.free_energy_on_cv_grid(
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


    print(f"used {len(importance_weights_cumulative)} datapoints")

    cv_coords = CV.cv_funct(coords_cumulative).flatten()
    # plt.hist(cv_coords, weights=importance_weights_cumulative)
    # plt.show()

    pops_1d = np.histogram(cv_coords, bins=CV.grid_n, range=(CV.cv_min[0], CV.cv_max[0]), weights=importance_weights_cumulative)

    plt.plot(cv_grid, -kT*np.log(pops_1d[0]/np.sum(pops_1d[0])), label = "importance sampling FE estimate")

    fe_norm = free_energy_grid + kT*np.log(np.sum(np.exp(-free_energy_grid/kT)))

    plt.plot(cv_grid, fe_norm, linestyle="dashed", color="black", linewidth="3", label="true FE")

    plt.xlabel("CV")
    plt.ylabel("Free Energy (kT)")

    plt.show()