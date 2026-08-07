#TODO write a variant of this that works by actual time stamp instead of WE round
#since WE round spacing may vary
def old_importance_sampling_fe_by_we_round(trj, discrete_trj, we_weights, potentials, metadata, macrostate_classifier, kB, T, delta_T):
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

    #TODO: can more of this be parallelized?

    #TODO: should these be combined into a single nested list since the entries must have a 1:1 correspondence?
    # This is on hold pending other issues
    trj_flattened_by_we_round = [[] for _ in range(len(trj))]
    importance_weights_flattened_by_we_round = [[] for _ in range(len(trj))]

    #beware that the variable naming convention here is the opposite of that used in multiplot_observable_convergence()
    # i.e. sub-arrays here have more suffix letters and sub-arrays in multiplot_observable_convergence() have fewer

    #loop over WE rounds
    for we_i, (trj_r, disc_trj_r, we_weights_r, potentials_r, metadata_r) in enumerate(zip(trj, discrete_trj, we_weights, potentials, metadata)):

        #loop over walkers
        for trj_rw, disc_trj_rw, we_weight, potentials_rw, metadata_rw in zip(trj_r, disc_trj_r, we_weights_r, potentials_r, metadata_r):

            #if the walker exists
            if metadata_rw: 

                #add trajectory to aggregated data
                trj_flattened_by_we_round[we_i].append(trj_rw)

                # #DONE: make metadynamics weight calculation into its own function
                # #calculate metadynamics weight
                # potential_along_trj = potentials_rw[disc_trj_rw]
                # exp_factor = np.exp(potential_along_trj/(kB*T))

                # Z0 = np.sum(np.exp(potentials_rw*(1/T + 1/delta_T)/kB), axis=tuple(range(1, potentials_rw.ndim)))
                # Z1 = np.sum(np.exp(potentials_rw*(1/delta_T)/kB), axis=tuple(range(1, potentials_rw.ndim)))
                # partition_ratio = np.divide(Z1,Z0)

                # mtd_weights_rw = np.multiply(exp_factor, partition_ratio)
                mtd_weights_rw = calc_MTD_importance_weights(potentials_rw, disc_trj_rw, kB, T, delta_T)

                #calculate importance weights and add them to aggregated data
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


#DEPRECATED; THIS HAS BEEN INTEGRATED INTO propagators_grid.propagate()
def calc_MTD_importance_weights(potentials, discrete_trajectory, kB, T, delta_T):
    """
    Calculate importance sampling weights from an MTD trajectory
    
    Parameters
    ----------
    potentials: (1+n_CV_dimensions)d numpy array of floats
        of shape (n_frames_per_we_round, [MTD grid dimensions])
        Each element is an MTD potential value at a specific walker, time, and place
        Because the MTD potential is not usually updated for every saved frame, this contains some redundant information

    discrete_trj: 2d numpy array of ints
        of shape (n_frames_per_we_round, n_coordinates)
        Each element is an MTD grid index along one axis.
        The i-th row indexes the last n_CV_dimensions dimensions of the potentials array (i.e. all but the first dimension)

    kB: float
        Boltzmann's constant 
    T: float
        Temperature
    delta_T: float
        The well-tempered metadynamics temperature factor

    Returns
    -------
    mtd_weights: 1d numpy array of floats
    
    """

    #calculate metadynamics weight
    potential_along_trj = potentials[discrete_trajectory]
    exp_factor = np.exp(potential_along_trj/(kB*T))

    Z0 = np.sum(np.exp(potentials*(1/T + 1/delta_T)/kB), axis=tuple(range(1, potentials.ndim)))
    Z1 = np.sum(np.exp(potentials*(1/delta_T)/kB), axis=tuple(range(1, potentials.ndim)))
    partition_ratio = np.divide(Z1,Z0)

    mtd_weights = np.multiply(exp_factor, partition_ratio)

    return mtd_weights



############################################################################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                        OLD CODE *ABOVE*
    #-------------------------------------------------------------------------------------------------
    #                                 PLOTTING CODE ONLY BELOW
    if False:
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

        #calculate the cumulative deltaG for all times
        coords_cumulative = np.concatenate(trj_flattened_by_we_round)
        importance_weights_cumulative = np.concatenate(importance_weights_flattened_by_we_round)

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