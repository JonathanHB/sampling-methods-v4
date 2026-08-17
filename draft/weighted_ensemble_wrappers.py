#from weighted_ensemble_v2.py


###################################################################################################
#                                      MERGING AND SPLITTING

#TODO: check resampling scheme in westpa 2.0 in case it's changed
#TODO: this function seems too long; can it be refactored?

#split and merge walkers to ensure that each bin has the target number of walkers
#parameters
# w = walker weights
# b = walker bins (either configurational or history augmented depending on the choice of binner)
# walkerdata_transposed = a tuple of walker-level parameters other than the weights 
#     (since the weights are modified by this function and then added later for output)
#     this contains redundant bin information but including that makes the code cleaner
# walkers_per_bin = target number of walkers per bin
#returns
# a tuple of weights, bins, and the other walker parameters from walkerdata_transposed, for the new set of split/merged walkers

def split_merge(w, b, walkerdata_transposed, walkers_per_bin):

    # excess_walkers = len(w)-n_gpus #added by JHB on 8/5/26 

    printdebug = False

    #this stops walkers with weights above this threshold from merging even it would leave a bin overpopulated, selecting lighter ones where possible
    #it's basically anti-trust legislation for WE walkers
    maxweight = False
    if maxweight:
        merge_threshold = 0.05

    #a list of length n_total_walkers
    #using numpy transposition here would not work because the coordinates are themselves arrays while other attributes are scalars
    walkerdata = [list(row) for row in zip(*walkerdata_transposed)]
    split_limit = 2.00001*sys.float_info.min #keep weights from going to 0
    #print(f"bins: {b}")
    inds_by_bin = [[] for _ in range(max(b)+1)]  #list of lists; each sublist contains the indices of walkers in the corresponding bin
    for walker_ind, bin_ind in enumerate(b):
        inds_by_bin[bin_ind].append(walker_ind)

    
    # weights and other walker information for each walker produced by the splitting/merging process (including unaltered ones)
    w_out = []
    walkerdata_out = []

    for isi, indset in enumerate(inds_by_bin):

        if printdebug:
            print(f"--------------------{isi}---------------------")
            for i in indset:
                print(walkerdata[i]+[w[i]])
        
        #continue simulations in bins with the right population
        if len(indset) == walkers_per_bin:
            for i in indset:
                walkerdata_out.append(walkerdata[i])
                w_out.append(w[i])
            

        #duplicate simulations in bins with too few walkers
        elif len(indset) < walkers_per_bin and len(indset) > 0:

            #select walkers to duplicate
            w_indset = [w[i] for i in indset]
            duplicated_walkers = random.choices(indset, weights=w_indset, k = walkers_per_bin-len(indset))
            
            #add coordinates and weights of walkers from this bin to the list for next round
            # coordinates are unchanged for duplicated walkers; weights are reduced
            for i in indset:
                #add multiple copies of walkers to be duplicated with proportionally smaller weights
                for j in range(1+duplicated_walkers.count(i)):
                    walkerdata_out.append(walkerdata[i])

                    if w[i] >= split_limit:
                        #this is the normal WE algorithm
                        w_out.append(w[i]/(1+duplicated_walkers.count(i)))
                        # if j>0: #added 8/5/26
                        #     #this is for the new setting which avoids going below n_gpus walkers
                        #     excess_walkers+=1 
                    else:
                        w_out.append(w[i])
                        break #do not duplicate too-light walkers


        #merge simulations in bins with too many walkers
        elif len(indset) > walkers_per_bin: # and excess_walkers > 0:

            #total bin weight; does not change because merging operations preserve weight
            w_bin = sum([w[i] for i in indset])
        
            #deepcopy; may be unnecessary
            local_indset = [i for i in indset]
            w_local_indset = [w[i] for i in indset]

            #TODO: why does this need to be done with a loop instead of choosing multiple things without replacement? 
            # Does it have to do with the weighting function used to determine what to remove?
            #remove walkers until only walkers_per_bin remain
            for i in range(len(indset)-walkers_per_bin):
                
                #weights for walker elimination from Huber and Kim 1996 appendix A
                w_removal = [(w_bin - w[i])/w_bin for i in local_indset]

                #-------------------------------------experimental--------------------------------------------------
                if maxweight:
                    #print("update line marked WEIGHTCAP")
                    w_removal_masked = [wr if wli < merge_threshold else 0 for wli, wr in zip(w_local_indset, w_removal)]
                    if sum(w_removal_masked) == 0:
                        print(w_local_indset)
                        continue
                #---------------------------------------------------------------------------------------
                
                #pick 1 walker to remove, most likely one with a low weight
                #the [0] eliminates an unnecessary list layer
                if not maxweight:
                    removed_walker = random.choices([j for j in range(len(local_indset))], weights=w_removal, k = 1)[0] #WEIGHTCAP: w_removal >>> w_removal_masked
                else:
                    removed_walker = random.choices([j for j in range(len(local_indset))], weights=w_removal_masked, k = 1)[0]

                #remove the walker
                local_indset = [i for ii, i in enumerate(local_indset) if ii != removed_walker]
                removed_weight = w_local_indset[removed_walker]
                w_local_indset = [i for ii, i in enumerate(w_local_indset) if ii != removed_walker]

                # #this is for the new setting which avoids going below n_gpus walkers
                # excess_walkers-=1 #added 8/5/26

                #-------------------------------------experimental--------------------------------------------------
                if maxweight:
                    #print("update line marked WEIGHTCAP")
                    w_local_indset_masked = [wli if wli < merge_threshold else 0 for wli in w_local_indset]
                    if sum(w_local_indset_masked) == 0:
                        print(w_local_indset)
                        continue
                #---------------------------------------------------------------------------------------

                #pick another walker to gain the removed walker's probability
                #selection chance is proportional to existing weight
                if not maxweight:
                    recipient_walker = random.choices([j for j in range(len(local_indset))], weights=w_local_indset, k = 1)[0] #WEIGHTCAP: w_local_indset >>> w_local_indset_masked
                else:
                    recipient_walker = random.choices([j for j in range(len(local_indset))], weights=w_local_indset_masked, k = 1)[0] #WEIGHTCAP: w_local_indset >>> w_local_indset_masked
                
                #transfer the removed walker's weight
                w_local_indset[recipient_walker] += removed_weight

            #add the remaining walkers with updated weights to the output list
            for i in range(walkers_per_bin):
                walkerdata_out.append(walkerdata[local_indset[i]])
                w_out.append(w_local_indset[i])


    #combine data for new walkers into a consistent list of lists and arrays
    outputs_all = [w_out] + [np.stack([wdi[0] for wdi in walkerdata_out])] + [[wdi[i] for wdi in walkerdata_out] for i in range(1,len(walkerdata_out[0]))]


    #----------------------------------debugging-------------------------------------------
    if printdebug:
        walkerdata2 = [list(row) for row in zip(*outputs_all)]
        print("                        outputs")
        for b in range(max(b)+1):
            print(f"--------------------{b}---------------------")
            for wdi in walkerdata2:
                if wdi[4] == b:
                    print(wdi)


    return outputs_all



#other versions will have metadynamics grids and updating functions 
#   TODO define a fancier version that uses a metadynamics grid
class we_propagator_1():
    
    def __init__(self, system, kT, timestep, nsteps):
        self.system = system
        self.kT = kT
        self.timestep = timestep
        self.nsteps = nsteps
        #self.save_period = save_period
    
    def propagate(self, x, w):
        return (propagators_v1.propagate_save1(self.system, self.kT, x, self.timestep, self.nsteps), np.ones(len(x)))
    
    def mtd_grid(self):
        return None


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#TODO: separate the code above and below this point into separate files, analogous to how metadynamics is structured
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


###################################################################################################

#a wrapper for running weighted ensemble in segments and feeding them through msm_trj_analysis.run_for_n_timepoints()
def we_histogram_msm(state, params):
    
    #TODO: add support for different WE and analysis bins, which is currently (and deceptively) half done
    
    #unpack inputs
    x, e, w, cb, b, propagator, cumulative_observables, cumulative_aggregate_time, cumulative_molecular_time = state
    split_merge, config_binner, ensemble_classifier, binner, calc_observables, nrounds, walkers_per_bin, aggregate_simulation_limit = params

    #run dynamics
    x, e, w, cb, b, propagator, new_observables = weighted_ensemble(x, e, w, cb, b, propagator, split_merge, config_binner, ensemble_classifier, binner, calc_observables, nrounds, walkers_per_bin)

    #update cumulative observables and aggregate time
    observables = cumulative_observables+new_observables
    cumulative_molecular_time += nrounds*propagator.nsteps
    cumulative_aggregate_time += sum([nobs[-1] for nobs in new_observables])*propagator.nsteps
    #cumulative_agg_t += sum([nobs[-1] for nobs in new_observables])*propagator.nsteps

    #----------------------------histogram-based population estimation----------------------------#

    #estimate state populations
    cumulative_config_bins = np.concatenate([o[0] for o in observables], axis = 1).transpose()

    est_bin_pops = np.zeros(config_binner.n_bins)  #initialize estimated bin populations to 0
    for cbi in cumulative_config_bins:
        est_bin_pops[int(cbi[0])] += cbi[1]

    est_bin_pops /= np.sum(est_bin_pops)  #normalize estimated bin populations


    #----------------------------MSM-based population estimation----------------------------#

    aggregate_transitions = np.concatenate([o[2] for o in observables], axis = 1).transpose()
    eqp_msm = MSM_methods.transitions_to_eq_probs_v2(aggregate_transitions, config_binner.n_bins, show_TPM=False)

    return (x, e, w, cb, b, propagator, observables, cumulative_aggregate_time, cumulative_molecular_time), (cumulative_aggregate_time, cumulative_molecular_time, eqp_msm), cumulative_aggregate_time >= aggregate_simulation_limit  #, est_bin_pops


############################ MAIN SAMPLER FUNCTION ############################

def sampler_we(system_args, resource_args, bin_args, sampler_params):

    #----------------------------------input handling--------------------------------

    system, kT, dt = system_args
    n_parallel, molecular_time_limit, min_communication_interval, save_period = resource_args
    n_timepoints, n_analysis_bins, binbounds, bincenters = bin_args #TODO these should be used
    walkers_per_bin, n_we_bins = sampler_params

    binbounds_we, bincenters_we, step_we = system.analysis_bins_1d(n_we_bins)

    #determine number of steps for each parallel simulation per timepoint
    nsteps = int(round(molecular_time_limit/n_timepoints))
    #number of frames to save for each parallel simulation per timepoint
    # = number of simulation segments of length save_period to run per timepoint
    n_rounds_per_timepoint = int(round(nsteps/min_communication_interval))

    #molecular and aggregate times accounting for rounding
    actual_molecular_time = n_rounds_per_timepoint*min_communication_interval*n_timepoints
    max_actual_aggregate_time = n_parallel*actual_molecular_time

    print("\n")
    print("---------------------WEIGHTED ENSEMBLE---------------------")
    print(f"weighted ensemble with {walkers_per_bin} walkers per bin in {len(binbounds_we)+1} bins for {n_rounds_per_timepoint*n_timepoints} WE rounds of {min_communication_interval} steps each")
    print(f"molecular time: {actual_molecular_time} steps;  maximum aggregate time: {max_actual_aggregate_time} steps")
    print(f"maximum data points saved: {n_rounds_per_timepoint*n_timepoints*walkers_per_bin*(len(binbounds_we)+1)} at {min_communication_interval}-step intervals")

    #--------------------------------set up and run system-----------------------------

    #initialize instances of classes
    config_binner = config_binner_1(binbounds_we)
    ensemble_classifier = ensemble_classifier_1(system.macro_class_parallel)
    binner = binner_1()
    propagator0 = we_propagator_1(system, kT, dt, min_communication_interval) #the propagator evolves over time in the case of metadynamics

    #initial state
    x0 = np.array([system.standard_init_coord for element in range(walkers_per_bin)]) #.reshape((walkers_per_bin, 1, len(system.standard_init_coord)))
    e0 = [system.macro_class(x0i) for x0i in x0] #initial ensemble is determined by the macrostate classifier
    w0 = [1/walkers_per_bin for element in range(walkers_per_bin)]
    cb0 = config_binner.bin(x0)  #configurational bins
    b0 = binner.bin(cb0, e0)
    #prop_out_0 = [1 for element in range(walkers_per_bin)]

    cumulative_observables0 = []  #list of lists; each sublist contains the observables calculated at each WE round

    #pack the initial state and parameters and run dynamics
    initial_state = (x0, e0, w0, cb0, b0, propagator0, cumulative_observables0, 0, 0) #the final 0 is the initial aggregate simulation time
    params = (split_merge, config_binner, ensemble_classifier, binner, calc_observables_1, n_rounds_per_timepoint, walkers_per_bin, max_actual_aggregate_time)
    time_x_observables = utility_v1.run_for_n_timepoints(we_histogram_msm, params, initial_state, n_timepoints)

    #effectively transpose the list of lists so the first axis is observable type rather than time
    #but without the data type/structure requirement of a numpy array
    observables_x_time = [list(row) for row in zip(*time_x_observables)]

    final_aggregate_time = observables_x_time[0][-1]
    print(f"aggregate simulation time: {final_aggregate_time} steps")
    print(f"aggregate number of walkers = number of data points saved = {final_aggregate_time/min_communication_interval} at {min_communication_interval}-step intervals")

    observable_names = ["WE: msm"]#, "histogram"]

    return observables_x_time, observable_names



# #ON SECOND THOUGHT TRY TO USE THE ORIGINAL VERSION
# def weighted_ensemble_trjout(x, e, w, cb, b, propagator, split_merge, config_binner, ensemble_classifier, binner, nrounds, walkers_per_bin):
#     """
#     Run a WE+MTD simulation
    
#     Parameters
#     ----------
#     TODO: add these
#     n_we_rounds: int
#         Number of WE rounds to run

#     Returns
#     -------
#     trj: 4d numpy array of floats
#         of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, n_coordinates)
#         Each element is a coordinate. NaN values denote walkers which did not exist 
#         (i.e. not all bins were filled at the start of the round so less than the maximum number of walkers were spawned)

#     discrete_trj: 4d numpy array of ints
#         of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, n_coordinates)
#         Each element is an MTD grid index along one axis. NaN values are distributed as above.
#         This is the discrete version of trj

#     #TODO; do we actually need this or can we just save the partition function at each timepoint and the weight for each frame on the fly?
#     potentials: (3+n_CV_dimensions)d numpy array of floats
#         of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, [MTD grid dimensions])
#         Each element is an MTD potential value at a specific walker, time, and place
#         NaN values are distributed as above.
#         Because the MTD potential is not usually updated for every saved frame, this contains some redundant information
    
#     we_weights: 2d numpy array of floats
#         of shape (n_we_rounds, max_n_walkers)
#         Each element is the WE weight of a walker. NaNs as above
    
#     metadata: 2d numpy array of ints
#         of shape (n_we_rounds, max_n_walkers)
#         Each element is 1 if the walker was spawned, 0 otherwise. This should match the NaNs in the above two parameters.
#         This could be encoded in the WE weights by setting the weight to 0 for non-spawned walkers, but this is not done here to avoid confusion with the actual WE weights.
#     """

#     trj = np.zeros((nrounds, walkers_per_bin*config_binner.n_bins, propagator.n_gaussians_per_round*propagator.mtd_params["n_frames_per_gaussian"], propagator.system.n_dim)) * np.nan
#     discrete_trj = np.zeros_like(trj, dtype=int) * np.nan
#     potentials = np.zeros((nrounds, walkers_per_bin*config_binner.n_bins, propagator.n_gaussians_per_round*propagator.mtd_params["n_frames_per_gaussian"], *(n_gridpoints for _ in propagator.CV.grid_n))) * np.nan

#     we_weights = np.zeros((nrounds, walkers_per_bin*config_binner.n_bins)) * np.nan
#     metadata =   np.zeros((nrounds, walkers_per_bin*config_binner.n_bins)) * np.nan

#     x = x.copy()    #positions and/or MSM state indices for trajectories generated by an MSM
#     e = e.copy()    #ensembles for history augmented analysis
#     w = w.copy()    #WE weights
#     cb = cb.copy()  #configurational bin indices for MSM analysis
#     b = b.copy()    #bin indices for haMSM analysis
#     #es_args = es_args.copy() #arguments for enhanced sampling methods, such as a metadynamics potential grid. 
#     # This is only needed as a variable outside the propagator if each walker has its own es_args.

#     #observables = []

#     for r in range(nrounds):

#         # #print a note every 1/10th of the way there
#         # if r%max(round(nrounds/10), 1) == 0:
#         #     print(f"WE round {r}")

#         #deepcopy variables for observable calculation (i.e. to get transitions)
#         x_last = x.copy()
#         e_last = e.copy()
#         cb_last = cb.copy()
#         b_last = b.copy()

#         #Propagate dynamics
#         # beware that this propagator modifies x in place
#         # w is only passed in because it may be used to update metadynamics grids
#         #TODO figure out if the following is needed:
#         # certain observables have to be computed after the trajectory is propagated 
#         # but before the propagator updates other internal variables like the metadynamics grid
#         # these are returned in propagator_outputs
#         x_md, mtd_data = propagator.propagate(x, w)

#         #Calculate configurational bins
#         cb_md = config_binner.bin(x_md)

#         #Determine which ensemble each walker belongs to based on the new coordinates or configurational bins and the last ensembles.
#         # This need not use both x_md and cb_md; both are included to support different ensemble_classifier objects.
#         e_md = ensemble_classifier.ensemble(x_md, cb_md, e_last)

#         #Determine which bin each walker belongs to based on the new coordinates or configurational bins and its current ensemble.
#         # For non-history-augmented binning schemes e is unused and this simply returns the configurational bins cb_md.
#         b_md = binner.bin(cb_md, e_md)

#         #Calculate total bin occupancies, MSM transitions, and/or whatever other observables are desired
#         #observables.append(calc_observables(x_last, x_md, e_last, e_md, w, cb_last, cb_md, b_last, b_md, propagator, mtd_data))

#         #Split and merge trajectories
#         (w, x, e, b, cb) = split_merge(w, b_md, (x_md, e_md, b_md, cb_md), walkers_per_bin)

#     #return the final coordinates, ensembles, weights, bins, propagator (for metadynamics purposes when it is modified) and observables
#     #return x, e, w, cb, b, propagator#, observables