#Jonathan Borowsky
#Grabe lab
#8/17/25
#sampling methods
#weighted ensemble

import numpy as np
import sys
import random
from collections import Counter
import matplotlib.pyplot as plt

from propagators_grid import propagate
from propagators_grid import propagate_shared_grid
import visualization
# import propagators_v1
# import utility_v1
# import MSM_methods




###################################################################################################
#                                      MSM RESAMPLER

def tram_reweight(cumulative_transitions_weights, w, b, walkers_per_bin, last_potential):
    """
    Reweight weighted ensemble walkers using TRAM (https://deeptime-ml.github.io/latest/notebooks/tram.html).
    In the case that there are multiple disconnected sets of states, 
    build a MSM for each one and redistribute the weight within it, 
    leaving the total for each disconnected set the same.

    Parameters
    ----------
    cumulative_transitions_weights: list of numpy arrays
    Each array is of shape (4, number of walkers in round i). 
    The indices along axis 0 are:
    0: the transition starting bin
    1: transition ending bin
    2: the MTD importance weight in the starting bin
    3: the MTD importance weight in the ending bin

    w: numpy array of floats
    The WE weight of each current walker, needed only if there are multiple disconnected sets of states

    b: numpy array of ints
    The bin occupied by each current walker. These indices should be one larger than the corresponding MTD potential grid indices 
    (b[i]=0 corresponds to a walker off the left edge of the MTD grid, at a MTD potential of zero)

    walkers_per_bin: int
    Target number of walkers per bin, technically inferrable from the count of each index in b but that's ugly

    last_potential: numpy array of floats
    of n_CV_dim dimensions
    The current MTD potential, for calculating WE weights

    Returns
    -------
    w_out: numpy array of floats
    The new WE weights of the current walkers, calculated as described below

    """

    ##########################################################
    # 1. USE DTRAM TO ESTIMATE THE EQUILIBRIUM BIN PROBABILITIES

    #This section should use:
    # cumulative_transitions_weights
    # w only if there are multiple disconnected sets of states


    ##########################################################
    #2. ADJUST WEIGHTED ENSEMBLE WEIGHTS ACCORDINGLY
    # accounting for the current MTD potential
    #i.e. 
    # if the MTD potential is zero, the WE weights equal the unbiased probabilities divided by the number of bins
    # if the MTD potential equals the true potential (i.e. the energy wells are all filled in), the WE weights are all equal
    
    #this section should use:
    # the unbiased bin probabilities from section 1
    # b
    # last_potential


    w_out = np.zeros(len(b))

    return w_out



###################################################################################################
#                                      NULL RESAMPLER
#                  for non-WE simulations using a consistent software framework
#return the inputs unchanged
def null_resample(w, b, walkerdata_transposed, walkers_per_bin):
    return [w]+walkerdata_transposed


###################################################################################################
#                                      RESAMPLER

#based on Aristoff et al 2023. 'Weighted ensemble, recent mathematical developments"
def resample(w, b, walkerdata_transposed, walkers_per_bin):

    #effectively transpose the walkerdata so that the first index is the walker rather than the attribute (i.e. bin, ensemble)
    #using numpy transposition here would not work because the coordinates are themselves arrays while other attributes are scalars
    walkerdata = [list(row) for row in zip(*walkerdata_transposed)]

    inds_by_bin = [[] for _ in range(max(b)+1)]  #list of lists; each sublist contains the indices of walkers in the corresponding bin
    for walker_ind, bin_ind in enumerate(b):
        inds_by_bin[bin_ind].append(walker_ind)

    # weights and other walker information for each walker produced by the splitting/merging process (including unaltered ones)
    w_out = []
    walkerdata_out = []

    for ib in np.unique(b):
        bin_inds = np.where(b==ib)[0]
        bin_weights = w[bin_inds]

        new_walker_inds = random.choices(bin_inds, weights=bin_weights, k=walkers_per_bin)

        new_weights = np.sum(bin_weights)/walkers_per_bin
        w_out += [new_weights] * walkers_per_bin
        walkerdata_out += [walkerdata[nwi] for nwi in new_walker_inds]


    #ensure that weights remain normalized in the face of accumulated floating point errors. 
    # So far these errors appear to be negligible and cancel out over the long run but this should be more reliable.
    w_out = np.array(w_out)
    w_out/=np.sum(w_out)

    #reverse the 'transposition' of walker data performed at the start of this function
    return [w_out] + [np.stack([wdi[0] for wdi in walkerdata_out])] + [np.array([wdi[i] for wdi in walkerdata_out]) for i in range(1,len(walkerdata_out[0]))]




#PARAMETERS
# x, e, w, cb, and b are all lists or arrays of length equal to the number of walkers at the start of the WE round
#      the i-th element of each array corresponds to the same walker
#   x = coordinates or state indices
#   e = ensembles (for history augmented analysis)
#   w = weights
#   cb = configurational bins
#   b = bins (equal to the configurational bins for non-history-augmented binning schemes)
# propagator: a class with a propagate() method that runs the dynamics (starting from x) 
#    and updates the metadynamics grid if it exists (this is why w is included as an argument)
# split_merge: a function that takes in weights, bins, a tuple of the other walker-level parameters, 
#    and the target number of walkers per bin and returns the above walker-level parameters for 
#    walkers which have been split and merged to yield the target walker number
# config_binner: a class with a bin() method that bins walkers based on their current coordinates
# ensemble_classifier: a class with an ensemble() method that determines which ensembles walkers are in 
#    based on their configurational bins plus their most recent ensembles
# binner: a class with a bin() method that determines which bins walkers are in based on their configurational bins and ensembles
#    the ensemble is ignored and the configurational bin is returned unchanged for non-history-augmented binning schemes
# calc_observables: a function that calculates observables based on the walker-level parameters from before and after dynamics were run
#    this is included here instead of just returning trajectories of the walker-level parameters because it avoids having to 
#    construct and use a parent-to-child walker mapping or pass information between WE segments from different function calls. 
#    This approach would break down if we needed to compute observables across more than two adjacent WE rounds (i.e. for an MSM of variable lag time)
#    but haMSMs work best with the shortest available lag time and are better for computing rates than regular ones so I see no occasion for that.
# nrounds: number of WE rounds to run
# walkers_per_bin: target number of walkers per bin after splitting and merging

#RETURNS
# x, e, w, cb, b: the walker-level parameters after the last WE round
# propagator: the propagator class after the last WE round, which may have updated its metadynamics grid but is otherwise unchanged
# observables: a list of the observables calculated at each WE round

def weighted_ensemble(x, e, w, cb, b, propagator, resampler, config_binner, ensemble_classifier, binner, calc_observables, nrounds, n_rounds_per_gaussian, walkers_per_bin, n_gpu_rounds_t_wall, n_gpus):

    x = x.copy()    #positions and/or MSM state indices for trajectories generated by an MSM
    e = e.copy()    #ensembles for history augmented analysis
    w = w.copy()    #WE weights
    cb = cb.copy()  #configurational bin indices for MSM analysis
    b = b.copy()    #bin indices for haMSM analysis
    w_mtd = np.ones(len(x))

    n_gpu_rounds = 0

    bin_pops = np.zeros((nrounds, config_binner.n_bins))
    bin_we_weights = np.zeros((nrounds, config_binner.n_bins))

    observables = []
    w_max = []
    transitions = [] #list of numpy arrays, 1 array per WE round. Elements are bin indices.

    deposit = 0

    for r in range(nrounds):
        # #print a note every 1/10th of the way there
        # if r%max(round(nrounds/10), 1) == 0:
        #     print(f"WE round {r}")

        #deepcopy variables for observable calculation (i.e. to get transitions)
        x_last = x.copy()
        e_last = e.copy()
        cb_last = cb.copy()
        b_last = b.copy()
        w_mtd_last = w_mtd.copy()

        deposit_last = deposit
        deposit = 0
        if n_rounds_per_gaussian == 1:
            deposit = 1
        elif n_rounds_per_gaussian > 1 and r > 0 and r % n_rounds_per_gaussian == 0:
            deposit = 1
        elif config_binner.n_bins == 1:
            deposit = 1

        print(config_binner.n_bins)
        print(f"deposit = {deposit}")
    
        #Propagate dynamics
        # beware that this propagator modifies x in place
        # w is only passed in because it may be used to update metadynamics grids
        #TODO figure out if the following is needed:
        # certain observables have to be computed after the trajectory is propagated 
        # but before the propagator updates other internal variables like the metadynamics grid
        # these are returned in propagator_outputs
        x_md, mtd_data = propagator.propagate(x, w, deposit)

        w_mtd_md = mtd_data[2][:,-1]

        #Calculate configurational bins
        cb_md = config_binner.bin(x_md)

        for cb_md_i, w_i in zip(cb_md, w):
            bin_pops[r,cb_md_i]+=1
            bin_we_weights[r,cb_md_i]+=w_i
            
        #Determine which ensemble each walker belongs to based on the new coordinates or configurational bins and the last ensembles.
        # This need not use both x_md and cb_md; both are included to support different ensemble_classifier objects.
        e_md = ensemble_classifier.ensemble(x_md, cb_md, e_last)

        #Determine which bin each walker belongs to based on the new coordinates or configurational bins and its current ensemble.
        # For non-history-augmented binning schemes e is unused and this simply returns the configurational bins cb_md.
        b_md = binner.bin(cb_md, e_md)

        if deposit_last != 1:
            # print(b_last.shape)
            # print(b_md.shape)
            # print(w_mtd_last.shape)
            # print(mtd_data[2].shape)
            # print(mtd_data[2][:,-1].shape)
            transitions.append(np.stack((b_last, b_md, w_mtd_last, w_mtd_md)))

        #Calculate total bin occupancies, MSM transitions, and/or whatever other observables are desired
        observables.append(calc_observables(x_last, x_md, e_last, e_md, w, cb_last, cb_md, b_last, b_md, propagator, mtd_data))

        n_gpu_rounds += len(x)/n_gpus #int(np.ceil(len(x)/n_gpus)) #note that this does not account for small additional costs if the number of walkers is not a multiple of the number of gpus
        if n_gpu_rounds >= n_gpu_rounds_t_wall:
            print(f"reached the maximum number of gpu (and hence WE) rounds permitted by the WE round length, number of GPUs, and wall clock time limit {n_gpu_rounds} >= {n_gpu_rounds_t_wall} after {r} WE rounds")

            rmax = 2000
            #walker distribution
            visualization.plot_masked_energies(data=bin_pops[0:rmax].transpose(), xlims=[0,rmax], ylims=[0,config_binner.n_bins], plot_shape=[16,8], aspect_ratio=10/4, vmax=10, labels=["WE round", "bin"])
            #weight distribution
            visualization.plot_masked_energies(data=bin_we_weights[0:rmax].transpose(), xlims=[0,rmax], ylims=[0,config_binner.n_bins], plot_shape=[16,8], aspect_ratio=10/4, vmax=0.1, labels=["WE round", "bin"])

            # import sys
            # sys.exit(0)


            # plt.plot(np.average(mtd_data[1][:,-1], axis=0, weights = w))
            # plt.show()
            # plt.plot(w_max)
            # plt.show()
            # plt.figure(figsize=(10, 20)) 
            # plt.imshow(bin_pops[:r+1], interpolation='none')
            # #plt.colorbar()
            # plt.show()

            # print(bin_pops[:r+1])
            # print(np.min(bin_pops[:r+1]))
            # print(np.max(bin_pops[:r+1]))

            break

        if r % 500 == 0:
            plt.plot(propagator.mtd_grid())
        #     plt.hist(w, bins=100, alpha = 0.3, range = (0,0.25))
        #     plt.show()
        #     # print(x_md)
        #     # print(x)
        #     plt.hist((x_md-x)[:,0], bins=16, range = (-0.2,0.2))
        #     plt.show()

        w_max.append(max(w)) #these diagnostics belong in the observables

        if r < nrounds-1:
            #Split and merge trajectories
            (w, x, e, b, cb, w_mtd) = resampler(w, b_md, (x_md, e_md, b_md, cb_md, w_mtd_md), walkers_per_bin)
            #if deposit == 1:
            #    w = msm_resample(transitions, b_md, walkers_per_bin, mtd_data[1][-1]) #the last argument is the most recent MTD potential



    #return the final coordinates, ensembles, weights, bins, propagator (for metadynamics purposes when it is modified) and observables
    return x, e, w, cb, b, propagator, observables



###################################################################################################
#                                      PROPAGATORS

#run dynamics and return the results
#for arguments and returns see the comments in propagators.py
#this version of this method exists to store the variables defined in __init__() without cluttering up the weighted_ensemble() function, not to do anything new

class we_propagator_2():
    
    def __init__(self, simulated_system, kB, T, mtd_params, n_gaussians_per_round):
        self.system = simulated_system
        self.kB = kB
        self.T = T
        self.mtd_params = mtd_params
        self.n_gaussians_per_round = n_gaussians_per_round
        self.CV = mtd_params["CV"] #to simplify referencing
        self.potential = np.zeros(self.CV.grid_n)

    def propagate(self, x, w, deposit):

        traj, pots, weights = propagate_shared_grid(
            G=self.system.G, kB=self.kB, T = self.T, dt=self.mtd_params["dt"], xi = self.system.xi, 
            init_coords=x, init_potentials=self.potential, we_weights = w,
            steps_per_saved_frame=self.mtd_params["n_steps_per_frame"],
            n_gaussians=self.n_gaussians_per_round, 
            frames_per_gaussian=self.mtd_params["n_frames_per_gaussian"],
            sigma=self.mtd_params["sigma"], omega=self.mtd_params["omega"]*deposit, delta_T=self.mtd_params["delta_T"],
            CV=self.CV.cv_funct, grad_CV=self.CV.cv_grad_funct, 
            cv_min=self.CV.cv_min, cv_max=self.CV.cv_max
        )

        self.potential = pots[-1]

        return traj[:,-1], (traj, pots, weights)

    def mtd_grid(self):
        return self.potential


###################################################################################################
#                                      BINS AND ENSEMBLES

#bin trajectory frames in configuration space
#for arguments and returns see the comments in msm_trj_analysis.py
#this version of this method exists to store the variables defined in __init__() without cluttering up the weighted_ensemble() function, not to do anything new
# other versions may be more complicated in order to support dynamic binning schemes, 
# and having this as its own method provides a modular way to implement such schemes 
class config_binner_1():
    
    def __init__(self, binbounds, CV):
        self.binbounds = binbounds
        self.n_bins = len(binbounds)+1
        self.CV = CV
        # self.n_bins = np.product([len(bbd)+1 for bbd in binbounds])
    
    def bin(self, x):
        return np.digitize(self.CV.cv_funct(x), self.binbounds).flatten()
        #utility_v1.binner_1d(self.binbounds, x)
    

#determine which ensemble a trajectory currently in ensemble e should be in upon moving to coordinate or state x
#this method exists to store the variables defined in __init__() without cluttering up the weighted_ensemble() function, 
#   which is sort of silly given that there is only one such variable and basically no new code 
#   and to my knowledge no competing macrostate classification schemes
class ensemble_classifier_1():

    def __init__(self, macrostate_classifier):
        self.macrostate_classifier = macrostate_classifier

    def ensemble(self, x, cb, e):
        #determine which ensemble a trajectory currently in ensemble e should be in upon moving to coordinate or state x
        macrostates = self.macrostate_classifier(x)
        ensembles = np.where(macrostates == -1, e, macrostates)  #if the macrostate is not -1, use it; otherwise use the current ensemble
        return ensembles


#for non-history-augmented binning schemes this simply returns the configurational bin
class binner_1():
    def bin(self, b, e):
        return b


#history augmented binning, where each config bin/ensemble pair is its own bin
class binner_2():

    def __init__(self, n_macrostates):
        self.n_macrostates = n_macrostates

    def bin(self, b, e):
        return [bi*self.n_macrostates + ei for bi, ei in zip(b, e)]
    

###################################################################################################
#                                      OBSERVABLES

#note that the _md variable suffix is not preserved here
def calc_observables_1(x_last, x, e_last, e, w, cb_last, cb, b_last, b, propagator, mtd_data):
    #for histogram methods
    trj_config_weighted = np.stack((cb, w))
    trj_weighted = np.stack((b, w))

    #for Markov state models
    cb_transitions = np.stack((cb_last, cb))
    bin_transitions = np.stack((b_last, b))

    if mtd_data is not None:
        #TODO add other correction factor from lab notebook
        mtd_transition_weights = np.sqrt(np.divide(mtd_data[1], mtd_data[0]))
    else:
        mtd_transition_weights = None

    #for metadynamics
    #currently unused; lacks sufficient temporal resolution to be useful
    #mtd_weights = propagator.mtd_grid()  #get the metadynamics grid if it exists (otherwise None) 

    return (trj_config_weighted, trj_weighted, cb_transitions, bin_transitions, mtd_data[1], mtd_transition_weights, mtd_data[2], len(w))


def calc_observables_2(x_last, x, e_last, e, w, cb_last, cb, b_last, b, propagator, mtd_data):
    #note that x is not returned here because mtd_data has the trajectories at better time resolution
    #specifically x is the final coordinates of each walker after the WE round, while mtd_data[0] contains the coordinates of each walker at every saved frame during the WE round
    return mtd_data[0], mtd_data[2], np.array(w)


