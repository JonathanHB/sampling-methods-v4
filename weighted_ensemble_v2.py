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

def msm_resample(w, b, walkerdata_transposed, walkers_per_bin):

    #effectively transpose the walkerdata so that the first index is the walker rather than the attribute (i.e. bin, ensemble)
    #using numpy transposition here would not work because the coordinates are themselves arrays while other attributes are scalars
    walkerdata = [list(row) for row in zip(*walkerdata_transposed)]
    #keep weights from going to 0
    split_limit = 2.00001*sys.float_info.min 

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
    return [w_out] + [np.stack([wdi[0] for wdi in walkerdata_out])] + [[wdi[i] for wdi in walkerdata_out] for i in range(1,len(walkerdata_out[0]))]



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
    #keep weights from going to 0
    split_limit = 2.00001*sys.float_info.min 

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
    return [w_out] + [np.stack([wdi[0] for wdi in walkerdata_out])] + [[wdi[i] for wdi in walkerdata_out] for i in range(1,len(walkerdata_out[0]))]


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

def weighted_ensemble(x, e, w, cb, b, propagator, resampler, config_binner, ensemble_classifier, binner, calc_observables, nrounds, walkers_per_bin, n_gpu_rounds_t_wall, n_gpus):

    x = x.copy()    #positions and/or MSM state indices for trajectories generated by an MSM
    e = e.copy()    #ensembles for history augmented analysis
    w = w.copy()    #WE weights
    cb = cb.copy()  #configurational bin indices for MSM analysis
    b = b.copy()    #bin indices for haMSM analysis

    n_gpu_rounds = 0

    bin_pops = np.zeros((nrounds, config_binner.n_bins))
    bin_we_weights = np.zeros((nrounds, config_binner.n_bins))

    observables = []
    w_max = []

    for r in range(nrounds):
        # #print a note every 1/10th of the way there
        # if r%max(round(nrounds/10), 1) == 0:
        #     print(f"WE round {r}")

        #deepcopy variables for observable calculation (i.e. to get transitions)
        x_last = x.copy()
        e_last = e.copy()
        cb_last = cb.copy()
        b_last = b.copy()

        #Propagate dynamics
        # beware that this propagator modifies x in place
        # w is only passed in because it may be used to update metadynamics grids
        #TODO figure out if the following is needed:
        # certain observables have to be computed after the trajectory is propagated 
        # but before the propagator updates other internal variables like the metadynamics grid
        # these are returned in propagator_outputs
        x_md, mtd_data = propagator.propagate(x, w)

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

        #Calculate total bin occupancies, MSM transitions, and/or whatever other observables are desired
        observables.append(calc_observables(x_last, x_md, e_last, e_md, w, cb_last, cb_md, b_last, b_md, propagator, mtd_data))

        n_gpu_rounds += len(x)/n_gpus #int(np.ceil(len(x)/n_gpus)) #note that this does not account for small additional costs if the number of walkers is not a multiple of the number of gpus
        if n_gpu_rounds >= n_gpu_rounds_t_wall:
            print(f"reached the maximum number of gpu (and hence WE) rounds permitted by the WE round length, number of GPUs, and wall clock time limit {n_gpu_rounds} >= {n_gpu_rounds_t_wall} after {r} WE rounds")

            rmax = 2000
            visualization.plot_masked_energies(data=bin_pops[0:rmax].transpose(), xlims=[0,rmax], ylims=[0,config_binner.n_bins], plot_shape=[16,8], aspect_ratio=10/4, vmax=10, labels=["WE round", "bin"])

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

        #Split and merge trajectories
        (w, x, e, b, cb) = resampler(w, b_md, (x_md, e_md, b_md, cb_md), walkers_per_bin)


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

    def propagate(self, x, w):

        traj, pots, weights = propagate_shared_grid(
            G=self.system.G, kB=self.kB, T = self.T, dt=self.mtd_params["dt"], xi = self.system.xi, 
            init_coords=x, init_potentials=self.potential, we_weights = w,
            steps_per_saved_frame=self.mtd_params["n_steps_per_frame"],
            n_gaussians=self.n_gaussians_per_round, 
            frames_per_gaussian=self.mtd_params["n_frames_per_gaussian"],
            sigma=self.mtd_params["sigma"], omega=self.mtd_params["omega"], delta_T=self.mtd_params["delta_T"],
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


