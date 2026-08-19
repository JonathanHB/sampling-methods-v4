import sys
import numpy as np

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




###################################################################################################
#                                      MSM RESAMPLER

# def tram_unbiased_energies(transitions, potential_functions, kT):
#     """
#     Calculate unbiased free energies of bins using TRAM (https://deeptime-ml.github.io/latest/notebooks/tram.html).

#     Parameters
#     ----------
#     transitions: list of 2d numpy arrays of floats
#     Each array has a first axis of length 2 and a variable second axis length depending on the number of walkers in that round
#     Each column of the array is a transition. The element in the first row is the index of the starting bin, 
#     and the element in the second row is the index of the ending bin
    
#     potential_functions: list of n_CV_dimensions - dimensional numpy arrays of floats
#     Each array is the metadynamics grid containing the metadynamics potential when a particular set of transitions occurred
#     The length of this is the number of thermodynamic states

#     kT: float
#     Boltzmann's constant times temperature

    
#     Returns
#     -------
#     bins: 1d numpy array of ints
#     Indices of the the bins for which TRAM potentials could be estimated

#     energies: 1d numpy array of floats
#     Potential energies for each of the above bins

#     """

#     #to implement TRAM with the provided arguments:
#     # concatenate all the transitions into a single array of shape (total_n_transitions, 2)
#     # make the 3d bias_matrices array, indexing potential_functions to get the potential of every sample in every state. 
#     # because there are not the same number of samples in all states, some elements of this array will be meaningless.
#     # I'm not sure of the top of my head what value to put for those elements.

#     return bins, energies


#THIS is on hold because the bias matrix required for tram grows quadratically with trajectory length 
# because its size includes the number of thermodynamic states in one dimension times the number of frames in the other
#as of this writing we're talking about matrices of size ~300 states * 300 states * 120 bins * 4 walkers per bin * 40 rounds per state = 1.7 billion elements for my toy system
#worse yet if you reweight at equally spaced intervals (thereby summing quadratic-cost operations), the total cost grows cubically
#in theory if the prefactor of this cost is not too large this might still be a small cost relative to simulation, 
# but I'm going to focus on more scalable approaches for the moment.

#The current arguments for this are wrong (i.e. they do not contain all the required information for dTRAM)
def tram_reweight(transitions, w, b, walkers_per_bin, last_potential):
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