import numpy as np
#add deeptime imports here

###################################################################################################
#                                      MSM RESAMPLER

def tram_reweight(cumulative_transitions_weights, w, b, walkers_per_bin, last_potential):
    """
    Reweight weighted ensemble walkers using TRAM (https://deeptime-ml.github.io/latest/notebooks/tram.html).
    In the case that there are multiple disconnected sets of states, 
    redistribute the weight within each, leaving the total for each disconnected set the same.

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