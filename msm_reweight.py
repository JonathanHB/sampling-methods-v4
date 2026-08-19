import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import normalize
import deeptime as deeptime
import matplotlib.pyplot as plt

#TODO get rid of walkers_per_bin argument if we don't find a need for it
def msm_reweight(transitions_we_weights, w, b, walkers_per_bin, r, n_rounds_per_gaussian):
    """
    Reweight weighted ensemble walkers using a Markov state model.
    For the sake of efficient testing fit the analytically defined non-reversible maximum likelihood MSM. Make this its own function to the extent possible.
    Use the deeptime package if/where it is suitable.
    In the case that there are multiple disconnected sets of states, 
    redistribute the weight within each, leaving the total for each disconnected set the same.

    Parameters
    ----------
    transitions_we_weights: list of numpy arrays
    Each array is of shape (3, number of walkers in round i). 
    The indices along axis 0 are:
    0: the transition starting bin
    1: transition ending bin
    2: the WE weight of the walker at the start of the transition
    The arrays should probably be concatenated along axis 1 into single array of shape (3, n_transitions)

    w: numpy array of floats
    The WE weight of each current walker, needed only if there are multiple disconnected sets of states

    b: numpy array of ints
    The bin occupied by each current walker.

    walkers_per_bin: int
    Target number of walkers per bin, technically inferrable from the count of each index in b but that's ugly

    Returns
    -------
    w_out: numpy array of floats
    The new WE weights of the current walkers, calculated as described below, in the same order as w

    """

    #print("-----------------reweight--------------")

    ##########################################################
    # 1. USE MSM TO ESTIMATE THE EQUILIBRIUM BIN PROBABILITIES

    #This section should use:
    # transitions
    # w
    # b

    transitions_all = np.concatenate(transitions_we_weights, axis=1).transpose()

    max_bin_index = int(np.max(transitions_all[:,:2]))+1

    #total WE weight in each bin as of the most recent WE round
    w_by_bin = np.zeros(max_bin_index)
    for ib in np.unique(b):
        bin_inds = np.where(b==ib)[0]
        w_by_bin[ib] = np.sum(w[bin_inds])


    #counts matrix for MSM
    counts = np.zeros((max_bin_index, max_bin_index))
    for (start, end, weight) in transitions_all:
        counts[int(end),int(start)] += weight

    #n_cc: int; number of connected components
    #cc: array of ints; which component each state belongs to
    n_cc, cc = connected_components(counts, directed=True, connection='strong')
    # print(n_cc)
    # print(cc)

    bins_all = []
    bin_weights_all = []

    #cwa = [] for debug

    for i in range(n_cc):
        #print(f"----------component {i}")
        inds = np.where(cc==i)

        #-----------------------------------------
        #bins of current component
        bins_all.append(inds[0])

        #-----------------------------------------
        #total WE weight of current component
        component_weight = np.sum(w_by_bin[inds])
        #print(f"w = {component_weight}")
        #cwa.append(component_weight) #for debug

        #-----------------------------------------
        #build MSM for current component

        #get block of the counts matrix belonging to current component
        # which need not actually be organized as a block since rows and columns can be permuted together without altering the dynamics
        # the last indexing [:,0,:] is to get rid of an excess array layer
        component_counts = counts[inds][:,inds][:,0,:]

        #if there are multiple bins build an MSM
        if len(component_counts)>1:
            #normalize to rate matrix and calculate equilibrium probabilities
            
            #normalize transition count matrix to transition rate matrix
            #each column (aka each feature in the documentation) is normalized so that its entries add to 1, 
            # so that the probability associated with each element of X(t) is preserved when X(t) is multiplied by the TPM
            tpm = normalize(component_counts, axis=0, norm='l1')

            eigenvectors = deeptime.markov.tools.analysis.eigenvectors(tpm)
            eq_probs = np.real(eigenvectors[:,0]/np.sum(eigenvectors[:,0]))

            #-----------------------------------------
            #reweighted weights of current component
            bin_weights_all.append(eq_probs*component_weight)

        #if there is only one bin its weight is unchanged
        else:
            bin_weights_all.append(np.array([component_weight]))

    #print(np.sum(cwa))

    bins_all = np.concatenate(bins_all)
    bin_weights_all = np.concatenate(bin_weights_all)

    #normalize since the MSM can give probability to currently-empty bins
    #bin_weights_all /= np.sum(bin_weights_all)

    # print(bins_all)
    # print(bin_weights_all)

    # plt.scatter(bins_all, bin_weights_all)
    # plt.show()


    ##########################################################
    #2. ADJUST WEIGHTED ENSEMBLE WEIGHTS SO THAT THE TOTAL WEIGHT IN EACH BIN EQUALS THE MSM-ESTIMATED BIN PROBABILITY

    #this section should use:
    # the unbiased bin probabilities from section 1
    # b
    # walkers_per_bin <not anymore

    if len(bins_all) != len(np.unique(bins_all)):
        print("error: overlapping 'disconnected' sets")
        raise ValueError

    #this may be implementable without the loop but it's not worth the trouble for now
    w_msm = np.zeros(len(w))
    for i, bi in enumerate(b):
        #print(np.where(bins_all==bi)[0][0])
        #zeros get rid of excess array layers
        #the denominator equals walkers_per_bin if the data was just resampled, but not if we're reweighting un-resampled MTD (non-WE) data
        w_msm[i] = bin_weights_all[np.where(bins_all==bi)[0][0]]/np.count_nonzero(b == bi)

    average_old_msm_weights=False
    if average_old_msm_weights:
        #average old and new weights to get smooth convergence of instantaneous weights
        w_out = (w*(r-n_rounds_per_gaussian) + w_msm*n_rounds_per_gaussian)/r
    else:
        w_out=w_msm

    #ensure normalized weights
    w_out /= np.sum(w_out)

    return w_out