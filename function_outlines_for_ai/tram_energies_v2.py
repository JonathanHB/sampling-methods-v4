import numpy as np
#add deeptime imports here

def tram_unbiased_energies(transitions, potential_functions, kT):
    """
    Calculate unbiased free energies of bins using TRAM (https://deeptime-ml.github.io/latest/notebooks/tram.html).

    Parameters
    ----------
    transitions: list of 2d numpy arrays of floats
    Each array has a first axis of length 2 and a variable second axis length depending on the number of walkers in that round
    Each column of the array is a transition. The element in the first row is the index of the starting bin, 
    and the element in the second row is the index of the ending bin
    
    potential_functions: list of n_CV_dimensions - dimensional numpy arrays of floats
    Each array is the metadynamics grid containing the metadynamics potential when a particular set of transitions occurred
    The length of this is the number of thermodynamic states

    kT: float
    Boltzmann's constant times temperature

    
    Returns
    -------
    bins: 1d numpy array of ints
    Indices of the the bins for which TRAM potentials could be estimated

    energies: 1d numpy array of floats
    Potential energies for each of the above bins

    """

    #to implement TRAM with the provided arguments:
    # concatenate all the transitions into a single array of shape (total_n_transitions, 2)
    # make the 3d bias_matrices array, indexing potential_functions to get the potential of every sample in every state. 
    # because there are not the same number of samples in all states, some elements of this array will be meaningless.
    # I'm not sure of the top of my head what value to put for those elements.

    return bins, energies