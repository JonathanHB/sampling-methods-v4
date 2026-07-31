import matplotlib.pyplot as plt
import numpy as np
import simulator_classes
import set_params
import estimate_observables
import make_figures
import draft.old_energy_landscapes as old_energy_landscapes

#TODO: fill out an entire project worth of function specs and then give them to Claude and see if it can fill them in

#global variable
kB=1


def multiplot_main_variable_WE():
    """
    Compare the macrostate free energy error of combined WE+MTD simulations for varying WE resampling intervals
    
    Parameters
    ----------
    None

    Returns
    -------
    None

    Files created
    -------------
    Saves a plot of error over time
    TODO: make this function save a numpy file to go with each graph, containing all the WE and MTD parameters
    
    """

    #specify system
    simulation_system = old_energy_landscapes.unit_double_well()

    #general parameters
    n_replicates = 3
    T=1

    #WE parameters
    max_we_rounds = 100
    we_intervals = list(range(100,500,50))
    walkers_per_bin = 6
    n_we_bins = 100

    CV = "some kind of collective variable object"

    #MTD parameters
    #TODO look at the original well-tempered MTD paper and what it says about how to set parameters, specifically delta T
    mtd_params = set_params.set_mtd_params_from_unbiased_literature_advice(simulation_system, T, CV)

    #for the time being stick with 1D CVs


    #initialize systems
    simulator_objects = []
    for wei in we_intervals:
        we_params = (walkers_per_bin, n_we_bins, wei, CV)
        simulator_objects.append(simulator_classes.we_mtd_simulator(wei, T, we_params, mtd_params))

    #run simulation
    conditions_replicate_time, max_we_rounds, time_axis_label = run_macrostate_dg_molecular_time(
        simulator_objects, 
        n_replicates, 
        max_we_rounds, 
        simulation_system.macrostate_classifier)

    #plot results
    make_figures.multiplot_observable_convergence(observables_all_crt = conditions_replicate_time, 
                                     condition_names = [f"t_WE = {wei}" for wei in we_intervals], 
                                     timepoints_all_crt = [wer for wer in range(max_we_rounds)], 
                                     time_axis_label = "WE round", 
                                     plottitle = "Macrostate delta G for variable WE interval", 
                                     true_value = simulation_system.true_macrostate_dg)



#TODO write a variant that saves the run output in a numpy file and another variant that loads them 
# so we can efficiently debug FE calculation and plotting code
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

    observables_all_crt = np.nan*np.ones([len(simulator_objects), n_replicates, max_we_rounds])

    for si, s in enumerate(simulator_objects):
        for ri in range(n_replicates):
            trj, discrete_trj, we_weights, potentials, metadata, kB, T, delta_T = s.run(max_we_rounds)
            fe_by_round = estimate_observables.importance_sampling_fe_by_we_round(trj, discrete_trj, we_weights, potentials, metadata, macrostate_classifier, kB, T, delta_T)
            observables_all_crt[si,ri] = fe_by_round


    return observables_all_crt, max_we_rounds, "WE rounds"