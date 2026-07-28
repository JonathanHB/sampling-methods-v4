import matplotlib.pyplot as plt
import numpy as np

#TODO: fill out an entire project worth of function specs and then give them to Claude and see if it can fill them in


def multiplot_observable_convergence(conditions_replicate_time, condition_names, timepoints, time_axis_label):
    """plot observables over time across multiple replicates and conditions
    
    Parameters
    ----------
    conditions_replicate_time: numpy array of shape (n_conditions, n_replicates, n_timepoints)
        The data to be plotted. Each entry in the array is an observable value. 
        If there is no data the value should be nan.
    
    condition_names: tuple of strings, of length n_conditions
        what to label each subplot

    timepoints: numpy array of times, of length n_timepoints
        The x axis data for each plot
    
    time_axis_label: string
        Time axis label. Used to provide units and distinguish aggregate and molecular time.

    Returns
    -------
    None
        The purpose of the function is to save a plot
    
    """
    pass


def run_macrostate_dg_molecular_time(simulator_objects, condition_names, n_replicates, max_n_timepoints):
    """plot macrostate free energy estimates over molecular time across multiple replicates and conditions
    
    Parameters
    ----------
    simulator_objects: tuple of simulator objects, of length n_conditions
    
    condition_names: tuple of strings, of length n_conditions
        what to label each subplot

    n_replicates: how many simulation replicates to run


    Returns
    -------
    conditions_replicate_time: numpy array of shape (n_conditions, n_replicates, n_timepoints)
        The data to be plotted. Each entry in the array is an observable value. 
        If there is no data the value should be nan.
    
    timepoints: numpy array of times, of length n_timepoints
        The x axis data for each plot

    time_axis_label: string
        Time axis label. Used to provide units and distinguish aggregate and molecular time.

    """

    conditions_replicate_time = np.nan*np.ones([len(simulator_objects), n_replicates, max_n_timepoints])

    pass

