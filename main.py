import matplotlib.pyplot as plt
import numpy as np
import simulator_classes
import set_params
import estimate_observables
import make_figures
import energy_landscapes
import collective_variables

#TODO: fill out an entire project worth of function specs and then give them to Claude and see if it can fill them in

#global variable
#kB=1


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
    simulation_system = energy_landscapes.diagonal_2well_2d_system

    #general parameters
    n_replicates = 3
    kB = 1
    T=1
    dt=0.01

    #WE parameters
    max_we_rounds = 10
    we_intervals = list(range(20,101,20))
    walkers_per_bin = 6
    n_we_bins = 50

    #MTD parameters
    n_steps_per_frame = 10

    #for the time being stick with 1D CVs
    #used for both WE and MTD, but in principle they could use different CVs
    CV = collective_variables.cv_coord0_2d_coord_1d_cv

    #MTD parameters
    #TODO look at the original well-tempered MTD paper and what it says about how to set parameters, specifically delta T
    delta_T, sigma, omega, n_frames_per_gaussian = set_params.set_mtd_params_from_unbiased_literature_advice(simulation_system, T, CV, dt, n_steps_per_frame)
    mtd_params = {'dt': dt, 
                  'n_steps_per_frame': n_steps_per_frame, 
                  'n_frames_per_gaussian': n_frames_per_gaussian, 
                  'delta_T': delta_T, 'sigma': sigma, 'omega': omega, 
                  'CV': CV}

    #dummy macrostate classifier
    def macrostate_classifier(coords):
        return np.where(coords[:,0] > 0.0, 1, 0)

    #initialize systems
    simulator_objects = []
    for n_gaussians_per_round in we_intervals:
        we_params = {'walkers_per_bin': walkers_per_bin, 
                     'n_we_bins': n_we_bins, 
                     'n_gaussians_per_round': n_gaussians_per_round, 
                     'CV': CV, 'macrostate_classifier': macrostate_classifier}
        simulator_objects.append(simulator_classes.we_mtd_simulator(kB, T, we_params, mtd_params, simulation_system))

    #run simulations and calculate observable
    conditions_replicate_time, timepoints, max_we_rounds, time_axis_label = run_macrostate_dg_molecular_time(
                    simulator_objects, 
                    n_replicates, 
                    max_we_rounds, 
                    macrostate_classifier)

    #plot results
    make_figures.multiplot_observable_convergence(observables_all_crt = conditions_replicate_time, 
                    condition_names = [f"ng_WE = {n_gaussians_per_round}" for n_gaussians_per_round in we_intervals], 
                    timepoints_all_crt = timepoints, 
                    time_axis_label = "WE round", 
                    plottitle = "Macrostate delta G for variable WE interval", 
                    true_value = 0)



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
    timepoints = np.nan*np.ones([len(simulator_objects), n_replicates, max_we_rounds])

    for si, s in enumerate(simulator_objects):
        print(f"running condition {si+1} of {len(simulator_objects)}")
        for ri in range(n_replicates):
            print(f"running replicate {ri+1} of {n_replicates}")

            we_observables = s.run(max_we_rounds)
            # print("obs lengths")
            # print([len(w) for w in we_observables])
            #trj, mtd_weights, we_weights, macrostate_classifier
            fe_by_round = estimate_observables.importance_sampling_fe_by_we_round(we_observables[0], we_observables[1], we_observables[2], macrostate_classifier)
            observables_all_crt[si,ri] = fe_by_round
            timepoints[si,ri] = [wer for wer in range(max_we_rounds)]


    return observables_all_crt, timepoints, max_we_rounds, "WE rounds"