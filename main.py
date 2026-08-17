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
    #simulation_system = energy_landscapes.diagonal_2well_2d_system
    simulation_system = energy_landscapes.cv0_2well_2d_system

    #general parameters
    n_replicates = 1
    n_timepoints = 100
    v_inherit=True

    kB = 1
    T=1

    #timescales
    dt=0.001
    
    n_gpus = 32
    t_wall = 200
    max_t_molecular = t_wall

    #for the time being stick with 1D CVs
    #used for both WE and MTD, but in principle they could use different CVs
    CV = collective_variables.cv_coord0_2d_coord_1d_cv

    #MTD parameters
    #TODO look at the original well-tempered MTD paper and what it says about how to set parameters, specifically delta T
    delta_T, sigma, omega, t_gaussian, walkers_per_bin, bin_width, t_we, t_frame = set_params.set_mtd_params_from_unbiased_literature_advice(simulation_system, kB, T, CV, dt, n_frames=10000, dG_ts = 10*kB*T)

    print("t_f, t_we, t_g")
    print(t_frame, t_we, t_gaussian)

    print("bin width:", bin_width)

    n_steps_per_frame = int(round(t_frame/dt))
    print(f"n_steps_per_frame={n_steps_per_frame}")

    n_frames_per_gaussian = int(round(t_gaussian/t_frame))
    print(f"n_frames_per_gaussian={n_frames_per_gaussian}")

    n_frames_per_we_gaussian = int(round(t_we/t_frame))
    print(f"n_frames_per_we_gaussian={n_frames_per_we_gaussian}")

    n_we_rounds_per_gaussian = int(round(t_gaussian/t_we))
    print(f"n_we_rounds_per_gaussian={n_we_rounds_per_gaussian}")

    omega_wemtd = omega*n_we_rounds_per_gaussian*t_we/t_gaussian
    print(f"omega = {omega}, omega_wemtd = {omega_wemtd}")

    macrostate_classifier = collective_variables.macrostate_classifier_coord0_eq0

    #WE parameters
    # walkers_per_bin = 6
    # n_we_bins = 100
    # if t_molecular/t_gaussian < 100:
    #     print(f"gaussian deposition interval is only {t_gaussian/t_molecular} of the molecular time, \
    #           so we can't test the sampling scheme as intended")
    #     return None
        
    # max_n_gaussians_per_round = int(round(t_wall/t_gaussian))
    #t_we = t_gaussian*max_n_gaussians_per_round*np.array([1, 1/4, 1/16, 1/64, 1/256, 1/1024, 1/4096])

    max_we_rounds = int(round(max_t_molecular/t_we))
    max_gaussians = int(round(max_t_molecular/t_gaussian))
    max_frames = int(round(max_t_molecular/t_frame))

    #np.linspace(t_gaussian, t_molecular, 5)

    #initialize systems

    #at some point we'll dispense with the max() and make this an array for compatibility with 2d binners
    n_we_bins = int(np.max(np.round(np.divide(CV.cv_max-CV.cv_min, bin_width))))

    #use MTD grid resolution matching the WE bins
    #CV.grid_n = n_we_bins
    #simulation_system.grid

    print(f"using {n_we_bins} bins for both WE and MTD grid")


    simulator_objects = []
    for i in range(1):

        #WE+MTD
        if i == 0:
            mtd_params = {'dt': dt, 
                        'n_steps_per_frame': n_steps_per_frame, 
                        'n_frames_per_gaussian': n_frames_per_we_gaussian, 
                        'delta_T': delta_T, 'sigma': sigma, 'omega': omega_wemtd, 
                        'CV': CV, 
                        'v_inherit': v_inherit
                        }
            we_params = {'walkers_per_bin': walkers_per_bin, 
                        'n_we_bins': n_we_bins, 
                        'n_gaussians_per_round': 1,
                        'n_we_rounds_per_gaussian': n_we_rounds_per_gaussian, 
                        't_we': t_we,
                        't_wall': t_wall,
                        'max_we_rounds': max_we_rounds,
                        'n_gpus': n_gpus,
                        'CV': CV, 'macrostate_classifier': macrostate_classifier
                        }

            
        #MTD
        elif i == 1:
            mtd_params = {'dt': dt, 
                        'n_steps_per_frame': n_steps_per_frame, 
                        'n_frames_per_gaussian': n_frames_per_gaussian, 
                        'delta_T': delta_T, 'sigma': sigma, 'omega': omega, 
                        'CV': CV, 
                        'v_inherit': v_inherit
                        }
            we_params = {'walkers_per_bin': -1, 
                        'n_we_bins': -1, 
                        'n_gaussians_per_round': max_gaussians, 
                        'n_we_rounds_per_gaussian': n_we_rounds_per_gaussian, 
                        't_we': t_we,
                        't_wall': t_wall,
                        'max_we_rounds': 1,
                        'n_gpus': n_gpus,
                        'CV': CV, 'macrostate_classifier': macrostate_classifier
                        }  
            #simulator_objects.append(simulator_classes.we_mtd_simulator(kB, T, we_params, mtd_params, simulation_system))
  
            #simulator_objects.append(simulator_classes.mtd_simulator(kB, T, mtd_params, max_gaussians, n_gpus, simulation_system))


        #WE
        if i == 2:
            mtd_params = {'dt': dt, 
                        'n_steps_per_frame': n_steps_per_frame, 
                        'n_frames_per_gaussian': n_frames_per_we_gaussian, 
                        'delta_T': delta_T, 'sigma': sigma, 'omega': 0, 
                        'CV': CV, 
                        'v_inherit': v_inherit
                        }
            we_params = {'walkers_per_bin': walkers_per_bin, 
                        'n_we_bins': n_we_bins, 
                        'n_gaussians_per_round': 1, 
                        'n_we_rounds_per_gaussian': n_we_rounds_per_gaussian, 
                        't_we': t_we,
                        't_wall': t_wall,
                        'max_we_rounds': max_we_rounds,
                        'n_gpus': n_gpus,
                        'CV': CV, 'macrostate_classifier': macrostate_classifier
                        }
            #simulator_objects.append(simulator_classes.we_mtd_simulator(kB, T, we_params, mtd_params, simulation_system))


        #unbiased
        elif i == 3:
            mtd_params = {'dt': dt, 
                        'n_steps_per_frame': n_steps_per_frame, 
                        'n_frames_per_gaussian': max_frames, 
                        'delta_T': delta_T, 'sigma': sigma, 'omega': 0, 
                        'CV': CV, 
                        'v_inherit': v_inherit
                        }
            we_params = {'walkers_per_bin': -1, 
                        'n_we_bins': -1, 
                        'n_gaussians_per_round': 1, 
                        'n_we_rounds_per_gaussian': n_we_rounds_per_gaussian, 
                        't_we': t_we,
                        't_wall': t_wall,
                        'max_we_rounds': 1,
                        'n_gpus': n_gpus,
                        'CV': CV, 'macrostate_classifier': macrostate_classifier
                        }                   

        simulator_objects.append(simulator_classes.we_mtd_simulator(kB, T, we_params, mtd_params, simulation_system))
                          


    max_analysis_frames = int(round(t_wall/t_frame))
    aggregate_time_increment = int(round(t_wall*n_gpus/(t_frame*n_timepoints)))
    #print(f"ati = {aggregate_time_increment}")

    #run simulations and calculate observable
    conditions_replicate_time, timepoints, conditions_replicate_time2, timepoints2, time_axis_label = run_macrostate_dg_molecular_time(
                    simulator_objects, 
                    n_replicates, 
                    macrostate_classifier,
                    n_timepoints,
                    max_analysis_frames,
                    aggregate_time_increment)

    #plot results
    # make_figures.multiplot_observable_convergence(observables_all_crt = conditions_replicate_time, 
    #                 condition_names = [f"t_WE = {t_we_i:.1f}" for t_we_i in t_we_round], 
    #                 timepoints_all_crt = timepoints, 
    #                 time_axis_label = "molecular time", 
    #                 plottitle = "Macrostate delta G for variable WE interval", 
    #                 true_value = 0)

    serial = "11"#1_cv_exact_noinh_100bins"

    #plot results
    make_figures.multiplot_observable_convergence(observables_all_crt = conditions_replicate_time, 
                    condition_names = ["WE+MTD", "MTD", "WE", "unbiased"],#[f'$t_{{WE}}$={t_we_i:.2f}' for t_we_i in [t_we]], 
                    timepoints_all_crt = timepoints, 
                    time_axis_label = "molecular time", 
                    savetitle = f"Macrostate delta G for variable WE interval molecular time v{serial}",
                    plottitle = f"$\\Delta G_{{mac}}(t_{{WE}})$ vs $t_{{mol}}$", # \n $V_{{inherit}}=${v_inherit}
                    true_value = 0,
                    we_round_lengths = [t_we, t_we*max_we_rounds], 
                    plot_we_boundaries=False)

    #plot results
    make_figures.multiplot_observable_convergence(observables_all_crt = conditions_replicate_time2, 
                    condition_names = ["WE+MTD", "MTD", "WE", "unbiased"], #[f'$t_{{WE}}$={t_we_i:.2f}' for t_we_i in [t_we]], 
                    timepoints_all_crt = timepoints2, 
                    time_axis_label = "aggregate time",
                    savetitle = f"Macrostate delta G for variable WE interval aggregate time v{serial}", 
                    plottitle = f"$\\Delta G_{{mac}}(t_{{WE}})$ vs $t_{{agg}}$", # \n $V_{{inherit}}=${v_inherit}
                    true_value = 0,
                    we_round_lengths = [t_we, t_we*max_we_rounds])


#TODO write a variant that saves the run output in a numpy file and another variant that loads them 
# so we can efficiently debug FE calculation and plotting code
def run_macrostate_dg_molecular_time(simulator_objects, n_replicates, macrostate_classifier, n_timepoints, max_n_frames, aggregate_time_increment):
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
    #n_timepoints = 10
    observables_all_crt = np.nan*np.ones([len(simulator_objects), n_replicates, n_timepoints])
    timepoints = np.nan*np.ones([len(simulator_objects), n_replicates, n_timepoints])

    observables_all_crt2 = np.nan*np.ones([len(simulator_objects), n_replicates, n_timepoints])
    timepoints2 = np.nan*np.ones([len(simulator_objects), n_replicates, n_timepoints])

    # observables_all_crt = np.nan*np.ones([len(simulator_objects), n_replicates, max_we_rounds])
    # timepoints = np.nan*np.ones([len(simulator_objects), n_replicates, max_we_rounds])

    for si, s in enumerate(simulator_objects):
        print(f"-- running condition {si+1} of {len(simulator_objects)}")
        print(f"with n_gaussians_per_round = {s.we_params['n_gaussians_per_round']}")
        #print(f"with n_we_rounds = {s.we_params['n_we_rounds']}")

        for ri in range(n_replicates):
            print(f"running replicate {ri+1} of {n_replicates}")

            we_observables = s.run()
            # print("obs lengths")
            # print([len(w) for w in we_observables])
            # trj, mtd_weights, we_weights, macrostate_classifier

            # fe_by_round = estimate_observables.importance_sampling_fe_by_we_round(we_observables[0], we_observables[1], we_observables[2], macrostate_classifier, sim_system=s.energy_landscape, CV=s.we_params["CV"], kT=s.kB*s.T)          
            # observables_all_crt[si,ri,:s.we_params['n_we_rounds']] = fe_by_round
            # timepoints[si,ri,:s.we_params['n_we_rounds']] = [wer*s.we_round_length for wer in range(1,s.we_params['n_we_rounds']+1)]

            fe_by_mol_timepoint = estimate_observables.importance_sampling_fe_by_molecular_timepoint(we_observables[0], we_observables[1], we_observables[2], macrostate_classifier, sim_system=s.energy_landscape, CV=s.we_params["CV"], kT=s.kB*s.T, n_timepoints=n_timepoints, max_n_frames=max_n_frames)
            observables_all_crt[si,ri] = fe_by_mol_timepoint
            timepoints[si,ri] = [s.we_params['t_wall'] * (ti+1)/n_timepoints for ti in range(n_timepoints)]

            fe_by_agg_timepoint = estimate_observables.importance_sampling_fe_by_aggregate_timepoint(we_observables[0], we_observables[1], we_observables[2], macrostate_classifier, sim_system=s.energy_landscape, CV=s.we_params["CV"], kT=s.kB*s.T, n_timepoints=n_timepoints, aggregate_time_increment=aggregate_time_increment)
            observables_all_crt2[si,ri] = fe_by_agg_timepoint
            timepoints2[si,ri] = [s.mtd_params['dt']*s.mtd_params['n_steps_per_frame'] * aggregate_time_increment*(ti+1) for ti in range(n_timepoints)]

            #timepoints[si,ri] = [s.we_params['n_we_rounds']*s.we_round_length * (ti+1)/n_timepoints for ti in range(n_timepoints)]

    return observables_all_crt, timepoints, observables_all_crt2, timepoints2, "WE rounds"