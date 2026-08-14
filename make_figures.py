import matplotlib.pyplot as plt
import numpy as np
import simulator_classes
import set_params

#TODO: fill out an entire project worth of function specs and then give them to Claude and see if it can fill them in


def multiplot_observable_convergence(observables_all_crt, condition_names, timepoints_all_crt, time_axis_label, savetitle, plottitle, true_value, we_round_lengths, plot_we_boundaries=False):
    """plot observables over time across multiple replicates and conditions
    
    Parameters
    ----------
    observables_all_crt: 3d numpy array of floats
        of shape (n_conditions, n_replicates, n_timepoints)
        The data to be plotted. Each entry in the array is an observable value. 
        If there is no data the value should be nan.
    
    condition_names: tuple of strings
        of length n_conditions
        what to label each subplot

    timepoints_all_crt: 3d numpy array of floats
        of shape (n_conditions, n_replicates, n_timepoints)
        The x axis data for each simulation
        This needs to be a 3d array because different WE simulations can have different round lengths, 
        leading to observables being saved at different times
    
    time_axis_label: string
        Time axis label. Used to provide units and distinguish aggregate and molecular time.
        This is the x axis label and need not actually be related to time.
    
    plottitle: string
        what to call the saved plot file

    true_values: float
        the true value of the observable

    Returns
    -------
    None
        The purpose of the function is to save a plot
    
    """

    #create plot
    fig, ax = plt.subplots(len(observables_all_crt), sharex='col', figsize=(5,10))

    #loop over conditions
    for i, (observables_all_rt, timepoints_all_rt, condition_name) in enumerate(zip(observables_all_crt, timepoints_all_crt, condition_names)):
        #print(f"we round length = {we_round_lengths[i]}")
        if plot_we_boundaries:
            for we_i in range(int(np.ceil(max(timepoints_all_crt.flatten())/we_round_lengths[i]))):
                #print(f"we round = {we_i}")
                #print(f"we round time = {we_i*we_round_lengths[i]}")
                
                ax[i].axvline(we_i*we_round_lengths[i], color="black", linewidth=0.1)
                #ax[i].axvline(10, color="black", linestyle="dotted")

        #loop over replicas
        for ri, (observables_all_t, timepoints_all_t) in enumerate(zip(observables_all_rt, timepoints_all_rt)):

            #plot data from each replicate
            #ax[i].scatter(timepoints_all_t, observables_all_t, alpha=0.5)
            ax[i].plot(timepoints_all_t, observables_all_t, alpha=0.7)
            # print(np.isnan(observables_all_t))
            # print(np.where(np.isnan(observables_all_t)))
            # print(timepoints_all_t[np.where(np.isnan(observables_all_t))])

            plot_nans = False
            if plot_nans:
                nan_times = timepoints_all_t[np.where(np.isinf(observables_all_t))]
                for nt in nan_times:
                    ax[i].axvline(nt, color = f'C{ri}', alpha=0.5)

            # print(f"{len(nan_times)} nan_times")
            # # if len(nan_times>0):
            # #     ax[i].axvline(nan_times)
            # # else:
            # print(observables_all_t)

            ax[i].axhline(true_value, linestyle="dashed", color="black")

            #label y axis
            ax[i].set_ylabel(condition_name)

            #set axis limits
            ax[i].set_xlim(0, max(timepoints_all_crt.flatten())) #use overall maximum time
            ax[i].set_ylim(-20,2)

    #overall plot labels and scaling
    plt.subplots_adjust(hspace=0.13, wspace=0.1, top=0.6, bottom=0, left=0, right=0.8)
    plt.xlabel(time_axis_label)

    ax[0].set_title(plottitle)

    #save and display plot
    plt.savefig(f"figures/{savetitle}.png", bbox_inches="tight", dpi=600)
    plt.show()
