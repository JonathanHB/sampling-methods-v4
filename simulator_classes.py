import numpy as np
import weighted_ensemble_v2
from propagators_grid import propagate_shared_grid


class we_mtd_simulator:

    def __init__(self, kB, T, we_params, mtd_params, energy_landscape):
        """
        Parameters
        ----------
        kB: float
            Boltzmann's constant
        T: float
            Temperature

        we_params: dictionary
            Parameters for weighted ensemble
            (
            walkers_per_bin: int,
            n_we_bins: int, 
            n_gaussians_per_round: int, 
                number of gaussians to add to the MTD potential per WE round
            t_wall: int,
            max_we_rounds: int,
            n_gpus: int,
            CV: collective_variable() object from collective_variables.py
            macrostate_classifier: a macrostate_classifier() object from macrostate_classifiers.py
                YET TO BE ACTUALLY IMPLEMENTED; currently uses a dummy function
            )

        mtd_params: dictionary
            Parameters for metadynamics
            (
            dt: float
                Time step for dynamics
            n_steps_per_frame: int,
            n_frames_per_gaussian: int, 
            delta_T: float,
                MTD tempering parameter
            sigma: float,
                MTD gaussian width parameter
            omega: float,
                MTD gaussian height parameter
            CV: collective_variable() object from collective_variables.py
            v_inherit: bool
                whether WE aalkers inherit the MTD potential from the previous round, using a sum weighted by WE weights
            )
        
        energy_landscape: energy_landscape() object from energy_landscapes.py
                
        macrostate_classifier: object
            The macrostate classifier object
        """

        self.kB = kB
        self.T = T
        self.we_params = we_params
        self.mtd_params = mtd_params
        self.energy_landscape = energy_landscape
        self.we_round_length = self.we_params["t_we"]
        #self.we_params["n_gaussians_per_round"]*self.mtd_params["n_frames_per_gaussian"]*self.mtd_params["n_steps_per_frame"]*self.mtd_params["dt"]


    def run(self):
        """
        Run a WE+MTD simulation for a specified number of WE rounds
        
        Parameters
        ----------
        n_we_rounds: int
            Number of WE rounds to run

        Returns
        -------
        trj: 4d numpy array of floats
            of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, n_coordinates)
            Each element is a coordinate. NaN values denote walkers which did not exist 
            (i.e. not all bins were filled at the start of the round so less than the maximum number of walkers were spawned)

        discrete_trj: 4d numpy array of ints
            of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, n_coordinates)
            Each element is an MTD grid index along one axis. NaN values are distributed as above.
            This is the discrete version of trj

        we_weights: 2d numpy array of floats
            of shape (n_we_rounds, max_n_walkers)
            Each element is the WE weight of a walker. NaNs as above

        potentials: (3+n_CV_dimensions)d numpy array of floats
            of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, [MTD grid dimensions])
            Each element is an MTD potential value at a specific walker, time, and place
            NaN values are distributed as above.
            Because the MTD potential is not usually updated for every saved frame, this contains some redundant information

        metadata: 2d numpy array of ints
            of shape (n_we_rounds, max_n_walkers)
            Each element is 1 if the walker was spawned, 0 otherwise. This should match the NaNs in the above two parameters.
        
        """

        #initialize instances of classes
        config_binner = weighted_ensemble_v2.config_binner_1(np.linspace(self.energy_landscape.coord_min[0], self.energy_landscape.coord_max[0], self.we_params['n_we_bins']+1), self.we_params['CV'])
        ensemble_classifier = weighted_ensemble_v2.ensemble_classifier_1(self.we_params['macrostate_classifier'])
        binner = weighted_ensemble_v2.binner_1()
        propagator0 = weighted_ensemble_v2.we_propagator_2(self.energy_landscape, self.kB, self.T, self.mtd_params, self.we_params['n_gaussians_per_round'])

        #initial state
        x = -np.ones((self.we_params['n_gpus'], self.energy_landscape.n_dim))
        #x0 = np.random.uniform(-1.5, -0.5, size=(n_walkers, sim_system.n_dim))
        #standard_init_coord = np.array([-1,-1])
        #x0 = np.array([standard_init_coord for element in range(walkers_per_bin)]) #.reshape((walkers_per_bin, 1, len(system.standard_init_coord)))
        e = self.we_params["macrostate_classifier"](x) #initial ensemble is determined by the macrostate classifier
        w = np.ones(self.we_params['n_gpus'])/self.we_params['n_gpus']  
        ##[1/walkers_per_bin for element in range(walkers_per_bin)]
        cb = config_binner.bin(x)  #configurational bins
        b = binner.bin(cb, e)
        #prop_out_0 = [1 for element in range(walkers_per_bin)]

        # print(x)
        # print(x.shape)
        # print(w)
        # print(w.shape)
        # print(e)
        # print(e.shape)
        # print(cb)
        # print(cb.shape)
        # print(b)
        # print(b.shape)

        n_gpu_rounds_t_wall = int(np.round(self.we_params["t_wall"]/self.we_round_length))

        x, e, w, cb, b, propagator, observables = weighted_ensemble_v2.weighted_ensemble(x, e, w, cb, b, 
                                                                                         propagator0, 
                                                                                         weighted_ensemble_v2.resample, 
                                                                                         config_binner, 
                                                                                         ensemble_classifier, 
                                                                                         binner, 
                                                                                         weighted_ensemble_v2.calc_observables_2, 
                                                                                         self.we_params['max_we_rounds'], self.we_params['n_we_rounds_per_gaussian'], self.we_params['walkers_per_bin'], n_gpu_rounds_t_wall, self.we_params['n_gpus'])

        #effectively transpose the list of lists so the first axis is observable type rather than time
        #but without the data type/structure requirement of a numpy array
        observables_over_time = [list(row) for row in zip(*observables)]

        # print("obs lengths 2")
        # print([len(lw) for lw in observables_over_time])
        return observables_over_time



class mtd_simulator:

    def __init__(self, kB, T, mtd_params, n_gaussians, n_gpus, energy_landscape):
        """
        Parameters
        ----------
        kB: float
            Boltzmann's constant
        T: float
            Temperature

        mtd_params: dictionary
            Parameters for metadynamics
            (
            dt: float
                Time step for dynamics
            n_steps_per_frame: int,
            n_frames_per_gaussian: int, 
            delta_T: float,
                MTD tempering parameter
            sigma: float,
                MTD gaussian width parameter
            omega: float,
                MTD gaussian height parameter
            CV: collective_variable() object from collective_variables.py
            v_inherit: bool
                whether WE walkers inherit the MTD potential from the previous round, using a sum weighted by WE weights
            )
        
        energy_landscape: energy_landscape() object from energy_landscapes.py
                
        macrostate_classifier: object
            The macrostate classifier object
        """

        self.kB = kB
        self.T = T
        self.mtd_params = mtd_params
        self.energy_landscape = energy_landscape
        self.n_gaussians = n_gaussians
        self.n_gpus = n_gpus
        self.we_parameters = n_gaussians_per_round = self.n_gaussians


    def run(self):
        """
        Run a WE+MTD simulation for a specified number of WE rounds
        
        Parameters
        ----------
        n_we_rounds: int
            Number of WE rounds to run

        Returns
        -------
        trj: 4d numpy array of floats
            of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, n_coordinates)
            Each element is a coordinate. NaN values denote walkers which did not exist 
            (i.e. not all bins were filled at the start of the round so less than the maximum number of walkers were spawned)

        discrete_trj: 4d numpy array of ints
            of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, n_coordinates)
            Each element is an MTD grid index along one axis. NaN values are distributed as above.
            This is the discrete version of trj

        we_weights: 2d numpy array of floats
            of shape (n_we_rounds, max_n_walkers)
            Each element is the WE weight of a walker. NaNs as above

        potentials: (3+n_CV_dimensions)d numpy array of floats
            of shape (n_we_rounds, max_n_walkers, n_frames_per_we_round, [MTD grid dimensions])
            Each element is an MTD potential value at a specific walker, time, and place
            NaN values are distributed as above.
            Because the MTD potential is not usually updated for every saved frame, this contains some redundant information

        metadata: 2d numpy array of ints
            of shape (n_we_rounds, max_n_walkers)
            Each element is 1 if the walker was spawned, 0 otherwise. This should match the NaNs in the above two parameters.
        
        """

        #initial state
        x = -np.ones((self.n_gpus, self.energy_landscape.n_dim))
        w = np.ones(self.n_gpus)/self.n_gpus

        traj, pots, weights = propagate_shared_grid(
            G=self.system.G, kB=self.kB, T = self.T, dt=self.mtd_params["dt"], xi = self.system.xi, 
            init_coords=x, init_potentials=self.potential, we_weights = w,
            steps_per_saved_frame=self.mtd_params["n_steps_per_frame"],
            n_gaussians=self.n_gaussians, 
            frames_per_gaussian=self.mtd_params["n_frames_per_gaussian"],
            sigma=self.mtd_params["sigma"], omega=self.mtd_params["omega"], delta_T=self.mtd_params["delta_T"],
            CV=self.CV.cv_funct, grad_CV=self.CV.cv_grad_funct, 
            cv_min=self.CV.cv_min, cv_max=self.CV.cv_max
        )



        #effectively transpose the list of lists so the first axis is observable type rather than time
        #but without the data type/structure requirement of a numpy array
        observables_over_time = (traj, weights, w)

        # print("obs lengths 2")
        # print([len(lw) for lw in observables_over_time])
        return observables_over_time