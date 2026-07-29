
#parameters
#   trj_coords: list of floats: initial coordinates of the trajectories on the progress coordinate
#   F: function of a float returning a float: 
#      the negative derivative of free energy function with respect to the progress coordinate as a function of the progress coordinate
#   D: float: brownian diffusion coefficient
#   kT: float: Boltzmann's constant times the temperature
#   timestep: float: the size of the timestep used for propagation
#   nsteps: nonnegative int: how many time steps to propagate for

#returns
#   trj_out: list of arrays: the coordinates of the trajectories at each time step
#      trj_out has size [nsteps//save_period, trj_coords.shape[0], trj_coords.shape[1]]

#Brownian diffusion
#nsteps must be an integer multiple of save_period
def propagate(system, kT, trj_coords, timestep, nsegs, save_period):
  
    nd = np.array(trj_coords.shape) #actually the number of walkers times the number of dimensions   
    D = system.diffusion_coefficient
    
    trj_out = np.zeros((nsegs, trj_coords.shape[0], trj_coords.shape[1]))
    for i in range(nsegs):
    
        for step in range(save_period):
            trj_coords += D/kT * system.F(trj_coords) * timestep + np.sqrt(2*D*timestep)*np.random.normal(size=nd)

        trj_out[i] = trj_coords
        #trj_out.append(trj_coords.copy())

    return trj_out