import numpy as np
import matplotlib.pyplot as plt
import collective_variables
import energy_landscapes
from propagators_grid import propagate
import visualization
import collective_variable_analysis


#TODO: rewrite this function (test_free_energy_on_cv_grid) to check collective_variable_analysis.free_energy_on_cv_grid() 
# against an analytically solvable case insread of another implementation of the same calculation.

#written by copilot
#CURRENTLY NOT RUNNABLE; SEE ABOVE
def test_free_energy_on_cv_grid():
    # def free_energy_fn(coords):
    #     return (coords[:, 0] - 1.0) ** 2 + 0.5 * coords[:, 1] ** 2

    # def cv_fn(coords):
    #     return np.asarray(coords, dtype=float)[:, :1]

    kT=2

    sim_system = energy_landscapes.diagonal_2well_2d_system
    CV = collective_variables.cv_coord0_2d_coord_1d_cv

    cv_grid, free_energy_grid = collective_variable_analysis.free_energy_on_cv_grid(
            free_energy_fn=sim_system.G,
            coord_min=sim_system.coord_min,
            coord_max=sim_system.coord_max,
            n_micro_grid=sim_system.grid_n,
            cv_fn=CV.cv_funct,
            cv_min=CV.cv_min,
            cv_max=CV.cv_max,
            n_cv_grid=CV.grid_n,
            kT = kT
        )

    print(cv_grid.shape)
    print(free_energy_grid.shape)

    assert cv_grid.shape == (CV.grid_n, 1)
    assert free_energy_grid.shape == (CV.grid_n,)

    micro_grid = np.linspace(sim_system.coord_min[1], sim_system.coord_max[1], sim_system.grid_n)
    expected = []
    for cv_value in cv_grid[:, 0]:
        values = (cv_value - 1.0) ** 2 + 0.5 * micro_grid ** 2 #contains a bug; see above
        expected.append(-kT*np.log(np.mean(np.exp(-values/kT))))

    plt.plot(cv_grid, free_energy_grid, label='Computed Free Energy')
    plt.plot(cv_grid, expected, linestyle='dashed', color='red', label='Expected Free Energy')

    assert np.allclose(free_energy_grid, np.asarray(expected))

    if np.allclose(free_energy_grid, np.asarray(expected)):
        print("Test passed: free_energy_on_cv_grid() matches gridcalculations in this test function; no dynamics were run.")



def test_propagators_grid_1d_cv():

    n_walkers = 4

    T=1
    kB=1
    kT = kB*T

    sim_system = energy_landscapes.diagonal_2well_2d_system

    CV = collective_variables.cv_coord0_2d_coord_1d_cv

    cv_grid, free_energy_grid = collective_variable_analysis.free_energy_on_cv_grid(
            free_energy_fn=sim_system.G,
            coord_min=sim_system.coord_min,
            coord_max=sim_system.coord_max,
            n_micro_grid=sim_system.grid_n,
            cv_fn=CV.cv_funct,
            cv_min=CV.cv_min,
            cv_max=CV.cv_max,
            n_cv_grid=CV.grid_n,
            kT = kT
        )

    plt.plot(cv_grid, free_energy_grid)
    plt.xlabel('CV')
    plt.ylabel('Free Energy(CV)')

    init_coords = np.random.uniform(-1.5, 1.5, size=(n_walkers, sim_system.n_dim))
    init_potentials = np.zeros((n_walkers, CV.grid_n))

    sigma = np.array([0.2])
    delta_T = 40
    omega = 0.4

    visualization.plot_G_surface(sim_system.G, x_limits = ((sim_system.coord_min[0], sim_system.coord_max[0]), (sim_system.coord_min[1], sim_system.coord_max[1])), n=200, center=None, slice_axis=0)

    traj, pots, weights = propagate(
        G=sim_system.G, kB=kB, T=T, dt=0.01, xi = sim_system.xi, init_coords=init_coords,
        init_potentials=init_potentials, steps_per_saved_frame=100,
        n_gaussians=300, frames_per_gaussian=10,
        CV=CV.cv_funct, grad_CV=CV.cv_grad_funct, sigma=sigma, omega=omega, delta_T=delta_T,
        cv_min=CV.cv_min, cv_max=CV.cv_max
    )

    print('trajectories shape', traj.shape)
    print('potentials shape', pots.shape)
    print('sample final coords', traj[:, -1, :])
    #print('max bias height walker0 over time', pots[0, :, :].max(axis=-1))
    print(np.isfinite(traj).all(), np.isfinite(pots).all())

    def plot_trj(trj, pot):
        
        print(trj[0].shape)
        plt.plot(trj[:,0], zorder=100, alpha=0.5, label='x0=CV')
        plt.plot(trj[:,1], alpha=0.5, label='x1')
        plt.xlabel('frame')
        plt.ylabel("position")
        plt.legend()
        plt.show()

        for i, Vi in enumerate(pot):
            if i%10==0:
                plt.plot(cv_grid, -Vi*(delta_T+T)/delta_T + kT*np.log(np.sum(np.exp(Vi*((delta_T+T)/delta_T)/kT))), color = [0.3+0.7*(i/len(pot)), 0.3, 0.3])

        plt.plot(cv_grid, free_energy_grid + kT*np.log(np.sum(np.exp(-free_energy_grid/kT))), linestyle="dashed", color="black", linewidth="3")

        plt.xlabel("CV")
        plt.ylabel('Free Energy')
        plt.show()

    plot_trj(traj[0], pots[0])
    plot_trj(traj[1], pots[1])

    plt.plot(traj[0,:,0], traj[0,:,1])
    plt.title('Walker 0 trajectory in 2D space')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.show()

    #pops = np.histogram2d(traj[0,:,0], traj[0,:,1], bins=CV.grid_n, range=[[CV.cv_min[0], CV.cv_max[0]], [CV.cv_min[1], CV.cv_max[1]]], density=True)
    pops_1d = np.histogram(traj[0,:,0], bins=CV.grid_n, range=(CV.cv_min[0], CV.cv_max[0]), weights=weights[0,:])

    plt.plot(cv_grid, -kT*np.log(pops_1d[0]/np.sum(pops_1d[0])), label = "importance sampling FE estimate")

    fe_norm = free_energy_grid + kT*np.log(np.sum(np.exp(-free_energy_grid/kT)))

    plt.plot(cv_grid, fe_norm, linestyle="dashed", color="black", linewidth="3", label="true FE")

    plt.legend()
    plt.xlabel("CV")
    plt.ylabel("Free Energy (kT)")

    plt.show()

def test_propagators_grid_2d_cv():

    n_walkers = 4

    T=1
    kB=1
    #kT = kB*T

    sim_system = energy_landscapes.diagonal_2well_2d_system
    CV2 = collective_variables.cv_coord01_2d_coord_2d_cv
    
    sigma2 = np.array([0.2, 0.2])
    grid_shape2 = (25, 25)
    init_coords = np.random.uniform(-1.5, 1.5, size=(n_walkers, sim_system.n_dim))
    init_potentials2 = np.zeros((n_walkers, *grid_shape2))


    traj2, pots2, weights2 = propagate(
        G=sim_system.G, kB=kB, T=T, dt=0.001, xi = sim_system.xi, init_coords=init_coords,
        init_potentials=init_potentials2, steps_per_saved_frame=100,
        n_gaussians=300, frames_per_gaussian=10,
        CV=CV2.cv_funct, grad_CV=CV2.cv_grad_funct, sigma=sigma2, omega=2, delta_T=40.0,
        cv_min=CV2.cv_min, cv_max=CV2.cv_max
    )

    print('2cv trajectories shape', traj2.shape)
    print('2cv potentials shape', pots2.shape)
    print(np.isfinite(traj2).all(), np.isfinite(pots2).all())

    plt.plot(traj2[0,:,0], traj2[0,:,1])
    plt.title('Walker 0 trajectory in 2D space (2 CVs)')
    plt.xlabel('x0=CV0')
    plt.ylabel('x1=CV1')
    plt.show()

    plt.imshow(pots2[0,-1], extent=(CV2.cv_min[0], CV2.cv_max[0], CV2.cv_min[1], CV2.cv_max[1]), origin='lower', aspect='auto')
    plt.colorbar(label='Bias Potential (kT)')
    plt.title("bias potential at final frame for walker 0 (2 CVs)")
    plt.xlabel('x0=CV0')
    plt.ylabel('x1=CV1')
    plt.show()

    plt.imshow(pots2[0,150], extent=(CV2.cv_min[0], CV2.cv_max[0], CV2.cv_min[1], CV2.cv_max[1]), origin='lower', aspect='auto')
    plt.colorbar(label='Bias Potential (kT)')
    plt.title("bias potential at frame 150 (halfway) for walker 0 (2 CVs)")
    plt.xlabel('x0=CV0')
    plt.ylabel('x1=CV1')
    plt.show()

    print("note that different-dimensional CVs may require different omega values to be effective, (copilot: 'and the bias potential may not be as smooth in higher dimensions due to fewer samples per grid point').")





