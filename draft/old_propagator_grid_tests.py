import numpy as np
import matplotlib.pyplot as plt


#written by claude
def deprecated_test_propagators_grid():

    n_walkers = 4
    n_dim = 2
    n_cv = 1

    def G(x):
        return (x[:, 0]**2 - 1)**2 + 0.5*x[:, 1]**2

    def CV(x):
        return x[:, :1]

    def grad_CV(x):
        n = x.shape[0]
        out = np.zeros((n, n_cv, n_dim))
        out[:, 0, 0] = 1.0
        return out

    grid_n = 81
    cv_min = np.array([-3.0])
    cv_max = np.array([3.0])

    init_coords = np.random.uniform(-1.5, 1.5, size=(n_walkers, n_dim))
    init_potentials = np.zeros((n_walkers, grid_n))
    sigma = np.array([0.2])

    traj, pots = propagate(
        G=G, kT=1.0, dt=0.001, init_coords=init_coords,
        init_potentials=init_potentials, steps_per_saved_frame=5,
        n_gaussians=20, frames_per_gaussian=3,
        CV=CV, grad_CV=grad_CV, sigma=sigma, omega=0.5, delta_T=5.0,
        cv_min=cv_min, cv_max=cv_max,
    )

    print('trajectories shape', traj.shape)
    print('potentials shape', pots.shape)
    print('sample final coords', traj[:, -1, :])
    print('max bias height walker0 over time', pots[0, :, :].max(axis=-1))
    print(np.isfinite(traj).all(), np.isfinite(pots).all())

    # quick 2-cv test
    n_cv2 = 2
    def CV2(x):
        return x[:, :2]
    def grad_CV2(x):
        n = x.shape[0]
        out = np.zeros((n, n_cv2, n_dim))
        out[:, 0, 0] = 1.0
        out[:, 1, 1] = 1.0
        return out

    sigma2 = np.array([0.3, 0.3])
    grid_shape2 = (25, 25)
    init_potentials2 = np.zeros((n_walkers, *grid_shape2))
    cv_min2 = np.array([-3.0, -3.0])
    cv_max2 = np.array([3.0, 3.0])

    traj2, pots2 = propagate(
        G=G, kT=1.0, dt=0.001, init_coords=init_coords,
        init_potentials=init_potentials2, steps_per_saved_frame=5,
        n_gaussians=5, frames_per_gaussian=2,
        CV=CV2, grad_CV=grad_CV2, sigma=sigma2, omega=0.5, delta_T=5.0,
        cv_min=cv_min2, cv_max=cv_max2,
    )
    print('2cv trajectories shape', traj2.shape)
    print('2cv potentials shape', pots2.shape)
    print(np.isfinite(traj2).all(), np.isfinite(pots2).all())




