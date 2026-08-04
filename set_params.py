import numpy as np

def set_mtd_params_from_unbiased_literature_advice(simulation_system, T, CV, dt, n_steps_per_frame):

    sigma = np.array([0.2])
    delta_T = 40
    omega = 0.4
    n_frames_per_gaussian = 10

    return delta_T, sigma, omega, n_frames_per_gaussian