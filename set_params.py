import numpy as np
from scipy.optimize import curve_fit

import propagators_grid
import collective_variables
import matplotlib.pyplot as plt


def autocorrelation_decay_constant(signal, dt, max_lag_frac=0.5):
    """
    Estimate the exponential decay constant tau of a signal's autocorrelation,
    assuming ACF(t) ~ exp(-t/tau).

    Parameters
    ----------
    signal : array_like
        Input time series (1D).
    dt : float
        Time step between samples.
    max_lag_frac : float
        Fraction of the signal length to use for the fit
        (autocorrelation gets noisy at large lags).

    Returns
    -------
    tau : float
        Fitted exponential decay constant (same units as dt).
    lags : ndarray
        Time lags used for the fit.
    acf : ndarray
        Normalized autocorrelation values at those lags.
    """

    x = np.asarray(signal) - np.mean(signal)
    n = len(x)

    # Full autocorrelation via FFT (fast, unbiased-normalized)
    corr = np.correlate(x, x, mode='full')[n - 1:]
    corr /= corr[0]  # normalize so ACF(0) = 1

    max_lag = int(n * max_lag_frac)
    lags = np.arange(max_lag) * dt
    acf = corr[:max_lag]

    # Fit only the positive part (log becomes undefined/noisy once ACF <= 0)
    mask = acf > 0
    lags_fit, acf_fit = lags[mask], acf[mask]

    def exp_decay(t, tau):
        return np.exp(-t / tau)

    popt, _ = curve_fit(exp_decay, lags_fit, acf_fit, p0=[lags_fit[-1] / 3])
    tau = popt[0]

    return tau, lags, acf


def set_mtd_params_from_unbiased_literature_advice(simulation_system, kB, T, CV, dt, n_frames):

    # sigma = np.array([0.2])
    # delta_T = 40
    # omega = 0.4

    init_coords = -np.ones((1, simulation_system.n_dim))
    init_potentials = np.zeros((len(init_coords), CV.grid_n))


    traj, pots, weights = propagators_grid.propagate(
        G=simulation_system.G, kB=kB, T=T, dt=dt, xi = simulation_system.xi, init_coords=init_coords,
        init_potentials=init_potentials, steps_per_saved_frame=1,
        n_gaussians=1, frames_per_gaussian=n_frames,
        CV=CV.cv_funct, grad_CV=CV.cv_grad_funct, sigma=np.array([1]), omega=0, delta_T=1,
        cv_min=CV.cv_min, cv_max=CV.cv_max
    )

    cv_trj = CV.cv_funct(traj[0])
    n_cv_dim = CV.n_cv_dim

    # print(cv_trj)
    # print(cv_trj.shape)

    sigma = np.zeros(n_cv_dim)
    taus = np.zeros(n_cv_dim)

    for dn in range(n_cv_dim):

        cv_trj_i = cv_trj[:,dn]

        #get variance
        sigma[dn] = np.std(cv_trj_i)

        #get autocorrelation
        tau, lags, acf = autocorrelation_decay_constant(cv_trj_i, dt, max_lag_frac=0.5)
        taus[dn] = tau

        # plt.plot(lags, acf)
        # plt.xlim(0,1)
        # plt.show()

        # plt.plot(cv_trj_i)


    delta_T=15*T
    omega = 0.1
    tau = np.max(taus)
    t_frame=tau/10
    t_gaussian=tau

    return delta_T, sigma, omega, t_frame, t_gaussian