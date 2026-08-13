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

    #TODO for a direct exponential fit this should not be necessary; (why) are we linearizing?
    #TODO just try without the mask
    #   Is the problem that the curve_fit function expects very good convergence but the exponential will never match the negative bits? That actually would make no sense.
    # Fit only the positive part (log becomes undefined/noisy once ACF <= 0)
    mask = acf > 0
    lags_fit, acf_fit = lags[mask], acf[mask]

    # tau0 = lags[-1] / 3
    # print(tau0)
    # plt.plot(-lags/tau0)
    # plt.show()

    def exp_decay(t, tau):
        # print(t)
        # print(tau)
        return np.exp(-t / tau)

    # popt, _ = curve_fit(exp_decay, lags_fit, acf_fit, p0=[lags_fit[-1] / 3])
    # tau = popt[0]

    popt2, _ = curve_fit(exp_decay, lags, acf, p0=[lags[-1] / 3])
    tau2 = popt2[0]

    # plt.plot(lags, acf)
    # plt.plot(lags, np.exp(-lags/tau))
    # plt.plot(lags, np.exp(-lags/tau2))
    # print(tau, tau2)
    # plt.xlim(0, 3)
    # plt.show()

    return tau2, lags, acf


def set_mtd_params_from_unbiased_literature_advice(simulation_system, kB, T, CV, dt, n_frames, dG_ts):

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
        sigma[dn] = np.std(cv_trj_i)/2

        #get autocorrelation
        tau, lags, acf = autocorrelation_decay_constant(cv_trj_i, dt, max_lag_frac=0.5)
        taus[dn] = tau

        # plt.plot(lags, acf)
        # plt.xlim(0,1)
        # plt.show()

        # plt.plot(cv_trj_i)

    #MTD params
    delta_T=dG_ts/kB - T
    omega = 0.1*np.sqrt(np.e)
    tau = np.max(taus)
    t_gaussian=tau

    #WE parameters, see 'WE review' overleaf document
    walkers_per_bin=4
    bin_width = sigma*np.sqrt(2*kB*T/dG_ts) #note that this is an array
    t_we = tau*kB*T/dG_ts

    if t_we > t_gaussian:
        print(f"Warning: MTD interval ({t_gaussian}) exceeds WE interval ({t_we}), probable input error")

    #frame save interval
    t_frame=t_we/5

    #how much potential to deposit each WE round (t_we) so that the deposition rate equals the rate you would get if you deposited a gaussian of height omega every t_gaussian
    #before accounting for the fact that well-tempering makes this like compouding interest at different intervals
    we_omega = omega*t_we/t_gaussian


    return delta_T, sigma, omega, we_omega, t_gaussian, walkers_per_bin, bin_width, t_we, t_frame