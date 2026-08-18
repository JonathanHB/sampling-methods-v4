#Chatgpt prompt 08/17/26 (free online version):
#[attached tram_reweight.py]
#Please implement the python function in the attached file


import numpy as np

try:
    from deeptime.markov.msm import MaximumLikelihoodMSM
except ImportError as exc:
    MaximumLikelihoodMSM = None
    _DEEPTIME_IMPORT_ERROR = exc

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
except ImportError as exc:
    csr_matrix = None
    connected_components = None
    _SCIPY_IMPORT_ERROR = exc


###################################################################################################
#                                      MSM RESAMPLER

def tram_reweight(cumulative_transitions_weights, w, b, walkers_per_bin, last_potential):
    """
    Reweight weighted-ensemble walkers using a reversible, transition-based
    estimator of the unbiased bin probabilities.

    Notes
    -----
    The supplied data contain transition records and endpoint MTD importance
    weights, but not the full bias-energy matrix required by a conventional
    dTRAM/TRAM implementation.  Therefore this function uses the closest
    transition-based estimator that is identifiable from the supplied inputs:

      1. Each transition is given the symmetric statistical weight
         sqrt(weight_start * weight_end).
      2. The resulting weighted transition-count matrix is passed to
         deeptime's reversible maximum-likelihood MSM estimator.
      3. Disconnected components are estimated independently.
      4. Within each disconnected component, the total current WE weight is
         preserved.
      5. The unbiased component distribution is reweighted by the current
         metadynamics potential, giving
             p_V(i) proportional to p_0(i) exp(-V_i)
         where ``last_potential`` is assumed to be expressed in units of kT.
      6. The probability of each bin is divided equally among the target
         number of walkers in that bin.

    This is a weighted reversible-MSM/TRAM-like estimator, not an exact
    dTRAM estimator. Exact dTRAM requires a bias energy for every
    thermodynamic state and configuration state, whereas the input format
    here supplies only the endpoint importance weights for each transition.

    Parameters
    ----------
    cumulative_transitions_weights : list of numpy arrays
        Each array has shape (4, number_of_walkers_in_round). Rows are:
          0: transition starting bin
          1: transition ending bin
          2: MTD importance weight at the starting point
          3: MTD importance weight at the ending point.

    w : numpy.ndarray
        WE weight of each current walker. Used to preserve the total weight
        in each disconnected state-space component.

    b : numpy.ndarray
        Bin occupied by each current walker. Bin 0 is the off-grid bin with
        zero MTD potential. For grid bins, ``b[i] - 1`` is interpreted as a
        flattened C-order index into ``last_potential``.

    walkers_per_bin : int
        Target number of walkers per bin.

    last_potential : numpy.ndarray
        Current metadynamics potential. It is assumed to be in units of kT.

    Returns
    -------
    w_out : numpy.ndarray
        New WE weights, one per current walker.
    """
    if MaximumLikelihoodMSM is None:
        raise ImportError(
            "tram_reweight requires deeptime. Install it with `pip install deeptime`."
        ) from _DEEPTIME_IMPORT_ERROR

    if csr_matrix is None:
        raise ImportError(
            "tram_reweight requires scipy. Install it with `pip install scipy`."
        ) from _SCIPY_IMPORT_ERROR

    # ----------------------------- input validation -----------------------------
    b = np.asarray(b, dtype=int)
    w = np.asarray(w, dtype=float)
    potential = np.asarray(last_potential, dtype=float)

    if b.ndim != 1 or w.ndim != 1:
        raise ValueError("b and w must be one-dimensional arrays.")
    if len(b) != len(w):
        raise ValueError("b and w must have the same length.")
    if walkers_per_bin <= 0 or int(walkers_per_bin) != walkers_per_bin:
        raise ValueError("walkers_per_bin must be a positive integer.")
    if potential.ndim == 0:
        raise ValueError("last_potential must be an array.")
    if not np.all(np.isfinite(w)):
        raise ValueError("w contains non-finite values.")
    if np.any(w < 0):
        raise ValueError("w must be non-negative.")
    if np.any(b < 0):
        raise ValueError("bin indices in b must be non-negative.")
    if np.any(b > potential.size):
        raise ValueError(
            "A bin index in b is larger than last_potential.size. "
            "Expected b=0 for off-grid and b=1..last_potential.size for grid bins."
        )

    if len(b) and np.sum(w) <= 0:
        raise ValueError("The current WE weights must have positive total weight.")

    n_states = potential.size + 1  # state 0 is the zero-potential off-grid state

    # ---------------------- 1. Build weighted transition counts -----------------
    #
    # A conventional dTRAM estimator needs ensemble-specific bias energies.
    # Here those are unavailable.  The endpoint MTD importance weights are
    # therefore used directly to assign a symmetric weight to each observed
    # transition.  The geometric mean is invariant under reversal of a
    # transition and is consequently compatible with the reversible MSM fit.
    counts = np.zeros((n_states, n_states), dtype=float)

    if cumulative_transitions_weights is None:
        raise ValueError("cumulative_transitions_weights cannot be None.")

    for round_data in cumulative_transitions_weights:
        arr = np.asarray(round_data, dtype=float)

        if arr.ndim != 2 or arr.shape[0] != 4:
            raise ValueError(
                "Every transition array must have shape (4, number_of_walkers)."
            )

        if arr.shape[1] == 0:
            continue

        starts = arr[0].astype(int)
        ends = arr[1].astype(int)
        ws = arr[2]
        we = arr[3]

        # Bin indices are discrete, so reject non-integral values rather than
        # silently truncating them.
        if not np.all(arr[0] == starts) or not np.all(arr[1] == ends):
            raise ValueError("Transition bin indices must be integers.")
        if np.any(starts < 0) or np.any(ends < 0):
            raise ValueError("Transition bin indices must be non-negative.")
        if np.any(starts >= n_states) or np.any(ends >= n_states):
            raise ValueError(
                "Transition bin index is outside the range implied by last_potential."
            )
        if not np.all(np.isfinite(arr[2:4])):
            raise ValueError("MTD importance weights contain non-finite values.")
        if np.any(ws <= 0) or np.any(we <= 0):
            raise ValueError(
                "MTD importance weights must be strictly positive."
            )

        transition_weights = np.sqrt(ws * we)

        # np.add.at handles repeated (start, end) pairs correctly.
        np.add.at(counts, (starts, ends), transition_weights)

    # States that have no transition data cannot have their stationary
    # probabilities inferred by the MSM. They will consequently receive zero
    # probability unless occupied by a current walker in a connected
    # component with transition data.
    if not np.any(counts > 0):
        raise ValueError("No positive transition weight was supplied.")

    # ---------------- 2. Find disconnected state-space components ---------------
    #
    # Reversibility couples i<->j, so connectivity is determined from the
    # undirected support of C + C^T.
    support = (counts + counts.T) > 0
    n_components, labels = connected_components(
        csr_matrix(support.astype(np.int8)), directed=False, return_labels=True
    )

    # ---------------- 3. Reversible MLE within each component -------------------
    p0 = np.zeros(n_states, dtype=float)

    for component in range(n_components):
        states = np.flatnonzero(labels == component)
        C = counts[np.ix_(states, states)]

        # A component containing a state with no self/transition counts can
        # occur only for isolated zero-count states. There is no information
        # with which to estimate its probability.
        if not np.any(C > 0):
            continue

        if len(states) == 1:
            p_local = np.array([1.0])
        else:
            estimator = MaximumLikelihoodMSM(
                reversible=True,
                allow_disconnected=False,
                maxiter=100000,
                maxerr=1e-10,
            )
            model = estimator.fit_fetch(C)
            p_local = np.asarray(model.stationary_distribution, dtype=float)

            if p_local.shape != (len(states),):
                raise RuntimeError(
                    "deeptime returned an unexpected stationary-distribution shape."
                )

        p_local = np.maximum(p_local, 0.0)
        total = p_local.sum()
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("Reversible MSM produced an invalid stationary distribution.")
        p0[states] = p_local / total

    # ---------------- 4. Preserve disconnected-set WE weights -------------------
    #
    # If the state space is connected, the usual normalization is simply
    # sum(p0)=1.  If it is disconnected, the MSM cannot infer the relative
    # masses of disconnected components.  We therefore retain the current WE
    # mass of every component and only replace its internal distribution.
    component_current_weight = np.zeros(n_components, dtype=float)
    for component in range(n_components):
        component_current_weight[component] = np.sum(
            w[labels[b] == component]
        )

    # If a component is not represented by a current walker, its mass is
    # necessarily zero for the purpose of the current WE ensemble.
    p_unbiased = np.zeros_like(p0)
    for component in range(n_components):
        states = np.flatnonzero(labels == component)
        if component_current_weight[component] <= 0:
            continue
        local = p0[states]
        if local.sum() > 0:
            p_unbiased[states] = (
                component_current_weight[component] * local / local.sum()
            )

    # ---------------- 5. Reweight by the current MTD potential ------------------
    #
    # For a bias V, p_V(i) ∝ p_0(i) exp(-V_i/kT).  last_potential is assumed
    # to be dimensionless (V/kT), so this is exp(-V).
    V = np.zeros(n_states, dtype=float)
    if potential.size:
        V[1:] = potential.ravel()

    # Work component-by-component so that disconnected component masses remain
    # exactly unchanged.
    p_biased = np.zeros_like(p_unbiased)

    for component in range(n_components):
        states = np.flatnonzero(labels == component)
        mass = component_current_weight[component]
        if mass <= 0:
            continue

        logp = np.full(len(states), -np.inf)
        positive = p_unbiased[states] > 0
        logp[positive] = np.log(p_unbiased[states][positive]) - V[states][positive]

        finite = np.isfinite(logp)
        if not np.any(finite):
            continue

        # Stable normalization of p0 * exp(-V).
        shift = np.max(logp[finite])
        relative = np.zeros(len(states), dtype=float)
        relative[finite] = np.exp(logp[finite] - shift)
        Z = relative.sum()

        p_biased[states] = mass * relative / Z

    # ---------------- 6. Divide bin probability among WE walkers ----------------
    w_out = np.zeros(len(b), dtype=float)

    for state in np.unique(b):
        mask = b == state
        n_walkers = np.count_nonzero(mask)
        if n_walkers == 0:
            continue
        w_out[mask] = p_biased[state] / walkers_per_bin

    # If the caller's current walker count differs from walkers_per_bin, preserve
    # the requested WE convention but make the discrepancy visible rather than
    # silently renormalizing the result.
    #
    # The normal WE case has n_walkers == walkers_per_bin for every occupied bin.
    occupied = np.unique(b)
    actual_counts = np.array([np.count_nonzero(b == state) for state in occupied])
    if np.any(actual_counts != walkers_per_bin):
        # This is intentionally a warning rather than an exception because WE
        # codes can temporarily violate the target count during resampling.
        import warnings
        warnings.warn(
            "Some occupied bins do not contain walkers_per_bin walkers; "
            "weights were divided by the requested target count.",
            RuntimeWarning,
            stacklevel=2,
        )

    return w_out
