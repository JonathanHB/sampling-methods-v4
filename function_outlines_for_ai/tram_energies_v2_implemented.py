import numpy as np

# TRAM is provided by deeptime.
from deeptime.markov.msm import TRAM, TRAMDataset


def tram_unbiased_energies(transitions, potential_functions, kT):
    """
    Calculate unbiased free energies of bins using TRAM.

    Parameters
    ----------
    transitions: list of 2d numpy arrays of floats
        Each array corresponds to one thermodynamic state.  Its shape is
        (2, n_transitions), and each column is a transition from the first
        row's bin to the second row's bin.

    potential_functions: list of numpy arrays of floats
        potential_functions[k] is the metadynamics bias potential on the
        configuration-space grid at thermodynamic state k.  All arrays must
        have the same shape.  Bin indices in ``transitions`` are interpreted
        as indices into the flattened potential grid.

    kT: float
        Boltzmann's constant times temperature, in the same energy units as
        the entries of potential_functions.

    Returns
    -------
    bins: 1d numpy array of ints
        Bin indices for which an unbiased TRAM free energy can be estimated.

    energies: 1d numpy array of floats
        Unbiased free energies of the bins, in the same energy units as kT.
        The minimum energy is shifted to zero.

    Notes
    -----
    The input contains transitions rather than complete trajectories.  To
    avoid introducing artificial transitions by concatenating independent
    transitions, every supplied transition is represented as a separate
    two-frame trajectory.  ``ttrajs`` is used to specify which thermodynamic
    state generated that transition.

    deeptime expects dimensionless bias energies (in units of kT), so the
    supplied potentials are divided by kT before constructing the TRAM data.
    """

    # -----------------------------
    # Validate the input.
    # -----------------------------
    if not np.isscalar(kT) or not np.isfinite(kT) or kT <= 0:
        raise ValueError("kT must be a finite positive scalar.")

    if len(transitions) != len(potential_functions):
        raise ValueError(
            "transitions and potential_functions must have the same length "
            "(one entry per thermodynamic state)."
        )

    n_therm_states = len(transitions)
    if n_therm_states == 0:
        raise ValueError("At least one thermodynamic state is required.")

    potentials = [np.asarray(V, dtype=float) for V in potential_functions]
    if any(V.ndim == 0 for V in potentials):
        raise ValueError("Each potential function must be an array.")

    grid_shape = potentials[0].shape
    if any(V.shape != grid_shape for V in potentials):
        raise ValueError("All potential_functions must have the same shape.")

    n_bins = int(np.prod(grid_shape))

    # Convert transitions to integer bin indices.  The documentation says
    # these are indices, so silently accepting non-integral values would be
    # dangerous.
    transition_arrays = []
    for k, tr in enumerate(transitions):
        tr = np.asarray(tr)
        if tr.ndim != 2 or tr.shape[0] != 2:
            raise ValueError(
                f"transitions[{k}] must have shape (2, n_transitions); "
                f"got {tr.shape}."
            )
        if not np.all(np.isfinite(tr)):
            raise ValueError(f"transitions[{k}] contains non-finite values.")
        if not np.all(tr == np.floor(tr)):
            raise ValueError(f"transitions[{k}] contains non-integer bin indices.")
        tr = tr.astype(np.int64, copy=False)
        if tr.size and (tr.min() < 0 or tr.max() >= n_bins):
            raise ValueError(
                f"transitions[{k}] contains a bin outside [0, {n_bins - 1}]."
            )
        transition_arrays.append(tr)

    # At least one transition is required for a meaningful TRAM estimate.
    if not any(tr.shape[1] > 0 for tr in transition_arrays):
        raise ValueError("No transitions were supplied.")

    # -----------------------------
    # Construct TRAM input.
    # -----------------------------
    # deeptime's TRAMDataset expects, for every sample, the bias evaluated at
    # that sample for *every* thermodynamic state.  Thus, for a sample in bin
    # i, bias_matrices[..., k] = potential_functions[k].flat[i] / kT.
    #
    # Each transition is kept as its own two-frame trajectory.  This is
    # important: concatenating all transitions into one trajectory would add
    # fictitious transitions between the end of one supplied transition and
    # the beginning of the next one.
    dtrajs = []
    bias_matrices = []
    ttrajs = []

    flat_potentials = np.asarray([V.reshape(-1) for V in potentials], dtype=float)
    # Shape: (n_therm_states, n_bins)
    flat_bias = flat_potentials / float(kT)

    for therm_state, tr in enumerate(transition_arrays):
        for start, end in tr.T:
            traj = np.array([start, end], dtype=np.int32)

            # bias[k] is the bias of both samples in this two-frame trajectory
            # evaluated at thermodynamic state k.
            bias = flat_bias[:, traj].T  # shape (2, n_therm_states)

            dtrajs.append(traj)
            bias_matrices.append(bias)
            ttrajs.append(np.full(2, therm_state, dtype=np.int32))

    # -----------------------------
    # Run TRAM.
    # -----------------------------
    # ``sample`` is appropriate here because each two-frame trajectory contains
    # exactly one supplied transition.  With lagtime=1, that produces exactly
    # one count for each input transition and no additional counts.
    dataset = TRAMDataset(
        dtrajs=dtrajs,
        bias_matrices=bias_matrices,
        ttrajs=ttrajs,
        n_therm_states=n_therm_states,
        n_markov_states=n_bins,
        lagtime=1,
        count_mode="sample",
    )

    # TRAM needs a connected state space to determine relative free energies.
    # Restrict to the largest connected set based on the summed transition
    # count matrix.  The returned state indices are the bins for which the
    # estimate is identifiable from the supplied transitions.
    dataset.restrict_to_largest_connected_set(connectivity="summed_count_matrix")

    tram = TRAM(
        lagtime=1,
        count_mode="sample",
        maxiter=10000,
        maxerr=1e-8,
        init_strategy="MBAR",
    )
    model = tram.fit_fetch(dataset)

    # -----------------------------
    # Extract the unbiased PMF.
    # -----------------------------
    # model.compute_PMF() returns a dimensionless PMF, i.e. -log(probability),
    # for the unbiased/reference state (therm_state=-1).  Use the original
    # Markov-state labels as the bin labels.
    connected_bins = np.asarray(dataset.dtrajs, dtype=object)
    bins = np.unique(np.concatenate(connected_bins)).astype(np.int64)

    # The model's connected-state representation is internally compact, while
    # dataset.dtrajs contains the corresponding original state labels.
    # Computing the PMF directly from the sample weights avoids assuming that
    # TRAM's internal state ordering is identical to the original grid labels.
    sample_weights = model.compute_sample_weights(
        dataset.dtrajs, dataset.bias_matrices, therm_state=-1
    )

    # Sum the TRAM weights over samples belonging to each original bin.
    # This gives p(bin), up to the common normalization already handled by
    # compute_sample_weights().
    probability = np.zeros(len(bins), dtype=float)
    bin_to_position = {int(b): i for i, b in enumerate(bins)}

    for traj, weights in zip(dataset.dtrajs, sample_weights):
        for state, weight in zip(traj, weights):
            probability[bin_to_position[int(state)]] += float(weight)

    if np.any(probability <= 0) or not np.all(np.isfinite(probability)):
        raise RuntimeError("TRAM produced invalid probabilities for the connected bins.")

    energies = -float(kT) * np.log(probability)
    energies -= energies.min()

    return bins, energies
