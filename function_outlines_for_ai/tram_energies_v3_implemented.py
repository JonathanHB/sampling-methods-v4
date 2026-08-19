import numpy as np
import matplotlib.pyplot as plt

from deeptime.markov.msm import TRAM, TRAMDataset


def tram_unbiased_energies(transitions, potential_functions, kT):
    """
    Calculate unbiased free energies of bins using TRAM.

    Parameters
    ----------
    transitions : list of 2d numpy arrays
        Each array corresponds to one thermodynamic state (i.e. one
        metadynamics bias potential).  It has shape (2, n_transitions).
        Column j contains one observed transition:
            transitions[k][0, j] -> transitions[k][1, j]

    potential_functions : list of numpy arrays
        One metadynamics bias-potential grid per thermodynamic state.
        All grids must have the same shape.  The flattened grid index is
        assumed to be the Markov-state/bin index used in ``transitions``.

        The values are bias energies (not dimensionless energies).

    kT : float
        Boltzmann's constant times temperature, in the same energy units as
        the entries of ``potential_functions``.

    Returns
    -------
    bins : 1d numpy array of ints
        Original bin indices belonging to the largest connected set.

    energies : 1d numpy array of floats
        Unbiased free energies of those bins, in the same energy units as
        ``kT``.  An arbitrary additive constant is removed so that the
        minimum energy is zero.

    Notes
    -----
    Each supplied transition is an independent observed transition.  We
    therefore cannot concatenate transitions directly, because doing so
    would create artificial transitions between the end of one supplied
    transition and the beginning of the next.

    Instead, each thermodynamic state is represented by one trajectory,
    with ``-1`` used as a separator between independent transitions:

        [i0, j0, -1, i1, j1, -1, ...]

    Deeptime treats negative states as excluded samples and, when finding
    trajectory fragments, splits trajectories at negative states.  Thus the
    separators prevent artificial transitions from being counted while
    retaining the individual i -> j transitions.

    ``ttrajs`` is deliberately omitted.  With no replica exchange,
    deeptime interprets dtrajs[k] as the trajectory sampled at
    thermodynamic state k.
    """

    # ------------------------------------------------------------------
    # Validate the basic inputs.
    # ------------------------------------------------------------------
    if not np.isscalar(kT) or not np.isfinite(kT) or kT <= 0:
        raise ValueError("kT must be a finite positive scalar.")

    n_therm_states = len(transitions)

    if n_therm_states == 0:
        raise ValueError("transitions must contain at least one thermodynamic state.")

    if len(potential_functions) != n_therm_states:
        raise ValueError(
            "transitions and potential_functions must have the same length."
        )

    # Flatten the bias grids.  Each flattened grid index is assumed to be
    # the Markov-state/bin index.
    flat_potentials = []
    grid_shape = None

    for k, potential in enumerate(potential_functions):
        potential = np.asarray(potential, dtype=float)

        if potential.ndim == 0:
            raise ValueError(
                f"potential_functions[{k}] must be an array, not a scalar."
            )

        if not np.all(np.isfinite(potential)):
            raise ValueError(
                f"potential_functions[{k}] contains non-finite values."
            )

        if grid_shape is None:
            grid_shape = potential.shape
        elif potential.shape != grid_shape:
            raise ValueError(
                "All potential_functions must have identical shapes."
            )

        flat_potentials.append(potential.ravel())

    flat_potentials = np.asarray(flat_potentials, dtype=float)
    n_markov_states = flat_potentials.shape[1]

    # ------------------------------------------------------------------
    # Construct one trajectory per thermodynamic state.
    #
    # For independent transitions
    #
    #     a -> b
    #     c -> d
    #
    # we use
    #
    #     [a, b, -1, c, d]
    #
    # rather than [a, b, c, d].  The -1 prevents deeptime from interpreting
    # b -> c as an observed transition.
    # ------------------------------------------------------------------
    dtrajs = []
    bias_matrices = []

    for therm_state, transition_array in enumerate(transitions):
        tr = np.asarray(transition_array)

        if tr.ndim != 2 or tr.shape[0] != 2:
            raise ValueError(
                f"transitions[{therm_state}] must have shape (2, n_transitions); "
                f"got {tr.shape}."
            )

        if tr.shape[1] == 0:
            # An empty thermodynamic state is allowed by the container, but
            # it cannot contribute information to TRAM.
            dtrajs.append(np.empty(0, dtype=np.int32))
            bias_matrices.append(
                np.empty((0, n_therm_states), dtype=float)
            )
            continue

        # Transitions should represent integer bin indices.  Allow float
        # arrays only when every value is exactly integral.
        if not np.all(np.isfinite(tr)):
            raise ValueError(
                f"transitions[{therm_state}] contains non-finite values."
            )

        if not np.all(tr == np.floor(tr)):
            raise ValueError(
                f"transitions[{therm_state}] contains non-integer bin indices."
            )

        tr = tr.astype(np.int64, copy=False)

        starts = tr[0]
        ends = tr[1]

        if np.any(starts < 0) or np.any(ends < 0):
            raise ValueError(
                "Transition bin indices must be non-negative. "
                "The value -1 is reserved internally as a trajectory separator."
            )

        if np.any(starts >= n_markov_states) or np.any(ends >= n_markov_states):
            bad = np.concatenate(
                [starts[starts >= n_markov_states],
                 ends[ends >= n_markov_states]]
            )
            raise ValueError(
                "A transition contains a bin index outside the potential grid. "
                f"Maximum valid index is {n_markov_states - 1}; "
                f"example invalid indices: {bad[:10]}."
            )

        n_transitions = tr.shape[1]

        # Interleave each transition with a -1 separator, except after the
        # final transition.
        dtraj = np.empty(3 * n_transitions - 1, dtype=np.int32)
        dtraj[0::3] = starts
        dtraj[1::3] = ends
        if n_transitions > 1:
            dtraj[2::3] = -1

        dtrajs.append(dtraj)

        # For every sampled configuration, TRAM needs the bias evaluated
        # under every thermodynamic state:
        #
        # bias_matrices[k][n, l] = b^l(x_n)
        #
        # The separator rows have no physical sample, so their values are
        # irrelevant.  Set them to zero.
        bias = np.zeros((len(dtraj), n_therm_states), dtype=float)

        # Actual sample rows occur at positions 0, 1, 3, 4, 6, 7, ...
        sample_positions = np.flatnonzero(dtraj >= 0)
        sample_bins = dtraj[sample_positions]

        # flat_potentials[:, sample_bins] has shape
        # (n_therm_states, n_samples); transpose to (n_samples, n_therm_states).
        bias[sample_positions] = flat_potentials[:, sample_bins].T

        bias_matrices.append(bias)

    # ------------------------------------------------------------------
    # Build the TRAM dataset.
    #
    # Because ttrajs=None, trajectory k is interpreted as thermodynamic
    # state k.  This is exactly the representation needed here.
    # ------------------------------------------------------------------

    #print(dtrajs)
    dataset = TRAMDataset(
        dtrajs=dtrajs,
        bias_matrices=bias_matrices,
        ttrajs=None,
        n_therm_states=n_therm_states,
        n_markov_states=n_markov_states,
        lagtime=1,
        count_mode="sample",
    )

    # ------------------------------------------------------------------
    # Restrict to the largest connected set.
    #
    # Deeptime changes samples outside the selected submodel to -1 and
    # splits trajectory fragments at those negative entries.  This is
    # why the -1 separators above are compatible with this operation.
    # ------------------------------------------------------------------
    # dataset.restrict_to_largest_connected_set(
    #     connectivity="summed_count_matrix"
    # )

    # Collect the ORIGINAL bin indices that survived the restriction.
    # restrict_to_submodel preserves the original state symbols.
    connected_bins = np.unique(
        np.concatenate(
            [traj[traj >= 0] for traj in dataset.dtrajs
             if len(traj) > 0]
        )
    ).astype(np.int64)

    if connected_bins.size == 0:
        raise ValueError(
            "No connected Markov states remain after applying the TRAM "
            "connectivity criterion."
        )

    # ------------------------------------------------------------------
    # Run TRAM.
    #
    # markov_state_energies are the dimensionless unbiased free energies
    # f^i estimated by the TRAM estimator.  They are preferable here to
    # reconstructing the PMF from sample weights and avoid relying on
    # TRAMModel.compute_sample_weights(), whose availability differs among
    # deeptime versions.
    # ------------------------------------------------------------------
    tram = TRAM(
        lagtime=1,
        count_mode="sample",
        maxiter=10000,
        maxerr=1e-8,
        init_strategy="MBAR",
    )

    model = tram.fit_fetch(dataset)

    markov_state_energies = np.asarray(
        model.markov_state_energies, dtype=float
    )

    if markov_state_energies.ndim != 1:
        raise RuntimeError(
            "TRAM returned markov_state_energies with an unexpected shape: "
            f"{markov_state_energies.shape}"
        )

    if np.max(connected_bins) >= len(markov_state_energies):
        raise RuntimeError(
            "The TRAM model returned fewer Markov-state energies than "
            "the connected-state indices require."
        )

    # TRAM energies are dimensionless. Convert to physical energy.
    energies = kT * markov_state_energies[connected_bins]

    # The zero of free energy is arbitrary.
    energies -= np.min(energies)

    plt.plot(energies)
    plt.show()

    return connected_bins, energies
