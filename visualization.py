import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import collective_variable_analysis

#written by copilot
def plot_G_surface(
    G,
    x_limits,
    n=200,
    center=None,
    slice_axis=0,
    name="",
    add_CV_FE=False,
    CV = None,
    sim_system = None,
    kT=1.0,
):
    """
    Plot a 2D function G(x) as a heatmap and a 1D slice through the center.

    Parameters
    ----------
    G : callable
        Function that accepts an array of shape (N, 2) and returns an array of shape (N,).
    x_limits : array-like
        Shape (2, 2): [[x_min, x_max], [y_min, y_max]]
    n : int
        Number of grid points per axis
    center : array-like or None
        Center point for the slice. If None, uses the midpoint of the domain.
    slice_axis : int
        0 for a slice along the first coordinate, 1 for a slice along the second coordinate.
    add_CV_FE : bool
        If True, overlay the CV-based free energy on the plot.
    cv_fn, cv_min, cv_max, n_cv_grid : callable, array-like, array-like, int or sequence
        CV definition and grid parameters passed to ``free_energy_on_cv_grid``.
        The plotting routine supports one-dimensional CVs.
    coord_min, coord_max : array-like or None
        Microscopic coordinate bounds passed to ``free_energy_on_cv_grid``. If
        omitted, the bounds in ``x_limits`` are used.
    n_micro_grid : int or sequence of ints or None
        Orthogonal microscopic grid resolution. Defaults to ``n``.
    kT : float
        Thermal energy passed to ``free_energy_on_cv_grid``.
    """
    x_limits = np.asarray(x_limits, dtype=float)
    if x_limits.shape != (2, 2):
        raise ValueError("x_limits must have shape (2, 2): [[x_min, x_max], [y_min, y_max]]")

    x_min = x_limits[:, 0]
    x_max = x_limits[:, 1]

    if center is None:
        center = 0.5 * (x_min + x_max)

    xs = np.linspace(x_min[0], x_max[0], n)
    ys = np.linspace(x_min[1], x_max[1], n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    pts = np.column_stack([X.ravel(), Y.ravel()])

    if not add_CV_FE:
        def plotted_energy(points):
            return np.asarray(G(points), dtype=float).reshape(-1)

        plot_title = "G(x)"
        colorbar_label = "G(x) (kT)"
    else:
        # required = {
        #     "cv_fn": cv_fn,
        #     "cv_min": cv_min,
        #     "cv_max": cv_max,
        #     "n_cv_grid": n_cv_grid,
        # }
        # missing = [name for name, value in required.items() if value is None]
        # if missing:
        #     raise ValueError(
        #         "CV plotting requires: " + ", ".join(missing)
        #     )

        # cv_min_array = np.asarray(cv_min, dtype=float)
        # cv_max_array = np.asarray(cv_max, dtype=float)
        # if cv_min_array.size != 1 or cv_max_array.size != 1:
        #     raise ValueError("plot_G_surface supports one-dimensional CVs")

        cv_grid, cv_free_energy = collective_variable_analysis.free_energy_on_cv_grid(
                free_energy_fn=sim_system.G,
                coord_min=sim_system.coord_min,
                coord_max=sim_system.coord_max,
                n_micro_grid=sim_system.grid_n,
                cv_fn=CV.cv_funct,
                cv_min=CV.cv_min,
                cv_max=CV.cv_max,
                n_cv_grid=CV.grid_n,
                kT=kT
    )

        # if coord_min is None:
        #     coord_min = x_min
        # if coord_max is None:
        #     coord_max = x_max
        # if n_micro_grid is None:
        #     n_micro_grid = n

        # cv_grid, cv_free_energy = free_energy_on_cv_grid(
        #     free_energy_fn=G,
        #     coord_min=coord_min,
        #     coord_max=coord_max,
        #     n_micro_grid=n_micro_grid,
        #     cv_fn=cv_fn,
        #     cv_min=cv_min_array,
        #     cv_max=cv_max_array,
        #     n_cv_grid=n_cv_grid,
        #     kT=kT,
        # )
        cv_grid = np.asarray(cv_grid, dtype=float)
        cv_free_energy = np.asarray(cv_free_energy, dtype=float).reshape(-1)
        if cv_grid.ndim != 2 or cv_grid.shape[1] != 1:
            raise ValueError("free_energy_on_cv_grid must return a one-dimensional CV grid")
        if cv_grid.shape[0] != cv_free_energy.size:
            raise ValueError("CV grid and free-energy grid must have matching lengths")

        def plotted_energy(points):
            points = np.asarray(points, dtype=float)
            sample_cv = np.asarray(CV.cv_funct(points), dtype=float)
            if sample_cv.shape != (points.shape[0], 1):
                raise ValueError("cv_fn must return an array with shape (n_points, 1)")
            cv_landscape = np.interp(
                sample_cv[:, 0], cv_grid[:, 0], cv_free_energy
            )
            return np.asarray(G(points), dtype=float).reshape(-1) - cv_landscape

        plot_title = "G(x) - F(CV)"
        colorbar_label = "G(x) - F(CV) (kT)"

    Z = plotted_energy(pts).reshape(n, n)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis", vmax = 20)
    ax.set_title(plot_title)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_aspect("equal")
    fig.colorbar(pcm, ax=ax, label=colorbar_label)

    #this could probably be written more elegantly
    ax2 = axes[1]
    if slice_axis == 0:
        coord_vals = np.linspace(x_min[0], x_max[0], n)
        slice_pts = np.column_stack([coord_vals, np.full(n, center[1])])
        slice_vals = plotted_energy(slice_pts)
        ax2.plot(coord_vals, slice_vals, lw=2)
        ax2.set_xlabel("x0")
        ax2.set_ylabel(f"{plot_title} (x0, x1={center[1]:.2f}) (kT)")
        ax2.set_title(f"Slice at x1={center[1]:.2f}")
    elif slice_axis == 1:
        coord_vals = np.linspace(x_min[1], x_max[1], n)
        slice_pts = np.column_stack([np.full(n, center[0]), coord_vals])
        slice_vals = plotted_energy(slice_pts)
        ax2.plot(coord_vals, slice_vals, lw=2)
        ax2.set_xlabel("x1")
        ax2.set_ylabel(f"{plot_title} (x0={center[0]:.2f}, x1) (kT)")
        ax2.set_title(f"Slice at x0={center[0]:.2f}")
    else:
        raise ValueError("slice_axis must be 0 or 1")

    plt.tight_layout()
    if name != "":
        plt.savefig(f"figures/{name}.png", dpi=600)
    plt.show()



#see the following stackoverfow posts: 
# https://stackoverflow.com/questions/22548813/python-color-map-but-with-all-zero-values-mapped-to-black
# https://stackoverflow.com/questions/56062299/how-to-add-axis-labels-to-imshow-plots-in-python
# https://stackoverflow.com/questions/13384653/imshow-extent-and-aspect

def plot_masked_energies(data, xlims, ylims, plot_shape, aspect_ratio, vmax, labels, savefn = ""):

    # mask 'bad' regions with no sampling
    masked_rfe = np.ma.masked_where(data == 0, data)

    #set color mapping for regions with sampling
    cmap = mpl.colormaps.get_cmap("viridis").copy()

    #set color for 'bad' regions with no sampling
    cmap.set_bad(color='grey')

    plt.figure(figsize=plot_shape)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    im = plt.imshow(masked_rfe, interpolation='none', cmap=cmap, extent = [xlims[0], xlims[1], ylims[0], ylims[1]], aspect = aspect_ratio, vmax=vmax, origin="lower")
    if savefn != "":
        plt.title(savefn)
        plt.savefig(f"figures/{savefn}.png", dpi=600, bbox_inches="tight")

    plt.show()