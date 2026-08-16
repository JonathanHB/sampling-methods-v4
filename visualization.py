import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


#written by copilot
def plot_G_surface(G, x_limits, n=200, center=None, slice_axis=0, name=""):
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
    Z = G(pts).reshape(n, n)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis", vmax = 20)
    ax.set_title("G(x)")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_aspect("equal")
    fig.colorbar(pcm, ax=ax, label="G(x) (kT)")

    ax2 = axes[1]
    if slice_axis == 0:
        coord_vals = np.linspace(x_min[0], x_max[0], n)
        slice_pts = np.column_stack([coord_vals, np.full(n, center[1])])
        slice_vals = G(slice_pts)
        ax2.plot(coord_vals, slice_vals, lw=2)
        ax2.set_xlabel("x0")
        ax2.set_ylabel(f"G(x0, x1={center[1]:.2f}) (kT)")
        ax2.set_title(f"Slice at x1={center[1]:.2f}")
    elif slice_axis == 1:
        coord_vals = np.linspace(x_min[1], x_max[1], n)
        slice_pts = np.column_stack([np.full(n, center[0]), coord_vals])
        slice_vals = G(slice_pts)
        ax2.plot(coord_vals, slice_vals, lw=2)
        ax2.set_xlabel("x1")
        ax2.set_ylabel(f"G(x0={center[0]:.2f}, x1) (kT)")
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

def plot_masked_energies(data, xlims, ylims, plot_shape, aspect_ratio, vmax, labels):

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
    plt.show()