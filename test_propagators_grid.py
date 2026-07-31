import numpy as np
from draft.old_energy_landscapes import free_energy_on_cv_grid
from propagators_grid import propagate
import matplotlib.pyplot as plt


#written by copilot
def test_free_energy_on_cv_grid():
    def free_energy_fn(coords):
        return (coords[:, 0] - 1.0) ** 2 + 0.5 * coords[:, 1] ** 2

    def cv_fn(coords):
        return np.asarray(coords, dtype=float)[:, :1]

    cv_grid, free_energy_grid = free_energy_on_cv_grid(
        free_energy_fn=free_energy_fn,
        cv_fn=cv_fn,
        coord_min=np.array([-2.0, -2.0]),
        coord_max=np.array([2.0, 2.0]),
        cv_min=np.array([-2.0]),
        cv_max=np.array([2.0]),
        n_grid=9,
        n_micro_grid=7,
        kT=1.0,
    )

    print(cv_grid.shape)
    print(free_energy_grid.shape)

    assert cv_grid.shape == (9, 1)
    assert free_energy_grid.shape == (9,)

    micro_grid = np.linspace(-2.0, 2.0, 7)
    expected = []
    for cv_value in cv_grid[:, 0]:
        values = (cv_value - 1.0) ** 2 + 0.5 * micro_grid ** 2
        expected.append(-np.log(np.mean(np.exp(-values))))

    assert np.allclose(free_energy_grid, np.asarray(expected))


#written by claude
def test_propagators_grid():

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




#written by copilot
def plot_G_surface(G, x_limits, n=200, center=None, slice_axis=0):
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
    fig.colorbar(pcm, ax=ax, label="G(x)")

    ax2 = axes[1]
    if slice_axis == 0:
        coord_vals = np.linspace(x_min[0], x_max[0], n)
        slice_pts = np.column_stack([coord_vals, np.full(n, center[1])])
        slice_vals = G(slice_pts)
        ax2.plot(coord_vals, slice_vals, lw=2)
        ax2.set_xlabel("x0")
        ax2.set_ylabel("G(x0, x1=center)")
        ax2.set_title(f"Slice at x1={center[1]:.2f}")
    elif slice_axis == 1:
        coord_vals = np.linspace(x_min[1], x_max[1], n)
        slice_pts = np.column_stack([np.full(n, center[0]), coord_vals])
        slice_vals = G(slice_pts)
        ax2.plot(coord_vals, slice_vals, lw=2)
        ax2.set_xlabel("x1")
        ax2.set_ylabel("G(x0=center, x1)")
        ax2.set_title(f"Slice at x0={center[0]:.2f}")
    else:
        raise ValueError("slice_axis must be 0 or 1")

    plt.tight_layout()
    plt.show()