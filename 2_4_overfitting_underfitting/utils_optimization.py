import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from ipywidgets import interactive
import matplotlib as mpl

# der wahre funktionale Zusammenhang
def true_function(x):
    f = 5 / (1 + np.exp(-x + 2))
    return f

rng = np.random.RandomState(1)

# Daten - Beobachtungen / Merkmale
x_train = 10 * rng.rand(20)
x_test = 10 * rng.rand(20)

# Daten - Zielvariablen
y_train = true_function(x_train) + 0.5 * rng.randn(20)
y_test = true_function(x_test) + 0.5 * rng.randn(20)

# Zum Plotten
xx = np.linspace(0, 10)
ff = true_function(xx)


def get_loss(x, y, w, bias):
    y_pred = x * w + bias
    return np.mean( (y - y_pred)**2)


# convenience functions and variables zur Darstellung der Verlustfunktion
def empirical_risk(w, b, x_sample, y_sample):
    # makes heavy use of broadcasting
    W = np.repeat(w[..., np.newaxis], x_sample.shape[0], axis=-1)
    B = np.repeat(b[..., np.newaxis], x_sample.shape[0], axis=-1)
    Y_pred = W * x_sample + B
    loss = np.mean((Y_pred - y_sample)**2, axis=-1)
    return loss


def weight_norm(W, B):
    return W**2 + B**2


def L1_norm(W, B):
    return np.abs(W) + np.abs(B)

ws = np.linspace(-3, 3, 1000)
bs = np.linspace(-10, 10, 1000)

W, B = np.meshgrid(ws, bs)
L = empirical_risk(W, B, x_train, y_train)
L_reg = weight_norm(W, B)
L_reg_l1 = L1_norm(W, B)


L_test = empirical_risk(W, B, x_test, y_test)

L_min, L_max = L.min(), L.max()


def get_trajectory(update_func, max_iter=50, **kwargs):
    
    w_init = np.random.uniform(-3, 3)
    bias_init = np.random.uniform(-4, 4)
    
    trajectory = np.zeros((max_iter, 3))
    trajectory[0, 0] = w_init
    trajectory[0, 1] = bias_init
    
    trajectory[0, 2] = get_loss(x_train, y_train, w_init, bias_init)
    
    for i in range(max_iter-1):
        w_new, bias_new = update_func(
            x_train, y_train,
            trajectory[i, 0],
            trajectory[i, 1],
            **kwargs
        )
        
        trajectory[i+1, 0] = w_new
        trajectory[i+1, 1] = bias_new
        
        trajectory[i+1, 2] = get_loss(x_train, y_train, w_new, bias_new)
        
    return trajectory


def visualize_optimization(update_func, max_iter=50, **kwargs):
    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)

    cax = fig.add_axes([0.93, 0.515, 0.02, 0.4])

    ax1.set_ylim(-1, 8)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")

    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel(r"$w$")
    ax2.set_ylabel(r"$b$")
    ax2.set_aspect("auto")

    fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=L_min, vmax=L_max),
            cmap="plasma_r"
        ),
        cax=cax)

    ax2.contourf(W, B, L, levels=np.linspace(0, 500, 50), cmap="plasma_r")

    traj = get_trajectory(update_func, max_iter=max_iter, **kwargs)

    w = traj[np.argmin(traj[:, 2]), 0]
    b = traj[np.argmin(traj[:, 2]), 1]
    
    ax2.plot(traj[:, 0], traj[:, 1], color="blue")
    ax2.scatter(w, b, color="red", marker="x", s=60, zorder=10)
    
    xx = np.linspace(0, 10, 100)
    yy_hat = w * xx + b
    y_hat_train = w * x_train + b

    line_handle, = ax1.plot(xx, yy_hat, color="orange")
    scatter_handle = ax1.scatter(x_train, y_train)
    vline_handles = ax1.vlines(x_train.T, ymin=y_train.T, ymax=y_hat_train,
                               linestyle="dashed", color='r', alpha=0.3)

    loss_points_handle = ax2.scatter([], [], s=10, alpha=0.5, cmap="plasma_r", vmin=L_min, vmax=L_max)


    ax3.plot(traj[:, 2], color="blue")
    ax3.set_xlabel("Iterationen")
    ax3.set_ylabel(r"$\mathcal{L}_E$")
    
    return fig, [ax1, ax2, ax3, cax]

    
    