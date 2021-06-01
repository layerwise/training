import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from ipywidgets import interactive, fixed
import ipywidgets as widgets
from functools import partial
from sklearn.decomposition import PCA

from sklearn.base import ClassifierMixin

from sklearn.compose import ColumnTransformer


def get_data(n_samples=200):
    rng = np.random.RandomState(1)
    X = np.dot(rng.rand(2, 2), rng.randn(2, n_samples)).T
    X -= X.mean(axis=0)
    return X
    
def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
    
def _draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    color="red",
                    shrinkA=0, shrinkB=0)
    h = ax.annotate('', v1, v0, arrowprops=arrowprops, zorder=10)
    return h

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrow_handle = ax.arrow(0, 0, v0, v1, color="red")
    return arrow_handle


def _draw_line(v0, v1, ax=None):
    ax = ax or plt.gca()
    m = v0/v1
    x = np.linspace(-3, 3)
    y = m * x
    line_handle, = ax.plot(x, y, color="cyan", alpha=0.5)
    return line_handle


def get_interactive_pca_images(images, n_components=8, normalize=True):
    N, width, height = images.shape

    n_components += 1

    X = images.reshape(N, -1)
    if normalize:
        X = X / 255

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(n_components + 2, n_components)

    axes_components = [fig.add_subplot(gs[-1, i]) for i in range(n_components)]
    axes_components2 = [fig.add_subplot(gs[-2, i]) for i in range(n_components)]
    ax_image = fig.add_subplot(gs[:-2, :])

    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_transformed = pca.transform(X)
    min_components = X_transformed.min(axis=0)
    max_components = X_transformed.max(axis=0)
    range_components = max_components - min_components

    components = np.row_stack((
        pca.mean_,
        pca.components_[:(n_components - 1), :]
    ))

    vmin = components.min(axis=1)
    vmax = components.max(axis=1)

    global weights
    weights = np.zeros(n_components)
    weights[0] = 1.0

    component_image_handles = []
    for i, ax in enumerate(axes_components):
        handle = ax.imshow(
            components[i, :].reshape(width, height),
            cmap="Greys",
            interpolation="none",
            vmin=vmin[i],
            vmax=vmax[i]
        )
        ax.axis("off")
        component_image_handles.append(handle)

    component_image_handles2 = []
    for i, ax in enumerate(axes_components2):
        handle = ax.imshow(
            weights[i] * components[i, :].reshape(width, height),
            cmap="Greys",
            interpolation="none",
            vmin=vmin[i],
            vmax=vmax[i]
        )
        ax.axis("off")
        component_image_handles2.append(handle)

    global image
    image = np.matmul(weights, components).reshape(width, height)

    image_handle = ax_image.imshow(
        image,
        cmap="Greys",
        interpolation="none"
    )
    ax_image.set_xticks([])
    ax_image.set_yticks([])

    def update(**kwargs):
        new_weights = np.array(list(kwargs.values()))

        global weights
        changed = (weights != new_weights)
        weights = new_weights

        global image
        image = np.matmul(weights, components).reshape(width, height)
        image_handle.set_array(image)

        for i, weight in enumerate(weights):
            if changed[i]:
                component_image_handles2[i].set_array(
                    weight * components[i, :].reshape(width, height)
                )

    weight_sliders = [widgets.FloatSlider(
            value=0.0 if i!=0 else 1.0,
            min=min_components[i-1] if i !=0 else 1.0,  # min=-25.0 if i!=0 else 1.0,
            max=max_components[i-1] if i !=0 else 1.0,  #max=25.0 if i!=0 else 1.0,
            step=range_components[i-1]/50. if i!=0 else 1.0,
            description='w%d' % i,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
        ) for i in range(n_components)]

    kwargs = {f"w{i}": slider for i, slider in enumerate(weight_sliders)}

    interactive_plot = interactive(update, **kwargs)

    return fig, axes_components + [ax_image], interactive_plot
    

def get_interactive_pca1(X):

    lim = 2.5

    fig, [ax, ax2] = plt.subplots(ncols=2, nrows=1, figsize=(8,4), constrained_layout=True)

    ax.set_ylim(-lim, lim)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Feature 1", fontsize=10)
    ax.set_ylabel("Feature 2", fontsize=10)
    ax.set_aspect("equal")

    ax2.set_ylim(-lim, lim)
    ax2.set_xlim(-lim, lim)
    ax2.set_xlabel("Feature 1 (rekonstruiert)", fontsize=10)
    ax2.set_ylabel("Feature 2 (rekonstruiert)", fontsize=10)
    ax2.set_aspect("equal")


    v0 = 1.0
    v1 = 1.0
    v = np.array([v0, v1])
    vv = v[np.newaxis, :]

    m = v1/v0
    x = np.linspace(-lim, lim)
    y = m * x
    null= np.array([0, 0])

    X_proj = np.dot(X, vv.T) / (np.linalg.norm(v)**2)
    X_reconst = np.dot(X_proj, vv)

    projections_x = np.stack((X[:, 0], X_reconst[:, 0]), axis=0)
    projections_y = np.stack((X[:, 1], X_reconst[:, 1]), axis=0)

    dummy = np.ones(X_proj.shape[0])   # Make all y values the same

    # plotting handles - ax
    scatter_handle = ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.2, color="blue", zorder=1)
    proj_scatter_handle = ax.scatter(X_reconst[:, 0], X_reconst[:, 1], s=10, color="blue", zorder=2)
    projection_vectors_handle = ax.plot(projections_x, projections_y, linestyle="dashed", color="r", alpha=0.3)
    line_handle, = ax.plot(x, y, color="cyan", alpha=0.5, zorder=3, linestyle="--")
    arrow_handle, = ax.plot([0, v0], [0, v1], color="red", zorder=10, marker="x", lw=2, label="projection vector")
    ax.legend(loc="upper left")
    
    # global arrow_handle
    # arrow_handle = ax.arrow(0, 0, v0, v1, color="red")

    # plotting handles - ax2
    proj_scatter_handle2 = ax2.scatter(X_reconst[:, 0], X_reconst[:, 1], s=10, color="blue", zorder=2)

    def update(v0=1.0, v1=1.0):
        m = v1/(v0 + 1e-8)
        v = np.array([v0, v1])
        line_handle.set_data(x, m*x)
        arrow_handle.set_data([0, v0], [0, v1])
        # global arrow_handle
        # arrow_handle.remove()
        # arrow_handle = draw_vector(v0, v1, ax=ax)

        vv = v[np.newaxis, :]
        X_proj = np.dot(X, vv.T) / (np.linalg.norm(v)**2)
        X_reconst = np.dot(X_proj, vv)
        
        for i, handle in enumerate(projection_vectors_handle):
            handle.set_data(
                [X[i, 0], X_reconst[i, 0]],
                [X[i, 1], X_reconst[i, 1]]
            )
        
        proj_scatter_handle.set_offsets(X_reconst)
        proj_scatter_handle2.set_offsets(X_reconst)
        
        fig.canvas.draw_idle()
        
    # plt.tight_layout()
    interactive_plot = interactive(update, v0=(-2.0, 2.0), v1=(-2.0, 2.0))
    return interactive_plot
    

def get_interactive_pca2(X):
    lim = 2.5

    fig = plt.figure(figsize=(8,8), constrained_layout=True)
    spec = fig.add_gridspec(ncols=2, nrows=2, height_ratios=[3, 1])


    ax = fig.add_subplot(spec[0, 0])
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Feature 1", fontsize=10)
    ax.set_ylabel("Feature 2", fontsize=10)
    ax.set_aspect("equal")
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_ylim(-lim, lim)
    ax2.set_xlim(-lim, lim)
    ax2.set_xlabel("Feature 1 (rekonstruiert)", fontsize=10)
    ax2.set_ylabel("Feature 2 (rekonstruiert)", fontsize=10)
    ax2.set_aspect("equal")

    ax3 = fig.add_subplot(spec[1, 0])
    a = [1,2,5,6,9,11,15,17,18]
    ax3.hlines(1, -lim, lim)  # Draw a horizontal line
    # ax3.axis('off')
    ax3.set_xlim(-lim, lim)
    ax3.set_yticks([])
    ax3.set_xlabel("Komponente 1")

    beta = 45

    v0 = np.cos(beta*np.pi/180)
    v1 = np.sin(beta*np.pi/180)

    v = np.array([v0, v1])
    vv = v[np.newaxis, :]

    m = v1/v0
    x = np.linspace(-lim, lim)
    y = m * x
    null= np.array([0, 0])

    X_proj = np.dot(X, vv.T) / (np.linalg.norm(v)**2)
    X_reconst = np.dot(X_proj, vv)

    dummy = np.ones(X_proj.shape[0])   # Make all y values the same
    component_handle, = ax3.plot(X_proj, dummy, 'D', alpha=0.5)


    projections_x = np.stack((X[:, 0], X_reconst[:, 0]), axis=0)
    projections_y = np.stack((X[:, 1], X_reconst[:, 1]), axis=0)

    # plotting handles - ax
    scatter_handle = ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.2, color="blue", zorder=1)
    proj_scatter_handle = ax.scatter(X_reconst[:, 0], X_reconst[:, 1], s=10, color="blue", zorder=2)
    projection_vectors_handle = ax.plot(projections_x, projections_y, linestyle="dashed", color="r", alpha=0.3)
    line_handle, = ax.plot(x, y, color="cyan", alpha=0.5, zorder=3)

    # arrow_handle = draw_vector(null, v, ax=ax)
    arrow_handle, = ax.plot([0, v0], [0, v1], color="red", zorder=10, marker="x", lw=2, label="projection vector")

    ax.legend(loc="upper left")

    # plotting handles - ax2
    proj_scatter_handle2 = ax2.scatter(X_reconst[:, 0], X_reconst[:, 1], s=10, color="blue", zorder=2)

    def update(beta=45):
        v0 = np.cos(beta*np.pi/180)
        v1 = np.sin(beta*np.pi/180)
        
        m = v1/(v0 + 1e-8)
        v = np.array([v0, v1])
        line_handle.set_data(x, m*x)

        vv = v[np.newaxis, :]
        X_proj = np.dot(X, vv.T) / (np.linalg.norm(v)**2)
        X_reconst = np.dot(X_proj, vv)
        
        length = np.std(X_proj) * 2
        
        #global arrow_handle
        #arrow_handle.remove()
        #arrow_handle = draw_vector(null, v * length, ax=ax)

        arrow_handle.set_data([0, v[0] * length], [0, v[1] * length])
        
        for i, handle in enumerate(projection_vectors_handle):
            handle.set_data(
                [X[i, 0], X_reconst[i, 0]],
                [X[i, 1], X_reconst[i, 1]]
            )
        
        proj_scatter_handle.set_offsets(X_reconst)
        proj_scatter_handle2.set_offsets(X_reconst)
        component_handle.set_data(X_proj, dummy)
        
        fig.canvas.draw_idle()
        
    # plt.tight_layout()
    interactive_plot = interactive(update, beta=(0, 360, 5))
    return interactive_plot


def get_interactive_pca3(X):
    lim = 2.5

    fig = plt.figure(figsize=(8,6), constrained_layout=False)

    ax = fig.add_subplot(1, 2, 1)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Feature 1", fontsize=10)
    ax.set_ylabel("Feature 2", fontsize=10)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Komponente 1", fontsize=10)
    ax2.set_ylabel("Komponente 2", fontsize=10)

    beta = 45

    v0 = np.array([np.cos(beta*np.pi/180), np.sin(beta*np.pi/180)])
    v1 = np.array([-v0[1], v0[0]])
    V = np.stack((v0, v1), axis=1)

    X_proj = np.dot(X, V)
    X_reconst = np.dot(X_proj, V.T)

    null= np.array([0, 0])

    scatter_handle = ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.2, color="blue", zorder=1)

    # v0_handle = draw_vector(v0[0], v0[1], ax=ax)
    # v1_handle = draw_vector(v1[0], v1[1], ax=ax)

    v0_handle, = ax.plot([0, v0[0]], [0, v0[1]], color="red", zorder=10, lw=2, label="projection vector 1")
    v1_handle, = ax.plot([0, v1[0]], [0, v1[1]], color="red", zorder=10, lw=2, label="projection vector 1")

    draw_vector(0, 2, ax=ax2)
    draw_vector(2, 0, ax=ax2)

    # plotting handles - ax2
    proj_scatter_handle2 = ax2.scatter(X_proj[:, 0], X_proj[:, 1], s=10, color="blue", zorder=2)

    def update(beta=45, v0_handle=None, v1_handle=None):
        v0 = np.array([np.cos(beta*np.pi/180), np.sin(beta*np.pi/180)])
        v1 = np.array([-v0[1], v0[0]])
        
        # m = v1/(v0 + 1e-8)
        # v = np.array([v0, v1])
        # line_handle.set_data(x, m*x)
        
        V = np.stack((v0, v1), axis=1)

        # vv = v0[np.newaxis, :]
        X_proj = np.dot(X, V)
        lengths = np.std(X_proj, axis=0)
        X_reconst = np.dot(X_proj, V.T)
        
        # global v0_handle
        # v0_handle.remove()
        # v0_handle = draw_vector(null, v0 * lengths[0] * 2, ax=ax)
        # v0_handle = draw_vector(v0[0] * lengths[0] * 2, v0[1] * lengths[0] * 2, ax=ax)
        v0_handle.set_data([0, v0[0] * 2 * lengths[0]], [0, v0[1] * 2 * lengths[0]])
        
        # global v1_handle
        # v1_handle.remove()
        # v1_handle = draw_vector(null, v1 * lengths[1] * 2, ax=ax)
        # v1_handle = draw_vector(v1[0] * lengths[1] * 2, v1[1] * lengths[1] * 2, ax=ax)
        v1_handle.set_data([0, v1[0] * 2 * lengths[1]], [0, v1[1] * 2 * lengths[1]])
        
        # proj_scatter_handle.set_offsets(X_reconst)
        proj_scatter_handle2.set_offsets(X_proj)
        # component_handle.set_data(X_proj, dummy)
        
        fig.canvas.draw_idle()
        
    plt.tight_layout()
    interactive_plot = interactive(update, beta=(0, 360, 5), v0_handle=fixed(v0_handle), v1_handle=fixed(v1_handle))
    return interactive_plot


def get_interactive_pca4(X):
    lim = 2.5

    fig = plt.figure(figsize=(8,6), constrained_layout=False)

    ax = fig.add_subplot(1, 2, 1)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Feature 1", fontsize=10)
    ax.set_ylabel("Feature 2", fontsize=10)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Komponente 1", fontsize=10)
    ax2.set_ylabel("Komponente 2", fontsize=10)

    beta = 45

    v0 = np.array([np.cos(beta*np.pi/180), np.sin(beta*np.pi/180)])
    v1 = np.array([-v0[1], v0[0]])
    V = np.stack((v0, v1), axis=1)

    X_proj = np.dot(X, V)
    X_reconst = np.dot(X_proj, V.T)

    scatter_handle = ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.2, color="blue", zorder=1)

    # v0_handle = draw_vector(null, v0, ax=ax)
    # v1_handle = draw_vector(null, v1, ax=ax)

    v0_handle, = ax.plot([0, v0[0]], [0, v0[1]], color="red", zorder=10, lw=2, label="projection vector 1")
    v1_handle, = ax.plot([0, v1[0]], [0, v1[1]], color="red", zorder=10, lw=2, label="projection vector 1")

    draw_vector(0, 2, ax=ax2)
    draw_vector(2, 0, ax=ax2)

    # plotting handles - ax2
    proj_scatter_handle2 = ax2.scatter(X_proj[:, 0], X_proj[:, 1], s=10, color="blue", zorder=2)

    def update(beta=45):
        v0 = np.array([np.cos(beta*np.pi/180), np.sin(beta*np.pi/180)])
        v1 = np.array([-v0[1], v0[0]])
        
        # m = v1/(v0 + 1e-8)
        # v = np.array([v0, v1])
        # line_handle.set_data(x, m*x)
        
        V = np.stack((v0, v1), axis=1)

        # vv = v0[np.newaxis, :]
        X_proj = np.dot(X, V)
        lengths = np.std(X_proj, axis=0)
        X_proj /= lengths
        X_reconst = np.dot(X_proj, V.T)
        
        #global v0_handle
        #v0_handle.remove()
        #v0_handle = draw_vector(null, v0 * lengths[0] * 2, ax=ax)
        v0_handle.set_data([0, v0[0] * 2 * lengths[0]], [0, v0[1] * 2 * lengths[0]])
        
        #global v1_handle
        #v1_handle.remove()
        #v1_handle = draw_vector(null, v1 * lengths[1] * 2, ax=ax)
        v1_handle.set_data([0, v1[0] * 2 * lengths[1]], [0, v1[1] * 2 * lengths[1]])
        
        # proj_scatter_handle.set_offsets(X_reconst)
        proj_scatter_handle2.set_offsets(X_proj)
        
        fig.canvas.draw_idle()
        
    plt.tight_layout()
    interactive_plot = interactive(update, beta=(0, 360, 5))
    return interactive_plot