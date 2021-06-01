import numpy as np
import matplotlib.pyplot as plt
try:
    from ipywidgets import interactive
except ModuleNotFoundError:
    print("No module named `ipywidgets`. Interactive plots cannot be used, otherwise you're fine")


rng = np.random.RandomState(1)

def get_xor_data(n_samples=80, scale=0.25):
    class1_samples = int(n_samples/2)
    class2_samples = n_samples - class1_samples
    
    class1_assign = rng.uniform(size=(class1_samples, 1)) < 0.5
    class2_assign = rng.uniform(size=(class1_samples, 1)) < 0.5
    
    class1 = (class1_assign * rng.normal(loc=[0, 0], scale=scale, size=(class1_samples, 2))
              + (1 - class1_assign) * rng.normal(loc=[1, 1], scale=scale, size=(class1_samples, 2)))
    class2 = (class2_assign * rng.normal(loc=[0, 1], scale=scale, size=(class2_samples, 2))
              + (1 - class2_assign) * rng.normal(loc=[1, 0], scale=scale, size=(class1_samples, 2)))
              
    x = np.concatenate((class1, class2), axis=0)
    y = np.concatenate((-1.0 * np.ones(class1_samples), np.ones(class2_samples)), axis=0)
    return x, y


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    ax.set_aspect("equal")
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=2, facecolors='none', edgecolors='black')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def get_interactive_svc(X, y, buffer=0.5):

    # preparation of figure and subplots including axes and labels
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)

    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    xrange_ = xmax - xmin
    lim_x = (xmin[0] - buffer * xrange_[0], xmax[0] + buffer * xrange_[0])

    ax.set_xlim(lim_x[0], lim_x[1])
    ax.set_ylim(xmin[1] - buffer * xrange_[1], xmax[1] + buffer * xrange_[1])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")

    # interactive plot handles
    scatter_handle = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

    w1 = 0.1
    w2 = 1.0
    bias = 0.0
    xx = np.linspace(lim_x[0], lim_x[1], num=100)

    def _yfunc(x, w1, w2, bias):
        return - (w1 / w2) * x - bias / w2

    def _dfunc(x, w1, w2, bias):
        return - (w1 / w2) * x - (bias - 1) / w2

    def _ufunc(x, w1, w2, bias):
        return - (w1 / w2) * x - (bias + 1)/ w2

    yy = _yfunc(xx, w1, w2, bias)
    ll = _dfunc(xx, w1, w2, bias)
    uu = _ufunc(xx, w1, w2, bias)
    v0 = _yfunc(0, w1, w2, bias)

    projection_vector, = ax.plot([0, w1], [v0, v0+w2], color="blue")
    decision_boundary, = ax.plot(xx, yy, color="k", alpha=0.5, linestyle="-")
    lower_boundary, = ax.plot(xx, ll, color="k", alpha=0.5, linestyle="--")
    upper_boundary, = ax.plot(xx, uu, color="k", alpha=0.5, linestyle="--")
    vector_tip, = ax.plot(w1, v0 + w2, marker="x", markersize=15, color="blue")
    # ax.plot(0, 0, markersize=10, color="red", marker="o")

    margin_filler = ax.fill_between(xx, y1=ll, y2=uu,
                                    edgecolor='none',
                                    color='#AAAAAA', alpha=0.4)
    # bottom_filler = ax.fill_between(xx, y1=yy, y2=10, color="yellow", alpha=0.1)

    def update(w1=0.1, w2=1.0, bias=0.0):        
        if not w1:
            w1 = 0.0001
        if not w2:
            w2 = 0.0001

        yy = _yfunc(xx, w1, w2, bias)
        ll = _dfunc(xx, w1, w2, bias)
        uu = _ufunc(xx, w1, w2, bias)
        v0 = _yfunc(0, w1, w2, bias)

        vector_tip.set_data(w1, v0 + w2)        
        projection_vector.set_data([0, w1], [v0, v0+w2])
        decision_boundary.set_data(xx, yy)
        lower_boundary.set_data(xx, ll)
        upper_boundary.set_data(xx, uu)

        #global margin_filler
        ax.collections = ax.collections[:1]
        margin_filler = ax.fill_between(xx, y1=ll, y2=uu,
                                        edgecolor='none',
                                        color='#AAAAAA', alpha=0.4)

        fig.canvas.draw_idle()

    interactive_plot = interactive(update, w1=(-2.0, 2.0, 0.01), w2=(-2.0, 2.0, 0.01), bias=(-5.0, 5.0, 0.01))
    return interactive_plot

