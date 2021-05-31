import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive, interactive_output, fixed, HBox, VBox
import ipywidgets as widgets


def true_function_old(x):
    x_copy = -1 * x
    f = 2 * x_copy * np.sin(0.8*x_copy) + 0.5 * x_copy**2 - 5
    return f


def sigmoid(x, L=10, k=2, x_0=20):
    return L / (1 + np.exp(-k * (x - x_0)))


def true_function(x):
    const = 17
    lin = -0.25 * x
    quad = 0.2*(x-20)**2
    sig = sigmoid(x, L=-20, k=0.6, x_0=30)
    # quad_sig = - sigmoid(xx, L=1, k=0.6, x_0=30) * (0.1 * (x-40)**2)
    sig2 = sigmoid(x, L=-50, k=0.8, x_0=37)
    f = const + lin + quad + sig + sig2
    return f


def generate_data(n_samples=50, random_state=None):
    rng = np.random.RandomState(random_state)
    # Beobachtungen
    x_sample = 40 * rng.rand(n_samples)

    # Kennzeichnungen/Labels
    f_sample = true_function(x_sample)
    noise = 7 * rng.randn(n_samples)

    y_sample = f_sample + noise    
    return x_sample, y_sample


def interactive_linear_model(x_sample, y_sample):

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-2, 42)
    ax.set_ylim(-10, 100)

    w = 1.0
    b = 0.0

    x = np.linspace(0, 40, 100)
    y_hat = w * x + b
    y_hat_sample = w * x_sample + b

    line_handle, = ax.plot(x, y_hat, color="orange")
    scatter_handle = ax.scatter(x_sample, y_sample)

    vline_handles = ax.vlines(x_sample.T, ymin=y_sample.T, ymax=y_hat_sample,
                linestyle="dashed",color='r',alpha=0.3)

    quadratic_error = np.mean((y_hat_sample - y_sample)**2)
    absolute_error = np.mean(np.abs(y_hat_sample - y_sample))

    # {:.2f}

    quadratic_error_handle = ax.text(25, 80, f"L2 error: {quadratic_error:.2f}", fontsize=12)
    absolute_error_handle = ax.text(25, 70, f"L1 error: {absolute_error:.2f}", fontsize=12)

    def update(w=1.0, b=0.0):
        y_hat = w * x + b
        y_hat_sample = w * x_sample + b
        line_handle.set_data(x, y_hat)
        array = np.concatenate((x_sample, y_sample, y_hat_sample))

        # does not work:
        # global vline_handles
        # vline_handles.remove()

        # hacky instead
        ax.collections = ax.collections[:1]

        vline_handles = ax.vlines(x_sample.T, ymin=y_sample.T, ymax=y_hat_sample,
                                linestyle="dashed",color='r',alpha=0.3)
        
        quadratic_error = np.mean((y_hat_sample - y_sample)**2)
        absolute_error = np.mean(np.abs(y_hat_sample - y_sample))
        quadratic_error_handle.set_text(f"L2 error: {quadratic_error:.2f}")
        absolute_error_handle.set_text(f"L1 error: {absolute_error:.2f}")
        fig.canvas.draw_idle()


    w1_slider = widgets.FloatSlider(
        value=1.0,
        min=-15.0,
        max=15.0,
        step=0.1,
        description="w1",
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    bias_slider =  widgets.FloatSlider(
        value=0.0,
        min=-5.0,
        max=120.0,
        step=1.0,
        description=r'$\theta$',
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    ui = VBox(
        children=[w1_slider, bias_slider]
    )

    interactive_plot = interactive_output(
        update,
        {"w": w1_slider, "b": bias_slider}
    )
    return interactive_plot, ui



def interactive_quadratic_model(x_sample, y_sample):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)

    w1 = 1.0
    w2 = 0.0
    b = 0.0

    ax.set_xlim(-2, 42)
    ax.set_ylim(-10, 100)

    x = np.linspace(0, 40, 100)
    y_hat = w2 * x**2 + w1 * x + b
    y_hat_sample = w2 * x_sample**2 + w1 * x_sample + b

    line_handle, = ax.plot(x, y_hat, color="orange")
    scatter_handle = ax.scatter(x_sample, y_sample)

    quadratic_error = np.mean((y_hat_sample - y_sample)**2)
    absolute_error = np.mean(np.abs(y_hat_sample - y_sample))

    vline_handles = ax.vlines(x_sample.T, ymin=y_sample.T, ymax=y_hat_sample,
                            linestyle="dashed",color='r',alpha=0.3)

    quadratic_error_handle = ax.text(25, 80, f"L2 error: {quadratic_error:.2f}", fontsize=12)
    absolute_error_handle = ax.text(25, 70, f"L1 error: {absolute_error:.2f}", fontsize=12)

    # {:.2f}

    def update(w2=0.0, w1=1.0, b=0.0):
        y_hat = w2 * x**2 + w1 * x + b
        y_hat_sample = w2 * x_sample**2 + w1 * x_sample + b
        line_handle.set_data(x, y_hat)
        array = np.concatenate((x_sample, y_sample, y_hat_sample))

        # does not work:
        # global vline_handles
        # vline_handles.remove()

        # hacky instead
        ax.collections = ax.collections[:1]

        vline_handles = ax.vlines(x_sample.T, ymin=y_sample.T, ymax=y_hat_sample,
                                linestyle="dashed",color='r',alpha=0.3)
        
        quadratic_error = np.mean((y_hat_sample - y_sample)**2)
        absolute_error = np.mean(np.abs(y_hat_sample - y_sample))
        quadratic_error_handle.set_text(f"L2 error: {quadratic_error:.2f}")
        absolute_error_handle.set_text(f"L1 error: {absolute_error:.2f}")
        fig.canvas.draw_idle()


    w2_slider = widgets.FloatSlider(
        value=0.0,
        min=-2.0,
        max=2.0,
        step=0.01,
        description="w2",
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    w1_slider = widgets.FloatSlider(
        value=1.0,
        min=-15.0,
        max=15.0,
        step=0.1,
        description="w1",
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    bias_slider =  widgets.FloatSlider(
        value=0.0,
        min=-5.0,
        max=120.0,
        step=1.0,
        description=r'$\theta$',
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    ui = VBox(
        children=[w2_slider, w1_slider, bias_slider]
    )

    interactive_plot = interactive_output(
        update,
        {"w2": w2_slider, "w1": w1_slider, "b": bias_slider}
    )
    return interactive_plot, ui


def interactive_cubic_model(x_sample, y_sample):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-2, 42)
    ax.set_ylim(-10, 100)

    w1 = 1.0
    w2 = 0.0
    w3 = 0.0
    b = 0.0

    x = np.linspace(0, 40, 100)
    y_hat = w3 * x**3 + w2 * x**2 + w1 * x + b
    y_hat_sample = w3 * x_sample**3 + w2 * x_sample**2 + w1 * x_sample + b

    line_handle, = ax.plot(x, y_hat, color="orange")
    scatter_handle = ax.scatter(x_sample, y_sample)

    vline_handles = ax.vlines(x_sample.T, ymin=y_sample.T, ymax=y_hat_sample,
                            linestyle="dashed",color='r',alpha=0.3)

    quadratic_error = np.mean((y_hat_sample - y_sample)**2)
    absolute_error = np.mean(np.abs(y_hat_sample - y_sample))

    quadratic_error_handle = ax.text(25, 80, f"L2 error: {quadratic_error:.2f}", fontsize=12)
    absolute_error_handle = ax.text(25, 70, f"L1 error: {absolute_error:.2f}", fontsize=12)


    def update(w3=0.0, w2=0.0, w1=1.0, b=0.0):
        y_hat = w3 * x**3 + w2 * x**2 + w1 * x + b
        y_hat_sample = w3 * x_sample**3 + w2 * x_sample**2 + w1 * x_sample + b
        line_handle.set_data(x, y_hat)
        array = np.concatenate((x_sample, y_sample, y_hat_sample))

        # does not work:
        # global vline_handles
        # vline_handles.remove()

        # hacky instead
        ax.collections = ax.collections[:1]

        vline_handles = ax.vlines(x_sample.T, ymin=y_sample.T, ymax=y_hat_sample,
                                linestyle="dashed",color='r',alpha=0.3)
        
        quadratic_error = np.mean((y_hat_sample - y_sample)**2)
        absolute_error = np.mean(np.abs(y_hat_sample - y_sample))
        quadratic_error_handle.set_text(f"L2 error: {quadratic_error:.2f}")
        absolute_error_handle.set_text(f"L1 error: {absolute_error:.2f}")
        fig.canvas.draw_idle()
    
    w3_slider = widgets.FloatSlider(
        value=0.0,
        min=-0.01,
        max=0.01,
        step=0.001,
        description="w3",
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.3f',
    )

    w2_slider = widgets.FloatSlider(
        value=0.0,
        min=-5.0,
        max=5.0,
        step=0.01,
        description="w2",
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    w1_slider = widgets.FloatSlider(
        value=1.0,
        min=-15.0,
        max=15.0,
        step=0.1,
        description="w1",
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    bias_slider =  widgets.FloatSlider(
        value=0.0,
        min=-5.0,
        max=120.0,
        step=1.0,
        description=r'$\theta$',
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    ui = VBox(
        children=[w3_slider, w2_slider, w1_slider, bias_slider]
    )

    interactive_plot = interactive_output(
        update,
        {"w3": w3_slider, "w2": w2_slider, "w1": w1_slider, "b": bias_slider}
    )
    return interactive_plot, ui



def true_function_2d(x1, x2):
    f = 2 * x1 * np.sin(x2) + 0.5 * x1**2 - np.cos(x2) - 5
    return f


def interactive_linear_2D_Model():
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection="3d")

    w1 = 1.0
    w2 = 1.0
    b = 0.0

    rng = np.random.RandomState(1)
    x1_sample = 10 * rng.rand(100)
    x2_sample = 10 * rng.rand(100)
    f_sample = true_function_2d(x1_sample, x2_sample)
    noise = 10 * rng.randn(100)
    y_sample = f_sample + noise
    ax.scatter(x1_sample, x2_sample, y_sample)


    x1 = np.linspace(0, 10, 100)
    x2 = np.linspace(0, 10, 100)
    X1, X2 = np.meshgrid(x1, x2)
    F = true_function_2d(X1, X2)

    Y_hat = w1 * X1 + w2 * X2 + b
    y_hat_sample = w1 * x1_sample + w2 * x2_sample + b

    contour_handle = ax.contour3D(X1, X2, Y_hat, 50, cmap="viridis")
    scatter_handle = ax.scatter(x1_sample, x2_sample, y_sample)

    error_lines_handles = [
        ax.plot3D(
            [xx1, xx1],
            [xx2, xx2],
            [yy_hat, yy],
            linestyle="dashed",
            color="r",
            alpha=0.3        
        )[0] for xx1, xx2, yy, yy_hat in zip(x1_sample, x2_sample, y_sample, y_hat_sample)
    ]

    def update(w1=1.0, w2=1.0, b=0.0):
        Y_hat = w1 * X1 + w2 * X2 + b
        y_hat_sample = w1 * x1_sample + w2 * x2_sample + b
        
        global contour_handle
        for collection in contour_handle.collections:
            collection.remove()
        contour_handle = ax.contour3D(X1, X2, Y_hat, 50, cmap="viridis")
        for i, error_line_handle in enumerate(error_lines_handles):
            error_line_handle.set_data_3d(
                [x1_sample[i], x1_sample[i]],
                [x2_sample[i], x2_sample[i]],
                [y_sample[i], y_hat_sample[i]]
            )
            
        fig.canvas.draw_idle()

    w2_slider = widgets.FloatSlider(
        value=0.0,
        min=-10.0,
        max=10.0,
        step=0.1,
        description="w2",
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    w1_slider = widgets.FloatSlider(
        value=1.0,
        min=-10.0,
        max=10.0,
        step=0.1,
        description="w1",
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    bias_slider =  widgets.FloatSlider(
        value=0.0,
        min=-15.0,
        max=15.0,
        step=1.0,
        description=r'$\theta$',
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    ui = VBox(
        children=[w2_slider, w1_slider, bias_slider]
    )

    interactive_plot = interactive_output(
        update,
        {"w2": w2_slider, "w1": w1_slider, "b": bias_slider}
    )
    return interactive_plot, ui
