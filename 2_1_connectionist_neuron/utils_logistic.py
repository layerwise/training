import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, interactive_output, fixed, HBox, VBox
import ipywidgets as widgets


def get_interactive_logistic_regression(X, y):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1, 1, 1)

    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    xrange_ = xmax - xmin
    lim_x = (xmin[0] - 0.1 * xrange_[0], xmax[0] + 0.1 * xrange_[0])

    ax.set_xlim(lim_x[0], lim_x[1])
    ax.set_ylim(xmin[1] - 0.1 * xrange_[1], xmax[1] + 0.1 * xrange_[1])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")

    scatter_handle = plt.scatter(X[:, 0], X[:, 1], c=y)

    w1 = 0.1
    w2 = 1.0
    projection_vector, = ax.plot([0, w1], [0, w2], color="blue")
    projection_vector2, = ax.plot([-w1, w1], [-w2, w2], linestyle="--", color="blue")
    decision_boundary, = ax.plot([0, -w2], [0, w1], color="orange")
    vector_tip, = ax.plot(w1, w2, marker="x", markersize=15, color="blue")
    ax.plot(0, 0, markersize=10, color="red", marker="o")

    xx = np.linspace(lim_x[0], lim_x[1], num=100)
    yy = -(w1/w2) * xx
    top_filler = ax.fill_between(xx, y1=-10, y2=yy, color="purple", alpha=0.1)
    bottom_filler = ax.fill_between(xx, y1=yy, y2=10, color="yellow", alpha=0.1)

    def update(w1=0.1, w2=1.0, bias=0.0):
        vector_tip.set_data(w1, w2)
        
        if not w1:
            w1 = 0.0001
        if not w2:
            w2 = 0.0001
        
        w = np.array([w1, w2])
        # bias_vec = w * bias / np.linalg.norm(w)  # TODO figure this out
        # b1, b2 = bias_vec
        # decision_boundary.set_data([w2+b1, -w2+b1], [-w1+b2, w1+b2])
        projection_vector.set_data([0, w1], [0, w2])
        decision_boundary.set_data([lim_x[0], lim_x[1]], [- w1/w2 * lim_x[0] + bias/w2, - w1/w2 * lim_x[1] + bias/w2])
        
        if not w2 == 0:
            # yy = -(w1/w2) * xx + b2 + w1/w2 * b1
            yy = -(w1/w2) * xx + bias/w2
            ax.collections = ax.collections[:1]

            if w2 > 0:
                top_filler = ax.fill_between(xx, y1=-10, y2=yy, color="purple", alpha=0.1)
                bottom_filler = ax.fill_between(xx, y1=yy, y2=10, color="yellow", alpha=0.1)
            else:
                top_filler = ax.fill_between(xx, y1=-10, y2=yy, color="yellow", alpha=0.1)
                bottom_filler = ax.fill_between(xx, y1=yy, y2=10, color="purple", alpha=0.1)

        w /= np.linalg.norm(w)
        w *= 5
        w1, w2 = w
        projection_vector2.set_data([-w1, w1], [-w2, w2])  

        fig.canvas.draw_idle()

    interactive_plot = interactive(update, w1=(-2.0, 2.0), w2=(-2.0, 2.0), bias=(-3.0, 3.0))
    return interactive_plot


def get_interactive_logistic_regression_advanced(X, y, X_test=None, y_test=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    w1 = 0.5
    w2 = 0.5
    bias = 0.0

    x1 = 0.0
    x2 = 0.0
    h = x1*w1 + x2*w2 - bias
    y_hat = int(h >= 0)

    w1_slider = widgets.FloatSlider(
        value=0.5,
        min=-5.0,
        max=5.0,
        step=0.1,
        description="w1",
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )

    w2_slider = widgets.FloatSlider(
        value=-1.0,
        min=-5.0,
        max=5.0,
        step=0.1,
        description="w2",
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )

    bias_slider =  widgets.FloatSlider(
        value=0.0,
        min=-5.0,
        max=5.0,
        step=0.1,
        description=r'$\theta$',
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )

    x1_slider = widgets.FloatSlider(
        value=0.0,
        min=-2.0,
        max=2.0,
        step=0.1,
        description="x1",
        disabled=False,
        continuous_update=True,
        # orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )

    x2_slider = widgets.FloatSlider(
        value=0.0,
        min=-2.0,
        max=2.0,
        step=0.1,
        description="x2",
        disabled=False,
        continuous_update=True,
        # orientation='horizontal',
        readout=True,
        readout_format='.1f',
    )

    show_train = widgets.Checkbox(
        value=False,
        description='Show train data',
        disabled=False
    )

    show_test = widgets.Checkbox(
        value=False,
        description='Show test data',
        disabled=False
    )

    show_boundary = widgets.Checkbox(
        value=False,
        description='Show decision boundary',
        disabled=False
    )

    show_prediction = widgets.Checkbox(
        value=False,
        description='Show prediction',
        disabled=False
    )

    show_weight_vector = widgets.Checkbox(
        value=False,
        description='Show weight vector',
        disabled=False
    )

    show_h = widgets.Checkbox(
        value=False,
        description=r'Show $h$',
        disabled=False
    )

    caption1 = widgets.Label(
        value=r"$h = w_1 \cdot x_1 + w_2 \cdot x_2 - \theta$"
    )
    caption2 = widgets.Label(
        #value=f"{w1} * {x1} + {w2} * {x2} - {bias}"
        # value=f"({w1}) * ({x1}) + ({w2}) * ({x2}) - ({bias})"
        value=f"{format(h, '.3f')} = ({w1}) * ({x1}) + ({w2}) * ({x2}) - ({bias})"
    )
    caption3 = widgets.Label(
        value=r"$\hat{y} = f(h)$"
    )
    caption4 = widgets.Label(
        value=f"{y_hat} = f({h})"
    )

    #label2 = widgets.Label(
    #    value=fr"${w1} * {x1} + {w2} * {x2} - {bias} = {y_hat} $"
    #)

    box1 = VBox(
        children=[x1_slider, x2_slider, show_train, show_test]
    )
    box2 = VBox(
        children=[w1_slider, w2_slider, bias_slider, show_boundary, show_prediction]
    )
    box3 = VBox(
        children=[caption1, caption2, caption3, caption4]
    )

    ui = HBox(
        children=[box3, box2, box1]
    )

    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    xrange_ = xmax - xmin
    lim_x = (xmin[0] - 0.1 * xrange_[0], xmax[0] + 0.1 * xrange_[0])

    ax.set_xlim(lim_x[0], lim_x[1])
    ax.set_ylim(xmin[1] - 0.1 * xrange_[1], xmax[1] + 0.1 * xrange_[1])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")

    training_data_handle = plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.0, marker="x", s=15)
    has_test = X_test is not None and y_test is not None
    if has_test:
        test_data_handle = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.0, marker="D", s=15)

    test_point_handle1 = plt.scatter([x1], [x2], s=150, linewidth=2, facecolors='none', edgecolors='black')
    test_point_handle2 = plt.scatter([x1], [x2], s=50, edgecolors='none', alpha=(y_hat==0), c=0.0, vmin=0.0, vmax=1.0)
    test_point_handle3 = plt.scatter([x1], [x2], s=50, edgecolors='none', alpha=(y_hat==1), c=1.0, vmin=0.0, vmax=1.0)
    # test_point_handle4 = plt.scatter([x1], [x2], s=50, edgecolors='none', alpha=0.0, c=h, vmin=-2.0, vmax=2.0)

    decision_boundary, = ax.plot([0, -w2], [0, w1], color="red", alpha=0.0)

    #projection_vector, = ax.plot([0, w1], [0, w2], color="blue")
    #projection_vector2, = ax.plot([-w1, w1], [-w2, w2], linestyle="--", color="blue")

    #vector_tip, = ax.plot(w1, w2, marker="x", markersize=15, color="blue")
    #ax.plot(0, 0, markersize=10, color="red", marker="o")

    xx = np.linspace(lim_x[0], lim_x[1], num=100)
    yy = -(w1/w2) * xx
    top_filler = ax.fill_between(xx, y1=-10, y2=yy, color="purple", alpha=0.0)
    bottom_filler = ax.fill_between(xx, y1=yy, y2=10, color="yellow", alpha=0.0)

    def update(w1=0.5, w2=0.5, bias=0.0, x1=0.0, x2=0.0, show_train=False, show_test=False, show_boundary=False, show_prediction=False):
        # vector_tip.set_data(w1, w2)
        
        h = x1*w1 + x2*w2 - bias
        y_hat = int(h >= 0)
        
        w = np.array([w1, w2])
        # bias_vec = w * bias / np.linalg.norm(w)  # TODO figure this out
        # b1, b2 = bias_vec

        # UPDATE HANDLES

        # training data scatterplot - set alpha to enable/disable
        training_data_handle.set_alpha(0.75 if show_train else 0.0)
        if has_test:
            test_data_handle.set_alpha(0.75 if show_test else 0.0)

        # test point - move along x1/x2 and switch color
        test_point_handle1.set_offsets([x1, x2])
        test_point_handle2.set_offsets([x1, x2])
        test_point_handle3.set_offsets([x1, x2])
        # test_point_handle4.set_offsets([x1, x2])
        #if not show_h:
        test_point_handle2.set_alpha((y_hat==0))
        test_point_handle3.set_alpha((y_hat==1))
        # test_point_handle4.set_alpha(0.0)
        # else:
            #test_point_handle2.set_alpha(0.0)
            #test_point_handle3.set_alpha(0.0)
            #test_point_handle4.set_alpha(1.0)           

        caption2.value = f"{format(round(h, 3), '.3f')} = ({round(w1, 2)}) * ({round(x1, 2)}) + ({round(w2, 2)}) * ({round(x2, 2)}) - ({round(bias, 2)})"
        caption4.value = f"{y_hat} = f({round(h, 2)})"

        # decision_boundary.set_data([w2+b1, -w2+b1], [-w1+b2, w1+b2])
        # projection_vector.set_data([0, w1], [0, w2])
        if w2:
            decision_boundary.set_data([lim_x[0], lim_x[1]], [- w1/w2 * lim_x[0] + bias/w2, - w1/w2 * lim_x[1] + bias/w2])
        else:
            if w1:
                decision_boundary.set_data([bias/w1, bias/w1], [lim_x[0], lim_x[1]])
            else:
                decision_boundary.set_data([], [])
                decision_boundary.set_alpha(0.0)
        
        decision_boundary.set_alpha(1.0 if show_boundary else 0.0)

        # yy = -(w1/w2) * xx + b2 + w1/w2 * b1
        ax.collections = ax.collections[:-2]
        alpha = 0.1 if show_prediction else 0.0
        if w2:
            yy = -(w1/w2) * xx + bias/w2
            if w2 > 0:
                top_filler = ax.fill_between(xx, y1=-10, y2=yy, color="purple", alpha=alpha)
                bottom_filler = ax.fill_between(xx, y1=yy, y2=10, color="yellow", alpha=alpha)
            else:
                top_filler = ax.fill_between(xx, y1=-10, y2=yy, color="yellow", alpha=alpha)
                bottom_filler = ax.fill_between(xx, y1=yy, y2=10, color="purple", alpha=alpha)
        else:
            if w1:
                if w1 > 0:
                    top_filler = ax.fill_betweenx([lim_x[0], lim_x[1]], x1=-10, x2=bias/w1, color="purple", alpha=alpha)
                    bottom_filler = ax.fill_betweenx([lim_x[0], lim_x[1]], x1=bias/w1, x2=10, color="yellow", alpha=alpha)
                else:
                    top_filler = ax.fill_betweenx([lim_x[0], lim_x[1]], x1=-10, x2=bias/w1, color="yellow", alpha=alpha)
                    bottom_filler = ax.fill_betweenx([lim_x[0], lim_x[1]], x1=bias/w1, x2=10, color="purple", alpha=alpha)
            else:
                if bias > 0:
                    top_filler = ax.fill_betweenx([lim_x[0], lim_x[1]], x1=-10, x2=10, color="purple", alpha=0.1)
                    bottom_filler = ax.fill_betweenx([lim_x[0], lim_x[1]], x1=-10, x2=10, color="yellow", alpha=0.0)
                else:
                    top_filler = ax.fill_betweenx([lim_x[0], lim_x[1]], x1=-10, x2=10, color="purple", alpha=0.0)
                    bottom_filler = ax.fill_betweenx([lim_x[0], lim_x[1]], x1=-10, x2=10, color="yellow", alpha=0.1)                                     

        #w /= np.linalg.norm(w)
        #w *= 5
        #w1, w2 = w
        #projection_vector2.set_data([-w1, w1], [-w2, w2])  

        fig.canvas.draw_idle()

    interactive_plot = interactive_output(
        update,
        {"w1":w1_slider,
        "w2":w2_slider,
        "bias":bias_slider,
        "x1":x1_slider,
        "x2":x2_slider,
        "show_train":show_train,
        "show_test":show_test,
        "show_boundary": show_boundary,
        "show_prediction": show_prediction
        }
    )

    #interactive_plot = interactive(
    #    update,
    #    w1=w1_slider,
    #    w2=w2_slider,
    #    bias=bias_slider,
    #    x1=x1_slider,
    #    x2=x2_slider,
    #    train=show_train,
    #    test=show_test
    #)

    return interactive_plot, ui


def stepwise(h):
    return (h >= 0).astype(np.int32)


def get_interactive_logistic_regression_univariate(x, y):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-2, 2)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$y_T$")
    ax.set_yticks([0, 1])
    # ax.set_aspect("equal")

    scatter_handle = plt.scatter(x, y, c=y)

    w1 = 1.0
    bias = 0.0

    if w1 == 0:
        w1 = 0.001
    
    # projection_vector, = ax.plot([0, w1], [0, w2])
    # decision_boundary, = ax.plot([0, -w2], [0, w1])
    # vector_tip, = ax.plot(w1, w2, marker="x", markersize=15, color="blue")
    # ax.plot(0, 0, markersize=10, color="red", marker="o")

    x_dfunc = np.linspace(-2, 2, num=1000)
    h_dfunc = w1 * x_dfunc - bias
    y_dfunc = stepwise(h_dfunc)
    
    decision_func, = ax.plot(x_dfunc, y_dfunc, color="black")
    if w1 >=0:
        left_filler = ax.fill_betweenx([-0.1, 1.1], x1=-5, x2=bias/w1, color="purple", alpha=0.1)
        right_filler = ax.fill_betweenx([-0.1, 1.1], x1=bias/w1, x2=5, color="yellow", alpha=0.1)
    else:
        left_filler = ax.fill_betweenx([-0.1, 1.1], x1=-5, x2=bias/w1, color="yellow", alpha=0.1)
        right_filler = ax.fill_betweenx([-0.1, 1.1], x1=bias/w1, x2=5, color="purple", alpha=0.1)


    def update(w1=1.0, bias=0.0):
        # vector_tip.set_data(w1, w2)
        
        if w1 == 0.0:
            w1 = 0.001
        
        # w = np.array([w1, w2])
        # bias_vec = w * bias/np.linalg.norm(w)
        # b1, b2 = bias_vec
        # w /= np.linalg.norm(w)
        # w *= 5
        # w1, w2 = w

        # projection_vector.set_data([-w1, w1], [-w2, w2])
        # decision_boundary.set_data([w2+b1, -w2+b1], [-w1+b2, w1+b2])

        h_dfunc = w1 * x_dfunc - bias
        y_dfunc = stepwise(h_dfunc)

        decision_func.set_data(x_dfunc, y_dfunc)

        # y = -(w1/w2)*x + b2 + w1/w2*b1
        ax.collections = ax.collections[:1]
        if w1 >=0:
            left_filler = ax.fill_betweenx([-0.1, 1.1], x1=-5, x2=bias/w1, color="purple", alpha=0.1)
            right_filler = ax.fill_betweenx([-0.1, 1.1], x1=bias/w1, x2=5, color="yellow", alpha=0.1)
        else:
            left_filler = ax.fill_betweenx([-0.1, 1.1], x1=-5, x2=bias/w1, color="yellow", alpha=0.1)
            right_filler = ax.fill_betweenx([-0.1, 1.1], x1=bias/w1, x2=5, color="purple", alpha=0.1)    

        fig.canvas.draw_idle()

    interactive_plot = interactive(update, w1=(-2.0, 2.0), bias=(-3.0, 3.0))
    return interactive_plot


class InteractiveConnectionistNeuron:
    def __init__(
            self,
            w1_range=(-5.0, 5.0, 0.05),
            w2_range=(-5.0, 5.0, 0.05),
            bias_range=(-3.0, 3.0, 0.01),
            xlabel=None,
            ylabel=None
        ):
        self.w1 = None
        self.w2 = None
        self.bias = None
        self.w1_range = w1_range
        self.w2_range = w2_range
        self.bias_range = bias_range
        if xlabel is None:
            self.xlabel = "price per sqft / maximum price per sqft"
        else:
            self.xlabel = xlabel
        if ylabel is None:
            self.ylabel = "elevation / maximum elevation"
        else:
            self.ylabel = ylabel
        
    def fit(self, X, y):
        if not X.ndim == 2:
            raise ValueError
        if not X.shape[1] == 2:
            raise ValueError("Matrix X should have only two features.")
            
        xx = np.linspace(0, 2 * np.pi)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlim(-0.2, 1.2)
        ax.set_aspect("equal")

        scatter_handle = plt.scatter(X[:, 0], X[:, 1], c=y)

        w1 = 0.1
        w2 = 1.0
        final_w1 = w1
        final_w2 = w2

        projection_vector, = ax.plot([0, w1], [0, w2], color="blue")
        projection_vector2, = ax.plot([-w1, w1], [-w2, w2], linestyle="--", color="blue")
        decision_boundary, = ax.plot([0, -w2], [0, w1], color="orange", label="Entscheidungsgrenze")
        vector_tip, = ax.plot(w1, w2, marker="x", markersize=15, color="blue", label="[w1, w2]")
        ax.plot(0, 0, markersize=10, color="red", marker="o", label="[0, 0]")

        xx = np.linspace(-2, 2, num=100)
        yy = - (w1 / w2) * xx
        top_filler = ax.fill_between(xx, y1=-10, y2=yy, color="purple", alpha=0.1)
        bottom_filler = ax.fill_between(xx, y1=yy, y2=10, color="yellow", alpha=0.1)
        
        handles, labels = scatter_handle.legend_elements(prop="colors", alpha=0.6)
        
        legend1 = ax.legend(loc="lower right")
        ax.add_artist(legend1)
        legend2 = ax.legend(handles, labels, loc="upper right", title="Labels")
        
        

        def update(w1=0.1, w2=1.0, bias=0.0):
            vector_tip.set_data(w1, w2)
            
            self.w1 = w1
            self.w2 = w2
            self.bias = bias

            if not w1:
                w1 = 0.0001
            if not w2:
                w2 = 0.0001

            w = np.array([w1, w2])

            projection_vector.set_data([0, w1], [0, w2])
            decision_boundary.set_data([-2, 2], [- w1/w2 * (-2) + bias/w2, - w1/w2 * 2 + bias/w2])

            if not w2 == 0:
                yy = - (w1 / w2) * xx + bias / w2
                ax.collections = ax.collections[:1]
                if w2 > 0:
                    top_filler = ax.fill_between(xx, y1=-10, y2=yy, color="purple", alpha=0.1)
                    bottom_filler = ax.fill_between(xx, y1=yy, y2=10, color="yellow", alpha=0.1)
                else:
                    top_filler = ax.fill_between(xx, y1=-10, y2=yy, color="yellow", alpha=0.1)
                    bottom_filler = ax.fill_between(xx, y1=yy, y2=10, color="purple", alpha=0.1)

            w /= np.linalg.norm(w)
            w *= 5
            w1, w2 = w
            projection_vector2.set_data([-w1, w1], [-w2, w2])         

            fig.canvas.draw_idle()

        interactive_plot = interactive(update, w1=(-6.0, 6.0, 0.05), w2=(-5.0, 5.0, 0.05), bias=(-3.0, 3.0, 0.01))
        return interactive_plot
    
    def predict(self, X):
        raise NotImplementedError
