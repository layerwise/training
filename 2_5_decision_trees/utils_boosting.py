import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, interactive_output, fixed, HBox, VBox
import ipywidgets as widgets

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.decomposition import PCA

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
    noise = 6 * rng.randn(n_samples)

    y_sample = f_sample + noise    
    return x_sample.reshape(-1, 1), y_sample


def make_all_predictions(model, X):
    init = model.init_
    estimators = model.estimators_
    estimators = [estimator[0] for estimator in estimators]

    lr = model.learning_rate

    init_prediction = init.predict(X)
    predictions = np.array([lr * estimator.predict(X) for estimator in estimators])
    cumulative_predictions = np.cumsum(predictions, axis=0)
    cumulative_predictions = cumulative_predictions + init_prediction

    return predictions, cumulative_predictions


def prepare_visuals(model, X, y, X_test, y_test, X_vis):
    
    train_single_predictions, train_predictions = make_all_predictions(model, X)
    test_single_predictions, test_predictions = make_all_predictions(model, X_test)
    vis_single_predictions, vis_cumulative_predictions = make_all_predictions(model, X_vis)

    train_scores = np.mean((train_predictions - y)**2, axis=1)
    test_scores = np.mean((test_predictions - y_test)**2, axis=1)

    return (
        train_scores,
        test_scores,
        train_predictions,
        train_single_predictions,
        test_predictions,
        test_single_predictions,
        vis_single_predictions,
        vis_cumulative_predictions
    )



def get_interactive_boosting(X, y, X_test, y_test, max_depth=2, n_estimators=100):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    lrs = [0.1, 0.5, 1.0]
    colors = ["green", "orange", "blue", "cyan", "yellow"]

    n_models = len(lrs)

    models = [
        GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth).fit(X, y)
        for lr in lrs
    ]

    estimator_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=n_estimators,
        step=1,
        description="# Iterations",
        disabled=False,
        continuous_update=False,
        # orientation='horizontal',
        readout=True,
        # readout_format='.1f',
    )

    show_train = widgets.Checkbox(
        value=True,
        description='Show train data',
        disabled=False
    )

    show_test = widgets.Checkbox(
        value=False,
        description='Show test data',
        disabled=False
    )

    show_fits = [False for __ in lrs]
    show_fits[-1] = True

    show_lrs = [
        widgets.Checkbox(
            value=v,
            description=f'Show fit GB(learning_rate={lr})',
            disabled=False
        ) for v, lr in zip(show_fits, lrs)
    ]

    box1 = VBox(
        children=[estimator_slider, show_train, show_test]
    )
    box2 = VBox(
        children=show_lrs
    )
    ui = HBox(
        children=[box1, box2]
    )

    xmin = X.min()
    xmax = X.max()
    xrange_ = xmax - xmin

    lim_x = (xmin - 0.05 * xrange_, xmax + 0.05 * xrange_)

    ymin = y.min()
    ymax = y.max()
    yrange_ = ymax - ymin
    lim_y = (ymin - 0.05 * yrange_, ymax + 0.05 * yrange_)

    ax1.set_xlim(lim_x[0], lim_x[1])
    ax1.set_ylim(lim_y[0], lim_y[1])
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")

    ax2.set_xlabel("# Iterations")
    ax2.set_ylabel("MSE")

    ax3.set_xlim(lim_x[0], lim_x[1])
    ax3.set_ylim(-0.5*ymax, 0.5*ymax)

    X_vis = np.linspace(lim_x[0], lim_x[1], 100).reshape(-1, 1)

    all_train_scores = []
    all_test_scores = []
    all_train_predictions = []
    all_train_single_predictions = []
    all_vis_single_predictions = []
    all_vis_cumulative_predictions = []

    for model in models:
        (
            train_scores,
            test_scores,
            train_predictions,
            train_single_predictions,
            test_predictions,
            test_single_predictions,
            vis_single_predictions,
            vis_cumulative_predictions
        ) = prepare_visuals(model, X, y, X_test, y_test, X_vis)

        all_train_scores.append(train_scores)
        all_test_scores.append(test_scores)
        all_train_predictions.append(train_predictions)
        all_train_single_predictions.append(train_single_predictions)
        all_vis_single_predictions.append(vis_single_predictions)
        all_vis_cumulative_predictions.append(vis_cumulative_predictions)

    cumulative_fit_handles = []
    i = 0
    for vis_cumulative_predictions in all_vis_cumulative_predictions:
        handle, = ax1.plot(X_vis, vis_cumulative_predictions[0, :], alpha=0.0, color=colors[i])
        cumulative_fit_handles.append(handle)
        i += 1

    training_data_handle = ax1.scatter(X, y, alpha=0.0, marker="x", s=15)
    test_data_handle = ax1.scatter(X_test, y_test, alpha=0.0, marker="D", s=15)

    train_score_handles = []
    test_score_handles = []

    i = 0
    for train_scores, test_scores in zip(all_train_scores, all_test_scores):
        train_handle, = ax2.plot(train_scores, alpha=0.0, color=colors[i])
        test_handle, = ax2.plot(test_scores, alpha=0.0, color=colors[i], linestyle="--")
        train_score_handles.append(train_handle)
        test_score_handles.append(test_handle)
        i += 1

    max_score = max(np.array(all_train_scores).max(), np.array(all_test_scores).max())
    hline_handle, = ax2.plot([1, 1], [0, max_score], linestyle="--", color="black", lw=0.5)

    single_fit_handles = []
    k = 0
    for vis_single_predictions in all_vis_single_predictions:
        handle, = ax3.plot(X_vis, vis_cumulative_predictions[0, :], alpha=0.0, color=colors[k])
        single_fit_handles.append(handle)
        k += 1

    residual_data_handles = []
    for k, model in enumerate(models):
        residual_data_handle = ax3.scatter(X, y, alpha=0.2, marker="x", s=15, color=colors[k])
        residual_data_handles.append(residual_data_handle)

    
    def update(iterations=1, show_train=True, show_test=False, **show_lrs):

        # UPDATE HANDLES

        # training data scatterplot - set alpha to enable/disable
        training_data_handle.set_alpha(0.75 if show_train else 0.0)
        test_data_handle.set_alpha(0.75 if show_test else 0.0)

        show_lrs = list(show_lrs.values())

        for k in range(n_models):
            train_score_handles[k].set_alpha(1.0 if (show_train and show_lrs[k]) else 0.0)
            test_score_handles[k].set_alpha(1.0 if (show_test and show_lrs[k]) else 0.0)

            cumulative_fit_handles[k].set_alpha(1.0 if show_lrs[k] else 0.0)
            cumulative_fit_handles[k].set_data(X_vis, all_vis_cumulative_predictions[k][iterations-1, :])

            residual_data = np.column_stack((X, y - all_train_predictions[k][iterations-1, :]))
            # print(residual_data)
            residual_data_handles[k].set_offsets(residual_data)
            residual_data_handles[k].set_alpha(0.75 if (show_train and show_lrs[k]) else 0.0)

            single_fit_handles[k].set_alpha(1.0 if show_lrs[k] else 0.0)
            single_fit_handles[k].set_data(X_vis, all_vis_single_predictions[k][iterations, :])


        hline_handle.set_data([iterations, iterations], [0, max_score])

        fig.canvas.draw_idle()

    args = {
        f"lr{lr}": show_lr for lr, show_lr in zip(lrs, show_lrs)
    }
    args.update({
        "iterations": estimator_slider,
        "show_train": show_train,
        "show_test": show_test,   
    })

    interactive_plot = interactive_output(update, args)

    return interactive_plot, ui