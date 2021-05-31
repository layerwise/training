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


def generate_data(n_samples=20, random_state=None):
    rng = np.random.RandomState(random_state)
    # Beobachtungen
    x_sample = 40 * rng.rand(n_samples)

    # Kennzeichnungen/Labels
    f_sample = true_function(x_sample)
    noise = 7 * rng.randn(n_samples)

    y_sample = f_sample + noise    
    return x_sample[:, np.newaxis], y_sample


