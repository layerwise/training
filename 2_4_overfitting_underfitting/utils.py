import numpy as np
import pandas as pd


rng = np.random.RandomState(1)


def true_function(x):
    f = - 2 * x * np.cos(x) + 0.5* x**2 - 5
    # f = x * np.sin(5*x**2) - 2 * x * np.cos(x) + 0.5* x**2 - 5
    # f = x * np.sin(0.1 * x**2) - 2 * x + 0.5* x**2 - 5 + 2 * np.cos(0.1*x)
    return f

def get_train_data(n_samples=50, mode="raw"):
    # x = np.random.uniform(-5, 5, size=n_samples)
    x = np.random.uniform(0, 10, size=n_samples)
    f = true_function(x)
    noise = np.random.normal(loc=0.0, scale=5.0, size=x.shape[0])
    y = f + noise
    if mode == "raw":
        return x[:, np.newaxis], y
    elif mode == "dataframe":
        return pd.DataFrame({"Zeit nach Betriebsbeginn": x, "Anzahl Kunden": y})
    elif mode == "dict":
        return {"Zeit nach Betriebsbeginn": x[:, np.newaxis], "Anzahl Kunden": y}
    else:
        raise ValueError
    

def get_test_data(n_samples=1000, mode="raw", mode2=1):
    rng = np.random.RandomState(5)
    if mode2==1:
        x = rng.uniform(0, 10, size=n_samples)
    elif mode2==2:
        x = rng.uniform(-2, 12, size=n_samples)
    f = true_function(x)
    noise = rng.normal(loc=0, scale=5.0, size=x.shape[0])
    y = f + noise
    if mode == "raw":
        return x[:, np.newaxis], y
    elif mode == "dataframe":
        return pd.DataFrame({"Zeit nach Betriebsbeginn": x, "Anzahl Kunden": y})
    elif mode == "dict":
        return {"Zeit nach Betriebsbeginn": x[:, np.newaxis], "Anzahl Kunden": y}
    else:
        raise ValueError
