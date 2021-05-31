import pandas as pd
import os

from learntools.core import *

class FruitDfCreation(EqualityCheckProblem):
    _var = 'fruits'
    _expected = (
            pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas']),
    )
    # TODO: This is a case where it would be nice to have a helper for creating 
    # a solution with multiple alternatives.
    _hint = 'Use the `pd.DataFrame` constructor to create the DataFrame.'
    _solution = CS(
            "fruits = pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas'])"
    )


class DataDownload(CodingProblem):
    show_solution_on_correct = False

    _var = 'data_path'
    _hint = 'Define a variable `data_path` for checking your solution. If the expected files are found in that path, the check will pass.'
    _solution = """Download the data and store it in a location specified by a variable `data_path`.
        There should be two files `house_location_test.csv` and `house_location_train.csv` at that location."""

    def check(self, *args):
        data_path = args[0]
        assert_file_exists(os.path.join(data_path, 'house_location_test.csv'))
        assert_file_exists(os.path.join(data_path, 'house_location_train.csv'))


class DataLoading(CodingProblem):
    show_solution_on_correct = False

    _vars = ['data_path', 'train_data', 'test_data']
    _hints = [
        """There are many alternatives to load in data in .csv format in python.
        Consider using the popular python libraries `pandas` or `numpy`.""",
        'Conventionally, the pandas library should be imported via `import pandas as pd` and the numpy library via `import numpy as np`',
        'Use the function `pd.read_csv` or `np.loadtxt` to load the .csv files',
        'Use the jupyter magic `pd.read_csv?` or `np.loadtxt?` if you get stuck.',
        'Consider a short tutorial on the use of the pandas libary [here](https://www.kaggle.com/learn/pandas)' 
    ]

    _solution = """
```python
import pandas as pd
train_data = pd.read_csv(f'{data_path}/house_location_train.csv', index_col=0)
test_data = pd.read_csv(f'{data_path}/house_location_test.csv', index_col=0)
```
or alternatively but not recommended
```python
import numpy as np
train_data = np.loadtxt(f'{data_path}/house_location_train.csv', skiprows=1, delimiter=",")
test_data = np.loadtxt(f'{data_path}/house_location_test.csv', skiprows=1, delimiter=",")
```
"""

    def check(self, *args):
        data_path = args[0]
        train_data = pd.read_csv(os.path.join(data_path, 'house_location_train.csv'), index_col=0)
        test_data = pd.read_csv(os.path.join(data_path, 'house_location_test.csv'), index_col=0)
        assert_df_equals(train_data, args[1],
            name="Dataframe loaded from `house_location_train.csv`")
        assert_df_equals(test_data, args[2],
            name="Dataframe loaded from `house_location_test.csv`")


class DataExploration(ThoughtExperiment):
    show_solution_on_correct = False

    _hints = [
        """Think about ways to characterize or summarize the data.
        For example, what is the average price of the houses? How many square feet is a house on average in San Francisco vs in New York?""",
        """To determine the average size in square feet for houses in SF vs in NY, use indexing and summary functions.
        Hop over to [this](https://www.kaggle.com/learn/pandas) tutorial fo a refresher.""",
        """Consider plotting a histogram of one of the 'variables', for example the price per square feet.
        Plot one histogram for houses in SF and one for houses in NY. Consider overlaying these histograms in one plot.""",
        """Use the matplotlib library for plotting histograms. By convention, the library is imported via
        `import matplotlib.pyplot as plt`. Use `plt.hist` to plot histograms.""",
        """An advanced way to plot histograms is to use the histogram functionality that comes with
        pandas dataframes. If `df` is a `pd.DataFrame` object, you can try `df.hist()`. Check out the
        docstring via `df.hist?` to gain additional insight into how that function can be used."""

    ]
    _solution = """
Exploring average values
```python
sqft = train_data.sqft  # alternatively: train_data["sqft"]
average_sqft_sf = sqft[train_data.in_sf == 1].mean()
average_sqft_ny = sqft[train_data.in_sf == 0].mean()
```
then plotting histograms
```python
import matplotlib.pyplot as plt
price_per_sqft = train_data.price_per_sqft

plt.hist(price_per_sqft[train_data.in_sf==1], alpha=0.5)
plt.hist(price_per_sqft[train_data.in_sf==0], alpha=0.5)
plt.xlabel("Price per square feet [USD]")
plt.ylabel("Count");
```
as well as advanced histograms
```python
columns_to_plot = ["beds", "bath", "price", "year_built", "sqft", "price_per_sqft", "elevation"]
n_columns = len(columns_to_plot)

axes = train_data[train_data.in_sf==1].hist(column=columns_to_plot, figsize=(15, 15), alpha=0.5)
train_data[train_data.in_sf==0].hist(column=columns_to_plot, ax=axes.flatten()[:n_columns], alpha=0.5);
```
"""

class SimpleMLModel(ThoughtExperiment):
    show_solution_on_correct = False

    _hints = [
        """Using your insights from the data exploration, how would you characterize the typical
        house in San Francisco - in terms of size, price, price per sqft etc. By contrast, how would you characterize
        the typical house in New York?""",
        """Which of the 'variables' in the data most clearly distinguishes houses in New York from those in San Francisco?""",
        """Use the 'variable' that most clearly distinguishes SF and NY houses. How can you use only this 'variable' to decide
        whether a house is in San Francisco or New York?""",
        """Use a 'decision value' for the 'variable' from before. If a house is below this value for the 'variable', 
        decide that it is a house in SF or NY (decide which one it should be). If it is above, decide that it is the opposite.""",
        """Use the data from `house_location_train.csv` cleverly to determine the 'threshold value' of the 'variable' you chose.""",
        """Can you improve by using another 'variable' in addition to the first?"""
    ]

    _solution = """
Use the elevation - it seems to be most different between NY and SF houses.
```python
average_elevation_sf = train_data.elevation[train_data.in_sf==1].mean()
average_elevation_ny = train_data.elevation[train_data.in_sf==0].mean()

decision_value = (average_elevation_sf + average_elevation_ny) / 2

# save the 'prediction' for the test data in a new column
test_data["in_sf_predicted"] = (test_data["elevation"] > decision_value).astype(np.int32)
```
"""


class Evaluation(CodingProblem):
    show_solution_on_correct = False

    _vars = ["data_path", "results"]
    _hints = [
        """Checking your solution requires that the file 'house_location_test_with_labels.csv' be available at `data_path`"""
    ]

    _solution = """
```python
test_data["in_sf_predicted"] = (test_data["elevation"] > decision_value).astype(np.int32)

results = dict(test_data["in_sf_predicted"])
```
"""
    def check(self, *args):
        data_path = args[0]
        results = args[1]
        data_file = 'house_location_test_with_labels.csv'
        assert_file_exists(os.path.join(data_path, data_file))
        
        test_df =  pd.read_csv(os.path.join(data_path, data_file), index_col=0)

        y_true = test_df.loc[:, "in_sf"]

        if not set(y_true.index).issubset(set(results.keys())):
            missing = set(y_true.index).difference(set(results.keys()))
            print(f"Es gibt noch keine Vorhersage für die folgenden `id`s")
            for missing_id in missing:
                print(missing_id)
            assert False

        if not set(results.keys()).issubset(set(y_true.index)):
            additional = set(results.keys()).difference(set(y_true.index))
            print("Es gibt Vorhersagen für nicht vorhandene `id`s. Diese sind")
            for additional_id in additional:
                print(additional_id)
            assert False

        correct = 0
        for id_, y_pred in results.items():
            correct += y_true.loc[id_] == y_pred
        accuracy = correct / len(y_true)

        print(f"Die Genauigkeit der Vorhersage ist {accuracy}")
        assert True
        

qvars = bind_exercises(globals(), [
    DataDownload,
    DataLoading,
    DataExploration,
    SimpleMLModel,
    Evaluation
    ],
    )
__all__ = list(qvars)
