import pandas as pd
import os
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OrdinalEncoder

from learntools.core import *

class OneDArrayCreation(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.arange(10)
    )
    _hints = [
        'Use `np.arange`.',
        'Make sure to declare a variable `out`'
    ]
    _solution = CS(
            "out = np.arange(10)"
    )


class BooleanArrayCreation(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.ones((3, 3), dtype=np.bool)
    )
    _hints = [
        'Use `np.ones`.',
        'Use the e.g. the `dtype` argument of `np.ones` or find some other way to convert to boolean type.',
        'Make sure to declare a variable `out`.'
    ]
    _solution = CS(
            "out = np.ones((3, 3), dtype=np.bool)"
    )


class SelectingValues(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.array([1, 3, 5, 7, 9])
    )
    _hints = [
        'Have a look at the modulo operation `%`.',
        'Arrays can be indexed by boolean arrays that represent a condition. E.g. `arr[condition]` if `condition is an array of booleans.',
    ]
    _solution = CS(
            "out = arr[arr % 2 == 1]"
    )


class ReplacingValues(EqualityCheckProblem):
    _var = 'arr'
    _expected = (
            np.array([0, -1, 2, -1, 4, -1, 6, -1, 8, -1])
    )
    _hints = [
        'Have a look at the modulo operation `%`.',
        'It is possible to assign values to an indexed array.',
    ]
    _solution = CS(
            "arr[arr % 2 == 1] = -1"
    )


class ReshapeArray(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = 'out'
    _expected = (
            np.arange(10).reshape(2, -1)
    )
    _hint = 'Use `np.reshape`.'
    _solution = """
```python
out = np.reshape(arr, (2, 5))
```
or (the -1 means that the number of columns is inferred automatically)
```python
out = np.reshape(arr, (2, -1))
```
"""


class StackVertically(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = 'out'
    _expected = (
            np.vstack((
                np.arange(10).reshape(2, -1),
                np.ones((2, 5))
            ))
    )
    _hints = [
        '"Vertically" means connecting the arrays by putting them on top of each other, letting their rows become the rows in the new array.',
        'There are many functions for this. Research them.',
        'Try `np.vstack`, `np.row_stack` or `np.concatenate`.'
    ]
    _solution = """
```python
out = np.concatenate([a, b], axis=0)
```
or
```python
out =  np.vstack([a, b])
```
or
```python
out = np.row_stack([a, b])
```
"""


class StackHorizontally(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = 'out'
    _expected = (
            np.hstack((
                np.arange(10).reshape(2, -1),
                np.ones((2, 5))
            ))
    )
    _hints = [
        '"Horizontally" means connecting the arrays by putting them left and right from each other, letting their columns become the columns in the new array.',
        'There are many functions for this. Research them.',
        'Try `np.hstack`, `np.column_stack` or `np.concatenate`.'
    ]
    _solution = """
```python
out = np.concatenate([a, b], axis=1)
```
or
```python
out =  np.hstack([a, b])
```
or
```python
out = np.column_stack([a, b])
```
"""


class TilingI(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    )
    _hint = 'Use `np.tile`'
    _solution = CS(
            "out = np.tile(a, 3)"
    )


class TilingII(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.array([
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3]
            ])
    )
    _hint = 'Use `np.tile`'
    _solution = CS(
            "out = np.tile(a, (3, 1))"
    )


class MatchingSubArray(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.array([2, 2, 4, 4])
    )
    _hint = 'Use elementwise comparison and indexing.'
    _solution = CS(
            "out = a[a==b]"
    )



class CountMatches(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = 'n_matches'
    _expected = 4
    _hint = 'Use elementwise comparison and an appropriate reduce-function.'
    _solution = """
```python
n_matches = np.sum(a == b)
```
or
```python
n_matches = (a == b).sum()
```
"""


class PercentageMatches(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = 'prop_matches'
    _expected = 0.4
    _hint = 'Use elementwise comparison and an appropriate reduce-function.'
    _solution = """
```python
prop_matches = np.sum(a == b) / len(a)
```
or
```python
prop_matches = np.mean(a == b)
```
or
```python
prop_matches = (a == b).mean()
```
"""


class FindingMatchingEntries(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = 'matching_positions'
    _expected = np.array([1, 3, 5, 7])
    _hint = 'Use `np.where` or `np.nonzero`.'
    _solution = """
```python
matching_positions = np.where(a == b)[0]
```
or
```python
prop_matches = np.nonzero(a == b)[0]
```
"""


class FindMostFrequent(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = 'most_frequent'
    _expected = 0
    _hints = [
        'Use `np.unique` in combination with `np.argmax`.',
        'Check the additional arguments of `np.unique`.'
        'Alternatively, use `scipy.stats.mode`.',
    ]
    _solution = """
```python
counts = np.unique(a, return_counts=True)[1]
most_frequent = np.argmax(counts)
```
or
```python
from scipy.stats import mode
most_frequent = mode(a).mode[0]
```
"""


class FindInRange(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.array([5, 7, 8, 7, 5, 9, 6])
    )
    _hints = [
        'Use elementwise comparisons, elementwise operations, and indexing.',
        'Look up the elementwise operations `&`, `|` and `~` and how they work with NumPy.'
    ]
    _solution = CS(
            "out = a[(a >= 5) & (a <= 10)]"
    )



class SwapColumns(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.array([
                [1, 0, 2],
                [4, 3, 5],
                [7, 6, 8]
            ])
    )
    _hint = 'Lists can be used to index into arrays.'
    _solution = CS(
            "out = arr[:, [1,0,2]]"
    )


class SwapRows(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.array([
                [3, 4, 5],
                [0, 1, 2],
                [6, 7, 8]
            ])
    )
    _hint = 'Lists can be used to index into arrays.'
    _solution = CS(
            "arr[[1,0,2], :]"
    )


class ReverseArray(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.array([12, 33, 2, 3, 5, 6, 8, 10, 11, 0, 11, 1])[::-1]
    )
    _hint = 'There is a canonical way to do this indexing. Research it.'
    _solution = CS(
            "out = arr[::-1]"
    )


class ReverseRows(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.arange(12).reshape(3, 4)[::-1, :]
    )
    _hint = 'Combine rowwise-indexing with the canonical way to reverse arrays.'
    _solution = CS(
            "out = arr[::-1, :]"
    )


class ReverseColumns(EqualityCheckProblem):
    _var = 'out'
    _expected = (
            np.arange(12).reshape(3, 4)[:, ::-1]
    )
    _hint = 'Combine columnwise-indexing with the canonical way to reverse arrays.'
    _solution = CS(
            "out = arr[:, ::-1]"
    )


class TwosArray(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = 'twos_arr'
    _expected = (
            2 * np.ones(100)
    )
    _hints = [
        'Use `np.ones`.',
        'Think about scalar multiplication for arrays.'
    ]
    _solution = CS(
        "twos_arr = 2 * np.ones(100)"
    )


class RandomArray(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = 'rand_arr'
    _expected = (
            np.random.RandomState(18).uniform(5, 10, size=(5,3))
    )
    _hint = 'Everything to do with random numbers is to be found in the submodule np.random. Research it.'
    _solution = """
```python
rand_arr = np.random.uniform(5, 10, size=(5,3))
```
or
```python
rand_arr = np.random.random((5,3)) * 5 + 5
```
"""


class PrintOptionsI(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['rand_arr']
    _expected = (
            '[[0.65  0.505 0.879]\n [0.182 0.852 0.75 ]\n [0.666 0.988 0.257]\n [0.028 0.636 0.847]\n [0.736 0.021 0.112]]'
    )
    _hint = 'There is some functionality to adapt print-outs for NumPy arrays: `np.set_printoptions`, `np.printoptions` and `np.get_printoptions`. Research them.'
    _solution = """
To change the display of NumPy arrays globally, use this
```python
np.set_printoptions(precision=3)
print(rand_arr)
```
However, for the purposes of this tutorial, displays should be changed only within the scope
of the relevant question. For this do
```python
with np.printoptions(precision=3):
    print(rand_arr)
```
so that the configuration reverts back to its original state afterwards. Use checking of your solution within this `with` block.
"""

    def check(self, *args):
        assert_equal(args[0].__str__(), self._expected, name="string representation of `rand_arr`")


class PrintOptionsII(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['rand_arr']
    with np.printoptions(suppress=True):
        _expected = (np.random.RandomState(22).random([3,3]) / 1e4).__str__()
    _hint = 'There is some functionality to adapt print-outs for NumPy arrays: `np.set_printoptions`, `np.printoptions` and `np.get_printoptions`. Research them.'
    _solution = """
To change the display of NumPy arrays globally, use this
```python
np.set_printoptions(suppress=True)
print(rand_arr)
```
However, for the purposes of this tutorial, displays should be changed only within the scope
of the relevant question. For this do
```python
with np.printoptions(suppress=True):
    print(rand_arr)
```
so that the configuration reverts back to its original state afterwards. Use checking of your solution within this `with` block.
"""

    def check(self, *args):
        assert_equal(args[0].__str__(), self._expected, name="string representation of `rand_arr`")


class PrintOptionsIII(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['a', 'b']
    with np.printoptions(threshold=10):
        _expected = [
            np.arange(11).__str__(),
            np.arange(10).__str__()
        ]
    _hint = 'There is some functionality to adapt print-outs for NumPy arrays: `np.set_printoptions`, `np.printoptions` and `np.get_printoptions`. Research them.'
    _solution = """
To change the display of NumPy arrays globally, use this
```python
np.set_printoptions(threshold=10)
print(rand_arr)
```
However, for the purposes of this tutorial, displays should be changed only within the scope
of the relevant question. For this do
```python
with np.printoptions(threshold=10):
    print(rand_arr)
```
so that the configuration reverts back to its original state afterwards. Use checking of your solution within this `with` block.
"""

    def check(self, *args):
        for arg, expected in zip(args, self._expected):
            assert_equal(arg.__str__(), expected, name="string representation")


class PrintOptionsIV(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['a']
    with np.printoptions(threshold=sys.maxsize):
        _expected = [np.arange(20000).reshape(100, 200).__str__()]
    _hints = [
        'There is some functionality to adapt print-outs for NumPy arrays: `np.set_printoptions`, `np.printoptions` and `np.get_printoptions`. Research them.',
        'Consider `import sys; sys.maxsize`.'
    ]
    _solution = """
```python
with np.printoptions(threshold=sys.maxsize):
    print(a)
```
"""

    def check(self, *args):
        for arg, expected in zip(args, self._expected):
            assert_equal(arg.__str__(), expected, name="string representation")


class LoadIris(ThoughtExperiment):
    _solution = """
```python
iris = load_iris()
```
"""


IRIS = load_iris()


class IrisDataMatrix(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = 'X'
    _expected = (
            IRIS["data"]
    )
    _hint = 'The dataset is a collection of different objects with different types. Try out some typical ways to access the elements of such collections.'
    _solution = """
The object behaves like a dictionary:
```python
X = iris["data"]
```
but also almost like a class instance with attributes:
```python
X = iris.data
```
"""


class IrisTargetVector(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = 'y'
    _expected = (
            IRIS["target"]
    )
    _hint = 'The dataset is a collection of different objects with different types. Try out some typical ways to access the elements of such collections.'
    _solution = """
The object behaves like a dictionary:
```python
y = iris["target"]
```
but also almost like a class instance with attributes:
```python
y = iris.target
```
"""


class MeanMedianSDI(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'mean', 'median', 'sd'] 
    _expected = [
            IRIS["data"],
            np.mean(IRIS["data"][:, 0]),
            np.median(IRIS["data"][:, 0]),
            np.std(IRIS["data"][:, 0])
    ]
    _hint = 'The relevant functions are `np.mean`, `np.median` and `np.std`.'
    _solution = """
```python
mean = np.mean(X[:, 0])
median = np.median(X[:, 0])
sd = np.std(X[:, 0])
```
"""


class MeanMedianSDII(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'mean', 'median', 'sd'] 
    _expected = [
            IRIS["data"],
            np.mean(IRIS["data"][:, 3]),
            np.median(IRIS["data"][:, 3]),
            np.std(IRIS["data"][:, 3])
    ]
    _hint = 'The relevant functions are `np.mean`, `np.median` and `np.std`.'
    _solution = """
```python
mean = np.mean(X[:, 3])
median = np.median(X[:, 3])
sd = np.std(X[:, 3])
```
or even more general
```python
col = iris["feature_names"].index("petal width (cm)")
mean = np.mean(X[:, col])
median = np.median(X[:, col])
sd = np.std(X[:, col])
```
"""


class MeanMedianSDIII(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'means', 'medians', 'sds'] 
    _expected = [
            IRIS["data"],
            np.mean(IRIS["data"], axis=0),
            np.median(IRIS["data"], axis=0),
            np.std(IRIS["data"], axis=0)
    ]
    _hint = 'The relevant functions are `np.mean`, `np.median` and `np.std` together with the `axis` argument.'
    _solution = """
```python
means = np.mean(X, axis=0)
medians = np.median(X, axis=0)
sds = np.std(X, axis=0)
```
"""


class StandardizeI(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'X_new'] 
    _expected = [
            IRIS["data"],
            IRIS["data"] - IRIS["data"].mean(axis=0)
    ]
    _hint = 'How does the mean change if you add a constant number to the value of a feature for all samples?'
    _solution = CS(
        "X_new = X - X.mean(axis=0)"
    )


class StandardizeII(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'X_new'] 
    _expected = [
            IRIS["data"],
            (IRIS["data"] - IRIS["data"].mean(axis=0)) / IRIS["data"].std(axis=0)
    ]
    _hint = 'How does the standard deviation change if you multiply the value of a feature for all samples by a constant number?'
    _solution = CS(
        "X_new = (X - X.mean(axis=0)) / X.std(axis=0)"
    )


class SelectingSamplesI(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'y', 'X_sub'] 
    _expected = [
            IRIS["data"],
            IRIS["target"],
            IRIS["data"][IRIS["target"]==1, :]
    ]
    _solution = CS(
        "X_sub = X[y==1, :]"
    )


class SelectingSamplesII(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'y', 'X_sub'] 
    _expected = [
            IRIS["data"],
            IRIS["target"],
            IRIS["data"][(IRIS["target"]==1) & (IRIS["data"][:, 2] > 4.5), :]
    ]
    _solution = CS(
        "X_sub = X[(y==1) & (X[:, 2] > 4.5) , :]"
    )


class Percentiles(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'percentiles'] 
    _expected = [
            IRIS["data"],
            np.percentile(IRIS["data"][:, 0], q=[5, 95])
    ]
    _hint = "Use `np.percentile`."
    _solution = CS(
        "percentiles = np.percentile(X[:, 0], q=[5, 95])"
    )


class Correlation(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'corr'] 
    _expected = [
            IRIS["data"],
            np.corrcoef(IRIS["data"][:, :2].T)[0, 1]
    ]
    _hint = "Use `np.corrcoef`."
    _solution = CS(
        "corr = np.corrcoef(X[:, :2].T)[0, 1]"
    )

class CountingAndUnique(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['y', 'unique', 'counts'] 
    _expected = [
            IRIS["target"],
            np.unique(IRIS["target"]),
            np.unique(IRIS["target"], return_counts=True)[1],
    ]
    _hint = "Use `np.unique` and check its arguments."
    _solution = CS(
        "unique, counts = np.unique(y, return_counts=True)"
    )

class Binning(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'x_string'] 
    _expected = [
            IRIS["data"],
            np.array([{1: 'small', 2: 'medium', 3: 'large'}[x] for x in np.digitize(IRIS["data"][:, 2], [0, 2, 5])]),
    ]
    _hints = [
        "Consider `np.digitize`.",
        """Transforming the binned values can be done in many ways.
        One possibility is to define a dictionary mapping integers to strings.
        This can be combined with a simple for-loop over the samples."""
    ]
    _solution = """
```python
label_map = {1: 'small', 2: 'medium', 3: 'large'}
binned = np.digitize(X[:, 2], [0, 2, 5])
x_string = np.array([label_map[x] for x in binned])
```
"""


class FeatureEngineeringI(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'petal_area', 'X_new'] 
    _expected = [
            IRIS["data"],
            IRIS["data"][:, 2] * IRIS["data"][:, 3],
            np.column_stack((IRIS["data"], IRIS["data"][:, 2] * IRIS["data"][:, 3]))
    ]
    _hint = "Revisit Q7."
    _solution = """
```python
petal_area = X[:, 2] * X[:, 3]
X_new = np.column_stack((X, petal_area))
```
"""


class SplittingI(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'X1', 'X2', 'y1', 'y2'] 
    _expected = [
            IRIS["data"],
            IRIS["data"][:100, :],
            IRIS["data"][100:, :],
            IRIS["target"][:100],
            IRIS["target"][100:]
    ]
    _solution = """
```python
X1 = X[:100, :]
X2 = X[100:, :]

y1 = y[:100]
y2 = y[100:]
```
"""


class SplittingII(CodingProblem):
    show_solution_on_correct = True
    _vars = ['X', 'X1', 'X2', 'y1', 'y2']

    _hint = "Use `np.random.permutation` or `np.random.shuffle`."
    _solution = """
```python
perm = np.random.permutation(X.shape[0])

X1 = X[perm[:100], :]
X2 = X[perm[100:], :]

y1 = y[perm[:100]]
y2 = y[perm[100:]]
```
"""

    def check(self, *args):
        assert_equal(args[0], IRIS["data"], "X")

        assert_equal(args[1].shape, (100, 4), "shape of X1")
        assert_equal(args[2].shape, (50, 4), "shape of X2")
        assert_equal(args[3].shape, (100,), "shape of y1")
        assert_equal(args[4].shape, (50,), "shape of y2")


class GroupSelect(EqualityCheckProblem):
    pass


class SortingI(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'X_sorted'] 
    _hint = "Use `np.argsort`."
    _expected = [
            IRIS["data"],
            IRIS["data"][np.argsort(IRIS["data"][:, 3]), :],
    ]
    _solution = CS(
        "X_sorted = X[np.argsort(X[:, 3]), :]"
    )


class SortingII(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['X', 'X_sorted'] 
    _hints = [
        "Use `np.argsort`.",
        "Revisit Q18."
    ]
    _expected = [
            IRIS["data"],
            IRIS["data"][np.argsort(IRIS["data"][:, 3])[::-1], :],
    ]
    _solution = CS(
        "X_sorted = X[np.argsort(X[:, 3])[::-1], :]"
    )


class MatrixMultiplicationI(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ['C', 'D']
    _hint = "Use `np.matmul` or the @ operator."
    _expected = [
            np.array([[-4,  2],  [ 1,  7]]),
            np.array([[-1, -2, -1, -3], [ 2,  3,  3,  5], [ 1,  5, -2,  6], [ 1,  2,  1,  3]])
    ]
    _solution = """
```python
C = A @ B
D = B @ A
```
or
```python
C = np.matmul(A, B)
D = np.matmul(B, A)
```
"""


class MatrixMultiplicationII(EqualityCheckProblem):
    _var = 'A'
    _expected = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    _solution = CS(
        """A = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])"""
    )


class MatrixMultiplicationIII(EqualityCheckProblem):
    _var = 'value'
    _expected = 10.8
    _hints = [
        "Construct an array `np.array([1, 2, -1, 0.5])`.",
        "Use `np.dot`."
    ]
    _solution = CS(
        "value = np.dot(X[0, :], np.array([1, 2, -1, 0.5]))"
    )

    def check(self, arg):
        assert np.allclose(arg, self._expected)


class MatrixMultiplicationIV(EqualityCheckProblem):
    _var = 'values'
    _expected = np.matmul(IRIS["data"], np.array([1, 2, -1, 0.5]))
    _hints = [
        "Construct an array `np.array([1, 2, -1, 0.5])`.",
        "Use `np.matmul`."
    ]
    _solution = CS(
        "values = np.matmul(X, np.array([1, 2, -1, 0.5]))"
    )

    def check(self, arg):
        assert np.allclose(arg, self._expected)


class DistancesI(EqualityCheckProblem):
    _vars = ["a", "b", "dist"] 
    _hint = "Use `np.linalg.norm`."
    _expected = [
            np.array([1,2,3,4,5]),
            np.array([4,5,6,7,8]),
            np.linalg.norm(np.array([1,2,3,4,5]) - np.array([4,5,6,7,8]))
    ]
    _solution = CS(
        "dist = np.linalg.norm(a - b)"
    )


class DistancesII(EqualityCheckProblem):
    _vars = ["X", "mean", "dists"] 
    _hints = [
        "Use `np.linalg.norm`.",
        "Investigate the `axis` argument of `np.linalg.norm`."
    ]
    _expected = [
            IRIS["data"],
            IRIS["data"].mean(axis=0),
            np.linalg.norm(IRIS["data"] - IRIS["data"].mean(axis=0), axis=1)
    ]
    _solution = CS(
        """mean = X.mean(axis=0)
dists = np.linalg.norm(X - mean, axis=1)"""
    )


class Clipping(EqualityCheckProblem):
    _var = "out" 
    _hint = "Use `np.clip`."
    _expected = np.clip(np.random.RandomState(5).uniform(1, 50, 20), a_min=10, a_max=30)
    _solution = CS(
        "out = np.clip(a, a_min=10, a_max=30)"
    )


class TopKValues(EqualityCheckProblem):
    _var = "top5" 
    _hints = [
        "Use `np.argsort`.",
        "Pay attention to whether `np.argsort` sorts in descending or ascending order."
    ]
    _expected = np.array([ 3, 14,  1,  6, 11])
    _solution = CS(
        "top5 = np.argsort(a)[::-1][:5]"
    )


class ReshapingII(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = "arr" 
    _hint = "Use the `.flatten()` attribute of arrays or `np.reshape`."
    _expected = np.array([
        [0, 1, 2, 1],
        [1, -1, 1, 3],
        [0, -2, 4, 1]
    ]).flatten()
    _solution = """
```python
arr = arr2d.flatten()  # or
arr = np.reshape(arr2d, -1)  # or
arr = np.reshape(arr2d, (-1,))  # or
arr = arr2d.reshape(-1)
```
"""


class ReshapingIII(EqualityCheckProblem):
    show_solution_on_correct = False
    _var = "arr2d" 
    _hint = "Use the `np.reshape`."
    _expected = np.reshape(np.random.RandomState(4).normal(size=(100, 4, 4)), (100, -1))
    _solution = """
```python
arr2d = np.reshape(arr3d, (arr3d.shape[0], -1))  # or
arr2d = np.reshape(arr3d, (arr3d.shape[0], arr3d.shape[1] * arr3d.shape[2]))
```
"""


class ReshapingIV(EqualityCheckProblem):
    show_solution_on_correct = False
    _vars = ["arr3d", "image35"]
    _hint = "Use the `np.reshape`."
    _expected = [
        np.reshape(np.random.RandomState(7).uniform(0, 1, size=3200), (50, 8, 8)),
        np.reshape(np.random.RandomState(7).uniform(0, 1, size=3200), (50, 8, 8))[34, :, :]
    ]
    _solution = """
```python
arr3d = np.reshape(arr, (50, 8, 8))
image35 = arr3d[34, :, :]
```
"""


class FindMaximumI(EqualityCheckProblem):
    _var = "position" 
    _hint = "Use `np.argmax`."
    _expected = 99
    _solution = CS(
        "position = np.argmax(a)"
    )


class FindMaximumII(EqualityCheckProblem):
    _var = "position" 
    _hints = [
        "Use `np.argmax` and try to recover the 2d coordinates from the index.",
        "Use `np.unravel_index` to recover 2d coordinates from the index."
    ]
    _expected = (34, 74)
    _solution = CS(
        "position = np.unravel_index(np.argmax(A), A.shape)"
    )


class ComputeMaximum(EqualityCheckProblem):
    _var = "maximums" 
    _hint = "Use `np.max` and investigate the `axis` argument."
    _expected = np.random.RandomState(3).randint(1, 10, [5, 3]).max(axis=1)
    _solution = CS(
        "maximums = np.max(A, axis=1)"
    )


class ComputeRange(EqualityCheckProblem):
    _var = "ranges" 
    _hint = "Use `np.max` together with `np.min`."
    _expected = np.random.RandomState(3).normal(size=(10, 20)).max(axis=1) - np.random.RandomState(3).normal(size=(10, 20)).min(axis=1)
    _solution = CS(
        "ranges = np.max(A, axis=1) - np.min(A, axis=1)"
    )


class CheckingEquality(ThoughtExperiment):
    _hint = "Consider `np.all` and `np.allclose`."
    _solution = """
```python
exactly_equal = np.all(A == B)
approx_equal = np.allclose(A, B)
```
    """
    


qvars = bind_exercises(globals(), [
    OneDArrayCreation,
    BooleanArrayCreation,
    SelectingValues,
    ReplacingValues,
    ReshapeArray,
    StackVertically,
    StackHorizontally,
    TilingI,
    TilingII,
    MatchingSubArray,
    CountMatches,
    PercentageMatches,
    FindingMatchingEntries,
    FindMostFrequent,
    FindInRange,
    SwapColumns,
    SwapRows,
    ReverseArray,
    ReverseRows,
    ReverseColumns,
    TwosArray,
    RandomArray,
    PrintOptionsI,
    PrintOptionsII,
    PrintOptionsIII,
    PrintOptionsIV,
    LoadIris,
    IrisDataMatrix,
    IrisTargetVector,
    MeanMedianSDI,
    MeanMedianSDII,
    MeanMedianSDIII,
    StandardizeI,
    StandardizeII,
    SelectingSamplesI,
    SelectingSamplesII,
    Percentiles,
    Correlation,
    CountingAndUnique,
    Binning,
    FeatureEngineeringI,
    SplittingI,
    SplittingII,
    GroupSelect,
    SortingI,
    SortingII,
    MatrixMultiplicationI,
    MatrixMultiplicationII,
    MatrixMultiplicationIII,
    MatrixMultiplicationIV,
    DistancesI,
    DistancesII,
    Clipping,
    TopKValues,
    ReshapingII,
    ReshapingIII,
    ReshapingIV,
    FindMaximumI,
    FindMaximumII,
    ComputeMaximum,
    ComputeRange,
    CheckingEquality
    ],
    )
__all__ = list(qvars)



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

        data_path = args[0]
        train_data = pd.read_csv(os.path.join(data_path, 'house_location_train.csv'), index_col=0)
        test_data = pd.read_csv(os.path.join(data_path, 'house_location_test.csv'), index_col=0)
        assert_df_equals(train_data, args[1],
            name="Dataframe loaded from `house_location_train.csv`")
        assert_df_equals(test_data, args[2],
            name="Dataframe loaded from `house_location_test.csv`")
