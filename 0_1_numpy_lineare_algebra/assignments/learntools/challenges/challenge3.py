import pandas as pd
import os
import numpy as np

from learntools.core import *

class Evaluation(CodingProblem):
    show_solution_on_correct = False

    _vars = ["data_path", "results"]
    _hints = [
        """Checking your solution requires that the file 'test_labels.csv' be available at `data_path`."""
    ]
    def check(self, *args):
        data_path = args[0]
        results = args[1]
        data_file = 'test_labels.csv'
        assert_file_exists(os.path.join(data_path, data_file))
        
        test_df = pd.read_csv(os.path.join(data_path, data_file), index_col=0, header=None)

        y_true = test_df.iloc[:, 0]

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
    Evaluation
    ],
    )
__all__ = list(qvars)

