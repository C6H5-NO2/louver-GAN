from functools import partial
from typing import List, Type, Union

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


class SpacedClassifier:
    def __init__(self, name: str, etype: Union[Type, partial], param_space: ConfigurationSpace):
        self._name = name
        self._params: List[dict] = [c.get_dictionary() for c in generate_grid(param_space)]
        if len(self._params) == 0:
            self._estimators = [etype()]
        else:
            self._estimators = [etype(**p) for p in self._params]

    @property
    def name(self):
        return self._name

    @property
    def n_estimators(self):
        return len(self._estimators)

    def fit(self, x: np.ndarray, y: np.ndarray):
        for e in self._estimators:
            e.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> List[np.ndarray]:
        return [e.predict(x) for e in self._estimators]

    def score(self, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        metrics = [accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score]
        y_preds = self.predict(x)
        scores = [[metric(y, p) for p in y_preds] for metric in metrics]
        return pd.DataFrame(scores, index=['acc', 'bacc', 'f1', 'auroc']).T
