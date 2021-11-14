import pickle
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder

from .util import CONTINUOUS, ColumnMeta, DISCRETE, TYPE


@dataclass
class TransformerMeta(ColumnMeta):
    estimator: Union[OneHotEncoder, BayesianGaussianMixture]


class Transformer:
    def __init__(self, vgm_components: int = 10, vgm_weight_threshold: float = .005):
        self.vgm_components = vgm_components
        self.vgm_weight_threshold = vgm_weight_threshold
        self._columns: List[str] = []
        self._dtypes: Optional[pd.Series] = None
        self._meta: List[TransformerMeta] = []

    @property
    def columns(self):
        return self._columns.copy()

    @property
    def meta(self):
        return [ColumnMeta(m.discrete, m.nmode) for m in self._meta]

    def fit(self, df: pd.DataFrame, desc: dict):
        self._columns = df.columns.to_list()
        self._dtypes = df.dtypes.copy()
        for col_name in self._columns:
            col_data = df[col_name].values.reshape(-1, 1)
            col_type = desc[col_name][TYPE]

            if col_type == DISCRETE:
                ohe = OneHotEncoder(sparse=False)
                ohe.fit(col_data)
                nmode = len(ohe.categories_[0])
                estimator = ohe

            elif col_type == CONTINUOUS:
                with warnings.catch_warnings():
                    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
                    vgm = BayesianGaussianMixture(n_components=self.vgm_components, weight_concentration_prior=.001)
                    vgm.fit(col_data)
                nmode = sum(vgm.weights_ > self.vgm_weight_threshold)
                estimator = vgm

            else:
                raise SyntaxError(f'names.json > {col_name} > {TYPE} is {col_type}')

            self._meta.append(TransformerMeta(col_type == DISCRETE, nmode, estimator))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert len(self._columns) == len(df.columns) and all(sc == dc for sc, dc in zip(self._columns, df.columns))
        data = []
        for col_name, col_meta in zip(self._columns, self._meta):
            col_data = df[col_name].values.reshape(-1, 1)

            if col_meta.discrete:
                data.append(col_meta.estimator.transform(col_data))

            else:
                vgm = col_meta.estimator
                mean = vgm.means_.reshape(1, -1)  # shape (1, #vgm_components)
                std = np.sqrt(vgm.covariances_).reshape(1, -1)  # shape (1, #vgm_components)
                mask = vgm.weights_ > self.vgm_weight_threshold

                normed_val = ((col_data - mean) / (4. * std))[:, mask]  # shape (#data_rows, #valid_components)
                eps = 1e-9  # dtype is np.float64
                prob = np.clip(vgm.predict_proba(col_data)[:, mask], eps, None)
                prob = prob / prob.sum(axis=1, keepdims=True)

                selected_component = np.zeros(df.shape[0], dtype=np.int)
                for i in range(df.shape[0]):
                    selected_component[i] = np.random.choice(col_meta.nmode, p=prob[i])

                selected_val = normed_val[np.arange(df.shape[0]), selected_component].reshape(-1, 1).clip(-.99, .99)
                selected_mode = np.zeros_like(prob)
                selected_mode[np.arange(df.shape[0]), selected_component] = 1
                data.append(selected_val)
                data.append(selected_mode)

        return np.concatenate(data, axis=1)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        inv_data = []
        col_idx = 0
        for col_meta in self._meta:
            if col_meta.discrete:
                sli = data[:, col_idx: col_idx + col_meta.nmode]
                inv = col_meta.estimator.inverse_transform(sli)  # shape (#data_rows, 1)
                inv = inv.reshape(-1)
                inv_data.append(inv)
                col_idx += col_meta.nmode

            else:
                vgm = col_meta.estimator
                mean = vgm.means_.reshape(-1)
                std = np.sqrt(vgm.covariances_).reshape(-1)
                mask = vgm.weights_ > self.vgm_weight_threshold

                selected_mode = data[:, col_idx + 1: col_idx + 1 + col_meta.nmode]
                prob = np.ones((data.shape[0], self.vgm_components)) * -100.
                prob[:, mask] = selected_mode
                selected_component = np.argmax(prob, axis=1)  # shape (#data_rows,)
                mean = mean[selected_component]  # shape (#data_rows,)
                std = std[selected_component]  # shape (#data_rows,)

                selected_val = data[:, col_idx].clip(-1, 1)  # shape (#data_rows,)
                inv_data.append(selected_val * 4. * std + mean)
                col_idx += 1 + col_meta.nmode

        inv_data = np.column_stack(inv_data)
        df = pd.DataFrame(inv_data, columns=self._columns).astype(self._dtypes)
        return df

    def save(self, path: str):
        with open(path, mode='wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, mode='rb') as f:
            o = pickle.load(f)
            return o if type(o) is cls else cls(o)
