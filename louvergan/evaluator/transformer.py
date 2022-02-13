from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from ..util import ColumnMeta


# todo: see LabelBinarizer & sklearn-pandas
class EvaluatorTransformer:
    def __init__(self):
        self._d_idx, self._c_idx = [], []
        self._oe = OrdinalEncoder()
        self._mm = MinMaxScaler(feature_range=(-1, 1))

    def fit(self, df: pd.DataFrame, meta: Iterable[ColumnMeta]):
        self._d_idx = [i for i, m in enumerate(meta) if m.discrete]
        self._c_idx = [i for i, m in enumerate(meta) if not m.discrete]
        self._oe.fit(df.iloc[:, self._d_idx])
        self._mm.fit(df.iloc[:, self._c_idx])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = np.zeros(df.shape)
        data[:, self._d_idx] = self._oe.transform(df.iloc[:, self._d_idx])
        data[:, self._c_idx] = self._mm.transform(df.iloc[:, self._c_idx])
        return pd.DataFrame(data, columns=df.columns)
