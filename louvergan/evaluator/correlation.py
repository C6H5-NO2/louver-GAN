import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

from .ipy import display
from .statistics import jensen_shannon_divergence
from ..util import ColumnMeta


def pairwise_mi(data: np.ndarray, meta: Iterable[ColumnMeta]) -> np.ndarray:
    data = data.copy()

    # discretization
    for col_idx, col_meta in enumerate(meta):
        if not col_meta.discrete:
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=UserWarning)
                kbd = KBinsDiscretizer(n_bins=20, encode='ordinal')
                dcr = kbd.fit_transform(data[:, col_idx].reshape(-1, 1))
            data[:, col_idx] = dcr.reshape(-1).astype(np.int64)

    mi = []
    n_col = data.shape[1]
    for i in range(n_col):
        for j in range(i + 1, n_col):
            mi.append(normalized_mutual_info_score(data[:, i], data[:, j]))
    mi = np.array(mi)
    r, c = np.triu_indices(n_col, 1)
    mat = np.eye(n_col)
    mat[r, c] = mi
    mat[c, r] = mi
    return mat


# todo: rewrite
def view_posterior(real_df: pd.DataFrame, fake_df: pd.DataFrame,
                   evidence_cols: Sequence[str], posterior_col: str):
    if type(evidence_cols) is str:
        raise TypeError
    real_group = real_df.groupby(evidence_cols)[posterior_col]
    fake_group = fake_df.groupby(evidence_cols)[posterior_col]
    for name, real_ser in real_group:
        fake_ser: pd.Series = fake_group.get_group(name)
        if len(evidence_cols) == 1:
            name = (name,)
        print('-' * 30)
        print(', '.join(f'{c} == {n}' for c, n in zip(evidence_cols, name)))
        jensen_shannon_divergence(real_ser, fake_ser, verbose=True)
    print('-' * 30)


# todo: rewrite
def posterior_jsd(real_df: pd.DataFrame, fake_df: pd.DataFrame,
                  evidence_cols: Sequence[str], posterior_col: str, *,
                  weight: bool = False, verbose: bool = False) -> np.float64:
    real_group = real_df.groupby(evidence_cols)[posterior_col]
    fake_group = fake_df.groupby(evidence_cols)[posterior_col]

    dist_func = jensen_shannon_divergence
    # dist_func = lambda r, f: earth_movers_distance(r.to_numpy(), f.to_numpy(), normalize=True)

    distances = []
    weights = []
    for name, real_ser in real_group:
        fake_ser: pd.Series = fake_group.get_group(name)
        dist = dist_func(real_ser, fake_ser)
        prior = len(real_ser) / len(real_df)  # use real as prior weight
        distances.append(dist)
        weights.append(prior)

    if verbose:
        dist_type = 'jsd'
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(pd.DataFrame({dist_type: distances, 'prior in real': weights},
                                 index=[name for name, _ in real_group]))

    if not weight:
        weights = None
    avg = np.average(distances, weights=weights)
    return avg
