from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from .plot import plot_cdf
from ..polyfill import display, zip_strict
from ..util import ColumnMeta


def earth_movers_distance(real_data: np.ndarray, fake_data: np.ndarray, normalize: bool = False) -> float:
    assert real_data.ndim == 1 and real_data.shape == fake_data.shape
    if normalize:
        maxi = np.max(real_data)
        mini = np.min(real_data)
        real_data = (real_data - mini) / (maxi - mini)
        fake_data = (fake_data - mini) / (maxi - mini)
    emd = wasserstein_distance(real_data, fake_data)
    return emd


# todo: rewrite
def jensen_shannon_divergence(real_ser: pd.Series, fake_ser: pd.Series, verbose: bool = False) -> float:
    # np.unique(real_data, return_counts=True)  # DOES NOT WORK. indices don't match
    counts = pd.merge(real_ser.value_counts(sort=False), fake_ser.value_counts(sort=False),
                      how='outer', left_index=True, right_index=True, suffixes=(' [real]', ' [fake]')).fillna(0)
    if verbose:
        tmp = pd.merge(real_ser.value_counts(normalize=True, sort=False),
                       fake_ser.value_counts(normalize=True, sort=False),
                       how='outer', left_index=True, right_index=True,
                       suffixes=(' [real normed]', ' [fake normed]')).fillna(0)
        tmp = pd.merge(counts, tmp, how='outer', left_index=True, right_index=True)
        tmp.iloc[:, [0, 1]] = tmp.iloc[:, [0, 1]].astype(np.int64)
        display(tmp)
    probs = counts.to_numpy().T
    jsd = np.square(jensenshannon(probs[0], probs[1]))
    return jsd.item()


def statistical_similarity(real_df: pd.DataFrame, fake_df: pd.DataFrame,
                           meta: Iterable[ColumnMeta], columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if columns is None:
        columns = real_df.columns
    d_col_names = set(c for m, c in zip_strict(meta, real_df.columns) if m.discrete)
    dist = []
    for col in columns:
        if col in d_col_names:
            dist_val = jensen_shannon_divergence(real_df[col], fake_df[col])
            dist_type = 'jsd'
        else:
            dist_val = earth_movers_distance(real_df[col].to_numpy(), fake_df[col].to_numpy(), normalize=True)
            dist_type = 'emd'
        dist.append({'column': col, 'distance': dist_val, 'type': dist_type})
        print('-' * 30)
        print(dist[-1])
        plot_cdf(real_df[col].to_numpy(), fake_df[col].to_numpy(), col)
    print('-' * 30)
    return pd.DataFrame(dist)
