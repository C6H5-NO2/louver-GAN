from typing import Iterable
from warnings import filterwarnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from .rank import plot_scatter, rank_binary_classifiers, spearman_score
from ..transformer import EvaluatorTransformer
from ...config import DATASET_EVAL, DATASET_NAME, HyperParam
from ...util import ColumnMeta, path_join


class RankEvaluator:
    def __init__(self, opt: HyperParam, meta: Iterable[ColumnMeta]):
        filterwarnings(action='once', category=FutureWarning)
        filterwarnings(action='always', category=ConvergenceWarning)
        filterwarnings(action='ignore', category=UndefinedMetricWarning)

        self._real_df = pd.read_csv(path_join(opt.dataset_path, f'{DATASET_NAME}/{DATASET_NAME}-train.csv'))
        self._transformer = EvaluatorTransformer()
        self._transformer.fit(self._real_df, meta)
        self._test_df = pd.read_csv(path_join(opt.dataset_path, f'{DATASET_NAME}/{DATASET_NAME}-test.csv'))
        self._meta = list(meta)

        self._cached_transformed_test_df = self._transformer.transform(self._test_df)
        self._cached_real_rank_score = rank_binary_classifiers(
            *self._train_test_cast(self._real_df, DATASET_EVAL['classification'])
        )

    def evaluate(self, df: pd.DataFrame):
        return self._evaluate_rank(df, DATASET_EVAL['classification'])

    def _evaluate_rank(self, fake_df: pd.DataFrame, y_column: str):
        x_train, x_test, y_train, y_test = self._train_test_cast(fake_df, y_column)
        fake_score = rank_binary_classifiers(x_train, x_test, y_train, y_test)
        plot_scatter(self._cached_real_rank_score, fake_score)
        rank_score = spearman_score(self._cached_real_rank_score, fake_score)
        return rank_score

    def _train_test_cast(self, train_df: pd.DataFrame, y_column: str):
        x_columns = list(self._real_df.columns)
        x_columns.remove(y_column)
        train_df = self._transformer.transform(train_df)
        x_train, y_train = train_df[x_columns].to_numpy(), train_df[y_column].to_numpy()
        x_test = self._cached_transformed_test_df[x_columns].to_numpy()
        y_test = self._cached_transformed_test_df[y_column].to_numpy()
        return x_train, x_test, y_train, y_test
