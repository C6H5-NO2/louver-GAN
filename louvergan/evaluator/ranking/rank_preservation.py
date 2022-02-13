from typing import List, Sequence
from warnings import filterwarnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from .rank import rank_binary_classifiers, Scores
from ..transformer import EvaluatorTransformer
from ...config import DATASET_EVAL, DATASET_NAME, HyperParam
from ...polyfill import display, Markdown, zip_strict
from ...util import ColumnMeta, path_join


class RankPreservationEvaluator:
    def __init__(self, opt: HyperParam, meta: Sequence[ColumnMeta]):
        filterwarnings(action='once', category=FutureWarning)
        filterwarnings(action='always', category=ConvergenceWarning)
        filterwarnings(action='ignore', category=UndefinedMetricWarning)

        self._real_df = pd.read_csv(path_join(opt.dataset_path, f'{DATASET_NAME}/{DATASET_NAME}-train.csv'))
        self._test_df = pd.read_csv(path_join(opt.dataset_path, f'{DATASET_NAME}/{DATASET_NAME}-test.csv'))
        self._meta = list(meta)

        self._transformer = EvaluatorTransformer()
        self._transformer.fit(self._real_df, meta)

        self._cached_transformed_test_df = self._transformer.transform(self._test_df)
        self._cached_real_metric_score, _ = \
            rank_binary_classifiers(*self._train_test_cast(self._real_df, DATASET_EVAL['classification']))

    def evaluate_one(self, fake_df: pd.DataFrame, solver_name: str):
        rank_scores, fake_metric_scores = self.evaluate_all([fake_df], [solver_name])
        # todo: merge rank scores
        return rank_scores, fake_metric_scores[0]

    def evaluate_all(self, fake_dfs: Sequence[pd.DataFrame], solver_names: Sequence[str] = ('none', 'ae', 'cgan')):
        assert len(fake_dfs) == len(solver_names)
        fake_scores_per_solver = []
        for fake_df, solver_name in zip_strict(fake_dfs, solver_names):
            x_train, x_test, y_train, y_test = self._train_test_cast(fake_df, DATASET_EVAL['classification'])
            fake_score, _ = rank_binary_classifiers(x_train, x_test, y_train, y_test)
            fake_scores_per_solver.append(fake_score)

        rank_scores_per_cls = dict()
        for cls_name in self._cached_real_metric_score:
            display(Markdown('# ' + cls_name))
            rank_scores_per_cls[cls_name] = self._calculate_rank(fake_scores_per_solver, solver_names, cls_name)
            display(rank_scores_per_cls[cls_name])
            display(Markdown('---'))

        if len(self._cached_real_metric_score) < 2:
            return rank_scores_per_cls, fake_scores_per_solver

        best_cls_name = 'best per classifier'
        display(Markdown('# ' + best_cls_name))
        best_real_scores: List[pd.Series] = []
        cor_fake_scores: List[List[pd.Series]] = [[] for _ in range(len(fake_scores_per_solver))]
        for cls_name in self._cached_real_metric_score:
            real_score = self._cached_real_metric_score[cls_name]
            best_real_scores.append(real_score.max())
            best_real_idx = real_score.idxmax()
            for fake_idx, fake_score in enumerate(fake_scores_per_solver):
                cor_fake_score = pd.Series({metric: fake_score[cls_name][metric][idx] for metric, idx in best_real_idx.items()})
                cor_fake_scores[fake_idx].append(cor_fake_score)
        # hack
        self._cached_real_metric_score[best_cls_name] = pd.concat(best_real_scores, axis=1, verify_integrity=True).T
        for fake_score, cor_fake_score in zip_strict(fake_scores_per_solver, cor_fake_scores):
            fake_score[best_cls_name] = pd.concat(cor_fake_score, axis=1, verify_integrity=True).T
        rank_score_best_cls = self._calculate_rank(fake_scores_per_solver, solver_names, best_cls_name)
        # remove
        self._cached_real_metric_score.pop(best_cls_name)
        for fake_score in fake_scores_per_solver:
            fake_score.pop(best_cls_name)
        display(rank_score_best_cls)
        display(Markdown('---'))

        display(Markdown('# overall'))
        # todo
        display(Markdown('---'))

        return rank_scores_per_cls, fake_scores_per_solver

    def _calculate_rank(self, fake_scores: List[Scores], solver_names: Sequence[str], classifier_name: str):
        scores_of_cls = [scores[classifier_name] for scores in fake_scores]
        # plot scatter
        _plot_scatter_cls(self._cached_real_metric_score[classifier_name], scores_of_cls, solver_names, classifier_name)
        # rank score
        sp_score_of_cls = pd.concat([_spearman_score(self._cached_real_metric_score[classifier_name], score) for score in scores_of_cls],
                                    axis=0, ignore_index=True, verify_integrity=True).assign(solver=solver_names)
        return sp_score_of_cls

    def _train_test_cast(self, train_df: pd.DataFrame, y_column: str):
        x_columns = list(self._real_df.columns)
        x_columns.remove(y_column)
        train_df = self._transformer.transform(train_df)
        x_train, y_train = train_df[x_columns].to_numpy(), train_df[y_column].to_numpy()
        x_test = self._cached_transformed_test_df[x_columns].to_numpy()
        y_test = self._cached_transformed_test_df[y_column].to_numpy()
        return x_train, x_test, y_train, y_test


# || ========================

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from .rank import Score


def _spearman_score(real_score: Score, fake_score: Score):
    """
    :return: DataFrame[0, metric name]
    """
    score = dict()
    for metric in ['acc', 'bacc', 'f1', 'auroc']:
        score[metric] = spearmanr(real_score[metric].to_numpy(), fake_score[metric].to_numpy()).correlation
    return pd.DataFrame(score, index=[0])


def _set_axis_range(lower: float, upper: float, step: float = .1):
    lower, upper, step = float(lower), float(upper), float(step)
    plt.axis([lower - step, upper + step, lower - step, upper + step])
    plt.xticks(np.arange(lower, upper + step, step))
    plt.yticks(np.arange(lower, upper + step, step))
    plt.plot([lower, upper], [lower, upper], '--', color='grey', alpha=.3)


def _plot_scatter_cls(real_score: Score, fake_scores: Sequence[Score], solver_names: Sequence[str], classifier_name: str):
    for metric_name in ['acc', 'bacc', 'f1', 'auroc']:
        plt.figure(figsize=np.array([700, 700]) / plt.rcParams['figure.dpi'])
        plt.title(f'{classifier_name} {metric_name} on test when trained on')
        plt.xlabel('real')
        plt.ylabel('fake')
        plt.axis('square')
        if metric_name == 'auroc':
            _set_axis_range(.5, 1, step=.05)
        else:
            _set_axis_range(0, 1)
        for fake_score in fake_scores:
            plt.scatter(real_score[metric_name].to_numpy(), fake_score[metric_name].to_numpy(), marker='x', alpha=.5)
        plt.legend(['Perfect match', *solver_names])
        plt.show()
