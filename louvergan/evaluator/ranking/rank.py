import time
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .classifier import SpacedClassifier
from .space import SpacedAdaBoostC, SpacedBernoulliNBC, SpacedDecisionTreeC, SpacedExtraTreesC, \
    SpacedGaussianNBC, SpacedHistGBC, SpacedMLPC, SpacedRandomForestC, SpacedSGDLRC, SpacedSGDSVMC

Score = pd.DataFrame
""" DataFrame[configuration index, metric name]  """

Scores = Dict[str, Score]
"""  Dict[classifier name, Score] """


def rank_binary_classifiers(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    classifiers: List[SpacedClassifier] = [
        SpacedSGDSVMC(),
        SpacedSGDLRC(),
        SpacedGaussianNBC(),
        SpacedBernoulliNBC(),
        SpacedMLPC(),
        SpacedAdaBoostC(),
        SpacedHistGBC(),
        SpacedRandomForestC(),
        SpacedExtraTreesC(),
        SpacedDecisionTreeC(),
    ]

    classifiers: List[SpacedClassifier] = [
        SpacedSGDSVMC(),
        SpacedSGDLRC(),
        # SpacedGaussianNBC(),
        # SpacedBernoulliNBC(),
        # SpacedMLPC(),
        SpacedAdaBoostC(),  # todo
        # SpacedHistGBC(),
        SpacedRandomForestC(),
        SpacedExtraTreesC(),
        SpacedDecisionTreeC(),
    ]

    scores: Scores = dict()
    for cls in classifiers:
        print(f'{cls.name} with {cls.n_estimators} configs')
        start = time.time()
        cls.fit(x_train, y_train)
        scores[cls.name] = cls.score(x_test, y_test)
        print(f'finished in {time.time() - start:.2f}s\n')
    return scores, classifiers


def _set_axis_range(lower: float, upper: float, step: float = .1):
    lower, upper, step = float(lower), float(upper), float(step)
    plt.axis([lower - step, upper + step, lower - step, upper + step])
    plt.xticks(np.arange(lower, upper + step, step))
    plt.yticks(np.arange(lower, upper + step, step))
    plt.plot([lower, upper], [lower, upper], '--', color='grey', alpha=.3)


def plot_scatter(real_scores: Scores, fake_scores: Scores):
    for metric in ['acc', 'bacc', 'f1', 'auroc']:
        plt.figure(figsize=np.array([700, 700]) / plt.rcParams['figure.dpi'])
        plt.title(metric + ' on test when trained on')
        plt.xlabel('real')
        plt.ylabel('fake')
        plt.axis('square')
        if metric == 'acc':
            _set_axis_range(.5, 1)
        elif metric == 'auroc':
            _set_axis_range(.6, .8)
        else:
            _set_axis_range(0, 1)
        legend = ['Perfect match']
        for classifier in real_scores:
            plt.scatter(real_scores[classifier][metric].to_numpy(), fake_scores[classifier][metric].to_numpy(), marker='x', alpha=.7)
            legend.append(classifier)
        plt.legend(legend)
        plt.show()


def plot_scatter_all(real_scores: Scores, fake_scores: Sequence[Scores]):
    assert len(fake_scores) == 3
    for metric in ['acc', 'bacc', 'f1', 'auroc']:
        plt.figure(figsize=np.array([700, 700]) / plt.rcParams['figure.dpi'])
        plt.title(metric + ' on test when trained on')
        plt.xlabel('real')
        plt.ylabel('fake')
        plt.axis('square')
        if metric == 'auroc':
            _set_axis_range(.5, 1, step=.05)
        else:
            _set_axis_range(0, 1)

        real_score_all = []
        none_score_all = []
        ae_score_all = []
        cgan_score_all = []
        for classifier in real_scores:
            real_score_all.append(real_scores[classifier][metric].to_numpy())
            none_score_all.append(fake_scores[0][classifier][metric].to_numpy())
            ae_score_all.append(fake_scores[1][classifier][metric].to_numpy())
            cgan_score_all.append(fake_scores[2][classifier][metric].to_numpy())
        real_score_all = np.hstack(real_score_all)
        plt.scatter(real_score_all, np.hstack(none_score_all), marker='x', alpha=.5)
        plt.scatter(real_score_all, np.hstack(ae_score_all), marker='x', alpha=.5)
        plt.scatter(real_score_all, np.hstack(cgan_score_all), marker='x', alpha=.5)

        plt.legend(['Perfect match', 'none', 'ae', 'cgan'])
        plt.show()


def spearman_score(real_scores: Scores, fake_scores: Scores):
    real_scores = pd.concat(real_scores.values(), ignore_index=True)
    fake_scores = pd.concat(fake_scores.values(), ignore_index=True)
    score = dict()
    for metric in ['acc', 'bacc', 'f1', 'auroc']:
        score[metric] = spearmanr(real_scores[metric].to_numpy(), fake_scores[metric].to_numpy()).correlation
    return pd.DataFrame(score, index=[0])
