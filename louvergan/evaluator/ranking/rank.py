from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .space import SpacedAdaBoostC, SpacedBernoulliNBC, SpacedDecisionTreeC, SpacedExtraTreesC, \
    SpacedGaussianNBC, SpacedHistGBC, SpacedMLPC, SpacedRandomForestC, SpacedSGDLRC, SpacedSGDSVMC

Scores = Dict[str, pd.DataFrame]


def rank_binary_classifiers(x_train: np.ndarray, x_test: np.ndarray,
                            y_train: np.ndarray, y_test: np.ndarray) -> Scores:
    classifiers = [
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

    import time
    scores = dict()
    for cls in classifiers:
        print(f'{cls.name} with {cls.n_estimators} configs')
        start = time.time()
        cls.fit(x_train, y_train)
        scores[cls.name] = cls.score(x_test, y_test)
        print(f'finished in {time.time() - start:.2f}s\n')
    return scores


def plot_scatter(real_scores: Scores, fake_scores: Scores):
    for metric in ['acc', 'bacc', 'f1', 'auroc']:
        plt.figure(figsize=np.array([700, 700]) / plt.rcParams['figure.dpi'])
        plt.title(metric + ' on test when trained on')
        plt.xlabel('real')
        plt.ylabel('fake')
        plt.axis('square')
        if metric == 'auroc':
            plt.axis([0.4, 1.1, 0.4, 1.1])
            plt.xticks(np.arange(0.5, 1.1, 0.1))
            plt.yticks(np.arange(0.5, 1.1, 0.1))
            plt.plot([0.5, 1], [0.5, 1], '--', color='grey', alpha=.3)
        else:
            plt.axis([-0.1, 1.1, -0.1, 1.1])
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.plot([0, 1], [0, 1], '--', color='grey', alpha=.3)
        legend = ['Perfect match']
        for classifier in real_scores:
            plt.scatter(real_scores[classifier][metric].to_numpy(), fake_scores[classifier][metric].to_numpy(),
                        marker='+', alpha=.5)
            legend.append(classifier)
        plt.legend(legend)
        plt.show()


def spearman_score(real_scores: Scores, fake_scores: Scores):
    real_scores = pd.concat(real_scores.values(), ignore_index=True)
    fake_scores = pd.concat(fake_scores.values(), ignore_index=True)
    score = dict()
    for metric in ['acc', 'bacc', 'f1', 'auroc']:
        score[metric] = spearmanr(real_scores[metric].to_numpy(), fake_scores[metric].to_numpy()).correlation
    return pd.DataFrame(score, index=[0])
