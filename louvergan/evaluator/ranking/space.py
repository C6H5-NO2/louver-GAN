from functools import partial

import numpy as np
import sklearn.experimental.enable_hist_gradient_boosting  # noqa
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.util import generate_grid
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import deprecated

from .classifier import SpacedClassifier


class LogStepHyperparameter(OrdinalHyperparameter):
    def __init__(self, name: str, lower_exponent: int, upper_exponent: int, base: int = 10, default_value: float = None):
        space = np.logspace(lower_exponent, upper_exponent, num=upper_exponent - lower_exponent + 1, base=base)
        super().__init__(name, space, default_value=default_value)


class SpacedSGDSVMC(SpacedClassifier):
    def __init__(self):
        # cs = ConfigurationSpace()
        # cs.add_hyperparameter(CategoricalHyperparameter('penalty', ('l2', 'l1'), default_value='l2'))
        # cs.add_hyperparameter(LogStepHyperparameter('alpha', -7, -1, default_value=1e-4))
        # cs.add_hyperparameter(LogStepHyperparameter('tol', -5, -1, default_value=1e-3))
        # super().__init__('SGD-SVM', partial(SGDClassifier, loss='hinge', n_jobs=-1, class_weight='balanced'), cs)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter('penalty', ('l2', 'l1')))
        cs.add_hyperparameter(CategoricalHyperparameter('alpha', (1e-4, 1e-3)))
        cs.add_hyperparameter(CategoricalHyperparameter('tol', (1e-4, 1e-3)))
        super().__init__('SGD-SVM', partial(SGDClassifier, loss='hinge', n_jobs=-1,
                                            n_iter_no_change=20, class_weight='balanced'), cs)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._scaler = x.max(axis=0).clip(min=1)
        return super().fit(x / self._scaler, y)

    def predict(self, x: np.ndarray):
        return super().predict(x / self._scaler)

    def score(self, x: np.ndarray, y: np.ndarray):
        return super().score(x / self._scaler, y)


class SpacedSGDLRC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter('penalty', ('l2', 'l1')))
        cs.add_hyperparameter(CategoricalHyperparameter('alpha', (1e-4, 1e-3)))
        cs.add_hyperparameter(CategoricalHyperparameter('tol', (1e-4, 1e-3)))
        super().__init__('SGD-LR', partial(SGDClassifier, loss='log', n_jobs=-1,
                                           n_iter_no_change=20, class_weight='balanced'), cs)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._scaler = x.max(axis=0).clip(min=1)
        return super().fit(x / self._scaler, y)

    def predict(self, x: np.ndarray):
        return super().predict(x / self._scaler)

    def score(self, x: np.ndarray, y: np.ndarray):
        return super().score(x / self._scaler, y)


@deprecated()
class SpacedLinearSVC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter('penalty', ['l1', 'l2'], default_value='l2'))
        cs.add_hyperparameter(LogStepHyperparameter('tol', -5, -2, default_value=1e-4))
        cs.add_hyperparameter(LogStepHyperparameter('C', -3, 2, default_value=1.))
        super().__init__('LinearSVC', partial(LinearSVC, dual=False, class_weight='balanced'), cs)


@deprecated()
class SpacedLRC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter('penalty', ['l1', 'l2', 'none'], default_value='l2'))
        cs.add_hyperparameter(LogStepHyperparameter('tol', -5, -2, default_value=1e-4))
        cs.add_hyperparameter(LogStepHyperparameter('C', -3, 2, default_value=1.))
        super().__init__(
            name='LR', param_space=cs,
            etype=partial(LogisticRegression, class_weight='balanced', solver='saga', max_iter=1000, n_jobs=-1)
        )


class SpacedGaussianNBC(SpacedClassifier):
    def __init__(self):
        super().__init__('GaussianNB', GaussianNB, ConfigurationSpace()),


@deprecated()
class SpacedMultinomialNBC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(LogStepHyperparameter('alpha', -3, 2, default_value=1.))
        super().__init__('MultinomialNB', MultinomialNB, cs)
        raise ValueError('MultinomialNB requires positive features')


class SpacedBernoulliNBC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(LogStepHyperparameter('alpha', -3, 2, default_value=1.))
        super().__init__('BernoulliNB', BernoulliNB, cs)


# broken
class SpacedMLPC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter(
            'hidden_layer_sizes',
            tuple((neuron,) * layer for layer in range(1, 4) for neuron in (2 ** exp for exp in range(5, 9)))
        ))
        # cs.add_hyperparameter(CategoricalHyperparameter('hidden_layer_sizes', [(32,), (32, 32,), (32, 32, 32,)]))
        # cs.add_hyperparameter(LogStepHyperparameter('alpha', -5, -3, default_value=1e-4))
        cs.add_hyperparameter(CategoricalHyperparameter('learning_rate_init', (1e-4, 1e-3)))
        # cs.add_hyperparameter(LogStepHyperparameter('learning_rate_init', -5, -2, default_value=1e-3))
        cs.add_hyperparameter(CategoricalHyperparameter('max_iter', (500,)))
        # cs.add_hyperparameter(LogStepHyperparameter('tol', -5, -3, default_value=1e-4))
        # cs.add_hyperparameter(CategoricalHyperparameter('early_stopping', [False, True], default_value=False))
        super().__init__('MLP', partial(MLPClassifier, early_stopping=True, n_iter_no_change=20), cs)


# checked
class SpacedAdaBoostC(SpacedClassifier):
    def __init__(self):
        super().__init__('AdaBoost', AdaBoostClassifier, ConfigurationSpace())
        cs = ConfigurationSpace()
        cs.add_hyperparameter(OrdinalHyperparameter('max_depth', (2,)))
        # cs.add_hyperparameter(OrdinalHyperparameter('max_depth', tuple(range(1, 5)), default_value=1))
        cs.add_hyperparameter(OrdinalHyperparameter('n_estimators', tuple(range(50, 300, 50)), default_value=50))
        cs.add_hyperparameter(OrdinalHyperparameter('learning_rate', (.2,)))
        # cs.add_hyperparameter(OrdinalHyperparameter('learning_rate', (2e-2, 2e-1, 1.), default_value=1.))
        self._params = [c.get_dictionary() for c in generate_grid(cs)]
        self._estimators = []
        for p in self._params:
            base_estimator = DecisionTreeClassifier(max_depth=p.pop('max_depth'))
            self._estimators.append(AdaBoostClassifier(base_estimator=base_estimator, **p))
            p['max_depth'] = base_estimator.max_depth


# broken
class SpacedHistGBC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        # cs.add_hyperparameter(LogStepHyperparameter('learning_rate', -2, 0, default_value=.1))
        # cs.add_hyperparameter(OrdinalHyperparameter('max_leaf_nodes', (2 ** (np.arange(10) + 2) - 1).tolist(), default_value=31))
        cs.add_hyperparameter(OrdinalHyperparameter('max_leaf_nodes', (3, 15, 63, 255, 1023)))
        # cs.add_hyperparameter(OrdinalHyperparameter('max_leaf_nodes', (31, 63, 127, 255), default_value=31))
        cs.add_hyperparameter(OrdinalHyperparameter('min_samples_leaf', (2, 6, 20, 62, 200), default_value=20))
        # cs.add_hyperparameter(OrdinalHyperparameter('min_samples_leaf', (2, 6, 20, 62, 200), default_value=20))
        super().__init__('HistGB', partial(HistGradientBoostingClassifier, max_iter=500), cs)


# checked
class SpacedRandomForestC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        # cs.add_hyperparameter(OrdinalHyperparameter('n_estimators', (50, 100, 200)))
        # cs.add_hyperparameter(OrdinalHyperparameter('criterion', ('gini', 'entropy')))
        # cs.add_hyperparameter(OrdinalHyperparameter('max_depth', (2, 4, 8, 16, 32, 64, 128)))
        cs.add_hyperparameter(OrdinalHyperparameter('min_samples_split', (2, 4, 8, 16, 32)))
        # cs.add_hyperparameter(OrdinalHyperparameter('min_samples_leaf', (1, 4, 16, 64)))
        super().__init__('RandomForest', partial(RandomForestClassifier, n_jobs=-1, class_weight='balanced'), cs)


# checked
class SpacedExtraTreesC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        # cs.add_hyperparameter(OrdinalHyperparameter('n_estimators', (50, 100, 200)))
        # cs.add_hyperparameter(OrdinalHyperparameter('criterion', ('gini', 'entropy')))
        cs.add_hyperparameter(OrdinalHyperparameter('min_samples_split', (2, 5, 8, 11, 14, 17, 20)))
        # cs.add_hyperparameter(OrdinalHyperparameter('min_samples_leaf', (1, 4, 16, 32)))
        cs.add_hyperparameter(OrdinalHyperparameter('min_samples_leaf', (1,)))
        super().__init__('ExtraTrees', partial(ExtraTreesClassifier, n_jobs=-1, class_weight='balanced'), cs)


# checked
class SpacedDecisionTreeC(SpacedClassifier):
    def __init__(self):
        super().__init__('DecisionTree', DecisionTreeClassifier, ConfigurationSpace())
        self._params = [
            # dict(min_samples_split=38, min_samples_leaf=37),
            # dict(min_samples_split=34, min_samples_leaf=33),
            # dict(min_samples_split=30, min_samples_leaf=29),
            # dict(min_samples_split=26, min_samples_leaf=25),
            dict(min_samples_split=22, min_samples_leaf=21),
            dict(min_samples_split=18, min_samples_leaf=17),
            dict(min_samples_split=14, min_samples_leaf=13),
            dict(min_samples_split=10, min_samples_leaf=9),
            dict(min_samples_split=6, min_samples_leaf=5),
            # dict(min_samples_split=2, min_samples_leaf=1),
        ]
        etype = partial(DecisionTreeClassifier, class_weight='balanced')
        self._estimators = [etype(**p) for p in self._params]

        # cs = ConfigurationSpace()
        # cs.add_hyperparameter(OrdinalHyperparameter('criterion', ('gini', 'entropy')))
        # cs.add_hyperparameter(OrdinalHyperparameter('max_depth', (2, 4, 8, 16, 32)))
        # cs.add_hyperparameter(OrdinalHyperparameter('min_samples_split', (2, 8, 32, 128)))
        # cs.add_hyperparameter(OrdinalHyperparameter('min_samples_leaf', (1, 4, 16, 64)))
        # super().__init__('DecisionTree', partial(DecisionTreeClassifier, class_weight='balanced'), cs)
