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
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter('penalty', ['l2', 'l1'], default_value='l2'))
        cs.add_hyperparameter(LogStepHyperparameter('alpha', -5, -3, default_value=1e-4))
        cs.add_hyperparameter(LogStepHyperparameter('tol', -4, -2, default_value=1e-3))
        super().__init__('SGD-SVM', partial(SGDClassifier, loss='hinge', n_jobs=-1, class_weight='balanced'), cs)


class SpacedSGDLRC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter('penalty', ['l2', 'l1'], default_value='l2'))
        cs.add_hyperparameter(LogStepHyperparameter('alpha', -5, -3, default_value=1e-4))
        cs.add_hyperparameter(LogStepHyperparameter('tol', -4, -2, default_value=1e-3))
        super().__init__('SGD-LR', partial(SGDClassifier, loss='log', n_jobs=-1, class_weight='balanced'), cs)


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


@deprecated
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


class SpacedMLPC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(
            CategoricalHyperparameter(
                'hidden_layer_sizes',
                tuple((neuron,) * layer for layer in range(1, 4) for neuron in (2 ** exp for exp in range(4, 8)))
            )
        )
        # cs.add_hyperparameter(LogStepHyperparameter('alpha', -5, -3, default_value=1e-4))
        cs.add_hyperparameter(LogStepHyperparameter('learning_rate_init', -5, -2, default_value=1e-3))
        # cs.add_hyperparameter(LogStepHyperparameter('tol', -5, -3, default_value=1e-4))
        # cs.add_hyperparameter(CategoricalHyperparameter('early_stopping', [False, True], default_value=False))
        super().__init__('MLP', partial(MLPClassifier, max_iter=500, early_stopping=True, n_iter_no_change=32), cs)


class SpacedAdaBoostC(SpacedClassifier):
    def __init__(self):
        super().__init__('AdaBoost', AdaBoostClassifier, ConfigurationSpace())
        cs = ConfigurationSpace()
        cs.add_hyperparameter(OrdinalHyperparameter('max_depth', list(range(1, 5)), default_value=1))
        cs.add_hyperparameter(OrdinalHyperparameter('n_estimators', list(range(50, 250, 50)), default_value=50))
        cs.add_hyperparameter(OrdinalHyperparameter('learning_rate', [2e-2, 2e-1, 1.], default_value=1.))
        self._params = [c.get_dictionary() for c in generate_grid(cs)]
        self._estimators = []
        for p in self._params:
            base_estimator = DecisionTreeClassifier(max_depth=p.pop('max_depth'))
            self._estimators.append(AdaBoostClassifier(base_estimator=base_estimator, **p))


# todo: consider constraint on `max_depth` instead of trivial properties


class SpacedHistGBC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(LogStepHyperparameter('learning_rate', -2, 0, default_value=.1))
        cs.add_hyperparameter(OrdinalHyperparameter('max_leaf_nodes', [31, 63, 127, 255], default_value=31))
        cs.add_hyperparameter(OrdinalHyperparameter('min_samples_leaf', [2, 8, 20, 32], default_value=20))
        super().__init__('HistGB', partial(HistGradientBoostingClassifier, max_iter=500, n_iter_no_change=16), cs)


class SpacedRandomForestC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(OrdinalHyperparameter('n_estimators', (50, 100, 200)))
        cs.add_hyperparameter(OrdinalHyperparameter('criterion', ('gini', 'entropy')))
        cs.add_hyperparameter(OrdinalHyperparameter('min_samples_split', (2, 8, 16)))
        cs.add_hyperparameter(OrdinalHyperparameter('min_samples_leaf', (1, 4, 16)))
        super().__init__('RandomForest', partial(RandomForestClassifier, n_jobs=-1, class_weight='balanced'), cs)


class SpacedExtraTreesC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(OrdinalHyperparameter('n_estimators', (50, 100, 200)))
        cs.add_hyperparameter(OrdinalHyperparameter('criterion', ('gini', 'entropy')))
        cs.add_hyperparameter(OrdinalHyperparameter('min_samples_split', (2, 8, 16)))
        cs.add_hyperparameter(OrdinalHyperparameter('min_samples_leaf', (1, 4, 16)))
        super().__init__('ExtraTrees', partial(ExtraTreesClassifier, n_jobs=-1, class_weight='balanced'), cs)


class SpacedDecisionTreeC(SpacedClassifier):
    def __init__(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(OrdinalHyperparameter('criterion', ('gini', 'entropy')))
        cs.add_hyperparameter(OrdinalHyperparameter('max_depth', (4, 16, 32)))
        cs.add_hyperparameter(OrdinalHyperparameter('min_samples_split', (2, 8, 16)))
        cs.add_hyperparameter(OrdinalHyperparameter('min_samples_leaf', (1, 4, 16)))
        super().__init__('DecisionTree', partial(DecisionTreeClassifier, class_weight='balanced'), cs)
