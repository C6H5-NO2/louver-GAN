from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# || binary classification

binary_classifiers = [
    (LogisticRegression, dict(class_weight='balanced', max_iter=300, n_jobs=-1)),
    (MLPClassifier, dict(max_iter=300)),
    (AdaBoostClassifier, dict()),
    (RandomForestClassifier, dict(max_depth=20, n_jobs=-1)),
    (DecisionTreeClassifier, dict(max_depth=30, class_weight='balanced')),
]

binary_metrics = {
    'acc': accuracy_score,
    'bacc': balanced_accuracy_score,
    'f1': f1_score,
    'auroc': roc_auc_score,
}


def binary_classification(x_train: np.ndarray, x_test: np.ndarray,
                          y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    score = []
    for cls, args in binary_classifiers:
        classifier = cls(**args)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        score.append({'classifier': cls.__name__})
        for name, metric in binary_metrics.items():
            score[-1][name] = metric(y_test, y_pred)
    return pd.DataFrame(score)


# || multiclass classification

multiclass_classifiers = [
    (LogisticRegression, dict(class_weight='balanced', max_iter=300, n_jobs=-1)),
    (MLPClassifier, dict(max_iter=300)),
    (AdaBoostClassifier, dict()),
    (RandomForestClassifier, dict(max_depth=20, n_jobs=-1)),
    (DecisionTreeClassifier, dict(max_depth=30, class_weight='balanced')),
]

multiclass_metrics = {
    'acc': accuracy_score,
    'macro f1': partial(f1_score, average='macro'),
    'micro f1': partial(f1_score, average='micro'),
    'macro auroc': partial(roc_auc_score, average='macro', multi_class='ovr'),
}


def multiclass_classification(x_train: np.ndarray, x_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    score = []
    auroc_ohe = OneHotEncoder(sparse=False).fit(y_test.reshape(-1, 1))
    for cls, args in multiclass_classifiers:
        classifier = cls(**args)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        score.append({'classifier': cls.__name__})
        for name, metric in multiclass_metrics.items():
            if name == 'macro auroc':
                y_pred = auroc_ohe.transform(y_pred.reshape(-1, 1))
            score[-1][name] = metric(y_test, y_pred)
    return pd.DataFrame(score)
