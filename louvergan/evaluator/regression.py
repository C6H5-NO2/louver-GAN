from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

# || regression

regressors = [
    (Lasso, dict()),
    (MLPRegressor, dict()),
    (GradientBoostingRegressor, dict()),
    (RandomForestRegressor, dict(max_depth=20, n_jobs=-1)),
]

regression_metrics = {
    'r2': r2_score,
    'rmse': partial(mean_squared_error, squared=False),
    'mae': mean_absolute_error,
}


def regression(x_train: np.ndarray, x_test: np.ndarray,
               y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    score = []
    for reg, args in regressors:
        regressor = reg(**args)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        score.append({'regressor': reg.__name__})
        for name, metric in regression_metrics.items():
            score[-1][name] = metric(y_test, y_pred)
    return pd.DataFrame(score)
