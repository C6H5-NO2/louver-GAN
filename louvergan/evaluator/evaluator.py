from typing import Dict, Iterable, Optional
from warnings import filterwarnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .classification import \
    binary_classification, binary_classifiers, binary_metrics, \
    multiclass_classification, multiclass_classifiers, multiclass_metrics
from .clustering import clustering, clustering_metrics, clusters
from .correlation import pairwise_mi, posterior_jsd
from .ipy import Markdown, display
from .plot import plot_mi
from .regression import regression, regression_metrics, regressors
from .statistics import statistical_similarity
from .trace import Trace
from .transformer import EvaluatorTransformer
from ..config import DATASET_CORR, DATASET_EVAL, DATASET_NAME, HyperParam
from ..util import ColumnMeta, path_join


class Evaluator:
    def __init__(self, opt: HyperParam, meta: Iterable[ColumnMeta]):
        filterwarnings(action='once', category=FutureWarning)
        filterwarnings(action='default', category=ConvergenceWarning)
        filterwarnings(action='ignore', category=UndefinedMetricWarning)

        self._real_df = pd.read_csv(path_join(opt.dataset_path, f'{DATASET_NAME}/{DATASET_NAME}-train.csv'))
        self._transformer = EvaluatorTransformer()
        self._transformer.fit(self._real_df, meta)
        self._test_df = pd.read_csv(path_join(opt.dataset_path, f'{DATASET_NAME}/{DATASET_NAME}-test.csv'))
        self._meta = list(meta)

        self._cached_real_mi_mat = pairwise_mi(self._transformer.transform(self._real_df).to_numpy(), meta)

        self._traces: Dict[str, Trace] = dict(
            stat_per_col=Trace(DATASET_EVAL.get('statistics')),
            mean_per_bin_cls_metric=Trace(f'mean {m}' for m in binary_metrics),
            mean_per_mul_cls_metric=Trace(f'mean {m}' for m in multiclass_metrics),
            mean_per_reg_metric=Trace(f'mean {m}' for m in regression_metrics),
            mean_per_clu_metric=Trace(f'mean {m}' for m in clustering_metrics),
            corr_nmi_matrix_error=Trace(['rmse', 'mae']),
            corr_posterior_jsd=Trace(f'{corr["A"]} => {corr["B"]}' for corr in DATASET_CORR),
        )
        self._traces.update({
            f'{m}_per_bin_cls': Trace(cls.__name__ for cls, _ in binary_classifiers) for m in binary_metrics
        })
        self._traces.update({
            f'{m}_per_mul_cls': Trace(cls.__name__ for cls, _ in multiclass_classifiers) for m in multiclass_metrics
        })
        self._traces.update({
            f'{m}_per_reg': Trace(reg.__name__ for reg, _ in regressors) for m in regression_metrics
        })
        self._traces.update({
            f'{m}_per_clu': Trace(clu.__name__ for clu, _ in clusters) for m in clustering_metrics
        })

    def evaluate(self, df: pd.DataFrame):
        """
        This will add to history
        """

        display(Markdown('# Statistics'))
        stat = self.evaluate_statistics(df, DATASET_EVAL.get('statistics'))
        display(stat)
        self._traces['stat_per_col'].collect(*stat['distance'])

        if DATASET_EVAL.get('classification'):
            display(Markdown('# Classification'))
            cls = self.evaluate_classification(df, DATASET_EVAL['classification'])
            display(cls)
            if len(self._real_df[DATASET_EVAL['classification']].unique()) == 2:  # is binary
                for metric in binary_metrics:
                    self._traces[f'{metric}_per_bin_cls'].collect(*cls[metric])
                self._traces['mean_per_bin_cls_metric'].collect(*(cls[m].mean() for m in binary_metrics))
            else:
                for metric in multiclass_metrics:
                    self._traces[f'{metric}_per_mul_cls'].collect(*cls[metric])
                self._traces['mean_per_mul_cls_metric'].collect(*(cls[m].mean() for m in multiclass_metrics))

        if DATASET_EVAL.get('regression'):
            display(Markdown('# Regression'))
            reg = self.evaluate_regression(df, DATASET_EVAL['regression'])
            display(reg)
            for metric in regression_metrics:
                self._traces[f'{metric}_per_reg'].collect(*reg[metric])
            self._traces['mean_per_reg_metric'].collect(*(reg[m].mean() for m in regression_metrics))

        if DATASET_EVAL.get('clustering'):
            display(Markdown('# Clustering'))
            clu = self.evaluate_clustering(df, DATASET_EVAL['clustering'])
            display(clu)
            for metric in clustering_metrics:
                self._traces[f'{metric}_per_clu'].collect(*clu[metric])
            self._traces['mean_per_clu_metric'].collect(*(clu[m].mean() for m in clustering_metrics))

        display(Markdown('# Correlation'))
        mi_err, pjsd = self.evaluate_correlation(df)
        display(mi_err)
        display(pjsd)
        self._traces['corr_nmi_matrix_error'].collect(mi_err['rmse'].item(), mi_err['mae'].item())
        self._traces['corr_posterior_jsd'].collect(*pjsd['pjsd'])

    def plot_history(self):
        for n, t in self._traces.items():
            t.plot(plot_mean='mean_per_' not in n, title=n.replace('_', ' '))
            display(Markdown('---'))

    def evaluate_statistics(self, fake_df: pd.DataFrame, columns: Optional[Iterable[str]] = None):
        return statistical_similarity(self._real_df, fake_df, self._meta, columns)

    def _train_test_cast(self, train_df: pd.DataFrame, y_column: str):
        x_columns = list(self._real_df.columns)
        x_columns.remove(y_column)

        train_df = self._transformer.transform(train_df)
        x_train, y_train = train_df[x_columns].to_numpy(), train_df[y_column].to_numpy()

        test_df = self._transformer.transform(self._test_df)
        x_test, y_test = test_df[x_columns].to_numpy(), test_df[y_column].to_numpy()

        return x_train, x_test, y_train, y_test

    def evaluate_classification(self, fake_df: pd.DataFrame, y_column: str):
        x_train, x_test, y_train, y_test = self._train_test_cast(fake_df, y_column)
        if len(self._real_df[y_column].unique()) == 2:
            return binary_classification(x_train, x_test, y_train, y_test)
        else:
            return multiclass_classification(x_train, x_test, y_train, y_test)

    def evaluate_regression(self, fake_df: pd.DataFrame, y_column: str):
        x_train, x_test, y_train, y_test = self._train_test_cast(fake_df, y_column)
        return regression(x_train, x_test, y_train, y_test)

    def evaluate_clustering(self, fake_df: pd.DataFrame, y_column: str):
        x_train, _, y_train, _ = self._train_test_cast(fake_df, y_column)
        return clustering(x_train, y_train)

    def evaluate_correlation(self, fake_df: pd.DataFrame):
        data = self._transformer.transform(fake_df)

        mi_mat = pairwise_mi(data.to_numpy(), self._meta)
        plot_mi(mi_mat)
        mi_err = pd.DataFrame([{
            'rmse': mean_squared_error(self._cached_real_mi_mat, mi_mat, squared=False),
            'mae': mean_absolute_error(self._cached_real_mi_mat, mi_mat)
        }])

        pjsd = []
        for corr in DATASET_CORR:
            assert len(corr['B']) == 1  # fixme
            pjsd.append({
                'corr': f'{corr["A"]} => {corr["B"]}',
                'pjsd': posterior_jsd(self._real_df, fake_df, corr['A'], corr['B'][0], weight=True, verbose=False)
            })
        pjsd = pd.DataFrame(pjsd)

        return mi_err, pjsd
