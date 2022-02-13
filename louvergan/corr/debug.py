from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd
import torch

from ..config import DATASET_NAME
from ..evaluator import Trace
from ..evaluator.correlation import posterior_jsd
from ..evaluator.statistics import jensen_shannon_divergence
from ..polyfill import zip_strict


def view_posterior(real_df: pd.DataFrame, fake_df: pd.DataFrame, evidence_cols: Sequence[str], posterior_col: str):
    if type(evidence_cols) is str:
        raise TypeError
    real_group = real_df.groupby(evidence_cols)[posterior_col]
    fake_group = fake_df.groupby(evidence_cols)[posterior_col]
    for name, real_ser in real_group:
        fake_ser: pd.Series = fake_group.get_group(name)
        if len(evidence_cols) == 1:
            name = (name,)
        print('-' * 30)
        print(', '.join(f'{c} == {n}' for c, n in zip_strict(evidence_cols, name)))
        jensen_shannon_divergence(real_ser, fake_ser, verbose=True)
    print('-' * 30)


@torch.no_grad()
def predict_inplace(solver, batch: torch.Tensor) -> torch.Tensor:
    corr_b_pred = solver.predict(batch)
    for pred, bias_span_s in zip(corr_b_pred, solver._corr.bias_span_b_per_slat):
        bias_in_slat = 0
        for bias, span in bias_span_s:
            batch[:, bias: bias + span] = pred[:, bias_in_slat: bias_in_slat + span]
            bias_in_slat += span
    return batch


@dataclass
class EvaluatorWrap:
    data: torch.Tensor
    real_df: pd.DataFrame
    solver: Any
    transformer: Any
    tracer: Trace

    @classmethod
    def parse(cls, conf, transformer, loader, solver):
        assert len(solver._corr.b_names) == 1
        data = torch.tensor(loader._data, dtype=torch.float32, device=conf.device)
        real_df = pd.read_csv(f'./dataset/{DATASET_NAME}/{DATASET_NAME}-train.csv')
        tracer = Trace(['loss corr d', 'loss corr g', 'pjsd'])
        return cls(data=data, real_df=real_df, solver=solver, transformer=transformer, tracer=tracer)

    def eval(self):
        data = predict_inplace(self.solver, self.data.clone())
        pred_df = self.transformer.inverse_transform(data.cpu().numpy())
        view_posterior(self.real_df, pred_df, self.solver._corr.a_names, self.solver._corr.b_names[0])
        pjsd = posterior_jsd(self.real_df, pred_df,
                             self.solver._corr.a_names, self.solver._corr.b_names[0],
                             weight=True, verbose=True)
        print('avg pjsd:', pjsd)
        print('*' * 30)
        return pjsd

    def collect(self, *args):
        self.tracer.collect(*args)

    def plot(self, **kwargs):
        self.tracer.plot(**kwargs)
