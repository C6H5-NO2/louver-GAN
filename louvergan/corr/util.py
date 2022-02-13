from enum import Enum, auto
from typing import Dict, List, NamedTuple
from typing import Iterable, Sequence, Tuple

import torch

from ..util import ColumnMeta

# || slicing

BiasSpan = Tuple[int, int]


def get_column_bias_span(col_name: str, columns: Sequence[str], meta: Sequence[ColumnMeta],
                         include_scalar: bool = False) -> BiasSpan:
    idx = columns.index(col_name)
    bias = sum(meta[i].nmode + int(not meta[i].discrete) for i in range(idx))
    span = meta[idx].nmode
    if include_scalar:
        span += int(not meta[idx].discrete)
    else:
        bias += int(not meta[idx].discrete)
    return bias, span


def get_slices(batch: torch.Tensor, bias_span_s: Iterable[BiasSpan]) -> torch.Tensor:
    slices = []
    for bias, span in bias_span_s:
        s = batch[..., bias: bias + span]
        slices.append(s)
    return torch.cat(slices, dim=-1)


# || corr

class CorrSolverType(Enum):
    NONE = auto()
    AE = auto()
    CGAN = auto()


class Corr(NamedTuple):
    bias_span_a_per_slat: List[List[BiasSpan]]
    bias_span_b_per_slat: List[List[BiasSpan]]
    a_names: List[str]
    b_names: List[str]

    @classmethod
    def from_dict(cls, corr_dict: Dict[str, Iterable[str]], columns: Sequence[str], meta: Sequence[ColumnMeta]):
        assert 'A' in corr_dict and 'B' in corr_dict
        bias_span_a = [get_column_bias_span(col, columns, meta, include_scalar=False) for col in corr_dict['A']]
        bias_span_b = [get_column_bias_span(col, columns, meta, include_scalar=False) for col in corr_dict['B']]
        return cls([bias_span_a], [bias_span_b], list(corr_dict['A']), list(corr_dict['B']))
