from enum import Enum, auto
from typing import Dict, Iterable, List, NamedTuple, Sequence

from ..util import BiasSpan, ColumnMeta, get_column_bias_span


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
