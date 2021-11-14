from enum import Enum, auto
from typing import Dict, List, NamedTuple

from ..util import BiasSpan, ColumnMeta, get_column_bias_span


class CondClampSolverType(Enum):
    NONE = auto()
    AE = auto()
    CGAN = auto()


class CondClamp(NamedTuple):
    bias_span_a_per_slat: List[List[BiasSpan]]
    bias_span_b_per_slat: List[List[BiasSpan]]

    # a_names: List[str]
    # b_names: List[str]

    @classmethod
    def from_dict(cls, ccdict: Dict[str, List[str]], columns: List[str], meta: List[ColumnMeta]):
        assert 'A' in ccdict and 'B' in ccdict
        bias_span_a = [get_column_bias_span(col, columns, meta, include_scalar=False) for col in ccdict['A']]
        bias_span_b = [get_column_bias_span(col, columns, meta, include_scalar=False) for col in ccdict['B']]
        return cls([bias_span_a], [bias_span_b])
        # return cls([bias_span_a], [bias_span_b], ccdict['A'], ccdict['B'])
