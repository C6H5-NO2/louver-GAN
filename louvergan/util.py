import os
import random
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch

path_join = os.path.join


def set_seed(seed: int = random.randint(1, 0xffff)):
    print(f'seed: {seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


# || constants for names.json

CONTINUOUS = 'continuous'
DISCRETE = 'discrete'
TYPE = 'type'
SPLIT = 'split'


# || data descriptions

@dataclass
class ColumnMeta:
    discrete: bool
    nmode: int  # one-hot part only


@dataclass
class SlatDim:
    attr_dim: int
    cond_dim: int


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


def get_slices(batch: torch.Tensor, bias_span_s: Sequence[BiasSpan]) -> torch.Tensor:
    slices = []
    for bias, span in bias_span_s:
        s = batch[..., bias: bias + span]
        slices.append(s)
    return torch.cat(slices, dim=-1)
