import os
import random
from dataclasses import dataclass

import numpy as np
import torch

path_join = os.path.join


def set_seed(seed: int = random.randint(1, 0xffff)):
    print(f'seed: {seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


CONTINUOUS = 'continuous'
DISCRETE = 'discrete'
TYPE = 'type'
SPLIT = 'split'


# def get_slices(batch: torch.Tensor, bias_span_list: List[Tuple[int, int]]):
#     slices = []
#     for bias, span in bias_span_list:
#         s = batch[:, bias: bias + span]
#         slices.append(s)
#     return torch.cat(slices, dim=1)


@dataclass
class ColumnMeta:
    discrete: bool
    nmode: int  # one-hot part only


@dataclass
class SlatDim:
    attr_dim: int
    cond_dim: int
