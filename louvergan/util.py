import os
import random
from dataclasses import dataclass

import numpy as np
import torch

path_join = os.path.join


def set_seed(seed=random.randint(1, 0xffff)):
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
