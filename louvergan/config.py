import os
from dataclasses import dataclass

import torch
from torch.backends import cudnn


# todo: change this file


@dataclass
class HyperParam:
    dataset_path: str = './dataset/'
    checkpoint_path: str = './ckpt/'
    device: str = 'cpu'

    n_epoch: int = 1000
    save_step: int = 50

    latent_dim: int = 128
    batch_size: int = 256

    gen_lr: float = 2e-4
    gen_weight_decay: float = 1e-6

    dis_lr: float = 2e-4
    dis_pac_size: int = 16

    lambda_cond: float = 1.
    lambda_corr: float = .1
    lambda_info: float = .2
    moving_avg_w: float = .99

    def __post_init__(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.device = 'cuda:0'


# || adult
# DATASET_NAME = 'adult'
# DATASET_CORR = [
#     {'A': ['education'], 'B': ['education-num']},
#     # {'A': ['sex', 'relationship'], 'B': ['marital-status']},
# ]
# DATASET_EVAL = {
#     'statistics': ['education', ],
#     'classification': 'label',
#     'regression': 'hours-per-week',
#     'clustering': 'label',
# }


# || bank
DATASET_NAME = 'bank'
DATASET_CORR = [
    {'A': ['education'], 'B': ['job']},
]
DATASET_EVAL = {
    'statistics': ['age', ],
    'classification': 'y',
    'regression': 'cons.conf.idx',
    'clustering': 'y',
}


# || student
# DATASET_NAME = 'student'
# DATASET_CORR = [
#     {'A': ['famsup'], 'B': ['paid']},
#     {'A': ['Pstatus'], 'B': ['famsize']},
# ]
# DATASET_EVAL = {
#     'statistics': ['age', ],
#     'regression': 'G3',
#     'clustering': 'G3',
# }
