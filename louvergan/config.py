import os
from dataclasses import dataclass

import torch
import torch.backends.cudnn as cudnn


# todo: change this file


@dataclass
class HyperParam:
    dataset_path: str = './dataset/'
    checkpoint_path: str = './ckpt/'
    device: str = 'cpu'

    n_epoch: int = 100  # 1600
    save_step: int = 50

    latent_dim: int = 128
    batch_size: int = 512

    gen_lr: float = 2e-4
    gen_weight_decay: float = 1e-6

    dis_lr: float = 2e-4
    dis_pac_size: int = 16

    lambda_cond: float = 1.
    lambda_corr: float = 1.
    lambda_info: float = .2
    moving_avg_w: float = .99

    def __post_init__(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.device = 'cuda:0'


DATASET_NAME = 'adult'
DATASET_CORR = [
    {'A': ['education'], 'B': ['education-num']},
    {'A': ['sex', 'relationship'], 'B': ['marital-status']},
]
