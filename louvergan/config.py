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

    batch_size: int = 512

    def __post_init__(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.device = 'cuda:0'


DATASET_NAME = 'adult'
DATASET_CC = [
    {'A': ['sex', 'relationship'], 'B': ['marital-status']},
    {'A': ['education'], 'B': ['education-num']},
]
