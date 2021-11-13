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

    def __post_init__(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.device = 'cuda:0'


DATASET_NAME = 'adult'
