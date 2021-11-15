from copy import copy
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .config import HyperParam
from .model import Generator
from .sampler import CondDataLoader
from .transformer import Transformer
from .util import ColumnMeta, SlatDim


def synthesize(opt: HyperParam, split: Sequence[SlatDim], meta: Sequence[ColumnMeta], syn_size: int,
               state_dict: dict, loader: CondDataLoader, transformer: Transformer,
               save_path: Optional[str] = None) -> pd.DataFrame:
    generator = Generator(opt, split, meta).to(opt.device)
    generator.load_state_dict(state_dict)
    generator.eval()

    loader = copy(loader).eval()  # intended shallow copy

    data = []
    with torch.no_grad():
        for _ in range(syn_size // opt.batch_size + 1):
            z = torch.randn(opt.batch_size, opt.latent_dim, device=opt.device)
            _, cond = loader.get_batch()
            cond = cond.to(opt.device)
            x = generator(z, cond)
            data.append(x.cpu().numpy())
    data = np.concatenate(data, axis=0)
    data = data[: syn_size]

    df = transformer.inverse_transform(data)
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df
