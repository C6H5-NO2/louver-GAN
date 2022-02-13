import json

import pandas as pd

from .config import DATASET_NAME, HyperParam
from .polyfill import zip_strict
from .transformer import Transformer
from .util import SPLIT, SlatDim, path_join


def preprocess(opt: HyperParam, cont_as_cond=True):
    assert cont_as_cond, 'cont_as_cond'

    file = path_join(opt.dataset_path, f'{DATASET_NAME}/{DATASET_NAME}-train.csv')
    df = pd.read_csv(file)

    file = path_join(opt.dataset_path, f'{DATASET_NAME}/{DATASET_NAME}-names.json')
    with open(file) as j:
        desc: dict = json.load(j)

    # transformer = Transformer.load(path_join(opt.checkpoint_path, f'{DATASET_NAME}-transformer.pkl'))
    transformer = Transformer().fit(df, desc)
    data = transformer.transform(df)

    file = path_join(opt.checkpoint_path, f'{DATASET_NAME}-transformer.pkl')
    transformer.save(file)

    data_info = {
        'data row': df.shape[0],
        'data col': df.shape[1],
        'attr dim': data.shape[1],
        'cond dim': sum(m.nmode for m in transformer.meta if cont_as_cond or m.discrete),
        'cont as cond': cont_as_cond
    }
    print(data_info)

    split = [SlatDim(0, 0)]
    for col_name, col_meta in zip_strict(transformer.columns, transformer.meta):
        if SPLIT in desc[col_name] and desc[col_name][SPLIT]:
            split.append(SlatDim(0, 0))
        if col_meta.discrete:
            split[-1].attr_dim += col_meta.nmode
            split[-1].cond_dim += col_meta.nmode
        else:
            split[-1].attr_dim += 1 + col_meta.nmode
            split[-1].cond_dim += col_meta.nmode if cont_as_cond else 0

    print(split)
    assert sum(s.attr_dim for s in split) == data_info['attr dim']
    assert sum(s.cond_dim for s in split) == data_info['cond dim']

    return transformer, data, split
