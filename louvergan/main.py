from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .config import DATASET_CORR, DATASET_NAME, HyperParam
from .corr import Corr, CorrSolver, CorrSolverConfig, CorrSolverType
from .evaluator import Evaluator
from .model import Discriminator, Generator
from .polyfill import display, zip_strict
from .preprocess import preprocess
from .sampler import CondDataLoader
from .train import train
from .util import ColumnMeta, set_seed


def pretrain_corr_solvers(opt: HyperParam, data: np.ndarray,
                          columns: Sequence[str], meta: Sequence[ColumnMeta], stype: CorrSolverType):
    corr_solvers = []
    corr_config = CorrSolverConfig(checkpoint_path=opt.checkpoint_path, device=opt.device)
    loader = CondDataLoader(data, meta, corr_config.batch_size)
    for corr in DATASET_CORR:
        print(f'CORR: {corr["A"]} => {corr["B"]}')
        solver = CorrSolver.from_type(stype, Corr.from_dict(corr, columns, meta))
        solver.fit(loader, corr_config, verbose=True)
        corr_solvers.append(solver)
    return corr_solvers


def load_corr_solvers(opt: HyperParam, paths: Sequence[str],
                      columns: Sequence[str], meta: Sequence[ColumnMeta], stype: CorrSolverType):
    assert len(paths) == len(DATASET_CORR)
    corr_solvers = []
    corr_config = CorrSolverConfig(checkpoint_path=opt.checkpoint_path, device=opt.device)
    for path, corr in zip_strict(paths, DATASET_CORR):
        print(f'CORR: {corr["A"]} => {corr["B"]}')
        solver = CorrSolver.from_type(stype, Corr.from_dict(corr, columns, meta))
        state_dict = torch.load(path, map_location=corr_config.device)
        print(f'    corr: load {path}')
        solver.load(state_dict, corr_config)
        corr_solvers.append(solver)
    return corr_solvers


def main():
    print(f'dataset: {DATASET_NAME}')
    opt = HyperParam()
    set_seed()
    print(opt)

    transformer, data, split = preprocess(opt)
    columns = transformer.columns
    meta = transformer.meta
    display(pd.DataFrame(meta, index=columns).T)

    corr_solvers = pretrain_corr_solvers(opt, data, columns, meta, CorrSolverType.NONE)
    # corr_solvers = load_corr_solvers(opt, [
    #     './ckpt/corr=education=education-num=cgan-g-0799.pt',
    #     './ckpt/corr=sex~relationship=marital-status=cgan-g-0799.pt',
    # ], columns, meta, CorrSolverType.CGAN)

    loader = CondDataLoader(data, meta, opt.batch_size)

    generator = Generator(opt, split, meta).to(opt.device).train()
    print('----- generator -----')
    print(generator)

    discriminator = Discriminator(opt, split).to(opt.device).train()
    print('----- discriminator -----')
    print(discriminator)

    evaluator = Evaluator(opt, meta)

    train(opt, split, meta, loader, generator, discriminator, evaluator, transformer, corr_solvers)

    evaluator.plot_history()


if __name__ == '__main__':
    main()
