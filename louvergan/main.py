from typing import Sequence

import numpy as np

from .config import DATASET_CORR, DATASET_NAME, HyperParam
from .corr import Corr, CorrSolver, CorrSolverConfig, CorrSolverType
from .preprocess import preprocess
from .sampler import CondDataLoader, SplitCondDataLoader
from .util import ColumnMeta, set_seed


def pretrain_corr_solvers(opt: HyperParam, data: np.ndarray, columns: Sequence[str], meta: Sequence[ColumnMeta]):
    corr_solvers = []
    corr_config = CorrSolverConfig(checkpoint_path=opt.checkpoint_path, device=opt.device)
    loader = CondDataLoader(data, meta, corr_config.batch_size)
    for corr in DATASET_CORR:
        print(f'CORR: {corr["A"]} => {corr["B"]}')
        solver = CorrSolver.from_type(CorrSolverType.CGAN, Corr.from_dict(corr, columns, meta))
        solver.fit(loader, corr_config)
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

    corr_solvers = pretrain_corr_solvers(opt, data, columns, meta)

    loader = SplitCondDataLoader(data, meta, opt.batch_size, split, opt.device)


if __name__ == '__main__':
    main()
