import torch.cuda

from .cc import CondClamp, CondClampSolver, CondClampSolverConfig, CondClampSolverType
from .config import DATASET_CC, DATASET_NAME, HyperParam
from .preprocess import preprocess
from .sampler import get_sampler
from .util import set_seed


def main():
    print(f'dataset: {DATASET_NAME}')
    opt = HyperParam()
    set_seed()
    print(opt)

    transformer, data, split = preprocess(opt)
    columns = transformer.columns
    meta = transformer.meta
    sampler = get_sampler(opt, data, meta, split)

    cc_solvers = []
    cc_config = CondClampSolverConfig(checkpoint_path=opt.checkpoint_path, device=opt.device)
    for cc in DATASET_CC:
        print(f'new CC: {cc["A"]} => {cc["B"]}')
        solver = CondClampSolver.from_type(CondClampSolverType.CGAN, CondClamp.from_dict(cc, columns, meta))
        solver.fit(sampler, cc_config)
        cc_solvers.append(solver)

if __name__ == '__main__':
    main()
