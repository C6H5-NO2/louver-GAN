from .config import DATASET_NAME, HyperParam
from .preprocess import preprocess
from .util import set_seed


def main():
    opt = HyperParam()
    set_seed()
    print(f'dataset: {DATASET_NAME}')
    print(opt)

    transformer, data, split = preprocess(opt)
    # todo: starts here


if __name__ == '__main__':
    main()
