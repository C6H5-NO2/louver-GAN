from dataclasses import dataclass


@dataclass
class CorrSolverConfig:
    checkpoint_path: str
    device: str

    n_epoch: int = 800
    save_step: int = 1000

    batch_size: int = 256
    lr: float = 1e-4
