from typing import List, Type

from torch.nn import Module
from torch.optim import Optimizer


class SplitOptimizer:
    def __init__(self, optims: List[Optimizer]):
        self._optims = optims

    @classmethod
    def from_module(cls, module: Module, optim: Type[Optimizer], **defaults):
        optims = [optim(slat.parameters(), **defaults) for slat in module._modules.values()]
        return cls(optims)

    def zero_grad(self):
        for optim in self._optims:
            optim.zero_grad()

    def step(self):
        for optim in self._optims:
            optim.step()
