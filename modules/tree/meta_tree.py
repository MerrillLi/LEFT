from abc import abstractmethod, ABC
from torch.nn import Module
import torch as t

class MetaTree(Module, ABC):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.narys = self.args.narys

    @abstractmethod
    def forward(self, nodeIdx):
        ...

    @abstractmethod
    def leaf2ravel(self, leafIdx):
        ...

    @abstractmethod
    def initialize(self, meta_tcom):
        ...

    @abstractmethod
    def setup_optimizer(self):
        ...

    def get_children(self, nodeIdx):
        children = []
        for i in range(self.narys):
            children.append(self.narys * nodeIdx + (i + 1))
        return t.cat(children, dim=-1)
