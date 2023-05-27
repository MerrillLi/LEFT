import torch as t

from abc import ABC, abstractmethod


class TreeIndexer(ABC):
    
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.narys = args.narys

        self.initialize()

    
    @abstractmethod
    def initialize(self):
        ...

    def leaf2ravel(self, leafIdx):
        return self.leaf_mapp[leafIdx]
    
    def get_children(self, nodeIdx):
        children = []
        for i in range(self.narys):
            children.append(self.narys * nodeIdx + (i + 1))
        return t.cat(children, dim=-1)
