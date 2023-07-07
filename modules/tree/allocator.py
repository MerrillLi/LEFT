import torch
import numpy as np

def get_children(node, narys, ith):
    """
    i-th child of node, 0-based
    :param node:
    :param narys:
    :param ith:
    :return:
    """
    return narys * node + ith + 1


def get_leaf_size(db_size, narys):
    leaf_size = 1
    while leaf_size < db_size:
        leaf_size *= narys
    return leaf_size


def get_tree_size(db_size, narys):
    leaf_size = get_leaf_size(db_size, narys)
    tree_depth = int(np.log(leaf_size) / np.log(narys))
    return (narys ** (tree_depth + 1) - 1) // (narys - 1)


def get_leaf_startIdx(db_size, narys):
    return get_tree_size(db_size, narys) - get_leaf_size(db_size, narys)


class BalanceAllocator:

    def __init__(self, narys, db_size):
        self.narys = narys
        self.db_size = db_size
        self.leaf_size = get_leaf_size(db_size, self.narys)
        self.leaf_startIdx = get_leaf_startIdx(db_size, self.narys)
        self.tree_size = get_tree_size(db_size, self.narys)
        self.allocation = {}

    def _recurrsive_helper(self, node, index):

        # 处理边界条件
        if len(index) <= self.narys:
            for i, idx in enumerate(index):
                estimated_node = get_children(node, self.narys, i)
                while estimated_node < self.leaf_startIdx:
                    estimated_node = get_children(estimated_node, self.narys, 0)
                self.allocation[estimated_node] = idx
            return

        # Numpy将index分成narys份
        slices = np.array_split(index, self.narys)

        # 分配到节点
        self.allocation[node] = index

        # 递归
        for i in range(self.narys):
            next_node = get_children(node, self.narys, i)
            self._recurrsive_helper(next_node, slices[i])


    def allocate(self):
        alocation = np.zeros(self.leaf_size, dtype=int) - 1
        index = np.arange(self.db_size)
        self._recurrsive_helper(0, index)
        for i in range(self.leaf_size):
            nodeId = i + self.leaf_startIdx
            if nodeId not in self.allocation:
                pass
            else:
                alocation[i] = self.allocation[nodeId]
        return torch.from_numpy(alocation)
