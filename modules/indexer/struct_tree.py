import math
import torch as t
import numpy as np
from modules.indexer.meta_treeindexer import TreeIndexer

class StructIndexTree(TreeIndexer):

    def __init__(self, args):
        super().__init__(args)

    def initialize(self):

        # 0. 初始化参数
        args = self.args

        # 1. 获取ktype的维度数量
        num_treenodes = 1
        subtree_size = 0
        for ktype in args.ktype:
            num_types_instance = getattr(args, f'num_{ktype}s')
            num_nodes_for_type = math.ceil(math.log(num_types_instance, args.narys))

            num_treenodes *= (args.narys ** num_nodes_for_type)
            subtree_size = (args.narys ** num_nodes_for_type)

        self.depth = int(math.log(num_treenodes, args.narys)) + 1

        self.index_size = int( (self.narys ** (self.depth - 1) - 1) / (self.narys - 1) )
        self.leaf_size = int( self.narys ** (self.depth - 1) )
        self.num_nodes = self.index_size + self.leaf_size
        self.leaf_startIdx = self.index_size

        # 2. 树结构数组：有效叶子节点数组、有效树节点数组
        self.leaf_mask = t.zeros((self.num_nodes, ), dtype=t.int32)
        self.node_mask = t.zeros((self.num_nodes, ), dtype=t.int32)
        self.leaf_mapp = t.zeros((self.num_nodes, ), dtype=t.int32)

        # 3. 分配叶子节点
        # 如果只有一个被查询维度，则不需要重构叶子节点
        if len(args.ktype) == 1:
            leaf_startIdx = self.leaf_startIdx
            num_types_instance = getattr(args, f'num_{args.ktype[0]}s')
            self.leaf_mask[leaf_startIdx: leaf_startIdx + num_types_instance] = 1
            self.node_mask[leaf_startIdx: leaf_startIdx + num_types_instance] = 1
            self.leaf_mapp[leaf_startIdx: leaf_startIdx + num_types_instance] = t.arange(num_types_instance)

        # 如果有两个被查询维度，则需要重构叶子节点，按照最后一个维度去安排叶子节点
        elif len(args.ktype) == 2:
            leaf_startIdx = self.leaf_startIdx
            num_types_instance_0 = getattr(args, f'num_{args.ktype[0]}s')
            num_types_instance_1 = getattr(args, f'num_{args.ktype[1]}s')
            for i in range(num_types_instance_0):
                subtree_offset = i * subtree_size
                self.leaf_mask[leaf_startIdx + subtree_offset: leaf_startIdx + subtree_offset + num_types_instance_1] = 1
                self.node_mask[leaf_startIdx + subtree_offset: leaf_startIdx + subtree_offset + num_types_instance_1] = 1
                self.leaf_mapp[leaf_startIdx + subtree_offset: leaf_startIdx + subtree_offset + num_types_instance_1] = t.arange(i * num_types_instance_1, (i+1) * num_types_instance_1)

        else:
            raise NotImplementedError(f'Not support {len(args.ktype)} x ktypes')

        # 4. 标记树节点有效性
        for i in range(leaf_startIdx - 1, -1, -1):
            valid_flag = 0
            for j in range(args.narys):
                valid_flag |= self.node_mask[args.narys * i + (j + 1)]
            self.node_mask[i] = valid_flag

        return


# 测试代码
if __name__ == '__main__':

    class Arguments:

        def __init__(self) -> None:

            self.num_users = 16
            self.num_items = 3
            self.num_times = 3
            self.qtypes = ['user']
            self.ktypes = ['item', 'time']
            self.narys = 3

    args = Arguments()

    index = StructIndexTree(args)

    print(index.node_mask)
    print(index.leaf_mapp[index.leaf_mask == 1])