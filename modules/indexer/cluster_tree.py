import math
import torch as t
import numpy as np
from modules.indexer.meta_treeindexer import TreeIndexer
from cluster import ScikitKMeans

class ClusterIndexTree(TreeIndexer):

    def __init__(self, args):
        super().__init__(args)


    def initialize(self):

        # 0. 初始化参数
        args = self.args

        # 1. 获取ktype的维度数量
        num_treenodes = 1
        for ktype in args.ktype:
            num_types_instance = getattr(args, f'num_{ktype}s')
            num_nodes_for_type = math.ceil(math.log(num_types_instance, args.narys))
            num_treenodes *= (args.narys ** num_nodes_for_type)

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
        leaf_startIdx = self.leaf_startIdx
        self.leaf_mask[leaf_startIdx: ] = 1
        self.node_mask[leaf_startIdx: ] = 1

        # 4. 标记树节点有效性
        for i in range(leaf_startIdx - 1, -1, -1):
            valid_flag = 0
            for j in range(args.narys):
                valid_flag |= self.node_mask[args.narys * i + (j + 1)]
            self.node_mask[i] = valid_flag

        # 5. 移动到目标设备
        self.leaf_mask = self.leaf_mask.to(self.args.device)
        self.node_mask = self.node_mask.to(self.args.device)


    def cluster(self, meta_tcom, select):

        # 1. 读取嵌入
        num_nodes = eval(f'self.args.num_{select}s')
        origin_embeds = meta_tcom.get_embeddings(t.arange(num_nodes, device=self.args.device), select=select)
        origin_index = t.arange(num_nodes, device=self.args.device)

        # 2. 嵌入采样
        full_tree_nums = 2 ** int(math.log(num_nodes, 2))
        sample_num = full_tree_nums - num_nodes
        sample_index = t.randint(0, num_nodes, (sample_num, ), device=self.args.device)

        sample_embeds = origin_embeds[sample_index]
        concat_type_embeds = t.cat([origin_embeds, sample_embeds], dim=0)
        concat_type_index = t.cat([origin_index, sample_index], dim=0)

        # 3. 聚类, 假设输入的数据编号为[0, num_nodes], 输出的是二叉聚类树的叶子节点的编号
        cluster = ScikitKMeans()
        cluster_index, _ = cluster.train(concat_type_embeds.cpu().numpy())
        cluster_index = np.array(cluster_index)

        # 4. 获取叶子节点的映射关系
        leaf_index_mapping = concat_type_index[cluster_index]

        # 5. 获取叶子节点的嵌入
        ordered_embeds = concat_type_embeds[cluster_index]

        return leaf_index_mapping, ordered_embeds


# 测试代码
if __name__ == '__main__':

    class Arguments:

        def __init__(self) -> None:

            self.num_users = 12
            self.num_items = 3
            self.num_times = 3
            self.qtype = ['user', ]
            self.ktype = ['item','time']
            self.narys = 2
            self.device = 'cpu'

    args = Arguments()

    index = ClusterIndexTree(args)



    print(index.node_mask)
    print(index.leaf_mapp[index.leaf_mask == 1])



    # 从叶子节点开始

