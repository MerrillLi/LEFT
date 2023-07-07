from modules.tree.meta_tree import MetaTree
import torch as t
import math
from torch.nn import *
from modules.tree.embed_transform import EmbedTransform


class SequentialTree(MetaTree):

    def __init__(self, args):
        super().__init__(args)

        # 创建嵌入转换网络
        self.chunks = len(self.args.ktype)
        self.transform = ModuleDict()
        for i in range(self.chunks):
            self.transform[str(i)] = EmbedTransform(self.args.rank, self.args.rank)



    def forward(self, nodeIdx):
        embeds = self.tree_node_embeddings(nodeIdx)
        embeds_list = list(embeds.chunk(self.chunks, dim=-1))

        output_embeds_list = [embed.clone() for embed in embeds_list]
        # Transform Only Non-Leaf Embeddings (Since Leaf Embeddings are Fixed)
        for i in range(self.chunks):
            nonLeafIdx = self.leaf_mask[nodeIdx] == 0
            output_embeds_list[i][nonLeafIdx] = self.transform[str(i)](embeds_list[i][nonLeafIdx])

        return output_embeds_list


    def leaf2ravel(self, leafIdx):
        return self.leaf_mapp[leafIdx]


    def initialize(self, meta_tcom):

        # 1. 初始化树结构
        self.__setup_index_tree()

        # 2. 初始化树嵌入
        self.__setup_tree_embeddings(meta_tcom)

        # 3. 初始化优化器
        self.setup_optimizer()

        # 移动网络到目标设备
        self.to(self.args.device)


    def setup_optimizer(self):
        self.optimizer = t.optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler = t.optim.lr_scheduler.ConstantLR(self.optimizer)


    def __setup_index_tree(self):
        # 0. 初始化参数
        args = self.args

        # 1. 获取被检索维度的大小
        num_treenodes = 1
        for ktype in args.ktype:
            num_types_instance = getattr(args, f'num_{ktype}s')
            num_nodes_for_type = math.ceil(math.log(num_types_instance, args.narys))
            num_treenodes *= (args.narys ** num_nodes_for_type)

        # 2. 计算所需要的索引树的参数[树高，叶子节点起始位置，索引节点数量，叶子节点数量]
        self.depth = math.ceil(math.log(num_treenodes, args.narys)) + 1
        self.index_size = int( (self.narys ** (self.depth - 1) - 1) / (self.narys - 1) )
        self.leaf_size = int( self.narys ** (self.depth - 1) )
        self.num_nodes = self.index_size + self.leaf_size
        self.leaf_startIdx = self.index_size

        # 3. 树结构数组：有效叶子节点数组、有效树节点数组
        self.leaf_mask = t.zeros((self.num_nodes, ), dtype=t.int32)
        self.node_mask = t.zeros((self.num_nodes, ), dtype=t.int32)
        self.leaf_mapp = t.zeros((self.num_nodes, ), dtype=t.int32)

        # 4. 分配叶子节点
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
            num_valid_leafs = num_types_instance_0 * num_types_instance_1
            self.leaf_mask[leaf_startIdx: leaf_startIdx + num_valid_leafs] = 1
            self.node_mask[leaf_startIdx: leaf_startIdx + num_valid_leafs] = 1
            self.leaf_mapp[leaf_startIdx: leaf_startIdx + num_valid_leafs] = t.arange(num_valid_leafs)

        else:
            raise NotImplementedError(f'Not support {len(args.ktypes)} x ktypes')

        # 5. 标记树节点有效性
        for i in range(leaf_startIdx - 1, -1, -1):
            valid_flag = 0
            for j in range(args.narys):
                valid_flag |= self.node_mask[args.narys * i + (j + 1)]
            self.node_mask[i] = valid_flag

        # 6. 移动到目标设备
        self.leaf_mask = self.leaf_mask.to(self.args.device)
        self.node_mask = self.node_mask.to(self.args.device)
        self.leaf_mapp = self.leaf_mapp.to(self.args.device)


    def __setup_tree_embeddings(self, meta_tcom):

        self.meta_tcom = meta_tcom
        self.tree_node_embeddings = Embedding(self.num_nodes, self.args.rank * self.chunks).to(self.args.device)

        if len(self.args.ktype) == 1:
            ktype = self.args.ktype[0]
            num_nodes = eval(f'self.args.num_{ktype}s')
            leaf_embeds = meta_tcom.get_embeddings(t.arange(num_nodes, device=self.device), select=ktype)

            # Set Leaf Embeddings
            valid_leaf = self.leaf_mask == 1
            self.tree_node_embeddings.weight.data[valid_leaf] = leaf_embeds

        elif len(self.args.ktype) == 2:

            type1 = self.args.ktype[0]
            type2 = self.args.ktype[1]

            num_type1 = eval(f'self.args.num_{type1}s')
            num_type2 = eval(f'self.args.num_{type2}s')

            type1_embeds = meta_tcom.get_embeddings(t.arange(num_type1, device=self.device), select=type1)
            type2_embeds = meta_tcom.get_embeddings(t.arange(num_type2, device=self.device), select=type2)

            # Shape : (num_type1, rank), (num_type2, rank)
            # type1_embeds_repeated: (num_type1 * %num_type2%, rank)
            # type2_embeds_repeated: (%num_type1% * num_type2, rank)
            type1_embeds_repeated = type1_embeds.repeat_interleave(num_type2, 0)
            type2_embeds_repeated = type2_embeds.repeat(num_type1, 1)

            # Shape: (num_type1 * num_type2, rank * 2)
            repeated_embeds = t.cat([type1_embeds_repeated, type2_embeds_repeated], dim=-1)

            # Set Leaf Embeddings
            valid_leaf = self.leaf_mask == 1
            self.tree_node_embeddings.weight.data[valid_leaf] = repeated_embeds

        else:
            raise NotImplementedError('only support search no more than two dimension!')

        # 构建父节点的embedding
        for i in range(self.leaf_startIdx - 1, -1, -1):

            # 跳过无效的索引节点
            if self.node_mask[i] == 0:
                continue

            # 计算有效的孩子节点数
            valid_leaf_num = 0
            for j in range(self.narys):
                if self.node_mask[self.narys * i + (j + 1)] == 1:
                    valid_leaf_num += 1

            # 父节点的Embedding为孩子节点Embedding的平均
            self.tree_node_embeddings.weight.data[i] = self.tree_node_embeddings.weight.data[self.narys * i + 1: self.narys * i + 1 + valid_leaf_num].mean(dim=0)
