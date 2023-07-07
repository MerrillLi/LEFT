import torch
from modules.tree.meta_tree import MetaTree
import torch as t
import math
from torch.nn import *
from modules.tree.embed_transform import EmbedTransform


def get_subtree_size(num_repos, narys):
    ktype_depth = math.ceil(math.log(num_repos, narys)) + 1
    num_subtree_nodes = int((narys ** ktype_depth - 1) / (narys - 1))
    return num_subtree_nodes


def get_subtree_leaf_start(num_repos, narys):
    ktype_depth = math.ceil(math.log(num_repos, narys)) + 1
    num_subtree_nodes = int((narys ** ktype_depth - 1) / (narys - 1))
    num_subtree_leaves = narys ** (ktype_depth - 1)
    return num_subtree_nodes - num_subtree_leaves


def get_children(node, k):
    children = []
    for i in range(k):
        children.append(node * k + i + 1)
    return torch.cat(children, dim=0)


def set_subtree_ids(tree, root_id, k):

    node = root_id
    while True:
        tree[node] = root_id
        children = get_children(node, k)
        node = children
        if node[0] >= len(tree):
            break


class StructTree(MetaTree):

    def __init__(self, args):
        super().__init__(args)

        # 1. 子树嵌入
        self.tree_node_embeddings = ModuleDict()
        for ktype in self.args.ktype:
            # 获取当前类型的数量
            num_ktype = getattr(self.args, f'num_{ktype}s')

            # 计算子树需要的节点个数
            ktype_depth = math.ceil(math.log(num_ktype, self.args.narys)) + 1
            num_subtree_nodes = int((self.args.narys ** ktype_depth - 1) / (self.args.narys - 1))

            # 初始化子树节点的嵌入
            self.tree_node_embeddings[ktype] = Embedding(num_subtree_nodes, self.args.rank)

        # 2. 嵌入转换
        self.transform = ModuleDict()
        rank = self.args.rank
        for ktype in self.args.ktype:
            self.transform[ktype] = EmbedTransform(2 * rank, rank).to(self.args.device)


    def forward(self, nodeIdx):
        # Get the node embedding
        node_embeds = []
        for ktype in self.args.ktype:
            ktype_index = self.subtreeIdx[ktype][nodeIdx]
            node_embed = self.tree_node_embeddings[ktype](ktype_index)
            node_embeds.append(node_embed)

        if torch.any(self.leaf_mask[nodeIdx] == 1):
            return node_embeds

        # Transform Node Embedding
        transformed = []
        for ktype, node_embed in zip(self.args.ktype, node_embeds):
            module = self.transform[ktype]
            node_embed = module(torch.cat(node_embeds, dim=-1)) + node_embed
            transformed.append(node_embed)
        return transformed


    def leaf2ravel(self, leafIdx):
        return self.leaf_mapp[leafIdx]


    def initialize(self, meta_tcom):

        # 1. 初始化树结构
        self.__setup_index_tree()
        self.initialize_subtree_mask()
        self.initialize_subtree_index()

        # 2. 初始化树节点嵌入
        self.__setup_tree_embeddings(meta_tcom)

        # 3. 初始化优化器
        self.setup_optimizer()


    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer)


    def __setup_index_tree(self):
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

        # 5. 移动到目标设备
        self.leaf_mask = self.leaf_mask.to(self.args.device)
        self.node_mask = self.node_mask.to(self.args.device)
        self.leaf_mapp = self.leaf_mapp.to(self.args.device)


    def __setup_tree_embeddings(self, meta_tcom):
        # 获取当前的ktype
        ktype = self.args.ktype[0]
        # 获取当前类型的数量
        num_ktype = getattr(self.args, f'num_{ktype}s')
        # 获取当前类型的嵌入
        ktype_embed = meta_tcom.get_embeddings(torch.arange(num_ktype, device=self.args.device), ktype)

        # 写入到node_embeddings子树中
        leafIdx = self.subtree0_leaf_mask == 1
        self.tree_node_embeddings[ktype].weight.data[leafIdx].requires_grad = False
        self.tree_node_embeddings[ktype].weight.data[leafIdx] = ktype_embed

        leafstart = get_subtree_leaf_start(num_ktype, self.args.narys)
        narys = self.args.narys
        for i in range(leafstart - 1, -1, -1):

            # 跳过无效的索引节点
            if self.subtree0_node_mask[i] == 0:
                continue

            # 计算有效的孩子节点数
            valid_leaf_num = 0
            for j in range(narys):
                if self.subtree0_node_mask[narys * i + (j + 1)] == 1:
                    valid_leaf_num += 1

            # 父节点的Embedding为孩子节点Embedding的平均
            self.tree_node_embeddings[ktype].weight.data[i] = self.tree_node_embeddings[ktype].weight.data[
                                                              narys * i + 1: narys * i + 1 + valid_leaf_num].mean(dim=0)

        if len(self.args.ktype) == 2:
            ktype1 = self.args.ktype[1]
            # 获取当前类型的数量
            num_ktype1 = getattr(self.args, f'num_{ktype1}s')
            # 获取当前类型的嵌入
            ktype1_embed = meta_tcom.get_embeddings(torch.arange(num_ktype1, device=self.args.device), ktype1)

            # 写入到tree_node_embeddings子树中
            leafIdx = self.subtree1_leaf_mask == 1
            self.tree_node_embeddings[ktype1].weight.data[leafIdx].requires_grad = False
            self.tree_node_embeddings[ktype1].weight.data[leafIdx] = ktype1_embed

            leafstart = get_subtree_leaf_start(num_ktype1, self.args.narys)
            narys = self.args.narys
            for i in range(leafstart - 1, -1, -1):

                # 跳过无效的索引节点
                if self.subtree1_node_mask[i] == 0:
                    continue

                # 计算有效的孩子节点数
                valid_leaf_num = 0
                for j in range(narys):
                    if self.subtree1_node_mask[narys * i + (j + 1)] == 1:
                        valid_leaf_num += 1

                # 父节点的Embedding为孩子节点Embedding的平均
                self.tree_node_embeddings[ktype1].weight.data[i] = self.tree_node_embeddings[ktype1].weight.data[
                                                                   narys * i + 1: narys * i + 1 + valid_leaf_num].mean(
                    dim=0)


    def initialize_subtree_mask(self):
        ktype0 = self.args.ktype[0]
        num_ktype0 = getattr(self.args, f'num_{ktype0}s')
        ktype0_treesize = get_subtree_size(num_ktype0, self.args.narys)
        ktype0_leafstart = get_subtree_leaf_start(num_ktype0, self.args.narys)

        self.subtree0_node_mask = torch.zeros((ktype0_treesize,), dtype=torch.int32)
        self.subtree0_leaf_mask = torch.zeros((ktype0_treesize,), dtype=torch.int32)
        self.subtree0_node_mask[ktype0_leafstart: ktype0_leafstart + num_ktype0] = 1
        self.subtree0_leaf_mask[ktype0_leafstart: ktype0_leafstart + num_ktype0] = 1

        for i in range(ktype0_leafstart - 1, -1, -1):
            valid_flag = 0
            for j in range(self.args.narys):
                valid_flag |= self.subtree0_node_mask[self.args.narys * i + (j + 1)]
            self.subtree0_node_mask[i] = valid_flag

        if len(self.args.ktype) == 2:
            ktype1 = self.args.ktype[1]
            num_ktype1 = getattr(self.args, f'num_{ktype1}s')
            ktype1_treesize = get_subtree_size(num_ktype1, self.args.narys)
            ktype1_leafstart = get_subtree_leaf_start(num_ktype1, self.args.narys)

            self.subtree1_node_mask = torch.zeros((ktype1_treesize,), dtype=torch.int32)
            self.subtree1_leaf_mask = torch.zeros((ktype1_treesize,), dtype=torch.int32)
            self.subtree1_node_mask[ktype1_leafstart: ktype1_leafstart + num_ktype1] = 1
            self.subtree1_leaf_mask[ktype1_leafstart: ktype1_leafstart + num_ktype1] = 1

            for i in range(ktype1_leafstart - 1, -1, -1):
                valid_flag = 0
                for j in range(self.args.narys):
                    valid_flag |= self.subtree1_node_mask[self.args.narys * i + (j + 1)]
                self.subtree1_node_mask[i] = valid_flag


    def initialize_subtree_index(self):
        self.subtreeIdx = dict()
        for ktype in self.args.ktype:
            self.subtreeIdx[ktype] = torch.zeros((self.num_nodes,), dtype=torch.int32, device=self.args.device)

        if len(self.args.ktype) == 1:
            ktype = self.args.ktype[0]
            self.subtreeIdx[ktype] = torch.arange(self.num_nodes, dtype=torch.int32, device=self.args.device)

        elif len(self.args.ktype) == 2:
            ktype0 = self.args.ktype[0]
            ktype1 = self.args.ktype[1]
            num_ktype0 = getattr(self.args, f'num_{ktype0}s')
            num_ktype1 = getattr(self.args, f'num_{ktype1}s')

            # 先建立一个节点与子树的对应关系的查询表
            node2subtree = torch.zeros((self.num_nodes,), dtype=torch.int32, device=self.args.device)
            ktype0_treesize = get_subtree_size(num_ktype0, self.args.narys)
            node2subtree[:ktype0_treesize] = torch.arange(ktype0_treesize, dtype=torch.int32, device=self.args.device)
            ktype0_leaf_start = get_subtree_leaf_start(num_ktype0, self.args.narys)

            for node in range(ktype0_leaf_start, ktype0_leaf_start + num_ktype0):
                set_subtree_ids(node2subtree, torch.tensor([node], dtype=torch.int32), self.args.narys)

            # 安排一下子树的索引
            self.subtreeIdx[ktype0] = torch.arange(self.num_nodes, dtype=torch.int32, device=self.args.device)
            self.subtreeIdx[ktype1] = torch.arange(self.num_nodes, dtype=torch.int32, device=self.args.device)

            # 设置上半部分ktype0部分索引
            ktype0_treesize = get_subtree_size(num_ktype0, self.args.narys)
            self.subtreeIdx[ktype0][:ktype0_treesize] = torch.arange(ktype0_treesize, dtype=torch.int32,
                                                                     device=self.args.device)
            self.subtreeIdx[ktype1][:ktype0_treesize] = 0

            # 设置下半部分ktype1部分索引
            for node in range(ktype0_leaf_start, ktype0_leaf_start + num_ktype0):
                self.subtreeIdx[ktype0][node2subtree == node] = node

                ktype1_treesize = get_subtree_size(num_ktype1, self.args.narys)
                self.subtreeIdx[ktype1][node2subtree == node] = torch.arange(ktype1_treesize, dtype=torch.int32,
                                                                             device=self.args.device)
