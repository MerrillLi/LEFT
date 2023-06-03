import torch
from torch.nn import *
from modules.embed.meta_embeds import TreeEmbeddings
from modules.tc import MetaTC
import math
from einops import rearrange


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



class BatchNorm1dLast(BatchNorm1d):
    def forward(self, input):
        # 将输入的维度顺序调整为（batch_size, feature_dim, length）
        if input.dim() == 2:
            input = rearrange(input, 'b l -> b l 1')
        input = input.permute(0, 2, 1)
        # 调用父类的forward函数进行批归一化
        output = super(BatchNorm1dLast, self).forward(input)
        # 将输出的维度再次调整为（batch_size, length, feature_dim）
        output = output.permute(0, 2, 1)
        return output



class FactorizedTreeEmbeddings(TreeEmbeddings):


    def __init__(self, args, tree):
        super().__init__(args, tree)
        self.args = args
        self.tree = tree

        # Initialize the embeddings
        self.initialize()


    def initialize(self):

        # Subtree Node Embeddings
        self.initialize_subtree_embeddings()

        # Node Embedding Transformation
        self.initialize_transform()

        # Virtual Unified Tree to Subtree Index Mapping
        self.initialize_subtree_index()

        # Initialize Subtree Mask
        self.initialize_subtree_mask()


    def initialize_subtree_embeddings(self):
        self.tree_node_embeddings = ModuleDict()
        for ktype in self.args.ktype:
            # 获取当前类型的数量
            num_ktype = getattr(self.args, f'num_{ktype}s')

            # 计算子树需要的节点个数
            ktype_depth = math.ceil(math.log(num_ktype, self.args.narys)) + 1
            num_subtree_nodes = int((self.args.narys ** ktype_depth - 1) / (self.args.narys - 1))

            # 初始化子树节点的嵌入
            self.tree_node_embeddings[ktype] = Embedding(num_subtree_nodes, self.args.rank)


    def initialize_transform(self):
        self.transform = ModuleDict()
        rank = self.args.rank
        for ktype in self.args.ktype:
            self.transform[ktype] = Sequential(
                Linear(2 * rank, rank),
                LayerNorm(rank),
                ReLU(),
                Dropout(p=0.2),
                Linear(rank, rank)
            )


    def initialize_subtree_index(self):
        self.subtreeIdx = dict()
        for ktype in self.args.ktype:
            self.subtreeIdx[ktype] = torch.zeros((self.tree.num_nodes,), dtype=torch.int32, device=self.args.device)

        if len(self.args.ktype) == 1:
            ktype = self.args.ktype[0]
            self.subtreeIdx[ktype] = torch.arange(self.tree.num_nodes, dtype=torch.int32, device=self.args.device)

        elif len(self.args.ktype) == 2:
            ktype0 = self.args.ktype[0]
            ktype1 = self.args.ktype[1]
            num_ktype0 = getattr(self.args, f'num_{ktype0}s')
            num_ktype1 = getattr(self.args, f'num_{ktype1}s')

            # 先建立一个节点与子树的对应关系的查询表
            node2subtree = torch.zeros((self.tree.num_nodes,), dtype=torch.int32, device=self.args.device)
            ktype0_treesize = get_subtree_size(num_ktype0, self.args.narys)
            node2subtree[:ktype0_treesize] = torch.arange(ktype0_treesize, dtype=torch.int32, device=self.args.device)
            ktype0_leaf_start = get_subtree_leaf_start(num_ktype0, self.args.narys)

            for node in range(ktype0_leaf_start, ktype0_leaf_start + num_ktype0):
                set_subtree_ids(node2subtree, torch.tensor([node], dtype=torch.int32), self.args.narys)

            # 安排一下子树的索引
            self.subtreeIdx[ktype0] = torch.arange(self.tree.num_nodes, dtype=torch.int32, device=self.args.device)
            self.subtreeIdx[ktype1] = torch.arange(self.tree.num_nodes, dtype=torch.int32, device=self.args.device)

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


    def prepare_leaf_embeddings(self, meta_tc:MetaTC):

        # 获取当前的ktype
        ktype = self.args.ktype[0]
        # 获取当前类型的数量
        num_ktype = getattr(self.args, f'num_{ktype}s')
        # 获取当前类型的嵌入
        ktype_embed = meta_tc.get_embeddings(torch.arange(num_ktype, device=self.args.device), ktype)

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
            self.tree_node_embeddings[ktype].weight.data[i] = self.tree_node_embeddings[ktype].weight.data[narys * i + 1: narys * i + 1 + valid_leaf_num].mean(dim=0)

        if len(self.args.ktype) == 2:
            ktype1 = self.args.ktype[1]
            # 获取当前类型的数量
            num_ktype1 = getattr(self.args, f'num_{ktype1}s')
            # 获取当前类型的嵌入
            ktype1_embed = meta_tc.get_embeddings(torch.arange(num_ktype1, device=self.args.device), ktype1)

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
                self.tree_node_embeddings[ktype1].weight.data[i] = self.tree_node_embeddings[ktype1].weight.data[narys * i + 1: narys * i + 1 + valid_leaf_num].mean(dim=0)


    def forward(self, nodeIdx):
        # Get the node embedding
        node_embeds = []
        for ktype in self.args.ktype:
            ktype_index = self.subtreeIdx[ktype][nodeIdx]
            node_embed = self.tree_node_embeddings[ktype](ktype_index)
            node_embeds.append(node_embed)

        if torch.any(self.tree.leaf_mask[nodeIdx] == 1):
            return node_embeds

        # Transform Node Embedding
        transformed = []
        for ktype, node_embed in zip(self.args.ktype, node_embeds):
            module = self.transform[ktype]
            node_embed = module(torch.cat(node_embeds, dim=-1)) + node_embed
            # node_embed = module(node_embed)
            transformed.append(node_embed)
        return transformed







