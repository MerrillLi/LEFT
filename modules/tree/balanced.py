import math
import torch
from torch.nn import *
from modules.tc import MetaTC
from modules.tree.allocator import *
from modules.tree.meta_tree import MetaTree
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


class BalancedTree(MetaTree):

    def __init__(self, args):
        super().__init__(args)


    def forward(self, nodeIdx):

        # Get the node embedding
        node_embeds = []
        for ktype in self.args.ktype:
            ktype_index = self.subtreeIdx[ktype][nodeIdx]
            node_embed = self.subtree_node_embeds[ktype](ktype_index)
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


    def _initialize_subtree_index(self):
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
            node2subtree[:ktype0_treesize] = torch.arange(ktype0_treesize, dtype=torch.int32,
                                                          device=self.args.device)
            ktype0_leaf_start = get_subtree_leaf_start(num_ktype0, self.args.narys)

            for node in range(ktype0_leaf_start, ktype0_treesize):
                set_subtree_ids(node2subtree, torch.tensor([node], dtype=torch.int32), self.args.narys)

            # 安排一下子树的索引
            self.subtreeIdx[ktype0] = torch.zeros((self.num_nodes, ), dtype=torch.int32, device=self.args.device)
            self.subtreeIdx[ktype1] = torch.zeros((self.num_nodes, ), dtype=torch.int32, device=self.args.device)

            # 设置上半部分ktype0部分索引
            ktype0_treesize = get_subtree_size(num_ktype0, self.args.narys)
            self.subtreeIdx[ktype0][:ktype0_treesize] = torch.arange(ktype0_treesize, dtype=torch.int32, device=self.args.device)
            self.subtreeIdx[ktype1][:ktype0_treesize] = 0

            # 设置下半部分ktype1部分索引
            for node in range(ktype0_leaf_start, ktype0_treesize):
                self.subtreeIdx[ktype0][node2subtree == node] = node

                ktype1_treesize = get_subtree_size(num_ktype1, self.args.narys)
                self.subtreeIdx[ktype1][node2subtree == node] = torch.arange(ktype1_treesize, dtype=torch.int32, device=self.args.device)

            self.node2subtree = node2subtree

    def initialize(self, meta_tcom: MetaTC):

        self.subtree_node_mask = {}
        self.subtree_leaf_mask = {}
        self.allocation = {}

        self.transform = ModuleDict()
        self.subtree_node_embeds = ModuleDict()

        ###############################
        # 1. 根据张量填充嵌入构建子树结构#
        ###############################
        for ktype in self.args.ktype:

            num_ktype = getattr(self.args, f'num_{ktype}s')
            index = torch.arange(num_ktype).to(self.args.device)
            ktype_embeds = meta_tcom.get_embeddings(index, ktype)
            tree_allocator = BalanceAllocator(self.args.narys, num_ktype)
            leaf_allocation = tree_allocator.allocate()
            self.allocation[ktype] = leaf_allocation

            # 1. 初始化子树结构
            subtree_size = get_subtree_size(num_ktype, self.args.narys)
            leaf_start = get_subtree_leaf_start(num_ktype, self.args.narys)
            self.subtree_node_mask[ktype] = torch.zeros((subtree_size, ), dtype=torch.int32).to(self.args.device)
            self.subtree_leaf_mask[ktype] = torch.zeros((subtree_size, ), dtype=torch.int32).to(self.args.device)

            # 2. 初始化叶子节点，将未分配节点设置为0
            self.subtree_leaf_mask[ktype][leaf_start:] = leaf_allocation != -1
            self.subtree_node_mask[ktype][leaf_start:] = leaf_allocation != -1

            for i in range(leaf_start - 1, -1, -1):
                valid_flag = 0
                for j in range(self.args.narys):
                    valid_flag |= self.subtree_node_mask[ktype][self.args.narys * i + (j + 1)]
                self.subtree_node_mask[ktype][i] = valid_flag

            # 3. 设置子树嵌入和转换网络
            self.transform[ktype] = EmbedTransform(2 * self.args.rank, self.args.rank).to(self.args.device)
            self.subtree_node_embeds[ktype] = Embedding(subtree_size, self.args.rank).to(self.args.device)

            # 按照leaf_allocation重新排列
            valid_index = leaf_allocation[leaf_allocation != -1]
            ordered_embeds = ktype_embeds[valid_index]
            init.zeros_(self.subtree_node_embeds[ktype].weight.data)
            self.subtree_node_embeds[ktype].weight.data[leaf_start:][leaf_allocation != -1] = ordered_embeds

            for i in range(leaf_start - 1, -1, -1):

                # 跳过无效的索引节点
                if self.subtree_node_mask[ktype][i] == 0:
                    continue

                # 计算有效的孩子节点数
                valid_leaf_idx = []
                for j in range(self.args.narys):
                    if self.subtree_node_mask[ktype][self.args.narys * i + (j + 1)] == 1:
                        valid_leaf_idx.append(self.args.narys * i + (j + 1))
                        # valid_leaf_num += 1

                # 父节点的Embedding为孩子节点Embedding的平均
                valid_leaf_idx = torch.tensor(valid_leaf_idx, dtype=torch.int64, device=self.args.device)
                self.subtree_node_embeds[ktype].weight.data[i] = self.subtree_node_embeds[ktype].weight.data[valid_leaf_idx].mean(dim=0)

        #####################
        # 2. 计算全检索树参数 #
        #####################
        num_treenodes = 1
        for ktype in self.args.ktype:
            num_types_instance = getattr(self.args, f'num_{ktype}s')
            num_nodes_for_type = math.ceil(math.log(num_types_instance, self.args.narys))
            num_treenodes *= (self.args.narys ** num_nodes_for_type)

        self.depth = int(math.log(num_treenodes, self.args.narys)) + 1
        self.index_size = int( (self.narys ** (self.depth - 1) - 1) / (self.narys - 1) )
        self.leaf_size = int( self.narys ** (self.depth - 1) )
        self.num_nodes = self.index_size + self.leaf_size
        self.leaf_startIdx = self.index_size

        #######################
        # 3. 初始化全检索树数组 #
        #######################
        # 初始化树结构数组：有效叶子节点数组、有效树节点数组
        self.leaf_mask = torch.zeros((self.num_nodes, ), dtype=torch.int32)
        self.node_mask = torch.zeros((self.num_nodes, ), dtype=torch.int32)
        self.leaf_mapp = torch.zeros((self.num_nodes, ), dtype=torch.int32) - 1

        # 建立TreeID -> Item Subtree ID, Time Subtree ID
        ###############################
        # 4. 设置全检索树到子树嵌入的映射 #
        ###############################
        self.tree2subtree = dict()
        for ktype in self.args.ktype:
            self.tree2subtree[ktype] = torch.zeros((self.num_nodes,), dtype=torch.int32, device=self.args.device)

        self._initialize_subtree_index()

        ###################################
        # 5. 设置全检索树的mask和叶子节点映射 #
        ###################################
        if len(self.args.ktype) == 1:
            # 如果只有一个被检索类型，则子树就是完整的检索树
            ktype = self.args.ktype[0]
            self.subtreeIdx[ktype] = torch.arange(self.num_nodes, dtype=torch.int32, device=self.args.device)
            self.leaf_mask[:] = self.subtree_leaf_mask[ktype]
            self.node_mask[:] = self.subtree_node_mask[ktype]
            self.leaf_mapp[self.leaf_startIdx:] = self.allocation[ktype]

        elif len(self.args.ktype) == 2:
            ktype0 = self.args.ktype[0]
            ktype1 = self.args.ktype[1]
            num_ktype0 = getattr(self.args, f'num_{ktype0}s')
            num_ktype1 = getattr(self.args, f'num_{ktype1}s')

            type0_allocation = self.allocation[ktype0]
            type1_allocation = self.allocation[ktype1]

            type1_treesize = get_subtree_size(num_ktype1, self.args.narys)
            type1_leafstart = get_subtree_leaf_start(num_ktype1, self.args.narys)
            type0_leafstart = get_subtree_leaf_start(num_ktype0, self.args.narys)
            type1_leafsize = type1_treesize - type1_leafstart

            for i in range(len(type0_allocation)):

                type0_id = type0_allocation[i]

                seg_id = type0_leafstart + i
                if type0_id == -1:
                    self.node_mask[self.node2subtree == seg_id] = 0
                    self.leaf_mask[self.node2subtree == seg_id] = 0
                    continue

                startIdx = self.leaf_startIdx + i * type1_leafsize
                endIdx = startIdx + type1_leafsize

                leaf_ravel_offset = type0_id * num_ktype1
                leaf2ravel = leaf_ravel_offset + type1_allocation[type1_allocation != -1]

                self.leaf_mask[self.node2subtree == seg_id] = self.subtree_leaf_mask[ktype1]
                self.node_mask[self.node2subtree == seg_id] = self.subtree_node_mask[ktype1]
                self.leaf_mapp[startIdx: endIdx][type1_allocation != -1] = leaf2ravel.type(torch.int32)

        for i in range(self.leaf_startIdx - 1, -1, -1):
            valid_flag = 0
            for j in range(self.args.narys):
                valid_flag |= self.node_mask[self.args.narys * i + (j + 1)]
            self.node_mask[i] = valid_flag

        ################
        # 6. 设置优化器 #
        ################
        self.setup_optimizer()

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer)
