import torch
from torch.nn import *
from modules.embed.meta_embeds import TreeEmbeddings
from modules.indexer import ClusterIndexTree
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


class FACTreeEmbeddings(TreeEmbeddings):


    def __init__(self, args, tree):
        super().__init__(args, tree)
        self.args = args
        self.tree = tree #type:ClusterIndexTree

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
        """
        通过计算子树的规模，生成树的嵌入表
        """
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
        """
        生成子树嵌入的转化器网络
        """
        self.transform = ModuleDict()
        rank = self.args.rank
        for ktype in self.args.ktype:
            self.transform[ktype] = Sequential(
                Linear(2 * rank, rank),
                ReLU(),
                Dropout(p=0.2),
                Linear(rank, rank),
            ).to(self.args.device)


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
        self.subtree0_node_mask[ktype0_leafstart: ] = 1
        self.subtree0_leaf_mask[ktype0_leafstart: ] = 1

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
            self.subtree1_node_mask[ktype1_leafstart: ] = 1
            self.subtree1_leaf_mask[ktype1_leafstart: ] = 1

            for i in range(ktype1_leafstart - 1, -1, -1):
                valid_flag = 0
                for j in range(self.args.narys):
                    valid_flag |= self.subtree1_node_mask[self.args.narys * i + (j + 1)]
                self.subtree1_node_mask[i] = valid_flag


    def prepare_leaf_embeddings(self, meta_tcom:MetaTC):

        # 获取ids和unordered_embeds
        ids, ordered_embeds = [], []
        for ktype in self.args.ktype:
            id, order_embed = self.tree.cluster(meta_tcom, select=ktype)
            ids.append(id)
            ordered_embeds.append(order_embed)

        ###############
        # 设置ktype0  #
        ###############

        # 获取当前类型的数量
        ktype0 = self.args.ktype[0]
        num_ktype0 = getattr(self.args, f'num_{ktype0}s')
        ktype0_embed = ordered_embeds[0]

        # 写入到node_embeddings子树中
        leafIdx = self.subtree0_leaf_mask == 1
        self.tree_node_embeddings[ktype0].weight.data[leafIdx] = ktype0_embed

        leafstart = get_subtree_leaf_start(num_ktype0, self.args.narys)
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
            self.tree_node_embeddings[ktype0].weight.data[i] = self.tree_node_embeddings[ktype0].weight.data[narys * i + 1: narys * i + 1 + valid_leaf_num].mean(dim=0)


        if len(self.args.ktype) == 2:
            ###############
            # 设置ktype1  #
            ###############

            # 获取当前的ktype
            ktype1 = self.args.ktype[1]
            # 获取当前类型的数量
            num_ktype1 = getattr(self.args, f'num_{ktype1}s')
            # 获取当前类型的嵌入
            ktype1_embed = ordered_embeds[1]

            # 写入到node_embeddings子树中
            leafIdx = self.subtree1_leaf_mask == 1
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


        # 设置leaf_mapp
        if len(self.args.ktype) == 1:
            self.tree.leaf_mapp = ids[0]

        else:
            id_mapping0 = ids[0]
            id_mapping1 = ids[1]
            subtree_size = len(id_mapping1)
            for id in id_mapping0:
                startIdx = id * subtree_size
                endIdx = startIdx + subtree_size
                self.tree.leaf_mapp[startIdx: endIdx] = id_mapping1 + startIdx


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
            # node_embed = module(node_embed) + node_embed
            transformed.append(node_embed)
        return transformed


    def setup_optimizer(self, lr=1e-4, select='all'):
        if select == 'all':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif select == 'time':
            params = [
                {'params': self.tree_node_embeddings['time'].parameters()},
                {'params': self.transform['time'].parameters()},
            ]
            self.optimizer = torch.optim.Adam(params, lr=lr)
        elif select == 'item':
            params = [
                {'params': self.tree_node_embeddings['item'].parameters()},
                {'params': self.transform['item'].parameters()},
            ]
            self.optimizer = torch.optim.Adam(params, lr=lr)

        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer)





















