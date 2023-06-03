import torch

from modules.embed.meta_embeds import TreeEmbeddings
from torch.nn import *


def murmurhash3_32_int(key, seed=0):
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    r1 = 15
    r2 = 13
    m = 5
    n = 0xe6546b64

    k1 = key & 0xffffffff

    k1 *= c1
    k1 = (k1 << r1) | (k1 >> (32 - r1))
    k1 *= c2

    h1 = seed

    h1 ^= k1
    h1 = (h1 << r2) | (h1 >> (32 - r2))
    h1 = h1 * m + n

    h1 ^= 4

    h1 ^= h1 >> 16
    h1 *= 0x85ebca6b
    h1 ^= h1 >> 13
    h1 *= 0xc2b2ae35
    h1 ^= h1 >> 16

    return h1 & 0xffffffff


class HashEmbeddings(TreeEmbeddings):


    def initialize(self):

        self.initialize_transform()

    def __init__(self, args, tree):
        super().__init__(args, tree)

        self.chunks = len(self.args.ktype)

        self.rank = self.args.rank
        self.num_ht = 5
        self.ht_size = 100

        self.hash_embeds = Embedding(self.num_ht * self.ht_size, self.rank * self.chunks)
        self.hash_trans = Linear(self.rank * self.chunks, self.rank * self.chunks)

        self.tree_node_embeddings = Embedding(self.tree.num_nodes, self.rank * self.chunks)

        self.initialize_transform()


    def forward(self, nodeIdx):

        # 读取节点的嵌入
        node_embeds = self.tree_node_embeddings(nodeIdx)

        output_embeds = node_embeds.clone()

        # 选出叶子节点和非叶子结点
        leafIdx = self.tree.leaf_mask[nodeIdx] == 1
        nonLeafIdx = self.tree.leaf_mask[nodeIdx] == 0

        # 写入叶子节点的值
        output_embeds[leafIdx] = self.tree_node_embeddings(nodeIdx[leafIdx])

        # 获取非叶子节点的哈希映射
        hashIdx = self.node_to_hash(nodeIdx[nonLeafIdx])
        hash_embeds = self.hash_embeds(hashIdx).mean(dim=-2)

        # 非叶子节点转换
        output_embeds[nonLeafIdx, :] = self.hash_trans(hash_embeds)
        return output_embeds

    def node_to_hash(self, nodeIdx):
        # NodeIdx = [batch, num_nodes] -> [batch, num_nodes, num_ht]
        hash_indices = []
        for i in range(self.num_ht):
            hashIdx = murmurhash3_32_int(nodeIdx, seed=i) % self.ht_size + i * self.ht_size
            hash_indices.append(hashIdx)
        return torch.stack(hash_indices, dim=-1)

    def initialize_transform(self):
        self.transform = ModuleDict()
        rank = self.args.rank
        for ktype in self.args.ktype:
            self.transform[ktype] = Sequential(
                Linear(rank, rank),
                BatchNorm1d(rank),
                ReLU(),
                Dropout(p=0.2),
                Linear(rank, rank)
            )
