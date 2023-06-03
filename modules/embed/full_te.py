import torch as t
from torch.nn import *
from modules.embed.meta_embeds import TreeEmbeddings


class FullTreeEmbeddings(TreeEmbeddings):


    def __init__(self, args, tree):
        super().__init__(args, tree)

        # Record Params
        self.args = args
        self.rank = self.args.rank
        self.qtype = self.args.qtype
        self.chunks = len(self.args.ktype)
        self.tree = tree
        self.num_nodes = self.tree.num_nodes

        # Initialize Tree
        self.initialize()


    def initialize(self):
        # Node Embedding Modules
        self.tree_node_embeddings = Embedding(self.num_nodes, self.rank * self.chunks)
        self.transform = ModuleDict()
        self.q_transform = ModuleDict()
        for i in range(self.chunks):
            self.transform[str(i)] = Sequential(Linear(self.rank, self.rank),
                                                BatchNorm1d(self.rank),
                                                ReLU(),
                                                Dropout(p=0.2),
                                                Linear(self.rank, self.rank))

        for i in range(self.chunks):
            self.q_transform[str(i)] = Sequential(Linear((len(self.qtype) + 1) * self.rank, 10 * self.rank), ReLU(),
                                                  Linear(10 * self.rank, self.rank))

        # Move to device
        self.to(self.args.device)

        # Allow Inplace Operation
        leafIdx = self.tree.leaf_mask == 1
        self.tree_node_embeddings.weight.data[leafIdx].requires_grad = False


    def forward(self, nodeIdx, query=None):
        embeds = self.tree_node_embeddings(nodeIdx)
        embeds_list = list(embeds.chunk(self.chunks, dim=-1))

        output_embeds_list = [ embed.clone() for embed in embeds_list ]
        # Transform Only Non-Leaf Embeddings (Since Leaf Embeddings are Fixed)
        for i in range(self.chunks):
            if query is None:
                nonLeafIdx = self.tree.leaf_mask[nodeIdx] == 0
                output_embeds_list[i][nonLeafIdx] = self.transform[str(i)](embeds_list[i][nonLeafIdx])

            else:
                nonLeafIdx = self.tree.leaf_mask[nodeIdx] == 0
                query_inputs = [ q.reshape(output_embeds_list[0].shape) for q in query.values() ]
                transform_inputs = t.cat(query_inputs, dim=-1)
                transform_inputs = t.cat([embeds_list[i][nonLeafIdx], transform_inputs[nonLeafIdx]], dim=-1)
                output_embeds_list[i][nonLeafIdx] = self.q_transform[str(i)](transform_inputs)

        return output_embeds_list
