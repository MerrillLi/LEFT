import math
import torch as t
from torch.nn import *
from modules.tc import get_model, MetaTC
from modules.indexer import SequentialIndexTree, StructIndexTree
from modules.embed import FullTreeEmbeddings, FactorizedTreeEmbeddings, HashEmbeddings
import numpy as np


class LEFT(Module):


    def __init__(self, args):
        super().__init__()

        # Config
        self.args = args
        self.device = self.args.device

        # Model
        self.meta_tcom = get_model(args) # type: MetaTC

        # Tree Index
        self.tree, self.tree_embs = self._setup_index_tree()

        # Loss
        self.loss = MSELoss()

        # Cache
        self.cache = {
            "QIndex0": {},
            "QIndex1": {}
        }


    def _setup_index_tree(self):

        # Record Tensor Size
        size_dict = {
            'user': self.args.num_users,
            'item': self.args.num_items,
            'time': self.args.num_times,
        }

        # Calculate the Repo Size
        num_repos = 1
        embed_chunks = 0
        for key in size_dict:
            if key not in self.args.qtype:
                num_repos *= size_dict[key]
                embed_chunks += 1

        # Init Neural Indexer Structure
        index_tree = StructIndexTree(self.args)
        tree_embeds = FactorizedTreeEmbeddings(self.args, index_tree)

        return index_tree, tree_embeds


    def setup_optimizer(self):
        self.meta_tcom.setup_optimizer()
        self.tree_embs.setup_optimizer()


    def _tensor_input(self, q_index:list, c_index):

        # Size: q_index: (Batch, 1), c_index: (Batch, Num Nodes)
        # Match Candidates Inputs

        num_nodes = c_index.shape[-1]

        q_index = [ index.view(-1, 1).repeat(1, num_nodes) for index in q_index ]

        # Create Query Input
        inputs = {}

        # Query Embeddings: Generated from MetaTC
        for i, type in enumerate(self.args.qtype):
            embeds = self.meta_tcom.get_embeddings(q_index[i].to(self.device), select=type)
            inputs[f'{type}_embeds'] = embeds

        # Candidate Embeddings: Generated from Tree Embeddings
        tree_embeds = self.tree_embs.forward(c_index.to(self.device))
        for i, type in enumerate(self.args.ktype):
            inputs[f'{type}_embeds'] = tree_embeds[i]

        return inputs


    @t.no_grad()
    def prepare_leaf_embeddings(self):

        if isinstance(self.tree_embs, FactorizedTreeEmbeddings):
            self._prepare_factorized_embeddings()
            return

        if len(self.args.ktype) == 1:
            ktype = self.args.ktype[0]
            num_nodes = eval(f'self.args.num_{ktype}s')
            leaf_embeds = self.meta_tcom.get_embeddings(t.arange(num_nodes, device=self.device), select=ktype)

            # Set Leaf Embeddings
            valid_leaf = self.tree.leaf_mask == 1
            self.tree_embs.tree_node_embeddings.weight.data[valid_leaf] = leaf_embeds

        elif len(self.args.ktype) == 2:

            type1 = self.args.ktype[0]
            type2 = self.args.ktype[1]

            num_type1 = eval(f'self.args.num_{type1}s')
            num_type2 = eval(f'self.args.num_{type2}s')

            type1_embeds = self.meta_tcom.get_embeddings(t.arange(num_type1, device=self.device), select=type1)
            type2_embeds = self.meta_tcom.get_embeddings(t.arange(num_type2, device=self.device), select=type2)

            # Shape : (num_type1, rank), (num_type2, rank)
            # type1_embeds_repeated: (num_type1 * %num_type2%, rank)
            # type2_embeds_repeated: (%num_type1% * num_type2, rank)
            type1_embeds_repeated = type1_embeds.repeat_interleave(num_type2, 0)
            type2_embeds_repeated = type2_embeds.repeat(num_type1, 1)

            # Shape: (num_type1 * num_type2, rank * 2)
            repeated_embeds = t.cat([type1_embeds_repeated, type2_embeds_repeated], dim=-1)

            # Set Leaf Embeddings
            valid_leaf = self.tree.leaf_mask == 1
            self.tree_embs.tree_node_embeddings.weight.data[valid_leaf] = repeated_embeds

        else:
            raise NotImplementedError('only support search no more than two dimension!')

        # 构建父节点的embedding
        for i in range(self.tree.leaf_startIdx - 1, -1, -1):

            # 跳过无效的索引节点
            if self.tree.node_mask[i] == 0:
                continue

            # 计算有效的孩子节点数
            valid_leaf_num = 0
            for j in range(self.tree.narys):
                if self.tree.node_mask[self.tree.narys * i + (j + 1)] == 1:
                    valid_leaf_num += 1

            # 父节点的Embedding为孩子节点Embedding的平均
            self.tree_embs.tree_node_embeddings.weight.data[i] = \
                self.tree_embs.tree_node_embeddings.weight.data[self.tree.narys * i + 1: self.tree.narys * i + 1 + valid_leaf_num].mean(dim=0)

        self.tree_embs.tree_node_embeddings.requires_grad_()


    @t.no_grad()
    def _prepare_factorized_embeddings(self):
        self.tree_embs.prepare_leaf_embeddings(self.meta_tcom)

    @t.no_grad()
    def beam_search(self, q_index, beam, return_scores=False):

        # Candidate Sets
        candidate = t.tensor([], dtype=t.int64, device=self.device)

        # Add Root Node
        queue = t.tensor([0]).to(self.device)

        # Layer-Skipping Optimization
        while len(queue) < beam:

            new_queue = self.tree.get_children(queue)
            validIdx = self.tree.node_mask[new_queue] == 1
            new_queue = new_queue[validIdx]
            if len(new_queue) > beam:
                break
            else:
                queue = new_queue

        while len(queue) > 0:
            nodeIdx = queue

            # if leaf node, add to candidate
            treeNode = nodeIdx[self.tree.leaf_mask[nodeIdx] == 1]
            if len(treeNode) > 0:
                leafIdx = treeNode[self.tree.node_mask[treeNode] == 1]
                candidate = t.cat([candidate, leafIdx], dim=-1)


            # if non leaf-node and active node, add children to queue
            nonleafIdx = self.tree.leaf_mask[nodeIdx] == 0
            treeNode = nodeIdx[nonleafIdx]
            treeNode = treeNode[self.tree.node_mask[treeNode] == 1]

            # if all leaf node, break
            if len(treeNode) == 0:
                break

            # search childrens and sort
            childrens = self.tree.get_children(treeNode)

            tensor_inputs = self._opt_tensor_input(q_index=q_index, c_index=childrens)
            scores = self.meta_tcom.get_score(**tensor_inputs).squeeze()
            beamIdx = t.argsort(scores, descending=True)[:beam]
            queue = childrens[beamIdx]


        # sort final candidate (leaf nodes)
        tensor_inputs = self._opt_tensor_input(q_index=q_index, c_index=candidate)
        scores = self.meta_tcom.get_score(**tensor_inputs).squeeze()
        sortIdx = t.argsort(scores, descending=True)
        beam_leaf = candidate[sortIdx]


        # convert to leaf index
        candidateIdx = self.tree.leaf2ravel(beam_leaf)

        if return_scores:
            return candidateIdx, scores[sortIdx]

        return candidateIdx


    def stochastic_beam_search_loss(self, q_index, beam, curriculum):
        """
        Stochastic Beam Search Loss
        :param q_index: Query Indices
        :param beam: Beam Size on Stochastic Beam Search
        :param curriculum: Curriculum Loss Control Parameters
        :return: Loss of Max-Heap
        """

        # Top Down Searching and Calibration
        bs = len(q_index[0])
        queue = t.zeros((bs, 1), dtype=t.int64, device=self.device)

        # Calculate Calibration Loss
        beam_search_loss = 0

        approx_heap_acc = []

        for depth in range(self.tree.depth - 1):

            nodeIdx = queue

            children = self.tree.get_children(nodeIdx)

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=children)
            origin_scores = self.meta_tcom.get_score(**tensor_inputs).detach()

            # Fix: 2023/06/05 - 未将无效的Children的Score设置为0
            origin_scores[self.tree.node_mask[children] == 0] = 0
            origin_scores = t.reshape(origin_scores, (len(nodeIdx), self.tree.narys, -1))
            label_scores = t.max(origin_scores, dim=1)[0]

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=nodeIdx)
            pred_scores = self.meta_tcom.get_score(**tensor_inputs)

            # Maskout the Node which is not valid in tree (to handle non full tree)
            curr_controller = self.tree.depth - 1 - (curriculum * 2 + 1)
            # randProb = np.random.rand() > 0.8
            if depth >= curr_controller or curriculum >= 6:
                node_mask = self.tree.node_mask[nodeIdx] == 1
                beam_search_loss += self.loss(pred_scores[node_mask], label_scores[node_mask]) * (depth / (self.tree.depth - 1))
                beam_search_loss += 0.1 * t.mean(t.relu(label_scores[node_mask] - pred_scores[node_mask]))# * (depth / (self.tree.depth - 1))
                approx_heap_acc += [t.sum(pred_scores[node_mask] > label_scores[node_mask]) / len(label_scores[node_mask])]
            queue = []
            for i in range(bs):
                # Select Only the Valid Children
                valid_children = children[i][self.tree.node_mask[children[i]] == 1]
                num_children = len(valid_children)
                randPerm = t.randperm(num_children)[:beam]
                queue.append(valid_children[randPerm])

            queue = t.stack(queue)

        heap_acc = t.mean(t.stack(approx_heap_acc))

        return beam_search_loss, heap_acc


    def _opt_tensor_input(self, q_index:list, c_index):

        # Size: q_index: (Batch, 1), c_index: (Batch, Num Nodes)
        # Match Candidates Inputs

        # Setup QIndex Cache
        for i, qindex in enumerate(q_index):
            qindex_int = qindex.item()
            if qindex_int not in self.cache[f"QIndex{i}"]:
                embeds = self.meta_tcom.get_embeddings(qindex.to(self.device), select=self.args.qtype[i])
                self.cache[f"QIndex{i}"][qindex_int] = embeds.unsqueeze(0).repeat(400 * self.tree.narys, 1)

        # Create Query Input
        inputs = {}

        # Query Embeddings: Generated from MetaTC
        num_nodes = c_index.shape[-1]
        for i, type in enumerate(self.args.qtype):
            embeds = self.cache[f"QIndex{i}"][q_index[i].item()][:num_nodes]
            inputs[f'{type}_embeds'] = embeds

        # Candidate Embeddings: Generated from Tree Embeddings
        tree_embeds = self.tree_embs.forward(c_index.to(self.device))
        for i, type in enumerate(self.args.ktype):
            inputs[f'{type}_embeds'] = tree_embeds[i]

        return inputs
