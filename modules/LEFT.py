import math
import torch as t
from torch.nn import *
from modules.tc import get_model, MetaTC
from modules.tree import SequentialTree, StructTree, ClusterTree, BalancedTree

class ExpectileLoss(Module):

    def __init__(self, w=0.5):
        super().__init__()
        self.w = w

    def forward(self, y_true, y_pred):
        err = y_true - y_pred
        weight = t.zeros_like(y_true, device=y_true.device)
        weight[err > 0] = self.w
        weight[err <= 0] = 1 - self.w
        exp_loss = err**2 * weight
        return exp_loss.mean()

class LEFT(Module):


    def __init__(self, args):
        super().__init__()

        # Config
        self.args = args
        self.device = self.args.device

        # Model
        self.meta_tcom = get_model(args) # type: MetaTC

        # Loss
        self.loss = MSELoss()

        # Cache
        self.cache = {
            "QIndex0": {},
            "QIndex1": {}
        }


    def setup_optimizer(self):
        self.meta_tcom.setup_optimizer()


    def setup_indexer(self):
        if self.args.tree_type == "sequential":
            self.indexer = SequentialTree(self.args)
        elif self.args.tree_type == "struct":
            self.indexer = StructTree(self.args)
        elif self.args.tree_type == "cluster":
            self.indexer = ClusterTree(self.args)
        elif self.args.tree_type == "balanced":
            self.indexer = BalancedTree(self.args)

        self.indexer.initialize(self.meta_tcom)


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
            target_shape = q_index[i].shape + (embeds.shape[-1],)
            inputs[f'{type}_embeds'] = embeds.reshape(target_shape)

        # Candidate Embeddings: Generated from Tree Embeddings
        tree_embeds = self.indexer.forward(c_index.to(self.device))
        for i, type in enumerate(self.args.ktype):
            inputs[f'{type}_embeds'] = tree_embeds[i]

        return inputs


    @t.no_grad()
    def beam_search(self, q_index, beam, return_scores=False):

        # Candidate Sets
        candidate = t.tensor([], dtype=t.int64, device=self.device)

        # Add Root Node
        queue = t.tensor([0]).to(self.device)

        # Layer-Skipping Optimization
        while len(queue) < beam:

            new_queue = self.indexer.get_children(queue)
            validIdx = self.indexer.node_mask[new_queue] == 1
            new_queue = new_queue[validIdx]
            if len(new_queue) > beam:
                break
            else:
                queue = new_queue

        while len(queue) > 0:
            nodeIdx = queue

            # if leaf node, add to candidate
            treeNode = nodeIdx[self.indexer.leaf_mask[nodeIdx] == 1]
            if len(treeNode) > 0:
                leafIdx = treeNode[self.indexer.node_mask[treeNode] == 1]
                candidate = t.cat([candidate, leafIdx], dim=-1)


            # if non leaf-node and active node, add children to queue
            nonleafIdx = self.indexer.leaf_mask[nodeIdx] == 0
            treeNode = nodeIdx[nonleafIdx]
            treeNode = treeNode[self.indexer.node_mask[treeNode] == 1]

            # if all leaf node, break
            if len(treeNode) == 0:
                break

            # search childrens and sort
            childrens = self.indexer.get_children(treeNode)
            validIdx = self.indexer.node_mask[childrens] == 1
            childrens = childrens[validIdx]

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
        candidateIdx = self.indexer.leaf2ravel(beam_leaf)
        assert t.mean(self.indexer.leaf_mask[beam_leaf].float()) == 1

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

        for depth in range(1, self.indexer.depth):

            nodeIdx = queue

            children = self.indexer.get_children(nodeIdx)

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=children)
            origin_scores = self.meta_tcom.get_score(**tensor_inputs).detach()

            # Fix: 2023/06/05 - 未将无效的Children的Score设置为0
            origin_scores[self.indexer.node_mask[children] == 0] = 0
            origin_scores = t.reshape(origin_scores, (len(nodeIdx), self.indexer.narys, -1))
            label_scores = t.max(origin_scores, dim=1)[0]

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=nodeIdx)
            pred_scores = self.meta_tcom.get_score(**tensor_inputs)

            # Maskout the Node which is not valid in tree (to handle non full tree)
            curr_controller = self.indexer.depth - 1 - (curriculum * 2 + 1)
            # randProb = np.random.rand() > 0.8
            if depth >= curr_controller or curriculum >= 5:
                node_mask = self.indexer.node_mask[nodeIdx] == 1
                beam_search_loss += self.loss(pred_scores[node_mask], label_scores[node_mask]) * (depth / (self.indexer.depth - 1))

                beam_search_loss += 0.01 * t.mean(t.relu(label_scores[node_mask] - pred_scores[node_mask])) * (1 / (curriculum + 1))
                approx_heap_acc += [t.sum(pred_scores[node_mask] > label_scores[node_mask]) / len(label_scores[node_mask])]
            queue = []
            for i in range(bs):
                # Select Only the Valid Children
                valid_children = children[i][self.indexer.node_mask[children[i]] == 1]
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
                if len(embeds.shape) == 1:
                    embeds = embeds.unsqueeze(0)
                self.cache[f"QIndex{i}"][qindex_int] = embeds.repeat(400 * self.indexer.narys, 1)


        # Create Query Input
        inputs = {}

        # Query Embeddings: Generated from MetaTC
        num_nodes = c_index.shape[-1]
        for i, type in enumerate(self.args.qtype):
            embeds = self.cache[f"QIndex{i}"][q_index[i].item()][:num_nodes]
            inputs[f'{type}_embeds'] = embeds

        # Candidate Embeddings: Generated from Tree Embeddings
        tree_embeds = self.indexer.forward(c_index.to(self.device))
        for i, type in enumerate(self.args.ktype):
            inputs[f'{type}_embeds'] = tree_embeds[i]

        return inputs
