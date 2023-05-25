import math
import torch as t
from torch.nn import *
from modules.tc import get_model, MetaTC
from utils.loss import raise_and_suppress_loss
from utils.radix import tree_path
from einops import rearrange


class HashTreeEmbeddings(Module):

    def __init__(self, qtype, num_nodes, rank, chunks, tree):
        super().__init__()

        # Record Params
        self.rank = rank
        self.qtype = qtype
        self.chunks = chunks
        self.num_nodes = num_nodes

        # Node Embedding Modules
        self.tree_node_embeddings = Embedding(num_nodes, rank * chunks)
        self.transform = ModuleDict()
        for i in range(chunks):
            self.transform[str(i)] = Sequential(Linear(rank, rank))

        # The Index Tree
        self.tree = tree

        # Allow Inplace Operation
        self.tree_node_embeddings.weight.data[tree.leaf_start_idx: tree.leaf_start_idx + tree.num_repos].requires_grad = False


        # Path Embeddings
        self.path_embeddings = Embedding(100, self.chunks * rank)

        self.lstm1 = GRU(rank, rank, batch_first=True)
        self.lstm2 = GRU(rank, rank, batch_first=True)



    def to_path(self, nodeIdx):
        paths = tree_path(nodeIdx, self.tree.narys)
        return paths


    def forward(self, nodeIdx):

        # Shape: nodeIdx: (batch_size, num_nodes)
        bsize = nodeIdx.shape[0]
        # 1. Check if the nodes are all leaf node
        nonLeafIdx = self.tree.leaf_mask[nodeIdx] == 0
        if t._is_all_true(nonLeafIdx):
            output_embeds_list = []

            # Read Embeddings
            paths = self.to_path(nodeIdx)

            # Shape: paths: (batch_size, num_nodes, path_len)
            # Shape: path_embeds: (batch_size, num_nodes, path_len, rank)
            path_embeds = self.path_embeddings(paths)

            path_embeds = path_embeds.chunk(self.chunks, dim=-1)

            # reshape to feed into rnn
            path_embeds_list = [rearrange(path_embed, 'b n p r -> (b n) p r') for path_embed in path_embeds]

            # LSTM to process
            for i in range(self.chunks):
                _, hn = self.lstm1.forward(path_embeds_list[i])

                # hn: (batch_size * num_nodes, rank)
                # reshape to (batch_size, num_nodes, rank)
                hn = rearrange(hn.squeeze(), '(b n) r -> b n r', b=bsize)
                output_embeds_list.append(hn)

            return output_embeds_list


        elif t.sum(nonLeafIdx) == 0:
            embeds = self.tree_node_embeddings(nodeIdx)
            embeds_list = list(embeds.chunk(self.chunks, dim=-1))
            return embeds_list

        else:
            raise ValueError("The input nodeIdx contains both leaf and non-leaf nodes.")



class TreeNodeEmbeddings(Module):

    def __init__(self, qtype, num_nodes, rank, chunks, tree):
        super().__init__()

        # Record Params
        self.rank = rank
        self.qtype = qtype
        self.chunks = chunks
        self.num_nodes = num_nodes

        # Node Embedding Modules
        self.tree_node_embeddings = Embedding(num_nodes, rank * chunks)
        self.transform = ModuleDict()
        self.q_transform = ModuleDict()
        for i in range(chunks):
            self.transform[str(i)] = Sequential(Linear(rank, 1 * rank), ReLU(), Linear(1 * rank, rank))

        for i in range(chunks):
            self.q_transform[str(i)] = Sequential(Linear((len(qtype) + 1) * rank, 10 * rank), ReLU(), Linear(10 * rank, rank))

        # The Index Tree
        self.tree = tree

        # Allow Inplace Operation
        self.tree_node_embeddings.weight.data[tree.leaf_start_idx: tree.leaf_start_idx + tree.num_repos].requires_grad = False


    def forward(self, nodeIdx, query:dict=None):
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



class IndexTree:

    def __init__(self, num_repos, narys):

        self.narys = narys
        self.num_repos = num_repos
        self.depth = math.ceil(math.log(num_repos, narys))

        # Build Full Nary-Tree by 1D-Array
        num_tree_nodes = narys ** (self.depth + 1) - 1
        leaf_start_idx = narys ** self.depth - 1
        self.num_tree_nodes = num_tree_nodes
        self.leaf_start_idx = leaf_start_idx

        # Leaf Mask: 1 for Leaf Node, 0 for Non-Leaf Node
        self.leaf_mask = t.zeros(num_tree_nodes, dtype=t.int)
        self.leaf_mask[leaf_start_idx:] = 1

        # Node Mask: 1 for Valid Node, 0 for Invalid Node
        self.node_mask = t.zeros(num_tree_nodes, dtype=t.int)
        self.node_mask[leaf_start_idx: leaf_start_idx + num_repos] = 1
        for i in range(leaf_start_idx - 1, -1, -1):
            valid_flag = 0
            for j in range(narys):
                valid_flag |= self.node_mask[narys * i + (j + 1)]
            self.node_mask[i] = valid_flag


    def get_children(self, nodeIdx):
        children = []
        for i in range(self.narys):
            children.append(self.narys * nodeIdx + (i + 1))
        return t.cat(children, dim=-1)


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
        index_tree = IndexTree(num_repos, self.args.narys)
        tree_embeds = TreeNodeEmbeddings(self.args.qtype, index_tree.num_tree_nodes,
                                         self.args.rank, embed_chunks, index_tree)

        return index_tree, tree_embeds


    def setup_optimizer(self):
        self.meta_tcom.setup_optimizer()
        self.tree_opt = t.optim.AdamW(self.tree_embs.parameters(), lr=0.01)
        # self.opt_scheduler = t.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.tree_opt, T_0=60, T_mult=1, eta_min=1e-3)


    def _tensor_input(self, q_index:list, c_index:list):

        # Size: q_index: (Batch, 1), c_index: (Batch, Num Nodes)
        # Match Candidates Inputs

        num_nodes = c_index[0].shape[-1]

        q_index = [ index.view(-1, 1).repeat(1, num_nodes) for index in q_index ]

        # Create Query Input
        inputs = {}

        # Query Embeddings: Generated from MetaTC
        for i, type in enumerate(self.args.qtype):
            embeds = self.meta_tcom.get_embeddings(q_index[i], select=type)
            inputs[f'{type}_embeds'] = embeds

        # Candidate Embeddings: Generated from Tree Embeddings
        tree_embeds = self.tree_embs.forward(c_index[0])
        for i, type in enumerate(self.args.ktype):
            inputs[f'{type}_embeds'] = tree_embeds[i]

        return inputs


    @t.no_grad()
    def prepare_leaf_embeddings(self):

        if len(self.args.ktype) == 1:
            ktype = self.args.ktype[0]
            num_nodes = eval(f'self.args.num_{ktype}s')
            leaf_embeds = self.meta_tcom.get_embeddings(t.arange(num_nodes, device=self.device), select=ktype)
            self.tree_embs.tree_node_embeddings.weight.data[self.tree.leaf_start_idx:self.tree.leaf_start_idx + num_nodes] = leaf_embeds

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

            num_nodes = num_type1 * num_type2
            self.tree_embs.tree_node_embeddings.weight.data[self.tree.leaf_start_idx:self.tree.leaf_start_idx + num_nodes] = repeated_embeds

        else:
            raise NotImplementedError('only support search no more than two dimension!')



    @t.no_grad()
    def beam_search(self, q_index, beam):

        # Candidate Sets
        candidate = t.tensor([], dtype=t.int64, device=self.device)

        # Add Root Node
        queue = t.tensor([0])
        while len(queue) < beam:

            new_queue = self.tree.get_children(queue)
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

            # search childrens and sort
            childrens = self.tree.get_children(treeNode)

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=[childrens])
            scores = self.meta_tcom.get_score(**tensor_inputs).squeeze()
            beamIdx = t.argsort(scores, descending=True)[:beam]
            queue = childrens[beamIdx]


        # sort final candidate (leaf nodes)
        tensor_inputs = self._tensor_input(q_index=q_index, c_index=[candidate])
        scores = self.meta_tcom.get_score(**tensor_inputs).squeeze()
        sortIdx = t.argsort(scores, descending=True)
        beam_leaf = candidate[sortIdx]

        beam_leaf -= self.tree.leaf_start_idx
        return beam_leaf


    def stochastic_beam_search_loss(self, q_index, beam, curriculum) -> t.Tensor:
        """
        Stochastic Beam Search Loss
        :param q_index: Query Indices
        :param beam: Beam Size on Stochastic Beam Search
        :return: Loss of Max-Heap
        """

        # Top Down Searching and Calibration
        bs = len(q_index[0])
        queue = t.zeros((bs, 1), dtype=t.int64, device=self.device)

        # Calculate Calibration Loss
        beam_search_loss = 0

        for depth in range(self.tree.depth):

            nodeIdx = queue

            children = self.tree.get_children(nodeIdx)

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=[children])
            origin_scores = self.meta_tcom.get_score(**tensor_inputs).detach()
            origin_scores = t.reshape(origin_scores, (len(nodeIdx), self.tree.narys, -1))
            label_scores = t.max(origin_scores, dim=1)[0]

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=[nodeIdx])
            pred_scores = self.meta_tcom.get_score(**tensor_inputs)

            # Maskout the Node which is not valid in tree (to handle non full tree)
            node_mask = self.tree.node_mask[nodeIdx] == 1

            if depth >= self.tree.depth - 1 - (curriculum * 2 + 1):
                # loss weights
                # weight = ( (1 + min(self.tree.depth, curriculum)) / (depth + 10))
                weight = 1
                beam_search_loss += self.loss(pred_scores[node_mask], label_scores[node_mask]) * weight

            # queue = t.zeros((bs, min(beam, num_children)), device=self.device, dtype=t.int64)
            queue = []
            for i in range(bs):

                # Select Only the Valid Children
                valid_children = children[i][self.tree.node_mask[children[i]] == 1]
                num_children = len(valid_children)
                randPerm = t.randperm(num_children)[:beam]
                queue.append(valid_children[randPerm])

            queue = t.stack(queue)

        return beam_search_loss


    def priority_beam_search_loss(self, q_index, beam):
        """
        Top-K Beam Search Loss
        :param q_index: Query Indices
        :param topk: TopK Size on TopK Beam Search
        :param beam: Beam Size on TopK Beam Search
        :return: Loss of Max-Heap
        """

        # Top Down Searching and Calibration
        bs = len(q_index[0])
        queue = t.zeros((bs, 1), dtype=t.int64, device=self.device)


        # Calculate Regret Loss
        beam_search_loss = 0

        for i in range(self.tree.depth):

            nodeIdx = queue
            children = self.tree.get_children(nodeIdx)

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=[children])
            origin_scores = self.meta_tcom.get_score(**tensor_inputs).detach()
            origin_scores = t.reshape(origin_scores, (len(nodeIdx), self.tree.narys, -1))
            label_scores = t.max(origin_scores, dim=1)[0]

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=[nodeIdx])
            pred_scores = self.meta_tcom.get_score(**tensor_inputs)
            label_scores[self.tree.node_mask[nodeIdx] == 0] = 0
            pred_scores = pred_scores.reshape(label_scores.shape)
            beam_search_loss += self.loss(pred_scores, label_scores)

            childrenIdx = t.argsort(origin_scores.reshape(bs, -1), descending=True)[:, :beam]
            queue = t.zeros_like(childrenIdx, device=self.device)
            for i in range(len(childrenIdx)):
                queue[i] = children[i][childrenIdx[i]]

        return beam_search_loss


    def beam_search_regret_loss(self, q_index, topk, beam):
        """
        Top-K Beam Search Loss to Minimize Regret@K
        :param q_index: Query Indices
        :param topk: TopK Size on TopK Beam Search
        :param beam: Beam Size on TopK Beam Search
        :return: Loss of Regret
        """

        # Top Down Searching and Calibration
        bs = len(q_index[0])
        queue_m = t.zeros((bs, 1), dtype=t.int64, device=self.device)
        queue_2m = t.zeros((bs, 1), dtype=t.int64, device=self.device)

        # Calculate Regret Loss
        beam_search_loss = 0
        rank_loss = 0
        regrets = []

        for depth in range(self.tree.depth):

            # Execute Beam Search with Beam=M
            nodeIdx_m = queue_m
            children_m = self.tree.get_children(nodeIdx_m)

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=[children_m])
            origin_scores = self.meta_tcom.get_score(**tensor_inputs).detach()
            origin_scores = t.reshape(origin_scores, (len(nodeIdx_m), self.tree.narys, -1))
            label_scores = t.max(origin_scores, dim=1)[0]

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=[nodeIdx_m])
            pred_scores = self.meta_tcom.get_score(**tensor_inputs)

            node_mask = self.tree.node_mask[nodeIdx_m] == 1
            beam_search_loss += self.loss(pred_scores[node_mask], label_scores[node_mask])

            scores_m = origin_scores.reshape(bs, -1)
            childrenIdx_m = t.argsort(scores_m, descending=True)[:, :beam]
            queue_m = t.zeros_like(childrenIdx_m, device=self.device)
            score_a = t.zeros_like(childrenIdx_m, device=self.device, dtype=t.float32)
            for i in range(len(childrenIdx_m)):
                queue_m[i] = children_m[i][childrenIdx_m[i]]
                score_a[i] = scores_m[i][childrenIdx_m[i]]

            # Execute Beam Search with Beam=2M
            nodeIdx_2m = queue_2m
            children_2m = self.tree.get_children(nodeIdx_2m)

            tensor_inputs_2m = self._tensor_input(q_index=q_index, c_index=[children_2m])
            origin_scores_2m = self.meta_tcom.get_score(**tensor_inputs_2m).detach()

            scores_2m = origin_scores_2m.reshape(bs, -1)
            childrenIdx_2m = t.argsort(scores_2m, descending=True)[:, :2 * beam]
            queue_2m = t.zeros_like(childrenIdx_2m, device=self.device)
            score_b = t.zeros_like(childrenIdx_2m, device=self.device, dtype=t.float32)

            for i in range(len(childrenIdx_2m)):
                queue_2m[i] = children_2m[i][childrenIdx_2m[i]]
                score_b[i] = scores_2m[i][childrenIdx_2m[i]]

            # Calculate Regret Loss
            if childrenIdx_2m.shape[-1] != childrenIdx_m.shape[-1]:
                rank_loss += raise_and_suppress_loss(queue_m[:, :topk], queue_2m[:, :topk],
                                                       score_a[:, :topk], score_b[:, :topk])


            regret = t.mean( t.sum(score_b[:, :topk], dim=-1) - t.sum(score_a[:, :topk], dim=-1))
            regrets.append(regret)

        return beam_search_loss, rank_loss, regrets[-1]
