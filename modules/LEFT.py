import math
import torch as t
from torch.nn import *
from modules.tc import get_model, MetaTC
from modules.indexer import SequentialIndexTree, StructIndexTree


class TreeNodeEmbeddings(Module):

    def __init__(self, args, tree):
        super().__init__()

        # Record Params
        self.args = args
        self.rank = self.args.rank
        self.qtype = self.args.qtype
        self.chunks = len(self.args.ktype)
        self.tree = tree
        self.num_nodes = self.tree.num_nodes

        # Node Embedding Modules
        self.tree_node_embeddings = Embedding(self.num_nodes, self.rank * self.chunks)
        self.transform = ModuleDict()
        self.q_transform = ModuleDict()
        for i in range(self.chunks):
            self.transform[str(i)] = Sequential(Linear(self.rank, 1 * self.rank), ReLU(), Linear(1 * self.rank, self.rank))

        for i in range(self.chunks):
            self.q_transform[str(i)] = Sequential(Linear((len(self.qtype) + 1) * self.rank, 10 * self.rank), ReLU(), Linear(10 * self.rank, self.rank))

        # The Index Tree
        self.tree = tree

        # Allow Inplace Operation
        leafIdx = self.tree.leaf_mask == 1
        self.tree_node_embeddings.weight.data[leafIdx].requires_grad = False


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
        tree_embeds = TreeNodeEmbeddings(self.args, index_tree)

        return index_tree, tree_embeds


    def setup_optimizer(self):
        self.meta_tcom.setup_optimizer()
        self.tree_opt = t.optim.AdamW(self.tree_embs.parameters(), lr=0.01)
        self.opt_scheduler = t.optim.lr_scheduler.ConstantLR(self.tree_opt)
        # self.opt_scheduler = t.optim.lr_scheduler.StepLR(self.tree_opt, step_size=150, gamma=0.8)


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


    @t.no_grad()
    def beam_search(self, q_index, beam):

        # Candidate Sets
        candidate = t.tensor([], dtype=t.int64, device=self.device)

        # Add Root Node
        queue = t.tensor([0])
        while len(queue) < beam:

            new_queue = self.tree.get_children(queue)
            if len(new_queue) > beam:
                queue = new_queue
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

            tensor_inputs = self._opt_tensor_input(q_index=q_index, c_index=[childrens])
            scores = self.meta_tcom.get_score(**tensor_inputs).squeeze()
            beamIdx = t.argsort(scores, descending=True)[:beam]
            queue = childrens[beamIdx]


        # sort final candidate (leaf nodes)
        tensor_inputs = self._opt_tensor_input(q_index=q_index, c_index=[candidate])
        scores = self.meta_tcom.get_score(**tensor_inputs).squeeze()
        sortIdx = t.argsort(scores, descending=True)
        beam_leaf = candidate[sortIdx]


        # convert to leaf index
        candidateIdx = self.tree.leaf2ravel(beam_leaf)
        return candidateIdx


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

        for depth in range(self.tree.depth - 1):

            nodeIdx = queue

            children = self.tree.get_children(nodeIdx)

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=[children])
            origin_scores = self.meta_tcom.get_score(**tensor_inputs).detach()

            # Fix: 2023/05/06 - 未将无效的Children的Score设置为0
            origin_scores[self.tree.node_mask[children] == 0] = 0
            origin_scores = t.reshape(origin_scores, (len(nodeIdx), self.tree.narys, -1))
            label_scores = t.max(origin_scores, dim=1)[0]

            tensor_inputs = self._tensor_input(q_index=q_index, c_index=[nodeIdx])
            pred_scores = self.meta_tcom.get_score(**tensor_inputs)

            # Maskout the Node which is not valid in tree (to handle non full tree)
            node_mask = self.tree.node_mask[nodeIdx] == 1

            if depth >= self.tree.depth - 1 - (curriculum * 2 + 1): 
                # loss weights
                weight = 1
                beam_search_loss += self.loss(pred_scores[node_mask], label_scores[node_mask]) * weight

            queue = []
            for i in range(bs):

                # Select Only the Valid Children
                valid_children = children[i][self.tree.node_mask[children[i]] == 1]
                num_children = len(valid_children)
                randPerm = t.randperm(num_children)[:beam]
                queue.append(valid_children[randPerm])

            queue = t.stack(queue)

        return beam_search_loss


    def _opt_tensor_input(self, q_index:list, c_index:list):

        # Size: q_index: (Batch, 1), c_index: (Batch, Num Nodes)
        # Match Candidates Inputs

        # Setup QIndex Cache
        for i, qindex in enumerate(q_index):
            qindex_int = qindex.item()
            if qindex_int not in self.cache[f"QIndex{i}"]:
                embeds = self.meta_tcom.get_embeddings(qindex, select=self.args.qtype[i])
                self.cache[f"QIndex{i}"][qindex_int] = embeds.unsqueeze(0).repeat(400 * self.tree.narys, 1)
            
        # Create Query Input
        inputs = {}

        # Query Embeddings: Generated from MetaTC
        num_nodes = c_index[0].shape[-1]
        for i, type in enumerate(self.args.qtype):
            embeds = self.cache[f"QIndex{i}"][q_index[i].item()][:num_nodes]
            inputs[f'{type}_embeds'] = embeds

        # Candidate Embeddings: Generated from Tree Embeddings
        tree_embeds = self.tree_embs.forward(c_index[0])
        for i, type in enumerate(self.args.ktype):
            inputs[f'{type}_embeds'] = tree_embeds[i]

        return inputs