import faiss
from modules.tc.meta_tc import MetaTC
import torch as t


class FaissIndexer:

    def __init__(self, args):
        self.args = args

        # Declare Variables
        self.index = None

        self.initialize()

    def initialize(self):

        if self.args.index == 'LSH':
            # IndexLSH(Vector Dim, LSH Bits)
            self.index = faiss.IndexLSH(self.args.rank, self.args.LSHbits)

        elif self.args.index == 'HNSW':
            # IndexHNSW(Vector Dim, HNSWx)
            self.index = faiss.IndexHNSWFlat(self.args.rank, self.args.HNSWx, faiss.METRIC_INNER_PRODUCT)
            self.index.hierarchy = self.args.hierarchy
            self.index.efConstruction = self.args.efConstruction
            self.index.efSearch = self.args.efSearch

        elif self.args.index == 'PQ':
            # IndexPQ(Vector Dim, M, nbits)
            self.index = faiss.IndexPQ(self.args.rank, self.args.PQm, self.args.PQbits)
        else:
            raise NotImplementedError(f"Index {self.args.index} is not supported yet.")

    def find_topk_by_query(self, query, topk):
        query = query.reshape(1, -1)
        distance, indices = self.index.search(query, topk)
        return indices.flatten()

    @t.no_grad()
    def train_indexer(self, model: MetaTC):
        device = model.device

        if len(self.args.ktype) == 1:

            ktype = self.args.ktype[0]
            num_type1 = eval(f'self.args.num_{ktype}s')
            index1 = t.arange(num_type1, device=device)
            embeds = model.get_embeddings(index1, ktype)
            database_embeds = embeds.cpu().numpy()
            self.index.train(database_embeds)
            self.index.add(database_embeds)

        elif len(self.args.ktype) == 2:

            ktype1 = self.args.ktype[0]
            ktype2 = self.args.ktype[1]

            num_type1 = eval(f'self.args.num_{ktype1}s')
            num_type2 = eval(f'self.args.num_{ktype2}s')

            index1 = t.arange(num_type1, device=device)
            index2 = t.arange(num_type2, device=device)

            type1_embeds = model.get_embeddings(index1, ktype1)
            type2_embeds = model.get_embeddings(index2, ktype2)

            type1_embeds_repeated = type1_embeds.repeat_interleave(num_type2, 0)
            type2_embeds_repeated = type2_embeds.repeat(num_type1, 1)

            database_embeds = type1_embeds_repeated * type2_embeds_repeated
            database_embeds = database_embeds.cpu().numpy()
            self.index.train(database_embeds)
            self.index.add(database_embeds)
        else:
            raise NotImplementedError("Length of KType is illegal !")



