import torch as t
from torch.nn import *
from modules.tc.meta_tc import MetaTC


class NTF(MetaTC):

    def __init__(self, args):
        super().__init__(args)
        self.lstm = LSTM(args.rank, args.rank, batch_first=False)
        self.rainbow = t.arange(-self.window + 1, 1).reshape(1, -1).to(self.device)
        self.inter = self.layers = Sequential(
            LayerNorm(3 * args.rank),
            Linear(3 * args.rank, 2 * args.rank),
            ReLU(),
            Linear(2 * args.rank, 2 * args.rank),
            ReLU(),
            Linear(2 * args.rank, 1),
            Sigmoid()
        )

    def to_seq_id(self, tids):
        tids = tids.reshape(-1, 1).repeat(1, self.window)
        tids += self.rainbow
        tids = tids.relu().permute(1, 0)
        return tids

    def forward(self, uIdx, iIdx, tIdx):
        user_embeds = self.get_embeddings(uIdx, 'user')
        item_embeds = self.get_embeddings(iIdx, 'item')
        time_embeds = self.get_embeddings(tIdx, 'time')
        return self.get_score(user_embeds, item_embeds, time_embeds)

    def get_embeddings(self, idx, select):
        if select == 'user':
            return self.user_embeds(idx)
        elif select == 'item':
            return self.item_embeds(idx)
        elif select == 'time':
            time_embeds = self.time_embeds(self.to_seq_id(idx))
            _, (time_embeds, _) = self.lstm.forward(time_embeds)
            return time_embeds.squeeze()


    def get_score(self, user_embeds, item_embeds, time_embeds):
        x = t.cat([time_embeds, user_embeds, item_embeds], dim=-1)
        x = self.layers(x)
        return x
