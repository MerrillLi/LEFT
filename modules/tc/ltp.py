import torch as t
from torch.nn import *
from modules.tc.meta_tc import MetaTC


class LTP(MetaTC):

    def __init__(self, args):
        super(LTP, self).__init__(args)
        self.lstm = LSTM(self.rank, self.rank, batch_first=False)
        self.rainbow = t.arange(-self.window + 1, 1).reshape(1, -1).to(self.device)
        self.attn = Sequential(Linear(2 * self.rank, 1), Tanh())
        self.user_linear = Linear(self.rank, self.rank)
        self.item_linear = Linear(self.rank, self.rank)
        self.time_linear = Linear(self.rank, self.rank)


    def to_seq_id(self, tids):
        tids = tids.reshape(-1, 1).repeat(1, self.window)
        tids += self.rainbow
        tids = tids.relu().permute(1, 0)
        return tids


    def forward(self, user, item, time):
        user_embeds = self.get_embeddings(user, "user")
        item_embeds = self.get_embeddings(item, "item")
        time_embeds = self.get_embeddings(time, "time")
        return self.get_score(user_embeds, item_embeds, time_embeds)

    def get_score(self, user_embeds, item_embeds, time_embeds):
        # Interaction Modules
        user_embeds = self.user_linear(user_embeds)
        item_embeds = self.item_linear(item_embeds)
        time_embeds = self.time_linear(time_embeds)
        pred = t.sum(user_embeds * item_embeds * time_embeds, dim=-1).sigmoid()
        return pred

    def get_embeddings(self, idx, select):

        if select == "user":
            return self.user_embeds(idx)

        elif select == "item":
            return self.item_embeds(idx)

        elif select == "time":
            # Read Time Embeddings
            time_embeds = self.time_embeds(self.to_seq_id(idx))
            outputs, (hs, cs) = self.lstm.forward(time_embeds)

            # Attention [seq_len, batch, dim] -> [seq_len, batch, 1]
            hss = hs.repeat(self.window, 1, 1)
            attn = self.attn(t.cat([outputs, hss], dim=-1))
            time_embeds = t.sum(attn * outputs, dim=0)
            return time_embeds
        else:
            raise NotImplementedError("Unknown select type: {}".format(select))
