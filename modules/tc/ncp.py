import torch as t
from torch.nn import *
from modules.tc.meta_tc import MetaTC


class NCP(MetaTC):

    def __init__(self, args):
        super(NCP, self).__init__(args)
        self.user_trans = Sequential(Linear(self.rank, self.rank))#, Tanh(), Linear(self.rank, self.rank))
        self.item_trans = Sequential(Linear(self.rank, self.rank))#, Tanh(), Linear(self.rank, self.rank))
        self.time_trans = Sequential(Linear(self.rank, self.rank))#, Tanh(), Linear(self.rank, self.rank))

    def get_embeddings(self, idx, select):
        if select == "user":
            user_embeds = self.user_embeds(idx)
            user_embeds = self.user_trans(user_embeds)
            return user_embeds
        elif select == "item":
            item_embeds = self.item_embeds(idx)
            item_embeds = self.item_trans(item_embeds)
            return item_embeds
        elif select == "time":
            time_embeds = self.time_embeds(idx)
            time_embeds = self.time_trans(time_embeds)
            return time_embeds
        else:
            raise NotImplementedError("Unknown select type: {}".format(select))


    def get_score(self, user_embeds, item_embeds, time_embeds):
        score = t.sum(user_embeds * item_embeds * time_embeds, dim=-1).sigmoid()
        return score


    def forward(self, user, item, time):
        user_embeds = self.get_embeddings(user, "user")
        item_embeds = self.get_embeddings(item, "item")
        time_embeds = self.get_embeddings(time, "time")
        return self.get_score(user_embeds, item_embeds, time_embeds)

