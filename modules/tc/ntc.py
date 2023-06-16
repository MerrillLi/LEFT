import math
import torch as t
from torch.nn import *
from modules.tc.meta_tc import MetaTC


class NTC(MetaTC):

    def __init__(self, args):
        super(NTC, self).__init__(args)
        self.cnn = ModuleList()
        for i in range(int(math.log2(self.rank))):
            in_channels = 1 if i == 0 else self.channels
            conv_layer = Conv3d(in_channels=in_channels, out_channels=self.channels,
                                kernel_size=2, stride=2)
            self.cnn.append(conv_layer)
            self.cnn.append(ReLU())

        self.score = Sequential(
            LazyLinear(1),
            Sigmoid()
        )


    def get_embeddings(self, idx, select):
        if select == "user":
            return self.user_embeds(idx)

        elif select == "item":
            return self.item_embeds(idx)

        elif select == "time":
            return self.time_embeds(idx)

    def get_score(self, user_embeds, item_embeds, time_embeds):
        outer_prod = t.einsum('ni, nj, nk-> nijk', user_embeds, item_embeds, time_embeds)
        outer_prod = t.unsqueeze(outer_prod, dim=1)
        rep = outer_prod
        for layer in self.cnn:
            rep = layer(rep)
        y = self.score(rep.squeeze())
        return y


    def forward(self, user, item, time):
        user_embeds = self.user_embeds(user)
        item_embeds = self.item_embeds(item)
        time_embeds = self.time_embeds(time)
        return self.get_score(user_embeds, item_embeds, time_embeds)
