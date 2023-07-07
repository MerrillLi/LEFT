#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 09/06/2023 21:13
# @Author : YuHui Li(MerylLynch)
# @File : costco.py
# @Comment : Created By Liyuhui,21:13
# @Completed : No
# @Tested : No

import torch as t
from torch.nn import *
from modules.tc.meta_tc import MetaTC


class CoSTCo(MetaTC):

    def __init__(self, args):
        super(CoSTCo, self).__init__(args)
        rank = args.rank
        self.conv1 = Sequential(LazyConv2d(self.channels, kernel_size=(3, 1)), ReLU())
        self.conv2 = Sequential(LazyConv2d(self.channels, kernel_size=(1, rank)), ReLU())
        self.flatten = Flatten()
        self.linear = Sequential(LazyLinear(rank), ReLU())
        self.output = Sequential(LazyLinear(1), Sigmoid())

    def forward(self, user, item, time):
        user_embeds = self.get_embeddings(user, "user")
        item_embeds = self.get_embeddings(item, "item")
        time_embeds = self.get_embeddings(time, "time")
        return self.get_score(user_embeds, item_embeds, time_embeds)

    def get_score(self, user_embeds, item_embeds, time_embeds):

        # Interaction Modules
        # stack as [batch, N, dim]
        x = t.stack([time_embeds, user_embeds, item_embeds], dim=1)

        # reshape to [batch, 1, N, dim]
        x = t.unsqueeze(x, dim=1)

        # conv
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x.flatten()

    def get_embeddings(self, idx, select):

        if select == "user":
            return self.user_embeds(idx)

        elif select == "item":
            return self.item_embeds(idx)

        elif select == "time":
            return self.time_embeds(idx)

        else:
            raise NotImplementedError("Unknown select type: {}".format(select))
