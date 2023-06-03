#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 02/06/2023 12:56
# @Author : YuHui Li(MerylLynch)
# @File : meta_embeds.py
# @Comment : Created By Liyuhui,12:56
# @Completed : No
# @Tested : No

from torch.nn import *
from abc import ABC, abstractmethod

class TreeEmbeddings(Module, ABC):

    def __init__(self, args, tree):
        super().__init__()
        self.args = args
        self.tree = tree

    @abstractmethod
    def initialize(self):
        ...

    @abstractmethod
    def forward(self, nodeIdx):
        ...
