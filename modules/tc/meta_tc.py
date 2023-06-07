import numpy as np
import torch
from torch.nn import *
from abc import ABC, abstractmethod
from torch.optim import *
from torch.utils.data import DataLoader
from utils.metrics import ErrorMetrics

def to_device(args, device):
    return [arg.to(device) for arg in args]


class MetaTC(Module, ABC):

    def __init__(self, args):

        super().__init__()

        # Basic Params
        self.args = args
        self.device = args.device
        self.window = args.window
        self.num_users = args.num_users
        self.num_items = args.num_items
        self.num_times = args.num_times
        self.rank = args.rank
        self.channels = args.channels

        # Shared Components
        self.user_embeds = Embedding(args.num_users, self.rank)
        self.item_embeds = Embedding(args.num_items, self.rank)
        self.time_embeds = Embedding(args.num_times, self.rank)

        # Optimization
        self.loss = MSELoss()

        # Misc
        self.errorMetrics = None

        # AMP
        self.amp = args.amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)



    @abstractmethod
    def forward(self, uIdx, iIdx, tIdx):
        pass

    @abstractmethod
    def get_embeddings(self, idx, select):
        pass

    @abstractmethod
    def get_score(self, user_embeds, item_embeds, time_embeds):
        pass

    def train_one_epoch(self, trainLoader:DataLoader):
        self.train()
        losses = []
        for batch in trainLoader:
            self.optimizer.zero_grad()
            uIdx, iIdx, tIdx, reals = to_device(batch, self.device)

            with torch.cuda.amp.autocast(enabled=self.amp):
                preds = self.forward(uIdx, iIdx, tIdx)
                loss = self.loss(preds, reals)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            losses.append(loss.item())
            self.optimizer.step()
        return sum(losses) / len(losses)

    @torch.no_grad()
    def valid_one_epoch(self, validLoader):
        self.eval()
        metrics = self._get_metrics_computer(len(validLoader.dataset))

        metrics.reset()
        for batch in validLoader:
            uIdx, iIdx, tIdx, reals = to_device(batch, self.device)
            preds = self.forward(uIdx, iIdx, tIdx)
            metrics.append(reals, preds)
        return metrics.compute()

    @torch.no_grad()
    def test_one_epoch(self, testLoader):
        self.eval()
        metrics = ErrorMetrics(len(testLoader.dataset), self.device)
        metrics.reset()
        for batch in testLoader:
            uIdx, iIdx, tIdx, reals = to_device(batch, self.device)
            preds = self.forward(uIdx, iIdx, tIdx)
            metrics.append(reals, preds)
        return metrics.compute()

    @torch.no_grad()
    def infer_full_tensor(self, fullLoader):
        self.eval()

        predTensor = torch.zeros(*fullLoader.dataset.tensor.shape, device=self.device)
        for batch in fullLoader:
            uIdx, iIdx, tIdx, reals = to_device(batch, self.device)
            preds = self.forward(uIdx, iIdx, tIdx)
            predTensor[tIdx, uIdx, iIdx] = preds
        return predTensor


    def _get_metrics_computer(self, size):
        if self.errorMetrics is None:
            self.errorMetrics = ErrorMetrics(size, self.device)
        return self.errorMetrics


    def setup_optimizer(self):
        self.optimizer = Adam(self.parameters(), lr=self.args.lr)
