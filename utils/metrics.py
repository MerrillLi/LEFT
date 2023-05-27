import numpy as np
import torch
from einops import rearrange
from utils.reshape import get_reshape_string

class ErrorMetrics:

    def __init__(self, size, device):
        self.true = torch.zeros((size,)).to(device)
        self.pred = torch.zeros((size,)).to(device)
        self.size = size
        self.device = device
        self.writeIdx = 0

    def append(self, true, pred):
        assert len(true) == len(pred)
        self.true[self.writeIdx:self.writeIdx + len(true)] = true
        self.pred[self.writeIdx:self.writeIdx + len(pred)] = pred
        self.writeIdx += len(true)

    def compute(self):
        true = self.true[:self.writeIdx].cpu().numpy()
        pred = self.pred[:self.writeIdx].cpu().numpy()
        return ErrMetrics(true, pred)

    def reset(self):
        self.true = torch.zeros(self.size).to(self.device)
        self.pred = torch.zeros(self.size).to(self.device)
        self.writeIdx = 0

class RankMetrics:

    def _list_to_string(self, l):
        return ' '.join([str(x) for x in l])

    def __init__(self, fullTensor, topk, args):

        self.fullTensor = fullTensor
        self.args = args

        # rearrange fullTensor [Time, User, Item] by QType and KType
        tgt_fmt = get_reshape_string(self.args.qtype, self.args.ktype)
        self.fullTensor = rearrange(self.fullTensor, f'time user item -> {tgt_fmt}')

        self.recalls = []
        self.precisions = []
        self.fmeasures = []
        self.topk = topk

    def append(self, query, predSet):
        trueSet = np.argsort(self.fullTensor[query])[::-1][:self.topk]

        # convert to list

        trueSet = trueSet.tolist()

        if isinstance(predSet, torch.Tensor):
            predSet = predSet.cpu().numpy().tolist()
        elif isinstance(predSet, np.ndarray):
            predSet = predSet.tolist()

        recall = len(set(trueSet) & set(predSet)) / self.topk
        precision = len(set(trueSet) & set(predSet)) / len(set(predSet))
        fmeasure = 2 * recall * precision / (recall + precision + 1e-5)

        self.recalls.append(recall)
        self.precisions.append(precision)
        self.fmeasures.append(fmeasure)

    def compute(self):
        return np.mean(self.recalls), np.mean(self.precisions), np.mean(self.fmeasures)

    def reset(self):
        self.recalls = []
        self.precisions = []
        self.fmeasures = []



def ErrMetrics(true, pred):
    nonzeroIdx = true.nonzero()
    true = true[nonzeroIdx]
    pred = pred[nonzeroIdx]
    NRMSE = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum(true ** 2))
    NMAE = np.sum(np.abs(true - pred)) / np.sum(true)
    return float(NRMSE), float(NMAE)



