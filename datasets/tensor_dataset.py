import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning import LightningDataModule

def get_tensor(dataset):
    return np.load(f'./datasets/{dataset}.npy').astype('float32')


def get_datasets(args):

    # get tensor
    data = get_tensor(args.dataset)[:args.num_times]
    trunc = np.percentile(data, q=99.9)
    data[data > trunc] = trunc
    data /= trunc

    # split tensors
    tIdx, rIdx, cIdx = data.nonzero()
    tSize = int(args.density * len(rIdx))
    vSize = int(0.05 * len(rIdx))
    split_indices = [tSize, tSize+vSize]
    p = np.random.permutation(len(rIdx))
    rIdx, cIdx, tIdx = rIdx[p], cIdx[p], tIdx[p]
    trRIdx, vaRIdx, tsRIdx = np.split(rIdx, split_indices)
    trCIdx, vaCIdx, tsCIdx = np.split(cIdx, split_indices)
    trTIdx, vaTIdx, tsTIdx = np.split(tIdx, split_indices)

    trainTensor = np.zeros_like(data)
    validTensor = np.zeros_like(data)
    testTensor = np.zeros_like(data)

    # copy data to train.valid.test tensor
    trainTensor[trTIdx, trRIdx, trCIdx] = data[trTIdx, trRIdx, trCIdx]
    validTensor[vaTIdx, vaRIdx, vaCIdx] = data[vaTIdx, vaRIdx, vaCIdx]
    testTensor[tsTIdx, tsRIdx, tsCIdx] = data[tsTIdx, tsRIdx, tsCIdx]
    full_tensor = np.copy(data)

    trainset = TensorDataset(trainTensor)
    validset = TensorDataset(validTensor)
    testset = TensorDataset(testTensor)
    fullset = TensorDataset(full_tensor)

    return trainset, validset, testset, fullset


class DataModule:

    def __init__(self, args, seed):

        np.random.seed(seed)
        trainset, validset, testset, fullset = get_datasets(args)

        self.trainset = trainset
        self.validset = validset
        self.testset = testset
        self.fullset = fullset

        self.trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=8)
        self.validloader = DataLoader(validset, batch_size=args.bs * 8, shuffle=False)
        self.testloader = DataLoader(testset, batch_size=args.bs * 8, shuffle=False)
        self.fullloader = DataLoader(fullset, batch_size=args.bs * 8, shuffle=False)

    def trainLoader(self):
        return self.trainloader

    def validLoader(self):
        return self.validloader

    def testLoader(self):
        return self.testloader

    def fullLoader(self):
        return self.fullloader


class TensorDataset(Dataset):

    def __init__(self, tensor:np.ndarray):
        self.tensor = tensor
        self.indices = np.array(tensor.nonzero()).transpose()

    def __getitem__(self, idx):
        tIdx, rIdx, cIdx,  = self.indices[idx]
        values = self.tensor[tIdx, rIdx, cIdx]
        return rIdx, cIdx, tIdx, values


    def __len__(self):
        return self.indices.shape[0]

