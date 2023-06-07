import collections
import time
import argparse
import numpy as np
from loguru import logger
from utils.logger import setup_logger
from lightning import seed_everything
from datasets.tensor_dataset import DataModule
from utils.metrics import ErrMetrics, RankMetrics, get_reshape_string
from einops import einsum, rearrange
import torch


def khatri_rao(A, B):
    C = einsum(A, B, 'i r, j r -> i j r')
    C = rearrange(C, 'i j k -> (i j) k')
    return C


class CPALS:

    def __init__(self, args):

        self.args = args
        self.U = torch.rand((args.num_users, args.rank), device=args.device)
        self.S = torch.rand((args.num_items, args.rank), device=args.device)
        self.T = torch.rand((args.num_times, args.rank), device=args.device)


    def update_y_hat(self):
        self.Y_hat = einsum(self.T, self.U, self.S, 'time rank, user rank, item rank -> time user item')


    def train(self, dataModule, maxIter, tol, verbose):

        self.Y = dataModule.trainset.tensor.copy()
        self.Y = torch.from_numpy(self.Y).to(self.args.device)
        mask = self.Y > 0
        lmda = self.args.lmda
        eps = 1e-8

        for iter in range(1, maxIter + 1):

            # 更新User
            self.update_y_hat()
            t1 = einsum(self.Y, self.S, self.T, "time user item, item rank, time rank -> user rank")
            t2 = einsum(self.Y_hat, self.S, self.T, "time user item, item rank, time rank -> user rank")
            self.U *= (t1 / (t2 + lmda * self.U + eps))

            # 更新Item
            self.update_y_hat()
            t1 = einsum(self.Y, self.U, self.T, "time user item, user rank, time rank -> item rank")
            t2 = einsum(self.Y_hat, self.U, self.T, "time user item, user rank, time rank -> item rank")
            self.S *= (t1 / (t2 + lmda * self.S + eps))

            # 更新Time
            self.update_y_hat()
            t1 = einsum(self.Y, self.U, self.S, "time user item, user rank, item rank -> time rank")
            t2 = einsum(self.Y_hat, self.U, self.S, "time user item, user rank, item rank -> time rank")
            self.T *= (t1 / (t2 + lmda * self.T + eps))

            # Loss
            self.update_y_hat()
            loss = torch.sum((self.Y - self.Y_hat) ** 2 * mask) / torch.sum(mask)

            if iter % verbose == 0:
                print('Iter: {}, Loss: {:.4f}'.format(iter, loss))

            if loss < tol:
                break


    def predict(self):
        return self.Y_hat


class CPSGD:

    def __init__(self, args):
        self.args = args
        self.U = torch.rand((args.num_users, args.rank), device=args.device)
        self.S = torch.rand((args.num_items, args.rank), device=args.device)
        self.T = torch.rand((args.num_times, args.rank), device=args.device)

    def update_y_hat(self):
        self.Y_hat = einsum(self.T, self.U, self.S, 'time rank, user rank, item rank -> time user item')


    def train(self, dataModule, maxIter, tol, verbose):
        self.Y = dataModule.trainset.tensor.copy()
        self.Y = torch.from_numpy(self.Y).to(self.args.device)
        mask = self.Y > 0
        lr = self.args.lr

        for iter in range(1, maxIter + 1):

            # 更新User
            self.update_y_hat()
            err = (self.Y - self.Y_hat) * mask
            kr_prod = khatri_rao(self.S, self.T)
            err = rearrange(err, 'time user item -> user (item time)')
            self.U += lr * einsum(err, kr_prod, "user temp, temp rank -> user rank")

            # 更新Item
            self.update_y_hat()
            err = (self.Y - self.Y_hat) * mask
            kr_prod = khatri_rao(self.U, self.T)
            err = rearrange(err, 'time user item -> item (user time)')
            self.S += lr * einsum(err, kr_prod, "item temp, temp rank -> item rank")


            # 更新Time
            self.update_y_hat()
            err = (self.Y - self.Y_hat) * mask
            kr_prod = khatri_rao(self.U, self.S)
            err = rearrange(err, 'time user item -> time (user item)')
            self.T += lr * einsum(err, kr_prod, "time temp, temp rank -> time rank")

            # Loss
            self.update_y_hat()
            loss = torch.sum((self.Y - self.Y_hat) ** 2 * mask) / torch.sum(mask)

            if iter % verbose == 0:
                print('Iter: {}, Loss: {:.4f}'.format(iter, loss))

            if loss < tol:
                break


    def predict(self):
        return self.Y_hat


def BruteForcePerf(Y, Y_hat, args, runId):

    fmt_str = get_reshape_string(args.qtype, args.ktype)
    Y_hat = rearrange(Y_hat, f'time user item -> {fmt_str}')

    top20_metrics = RankMetrics(Y, topk=20, args=args)
    top50_metrics = RankMetrics(Y, topk=50, args=args)
    top75_metrics = RankMetrics(Y, topk=75, args=args)
    top100_metrics = RankMetrics(Y, topk=100, args=args)
    top200_metrics = RankMetrics(Y, topk=200, args=args)


    num_queries = 1
    for single_type in args.qtype:
        num_queries *= eval(f'args.num_{single_type}s')

    for i in range(num_queries):

        top20beam = torch.argsort(Y_hat[i], descending=True)[:20]
        top50beam = torch.argsort(Y_hat[i], descending=True)[:50]
        top75beam = torch.argsort(Y_hat[i], descending=True)[:75]
        top100beam = torch.argsort(Y_hat[i], descending=True)[:100]
        top200beam = torch.argsort(Y_hat[i], descending=True)[:200]

        top20_metrics.append(i, top20beam)
        top50_metrics.append(i, top50beam)
        top75_metrics.append(i, top75beam)
        top100_metrics.append(i, top100beam)
        top200_metrics.append(i, top200beam)

    top20_recall, top20_precision, top20_fmeasure = top20_metrics.compute()
    top50_recall, top50_precision, top50_fmeasure = top50_metrics.compute()
    top75_recall, top75_precision, top75_fmeasure = top75_metrics.compute()
    top100_recall, top100_precision, top100_fmeasure = top100_metrics.compute()
    top200_recall, top200_precision, top200_fmeasure = top200_metrics.compute()

    return top20_recall, top50_recall, top75_recall, top100_recall, top200_recall


def RunOnce(args, runId, runHash):
    seed = runId + args.seed
    seed_everything(args.seed)
    dataModule = DataModule(args, seed=seed)

    model = None
    if args.model == 'CPals':
        model = CPALS(args)
        model.train(dataModule, maxIter=300, tol=1e-4, verbose=20)
    elif args.model == 'CPsgd':
        model = CPSGD(args)
        model.train(dataModule, maxIter=1000, tol=1e-4, verbose=100)
    Y_hat = model.predict().ravel().cpu().numpy()
    Y = dataModule.testset.tensor.ravel()
    tNRMSE, tNMAE = ErrMetrics(Y, Y_hat)

    print('Run: {}, NRMSE: {:.4f}, NMAE: {:.4f}'.format(runId, tNRMSE, tNMAE))

    retrieval_Y_hat = model.predict().cpu()
    retrieval_Y = dataModule.fullset.tensor
    recalls = BruteForcePerf(retrieval_Y, retrieval_Y_hat, args, runId)

    logger.info(f"Run={runId} NRMSE={tNRMSE:.4f} NMAE={tNMAE:.4f}")
    logger.info(f"Run={runId} Recall@20={recalls[0]:.4f} Recall@50={recalls[1]:.4f} Recall@75={recalls[2]:.4f} Recall@100={recalls[3]:.4f} Recall@200={recalls[4]:.4f}")


    return {
        'NRMSE': tNRMSE,
        'NMAE': tNMAE,
        'Recall@20': recalls[0],
        'Recall@50': recalls[1],
        'Recall@75': recalls[2],
        'Recall@100': recalls[3],
        'Recall@200': recalls[4],
    }


def RunExperiments(args):

    metrics = collections.defaultdict(list)

    for runId in range(args.rounds):
        runHash = int(time.time())
        results = RunOnce(args, runId, runHash)

        for key in results:
            metrics[key].append(results[key])

    logger.info('*' * 10 + 'Experiment Results:' + '*' * 10)
    for key in metrics:
        logger.info(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Experiments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    # MetaTC
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--lmda', type=float, default=0.01)
    parser.add_argument('--window', type=int, default=12)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--model', type=str, default='CPsgd')

    # Retrieval
    parser.add_argument('--narys', type=int, default=2)
    parser.add_argument('--beam', type=int, default=50)
    parser.add_argument('--curr', type=int, default=40)
    parser.add_argument('--qtype', type=list, default=['user'])
    parser.add_argument('--ktype', type=list, default=['item', 'time'])

    # Dataset
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--num_users', type=int, default=32)
    parser.add_argument('--num_items', type=int, default=32)
    parser.add_argument('--num_times', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='mock')

    # Training
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # Setup Logger
    setup_logger(args, "math_baseline")

    # Record Experiments Config
    logger.info(args)

    # Run Experiments
    RunExperiments(args=args)
