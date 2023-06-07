import os
import argparse
import time
import einops
import numpy as np
import torch as t
from loguru import logger
from datasets.tensor_dataset import DataModule
from utils.monitor import EarlyStopMonitor
from lightning import seed_everything
from utils.logger import setup_logger
from modules.LEFT import LEFT
from utils.metrics import RankMetrics
from utils.reshape import get_reshape_string
import collections

@t.no_grad()
def BruteForcePerf(model, dataModule, args, runId):
    model.eval()
    fullTensor = dataModule.fullset.tensor
    predTensor = model.meta_tcom.infer_full_tensor(dataModule.fullLoader())

    predTensor = einops.rearrange(predTensor, f'time user item -> {get_reshape_string(args.qtype, args.ktype)}')

    top20_metrics = RankMetrics(fullTensor, topk=20, args=args)
    top50_metrics = RankMetrics(fullTensor, topk=50, args=args)
    top75_metrics = RankMetrics(fullTensor, topk=75, args=args)
    top100_metrics = RankMetrics(fullTensor, topk=100, args=args)
    top200_metrics = RankMetrics(fullTensor, topk=200, args=args)

    num_queries = 1
    for single_type in args.qtype:
        num_queries *= eval(f'args.num_{single_type}s')

    for i in range(num_queries):

        top20beam = t.argsort(predTensor[i], descending=True)[:20]
        top50beam = t.argsort(predTensor[i], descending=True)[:50]
        top75beam = t.argsort(predTensor[i], descending=True)[:75]
        top100beam = t.argsort(predTensor[i], descending=True)[:100]
        top200beam = t.argsort(predTensor[i], descending=True)[:200]

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

    # Initialize
    model = LEFT(args).to(args.device)
    model.setup_optimizer()
    dataModule = DataModule(args, seed=seed)
    monitor = EarlyStopMonitor(args.patience)


    ################
    # Train MetaTC #
    ################
    expected_ckpt_name = f"{args.model}_{args.rank}_{seed}.pt"
    saved_model_path = os.path.join("./saved/ntc", expected_ckpt_name)

    if os.path.exists(saved_model_path):
        model.load_state_dict(t.load(saved_model_path))
        logger.info(f"Loaded {saved_model_path}")
        monitor.params = model.state_dict()

    else:
        for epoch in range(args.epochs):
            epoch_loss = model.meta_tcom.train_one_epoch(dataModule.trainLoader())
            vNRMSE, vNMAE = model.meta_tcom.valid_one_epoch(dataModule.validLoader())
            monitor.track(epoch, model.meta_tcom.state_dict(), vNRMSE)

            if epoch % 10 == 0:
                print(f"Round={runId} Epoch={epoch:02d} Loss={epoch_loss:.4f} vNRMSE={vNRMSE:.4f} vNMAE={vNMAE:.4f}")

            if monitor.early_stop():
                break

        t.save(monitor.params, saved_model_path)

    # Test
    model.meta_tcom.load_state_dict(monitor.params)
    tNRMSE, tNMAE = model.meta_tcom.test_one_epoch(dataModule.testLoader())
    recalls = BruteForcePerf(model, dataModule, args, runId)

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
    parser.add_argument('--window', type=int, default=12)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--model', type=str, default='LTP')

    # LEFT
    parser.add_argument('--narys', type=int, default=2)
    parser.add_argument('--beam', type=int, default=50)
    parser.add_argument('--curr', type=int, default=40)
    # parser.add_argument('--qtype', type=list, default=['user', 'item'])
    # parser.add_argument('--ktype', type=list, default=['time'])
    parser.add_argument('--qtype', type=list, default=['user'])
    parser.add_argument('--ktype', type=list, default=['item', 'time'])

    # Dataset
    parser.add_argument('--density', type=float, default=0.02)
    parser.add_argument('--num_users', type=int, default=144)
    parser.add_argument('--num_items', type=int, default=168)
    parser.add_argument('--num_times', type=int, default=288)
    parser.add_argument('--dataset', type=str, default='abilene_rs')

    # Training
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # Setup Logger
    setup_logger(args, "ntc_baselines")

    # Record Experiments Config
    logger.info(args)

    # Run Experiments
    RunExperiments(args=args)

