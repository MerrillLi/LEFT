import argparse
import time
import os
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
from utils.timer import PerfTimer
from utils.reshape import get_reshape_string
import collections


@t.no_grad()
def RankPerf(model, dataModule, args, runId):
    model.eval()
    qtype = args.qtype

    top20_metrics = RankMetrics(dataModule.fullset.tensor, topk=20, args=args)
    top50_metrics = RankMetrics(dataModule.fullset.tensor, topk=50, args=args)
    top75_metrics = RankMetrics(dataModule.fullset.tensor, topk=75, args=args)
    top100_metrics = RankMetrics(dataModule.fullset.tensor, topk=100, args=args)
    top200_metrics = RankMetrics(dataModule.fullset.tensor, topk=200, args=args)

    perfTimer20 = PerfTimer()
    perfTimer50 = PerfTimer()
    perfTimer75 = PerfTimer()
    perfTimer100 = PerfTimer()
    perfTimer200 = PerfTimer()

    if len(args.qtype) == 1:

        for i in range(eval(f'args.num_{qtype[0]}s')):
            q_index = [t.tensor(i)]

            perfTimer20.start()
            top20beam = model.beam_search(q_index, beam=40)[:20]
            perfTimer20.end()

            perfTimer50.start()
            top50beam = model.beam_search(q_index, beam=100)[:50]
            perfTimer50.end()

            perfTimer75.start()
            top75beam = model.beam_search(q_index, beam=150)[:75]
            perfTimer75.end()

            perfTimer100.start()
            top100beam = model.beam_search(q_index, beam=200)[:100]
            perfTimer100.end()

            perfTimer200.start()
            top200beam = model.beam_search(q_index, beam=400)[:200]
            perfTimer200.end()

            top20_metrics.append(i, top20beam)
            top50_metrics.append(i, top50beam)
            top75_metrics.append(i, top75beam)
            top100_metrics.append(i, top100beam)
            top200_metrics.append(i, top200beam)

    elif len(args.qtype) == 2:
        num_type0 = eval(f'args.num_{qtype[0]}s')
        num_type1 = eval(f'args.num_{qtype[1]}s')

        for i in range(num_type0):
            for j in range(num_type1):
                idx = i * num_type1 + j

                q_index = [t.tensor(i), t.tensor(j)]

                perfTimer20.start()
                top20beam = model.beam_search(q_index, beam=40)[:20]
                perfTimer20.end()

                perfTimer50.start()
                top50beam = model.beam_search(q_index, beam=100)[:50]
                perfTimer50.end()

                perfTimer75.start()
                top75beam = model.beam_search(q_index, beam=150)[:75]
                perfTimer75.end()

                perfTimer100.start()
                top100beam = model.beam_search(q_index, beam=200)[:100]
                perfTimer100.end()

                perfTimer200.start()
                top200beam = model.beam_search(q_index, beam=400)[:200]
                perfTimer200.end()

                top20_metrics.append(idx, top20beam)
                top50_metrics.append(idx, top50beam)
                top75_metrics.append(idx, top75beam)
                top100_metrics.append(idx, top100beam)
                top200_metrics.append(idx, top200beam)


    top20_recall, top20_precision, top20_fmeasure = top20_metrics.compute()
    top50_recall, top50_precision, top50_fmeasure = top50_metrics.compute()
    top75_recall, top75_precision, top75_fmeasure = top75_metrics.compute()
    top100_recall, top100_precision, top100_fmeasure = top100_metrics.compute()
    top200_recall, top200_precision, top200_fmeasure = top200_metrics.compute()

    top20_ms = perfTimer20.compute()
    top50_ms = perfTimer50.compute()
    top75_ms = perfTimer75.compute()
    top100_ms = perfTimer100.compute()
    top200_ms = perfTimer200.compute()

    recalls = [top20_recall, top50_recall, top75_recall, top100_recall, top200_recall]
    ret_mss = [top20_ms, top50_ms, top75_ms, top100_ms, top200_ms]

    mss = [ret_ms for ret_ms in ret_mss]
    return recalls, mss


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


@t.no_grad()
def CoveragePerf(model, dataModule, args, runId):
    model.eval()
    qtype = args.qtype

    predTensor = model.meta_tcom.infer_full_tensor(dataModule.fullLoader())
    predTensor = einops.rearrange(predTensor, f'time user item -> {get_reshape_string(args.qtype, args.ktype)}')


    coverage_20 = []
    coverage_50 = []
    coverage_75 = []
    coverage_100 = []
    coverage_200 = []

    if len(args.qtype) == 1:

        for i in range(eval(f'args.num_{qtype[0]}s')):
            q_index = [t.tensor(i)]

            top20beam_left = model.beam_search(q_index, beam=40)[:20]
            top50beam_left = model.beam_search(q_index, beam=100)[:50]
            top75beam_left = model.beam_search(q_index, beam=150)[:75]
            top100beam_left = model.beam_search(q_index, beam=200)[:100]
            top200beam_left = model.beam_search(q_index, beam=400)[:200]

            top20beam_bf = t.argsort(predTensor[i], descending=True)[:20]
            top50beam_bf = t.argsort(predTensor[i], descending=True)[:50]
            top75beam_bf = t.argsort(predTensor[i], descending=True)[:75]
            top100beam_bf = t.argsort(predTensor[i], descending=True)[:100]
            top200beam_bf = t.argsort(predTensor[i], descending=True)[:200]

            coverage_20.append(len(set(top20beam_left).intersection(set(top20beam_bf))) / 20)
            coverage_50.append(len(set(top50beam_left).intersection(set(top50beam_bf))) / 50)
            coverage_75.append(len(set(top75beam_left).intersection(set(top75beam_bf))) / 75)
            coverage_100.append(len(set(top100beam_left).intersection(set(top100beam_bf))) / 100)
            coverage_200.append(len(set(top200beam_left).intersection(set(top200beam_bf))) / 200)


    elif len(args.qtype) == 2:
        num_type0 = eval(f'args.num_{qtype[0]}s')
        num_type1 = eval(f'args.num_{qtype[1]}s')

        for i in range(num_type0):
            for j in range(num_type1):
                idx = i * num_type1 + j

                q_index = [t.tensor(i), t.tensor(j)]

                top20beam_left = model.beam_search(q_index, beam=40)[:20]
                top50beam_left = model.beam_search(q_index, beam=100)[:50]
                top75beam_left = model.beam_search(q_index, beam=150)[:75]
                top100beam_left = model.beam_search(q_index, beam=200)[:100]
                top200beam_left = model.beam_search(q_index, beam=400)[:200]

                top20beam_bf = t.argsort(predTensor[i], descending=True)[:20]
                top50beam_bf = t.argsort(predTensor[i], descending=True)[:50]
                top75beam_bf = t.argsort(predTensor[i], descending=True)[:75]
                top100beam_bf = t.argsort(predTensor[i], descending=True)[:100]
                top200beam_bf = t.argsort(predTensor[i], descending=True)[:200]

                coverage_20.append(len(set(top20beam_left).intersection(set(top20beam_bf))) / 20)
                coverage_50.append(len(set(top50beam_left).intersection(set(top50beam_bf))) / 50)
                coverage_75.append(len(set(top75beam_left).intersection(set(top75beam_bf))) / 75)
                coverage_100.append(len(set(top100beam_left).intersection(set(top100beam_bf))) / 100)
                coverage_200.append(len(set(top200beam_left).intersection(set(top200beam_bf))) / 200)


    coverage_20 = np.mean(coverage_20)
    coverage_50 = np.mean(coverage_50)
    coverage_75 = np.mean(coverage_75)
    coverage_100 = np.mean(coverage_100)
    coverage_200 = np.mean(coverage_200)

    return coverage_20, coverage_50, coverage_75, coverage_100, coverage_200


def RunOnce(args, runId, runHash):

    seed = runId + args.seed
    seed_everything(seed)

    # Initialize
    model = LEFT(args).to(args.device)

    model.setup_optimizer()
    dataModule = DataModule(args, seed=seed)
    monitor = EarlyStopMonitor(args.patience)

    ################
    # Train MetaTC #
    ################
    expected_ckpt_name = f"{args.dataset}_d{args.density}_{args.model}_r{args.rank}_s{seed}.pt"
    saved_model_path = os.path.join(f"./saved/ntc/{args.model}", expected_ckpt_name)

    if os.path.exists(saved_model_path):
        logger.info(f"Round={runId} Loading Pretrained Model")
        model.meta_tcom.load_state_dict(t.load(saved_model_path))
        monitor.params = model.meta_tcom.state_dict()
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
    logger.info(f"Round={runId} tNRMSE={tNRMSE:.4f} tNMAE={tNMAE:.4f}")

    recalls = BruteForcePerf(model, dataModule, args, runId)

    logger.info(f"Run={runId} NRMSE={tNRMSE:.4f} NMAE={tNMAE:.4f}")
    logger.info(f"Run={runId} Recall@20={recalls[0]:.4f} Recall@50={recalls[1]:.4f} Recall@75={recalls[2]:.4f} Recall@100={recalls[3]:.4f} Recall@200={recalls[4]:.4f}")

    return_metrics = {
        "tNRMSE": tNRMSE,
        "tNMAE": tNMAE,
        "BF-Recall@20": recalls[0],
        "BF-Recall@50": recalls[1],
        "BF-Recall@75": recalls[2],
        "BF-Recall@100": recalls[3],
        "BF-Recall@200": recalls[4],
    }


    ######################
    # Train Tree Indexer #
    ######################
    model.setup_indexer()

    # Prepare Early Stop Monitor
    index_monitor = EarlyStopMonitor(5)

    # Train Indexer
    for i in range(5000):

        model.train()

        # 创建输入
        q_index = [t.randint(low=0, high=eval(f"args.num_{qtype}s"), size=(16, )) for qtype in args.qtype]

        # 上溯逐层学习, 使用Curriculum控制学习的层次大小
        curriculum = i // args.curr
        sbs_loss, heap_acc = model.stochastic_beam_search_loss(q_index, beam=args.beam, curriculum=curriculum)
        loss = sbs_loss

        # 梯度下降
        model.indexer.optimizer.zero_grad()
        loss.backward()
        t.nn.utils.clip_grad_norm_(model.indexer.parameters(), 2.0)

        model.indexer.optimizer.step()
        model.indexer.scheduler.step()

        if i % 20 == 0:
            print(f"Round={runId} Iter={i} sbs_loss={sbs_loss:.4f} heap_acc={heap_acc:.4f}")
            # Early Stop
            if i > 200:
                index_monitor.track(i, model.indexer.state_dict(), -heap_acc)

        if index_monitor.early_stop():
            break

        if i % 100 == 0:
            recalls, _ = RankPerf(model, dataModule, args, runId)
            print(f"Run={runId} Recall@20={recalls[0]:.4f} Recall@50={recalls[1]:.4f} Recall@75={recalls[2]:.4f} Recall@100={recalls[3]:.4f} Recall@200={recalls[4]:.4f}")

    # Test Indexer
    model.indexer.load_state_dict(index_monitor.params)
    recalls, mss = RankPerf(model, dataModule, args, runId)
    print(f"Run={runId} Recall@20={recalls[0]:.4f} Recall@50={recalls[1]:.4f} Recall@75={recalls[2]:.4f} Recall@100={recalls[3]:.4f} Recall@200={recalls[4]:.4f}")

    return_metrics.update({
        "LEFT-Recall@20": recalls[0],
        "LEFT-Recall@50": recalls[1],
        "LEFT-Recall@75": recalls[2],
        "LEFT-Recall@100": recalls[3],
        "LEFT-Recall@200": recalls[4],
        "LEFT-Time@20": mss[0],
        "LEFT-Time@50": mss[1],
        "LEFT-Time@75": mss[2],
        "LEFT-Time@100": mss[3],
        "LEFT-Time@200": mss[4],
    })

    coverages = CoveragePerf(model, dataModule, args, runId)
    return_metrics.update({
        "LEFT-Coverage@20": coverages[0],
        "LEFT-Coverage@50": coverages[1],
        "LEFT-Coverage@75": coverages[2],
        "LEFT-Coverage@100": coverages[3],
        "LEFT-Coverage@200": coverages[4],
    })

    return return_metrics


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
    parser.add_argument('--rounds', type=int, default=1)

    # MetaTC
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--model', type=str, default='LTP')

    # LEFT
    parser.add_argument('--narys', type=int, default=2)
    parser.add_argument('--beam', type=int, default=50)
    parser.add_argument('--curr', type=int, default=50)
    parser.add_argument('--qtype', type=list, default=['time'])
    parser.add_argument('--ktype', type=list, default=['user', 'item'])
    parser.add_argument('--tree_type', type=str, default='balanced')

    # Dataset
    parser.add_argument('--density', type=float, default=0.02)
    parser.add_argument('--num_times', type=int, default=65)
    parser.add_argument('--num_users', type=int, default=50)
    parser.add_argument('--num_items', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='mock')

    # Training
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--amp', type=bool, default=False)

    args = parser.parse_args()

    # Setup Logger
    setup_logger(args, "LEFT")

    # Record Experiments Config
    logger.info(args)

    # Run Experiments
    RunExperiments(args=args)

