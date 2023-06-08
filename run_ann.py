import os
import time
import einops
import argparse
import torch as t
import collections
import numpy as np
from loguru import logger
from modules.tc import get_model
from utils.timer import PerfTimer
from lightning import seed_everything
from utils.logger import setup_logger
from modules.tc.meta_tc import MetaTC
from utils.monitor import EarlyStopMonitor
from datasets.tensor_dataset import DataModule
from modules.indexer.faiss_indexer import FaissIndexer
from utils.metrics import RankMetrics, get_reshape_string


@t.no_grad()
def FaissRankPerf(model: MetaTC, dataModule, args, runId):

    top20_metrics = RankMetrics(dataModule.fullset.tensor, topk=20, args=args)
    top50_metrics = RankMetrics(dataModule.fullset.tensor, topk=50, args=args)
    top75_metrics = RankMetrics(dataModule.fullset.tensor, topk=75, args=args)
    top100_metrics = RankMetrics(dataModule.fullset.tensor, topk=100, args=args)
    top200_metrics = RankMetrics(dataModule.fullset.tensor, topk=200, args=args)

    # 思路：
    # 1. 構建索引

    commonTimer = PerfTimer()

    commonTimer.start()
    indexer = FaissIndexer(args)
    indexer.train_indexer(model)
    commonTimer.end()


    # 2. 查詢結果並判斷結果的準確性
    perfTimer20 = PerfTimer()
    perfTimer50 = PerfTimer()
    perfTimer75 = PerfTimer()
    perfTimer100 = PerfTimer()
    perfTimer200 = PerfTimer()

    if len(args.qtype) == 1:
        qtype = args.qtype[0]
        num_type = eval(f'args.num_{qtype}s')
        typeIndex = t.arange(0, num_type, device=model.device)
        query_embeds = model.get_embeddings(typeIndex, select=qtype).cpu().numpy()

        for i in range(num_type):
            query = query_embeds[i]
            perfTimer20.start()
            top20_beam = indexer.find_topk_by_query(query, topk=20)
            perfTimer20.end()

            perfTimer50.start()
            top50_beam = indexer.find_topk_by_query(query, topk=50)
            perfTimer50.end()

            perfTimer75.start()
            top75_beam = indexer.find_topk_by_query(query, topk=75)
            perfTimer75.end()

            perfTimer100.start()
            top100_beam = indexer.find_topk_by_query(query, topk=100)
            perfTimer100.end()

            perfTimer200.start()
            top200_beam = indexer.find_topk_by_query(query, topk=200)
            perfTimer200.end()

            top20_metrics.append(i, top20_beam)
            top50_metrics.append(i, top50_beam)
            top75_metrics.append(i, top75_beam)
            top100_metrics.append(i, top100_beam)
            top200_metrics.append(i, top200_beam)

    elif len(args.qtype) == 2:

        qtype1 = args.qtype[0]
        qtype2 = args.qtype[1]
        num_type1 = eval(f'args.num_{qtype1}s')
        num_type2 = eval(f'args.num_{qtype2}s')

        type1Index = t.arange(0, num_type1, device=model.device)
        type2Index = t.arange(0, num_type2, device=model.device)

        query1_embeds = model.get_embeddings(type1Index, select=qtype1).cpu().numpy()
        query2_embeds = model.get_embeddings(type2Index, select=qtype2).cpu().numpy()

        for i in range(num_type1):
            for j in range(num_type2):
                idx = i * num_type2 + j
                query = query1_embeds[i] * query2_embeds[j]

                perfTimer20.start()
                top20_beam = indexer.find_topk_by_query(query, topk=20)
                perfTimer20.end()

                perfTimer50.start()
                top50_beam = indexer.find_topk_by_query(query, topk=50)
                perfTimer50.end()

                perfTimer75.start()
                top75_beam = indexer.find_topk_by_query(query, topk=75)
                perfTimer75.end()

                perfTimer100.start()
                top100_beam = indexer.find_topk_by_query(query, topk=100)
                perfTimer100.end()

                perfTimer200.start()
                top200_beam = indexer.find_topk_by_query(query, topk=200)
                perfTimer200.end()

                top20_metrics.append(idx, top20_beam)
                top50_metrics.append(idx, top50_beam)
                top75_metrics.append(idx, top75_beam)
                top100_metrics.append(idx, top100_beam)
                top200_metrics.append(idx, top200_beam)

    top20_recall, top20_precision, top20_fmeasure = top20_metrics.compute()
    top50_recall, top50_precision, top50_fmeasure = top50_metrics.compute()
    top75_recall, top75_precision, top75_fmeasure = top75_metrics.compute()
    top100_recall, top100_precision, top100_fmeasure = top100_metrics.compute()
    top200_recall, top200_precision, top200_fmeasure = top200_metrics.compute()

    top20_ms= perfTimer20.compute()
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

    fullTensor = dataModule.fullset.tensor

    commonTimer = PerfTimer()
    commonTimer.start()
    predTensor = model.infer_full_tensor(dataModule.fullLoader())
    commonTimer.end()

    fmt_str = get_reshape_string(args.qtype, args.ktype)
    predTensor = einops.rearrange(predTensor, f"time user item -> {fmt_str}")

    top20_metrics = RankMetrics(fullTensor, topk=20, args=args)
    top50_metrics = RankMetrics(fullTensor, topk=50, args=args)
    top75_metrics = RankMetrics(fullTensor, topk=75, args=args)
    top100_metrics = RankMetrics(fullTensor, topk=100, args=args)
    top200_metrics = RankMetrics(fullTensor, topk=200, args=args)

    top20_timer = PerfTimer()
    top50_timer = PerfTimer()
    top75_timer = PerfTimer()
    top100_timer = PerfTimer()
    top200_timer = PerfTimer()

    num_queries = 1
    for single_type in args.qtype:
        num_queries *= eval(f'args.num_{single_type}s')

    for i in range(num_queries):

        top20_timer.start()
        top20beam = t.argsort(predTensor[i], descending=True)[:20]
        top20_timer.end()

        top50_timer.start()
        top50beam = t.argsort(predTensor[i], descending=True)[:50]
        top50_timer.end()

        top75_timer.start()
        top75beam = t.argsort(predTensor[i], descending=True)[:75]
        top75_timer.end()

        top100_timer.start()
        top100beam = t.argsort(predTensor[i], descending=True)[:100]
        top100_timer.end()

        top200_timer.start()
        top200beam = t.argsort(predTensor[i], descending=True)[:200]
        top200_timer.end()

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

    top20_ms = top20_timer.compute()
    top50_ms = top50_timer.compute()
    top75_ms = top75_timer.compute()
    top100_ms = top100_timer.compute()
    top200_ms = top200_timer.compute()

    common_ms = commonTimer.compute() / num_queries

    recalls = [top20_recall, top50_recall, top75_recall, top100_recall, top200_recall]
    ret_mss = [top20_ms, top50_ms, top75_ms, top100_ms, top200_ms]
    mss = [ ret_ms + common_ms for ret_ms in ret_mss]

    return recalls, mss


def RunOnce(args, runId, runHash):

    # Initialize
    seed = runId + args.seed
    seed_everything(seed)

    model = get_model(args)
    model.setup_optimizer()
    dataModule = DataModule(args, seed=seed)
    monitor = EarlyStopMonitor(args.patience)

    ################
    # Train MetaTC #
    ################
    expected_ckpt_name = f"{args.dataset}_{args.model}_{args.rank}_{seed}.pt"
    saved_model_path = os.path.join("./saved", expected_ckpt_name)

    if os.path.exists(saved_model_path):
        model.load_state_dict(t.load(saved_model_path))
        logger.info(f"Loaded {saved_model_path}")
        monitor.params = model.state_dict()

    else:
        for epoch in range(args.epochs):
            epoch_loss = model.train_one_epoch(dataModule.trainLoader())
            vNRMSE, vNMAE = model.valid_one_epoch(dataModule.validLoader())
            monitor.track(epoch, model.state_dict(), vNRMSE)

            if epoch % 10 == 0:
                print(f"Round={runId} Epoch={epoch:02d} Loss={epoch_loss:.4f} vNRMSE={vNRMSE:.4f} vNMAE={vNMAE:.4f}")

            if monitor.early_stop():
                break

        t.save(monitor.params, saved_model_path)

    # Test
    model.load_state_dict(monitor.params)
    tNRMSE, tNMAE = model.test_one_epoch(dataModule.testLoader())

    # Brute Force Performance
    recalls, mss = BruteForcePerf(model, dataModule, args, runId)
    logger.info(f"Round={runId} tNRMSE={tNRMSE:.4f} tNMAE={tNMAE:.4f}")
    logger.info(f"Run={runId} Recall@20={recalls[0]:.4f} Recall@50={recalls[1]:.4f} Recall@75={recalls[2]:.4f} Recall@100={recalls[3]:.4f} Recall@200={recalls[4]:.4f}")

    brute_metrics = {
        'NRMSE': tNRMSE,
        'NMAE': tNMAE,
        'BF-Recall@20': recalls[0],
        'BF-Recall@50': recalls[1],
        'BF-Recall@75': recalls[2],
        'BF-Recall@100': recalls[3],
        'BF-Recall@200': recalls[4],
        'BF-Time@20': mss[0],
        'BF-Time@50': mss[1],
        'BF-Time@75': mss[2],
        'BF-Time@100': mss[3],
        'BF-Time@200': mss[4],
    }

    # Faiss-based ANN Searching
    recalls, mss = FaissRankPerf(model, dataModule, args, runId)


    faiss_metrics = {
        f'{args.index}-Recall@20': recalls[0],
        f'{args.index}-Recall@50': recalls[1],
        f'{args.index}-Recall@75': recalls[2],
        f'{args.index}-Recall@100': recalls[3],
        f'{args.index}-Recall@200': recalls[4],
        f'{args.index}-Time@20': mss[0],
        f'{args.index}-Time@50': mss[1],
        f'{args.index}-Time@75': mss[2],
        f'{args.index}-Time@100': mss[3],
        f'{args.index}-Time@200': mss[4],
    }

    metrics = {}
    metrics.update(brute_metrics)
    metrics.update(faiss_metrics)

    return metrics


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
    parser.add_argument('--rank', type=int, default=30)
    parser.add_argument('--window', type=int, default=12)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--model', type=str, default='NCP')

    # Index Params
    parser.add_argument('--index', type=str, default='PQ')
    parser.add_argument('--qtype', type=list, default=['user'])
    parser.add_argument('--ktype', type=list, default=['item', 'time'])

    # Specify for LSH
    parser.add_argument('--LSHbits', type=int, default=64)

    # Specify for HNSW
    parser.add_argument('--HNSWx', type=int, default=128)
    parser.add_argument('--hierarchy', type=int, default=4)
    parser.add_argument('--efConstruction', type=int, default=1600)
    parser.add_argument('--efSearch', type=int, default=800)

    # Specify for PQ
    parser.add_argument('--PQm', type=int, default=10)
    parser.add_argument('--PQbits', type=int, default=8)

    # Dataset
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--num_users', type=int, default=144)
    parser.add_argument('--num_items', type=int, default=168)
    parser.add_argument('--num_times', type=int, default=288)
    parser.add_argument('--dataset', type=str, default='abilene_rs')

    # Training
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp', type=bool, default=True)

    args = parser.parse_args()

    # Setup Logger
    setup_logger(args, f"Faiss/{args.index}")

    # Record Experiments Config
    logger.info(args)

    # Run Experiments
    RunExperiments(args=args)
