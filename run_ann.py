import argparse
import time
import json

import einops
import torch as t
from loguru import logger
from datasets.tensor_dataset import DataModule
from utils.monitor import EarlyStopMonitor
from lightning import seed_everything
from utils.logger import setup_logger
from modules.tc import get_model
from modules.tc.meta_tc import MetaTC
from utils.metrics import RankMetrics
from modules.faiss_indexer import FaissIndexer
from utils.timer import PerfTimer


@t.no_grad()
def FaissRankPerf(model: MetaTC, dataModule, args, runId):
    top20_metrics = RankMetrics(dataModule.fullset.tensor, topk=20, args=args)
    top50_metrics = RankMetrics(dataModule.fullset.tensor, topk=50, args=args)
    top100_metrics = RankMetrics(dataModule.fullset.tensor, topk=100, args=args)

    # 思路：
    # 1. 構建索引
    indexer = FaissIndexer(args)
    indexer.train_indexer(model)

    # 2. 查詢結果並判斷結果的準確性

    perfTimer20 = PerfTimer()
    perfTimer50 = PerfTimer()
    perfTimer100 = PerfTimer()

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

            perfTimer100.start()
            top100_beam = indexer.find_topk_by_query(query, topk=100)
            perfTimer100.end()

            top20_metrics.append(i, top20_beam)
            top50_metrics.append(i, top50_beam)
            top100_metrics.append(i, top100_beam)

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

                perfTimer100.start()
                top100_beam = indexer.find_topk_by_query(query, topk=100)
                perfTimer100.end()

                top20_metrics.append(idx, top20_beam)
                top50_metrics.append(idx, top50_beam)
                top100_metrics.append(idx, top100_beam)

    top20_recall, top20_precision, top20_fmeasure = top20_metrics.compute()
    top50_recall, top50_precision, top50_fmeasure = top50_metrics.compute()
    top100_recall, top100_precision, top100_fmeasure = top100_metrics.compute()

    top20_ms, top50_ms, top100_ms = perfTimer20.compute(), perfTimer50.compute(), perfTimer100.compute()

    logger.info("***" * 22)
    logger.info(f"Round={runId} {args.index} Indexer Retrieval Performance")
    logger.info(f"Round={runId} Top  20  R={top20_recall:.4f} P={top20_precision:.4f} F={top20_fmeasure:.4f} T={top20_ms:.1f}ms")
    logger.info(f"Round={runId} Top  50  R={top50_recall:.4f} P={top50_precision:.4f} F={top50_fmeasure:.4f} T={top50_ms:.1f}ms")
    logger.info(f"Round={runId} Top 100  R={top100_recall:.4f} P={top100_precision:.4f} F={top100_fmeasure:.4f} T={top100_ms:.1f}ms")
    logger.info("***" * 22)


@t.no_grad()
def BruteForcePerf(model, dataModule, args, runId):
    qtype = args.qtype

    fullTensor = dataModule.fullset.tensor


    commonTimer = PerfTimer()
    commonTimer.start()
    predTensor = model.infer_full_tensor(dataModule.fullLoader())
    commonTimer.end()

    predTensor = einops.rearrange(predTensor, 'time user item -> (user item) (time)')

    top20_metrics = RankMetrics(fullTensor, topk=20, args=args)
    top50_metrics = RankMetrics(fullTensor, topk=50, args=args)
    top100_metrics = RankMetrics(fullTensor, topk=100, args=args)

    top20_timer = PerfTimer()
    top50_timer = PerfTimer()
    top100_timer = PerfTimer()

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

        top100_timer.start()
        top100beam = t.argsort(predTensor[i], descending=True)[:100]
        top100_timer.end()

        top20_metrics.append(i, top20beam)
        top50_metrics.append(i, top50beam)
        top100_metrics.append(i, top100beam)

    top20_recall, top20_precision, top20_fmeasure = top20_metrics.compute()
    top50_recall, top50_precision, top50_fmeasure = top50_metrics.compute()
    top100_recall, top100_precision, top100_fmeasure = top100_metrics.compute()

    top20_ms, top50_ms, top100_ms = top20_timer.compute(), top50_timer.compute(), top100_timer.compute()
    common_ms = commonTimer.compute() / num_queries

    logger.info("***" * 22)
    logger.info(f"Round={runId} Brute Force Retrieval Performance")
    logger.info(f"Round={runId} Top  20  R={top20_recall:.4f} P={top20_precision:.4f} F={top20_fmeasure:.4f} T={common_ms+top20_ms:.1f}ms")
    logger.info(f"Round={runId} Top  50  R={top50_recall:.4f} P={top50_precision:.4f} F={top50_fmeasure:.4f} T={common_ms+top50_ms:.1f}ms")
    logger.info(f"Round={runId} Top 100  R={top100_recall:.4f} P={top100_precision:.4f} F={top100_fmeasure:.4f} T={common_ms+top100_ms:.1f}ms")
    logger.info("***" * 22)


def RunOnce(args, runId, runHash):

    seed = runId + args.seed

    seed_everything(args.seed)

    # Initialize
    model = get_model(args)
    model.setup_optimizer()
    dataModule = DataModule(args, seed=seed)
    monitor = EarlyStopMonitor(args.patience)

    ################
    # Train MetaTC #
    ################

    for epoch in range(args.epochs):
        epoch_loss = model.train_one_epoch(dataModule.trainLoader())
        vNRMSE, vNMAE = model.valid_one_epoch(dataModule.validLoader())
        monitor.track(epoch, model.state_dict(), vNRMSE)

        logger.info(f"Round={runId} Epoch={epoch:02d} Loss={epoch_loss:.4f} vNRMSE={vNRMSE:.4f} vNMAE={vNMAE:.4f}")

        if monitor.early_stop():
            break

    # Test
    model.load_state_dict(monitor.params)
    tNRMSE, tNMAE = model.test_one_epoch(dataModule.testLoader())
    logger.info(f"Round={runId} tNRMSE={tNRMSE:.4f} tNMAE={tNMAE:.4f}")

    # Save
    t.save(model.state_dict(), f"./results/Faiss/{args.model}_{runHash}.pt")

    # Brute Force Performance
    BruteForcePerf(model, dataModule, args, runId)

    # Faiss-based ANN Searching
    FaissRankPerf(model, dataModule, args, runId)
    return tNRMSE, tNMAE


def RunExperiments(args):

    for runId in range(args.rounds):
        runHash = int(time.time())
        tNRMSE, tNMAE = RunOnce(args, runId, runHash)

        # Write to CSV
        fp = open(f"./results/baselines/{runHash}.json", "w")
        exp_logs = {
            "RunHash": runHash,
            "Model": args.model,
            "Dataset": args.dataset,
            "Window": args.window,
            "Rank": args.rank,
            "RunId": runId,
            "LR": args.lr,
            "Density": args.density,
            "tNRMSE": tNRMSE,
            "tNMAE": tNMAE,
        }
        json.dump(exp_logs, fp)
        fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Experiments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    # MetaTC
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--window', type=int, default=12)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--model', type=str, default='NCP')

    # Index Params
    parser.add_argument('--index', type=str, default='PQ')
    parser.add_argument('--qtype', type=list, default=['user', 'item'])
    parser.add_argument('--ktype', type=list, default=['time'])

    # Specify for LSH
    parser.add_argument('--LSHbits', type=int, default=32)

    # Specify for HNSW
    parser.add_argument('--HNSWx', type=int, default=8)

    # Specify for PQ
    parser.add_argument('--PQm', type=int, default=5)
    parser.add_argument('--PQbits', type=int, default=4)

    # Dataset
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--num_users', type=int, default=12)
    parser.add_argument('--num_items', type=int, default=12)
    parser.add_argument('--num_times', type=int, default=3000)
    parser.add_argument('--dataset', type=str, default='abilene')

    # Training
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # Setup Logger
    setup_logger(args, "Faiss")

    # Record Experiments Config
    logger.info(args)

    # Run Experiments
    RunExperiments(args=args)
