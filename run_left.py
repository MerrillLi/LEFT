import argparse
import time
import json
import einops
import numpy as np
import torch as t
from loguru import logger
from datasets.tensor_dataset import DataModule
from utils.monitor import EarlyStopMonitor
from lightning import seed_everything
from utils.logger import setup_logger
from modules.LEFT import LEFT
from utils.metrics import RankMetrics, TopKRegret
from utils.timer import PerfTimer
from utils.reshape import get_reshape_string

@t.no_grad()
def RegretPerf(model, args):
    qtype = args.qtype

    regrets = []

    if len(args.qtype) == 1:

        for i in range(eval(f'args.num_{qtype[0]}s')):
            q_index = [t.tensor(i)]
            _, scores_pred = model.beam_search(q_index, beam=args.beam, return_scores=True)
            _, scores_refs = model.beam_search(q_index, beam=args.beam*2, return_scores=True)

            scores_pred = scores_pred[:args.beam]
            scores_refs = scores_refs[:args.beam]
            regret = TopKRegret(scores_pred, scores_refs)
            regrets.append(regret)

    elif len(args.qtype) == 2:
        num_type0 = eval(f'args.num_{qtype[0]}s')
        num_type1 = eval(f'args.num_{qtype[1]}s')

        for i in range(num_type0):
            for j in range(num_type1):

                q_index = [t.tensor(i), t.tensor(j)]

                _, scores_pred = model.beam_search(q_index, beam=args.beam, return_scores=True)
                _, scores_refs = model.beam_search(q_index, beam=args.beam*2, return_scores=True)

                scores_pred = scores_pred[:args.beam]
                scores_refs = scores_refs[:args.beam]
                regret = TopKRegret(scores_pred, scores_refs)
                regrets.append(regret)

    regret = np.mean(regrets)
    return regret


@t.no_grad()
def RankPerf(model, dataModule, args, runId):
    qtype = args.qtype

    top20_metrics = RankMetrics(dataModule.fullset.tensor, topk=20, args=args)
    top50_metrics = RankMetrics(dataModule.fullset.tensor, topk=50, args=args)
    top100_metrics = RankMetrics(dataModule.fullset.tensor, topk=100, args=args)

    perfTimer20 = PerfTimer()
    perfTimer50 = PerfTimer()
    perfTimer100 = PerfTimer()

    if len(args.qtype) == 1:

        for i in range(eval(f'args.num_{qtype[0]}s')):
            q_index = [t.tensor(i)]

            perfTimer20.start()
            top20beam = model.beam_search(q_index, beam=50)[:20]
            perfTimer20.end()

            perfTimer50.start()
            top50beam = model.beam_search(q_index, beam=100)[:50]
            perfTimer50.end()

            perfTimer100.start()
            top100beam = model.beam_search(q_index, beam=200)[:100]
            perfTimer100.end()

            top20_metrics.append(i, top20beam)
            top50_metrics.append(i, top50beam)
            top100_metrics.append(i, top100beam)

    elif len(args.qtype) == 2:
        num_type0 = eval(f'args.num_{qtype[0]}s')
        num_type1 = eval(f'args.num_{qtype[1]}s')

        for i in range(num_type0):
            for j in range(num_type1):
                idx = i * num_type1 + j

                q_index = [t.tensor(i), t.tensor(j)]

                perfTimer20.start()
                top20beam = model.beam_search(q_index, beam=50)[:20]
                perfTimer20.end()

                perfTimer50.start()
                top50beam = model.beam_search(q_index, beam=100)[:50]
                perfTimer50.end()

                perfTimer100.start()
                top100beam = model.beam_search(q_index, beam=200)[:100]
                perfTimer100.end()

                top20_metrics.append(idx, top20beam)
                top50_metrics.append(idx, top50beam)
                top100_metrics.append(idx, top100beam)


    top20_recall, top20_precision, top20_fmeasure = top20_metrics.compute()
    top50_recall, top50_precision, top50_fmeasure = top50_metrics.compute()
    top100_recall, top100_precision, top100_fmeasure = top100_metrics.compute()


    top20_ms, top50_ms, top100_ms = perfTimer20.compute(), perfTimer50.compute(), perfTimer100.compute()

    logger.info("***" * 20)
    logger.info(f"Round={runId} Tree Indexer Retrieval Performance")
    logger.info(f"Round={runId} Top  20  R={top20_recall:.4f} P={top20_precision:.4f} F={top20_fmeasure:.4f} T={top20_ms:.1f}ms")
    logger.info(f"Round={runId} Top  50  R={top50_recall:.4f} P={top50_precision:.4f} F={top50_fmeasure:.4f} T={top50_ms:.1f}ms")
    logger.info(f"Round={runId} Top 100  R={top100_recall:.4f} P={top100_precision:.4f} F={top100_fmeasure:.4f} T={top100_ms:.1f}ms")
    logger.info("***" * 20)


@t.no_grad()
def BruteForcePerf(model, dataModule, args, runId):

    fullTensor = dataModule.fullset.tensor

    commonTimer = PerfTimer()

    commonTimer.start()
    predTensor = model.meta_tcom.infer_full_tensor(dataModule.fullLoader())
    commonTimer.end()

    predTensor = einops.rearrange(predTensor, f'time user item -> {get_reshape_string(args.qtype, args.ktype)}')

    top20_metrics = RankMetrics(fullTensor, topk=20, args=args)
    top50_metrics = RankMetrics(fullTensor, topk=50, args=args)
    top100_metrics = RankMetrics(fullTensor, topk=100, args=args)

    perfTimer20 = PerfTimer()
    perfTimer50 = PerfTimer()
    perfTimer100 = PerfTimer()

    num_queries = 1
    for single_type in args.qtype:
        num_queries *= eval(f'args.num_{single_type}s')

    for i in range(num_queries):

        perfTimer20.start()
        top20beam = t.argsort(predTensor[i], descending=True)[:20]
        perfTimer20.end()

        perfTimer50.start()
        top50beam = t.argsort(predTensor[i], descending=True)[:50]
        perfTimer50.end()

        perfTimer100.start()
        top100beam = t.argsort(predTensor[i], descending=True)[:100]
        perfTimer100.end()

        top20_metrics.append(i, top20beam)
        top50_metrics.append(i, top50beam)
        top100_metrics.append(i, top100beam)

    top20_recall, top20_precision, top20_fmeasure = top20_metrics.compute()
    top50_recall, top50_precision, top50_fmeasure = top50_metrics.compute()
    top100_recall, top100_precision, top100_fmeasure = top100_metrics.compute()
    top20_ms, top50_ms, top100_ms = perfTimer20.compute(), perfTimer50.compute(), perfTimer100.compute()
    common_time = commonTimer.compute() / num_queries

    logger.info("***" * 20)
    logger.info(f"Round={runId} Brute Force Retrieval Performance")
    logger.info(f"Round={runId} Top  20  R={top20_recall:.4f} P={top20_precision:.4f} F={top20_fmeasure:.4f} T={top20_ms+common_time:.1f}ms")
    logger.info(f"Round={runId} Top  50  R={top50_recall:.4f} P={top50_precision:.4f} F={top50_fmeasure:.4f} T={top50_ms+common_time:.1f}ms")
    logger.info(f"Round={runId} Top 100  R={top100_recall:.4f} P={top100_precision:.4f} F={top100_fmeasure:.4f} T={top100_ms+common_time:.1f}ms")
    logger.info("***" * 20)


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
    for epoch in range(args.epochs):

        epoch_loss = model.meta_tcom.train_one_epoch(dataModule.trainLoader())

        vNRMSE, vNMAE = model.meta_tcom.valid_one_epoch(dataModule.validLoader())
        monitor.track(epoch, model.meta_tcom.state_dict(), vNRMSE)

        print(f"Round={runId} Epoch={epoch:02d} Loss={epoch_loss:.4f} vNRMSE={vNRMSE:.4f} vNMAE={vNMAE:.4f}")

        if monitor.early_stop():
            break


    # Test
    model.meta_tcom.load_state_dict(monitor.params)
    tNRMSE, tNMAE = model.meta_tcom.test_one_epoch(dataModule.testLoader())
    logger.info(f"Round={runId} tNRMSE={tNRMSE:.4f} tNMAE={tNMAE:.4f}")

    # Save
    t.save(model.meta_tcom.state_dict(), f"./results/LEFT/{args.model}_{runHash}.pt")


    BruteForcePerf(model, dataModule, args, runId)

    ######################
    # Train Tree Indexer #
    ######################


    model.prepare_leaf_embeddings()

    for i in range(5000):

        if i % 100 == 0:
            RankPerf(model, dataModule, args, runId)

        model.tree_opt.zero_grad()
        q_index = [t.randint(low=0, high=eval(f"args.num_{qtype}s"), size=(16, )) for qtype in args.qtype]

        # 上溯逐层学习, 使用Curriculum控制学习的层次大小
        sbs_loss = model.stochastic_beam_search_loss(q_index, beam=args.beam, curriculum=i // args.curr)
        loss = sbs_loss
        loss.backward()

        # gradient clipping
        t.nn.utils.clip_grad_norm_(model.tree_embs.parameters(), 2.0)
        model.tree_opt.step()
        model.opt_scheduler.step()

        if i % 20 == 0:
            regret = RegretPerf(model, args)
            print(f"Round={runId} Iter={i} sbs_loss={sbs_loss:.4f} regret={regret:.4f}")


    return tNRMSE, tNMAE


def RunExperiments(args):

    tNRMSEs, tNMAEs = [], []
    for runId in range(args.rounds):
        runHash = int(time.time())
        tNRMSE, tNMAE = RunOnce(args, runId, runHash)
        tNRMSEs += [tNRMSE]
        tNMAEs += [tNMAE]


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
    parser.add_argument('--curr', type=int, default=80)
    # parser.add_argument('--qtype', type=list, default=['user', 'item'])
    # parser.add_argument('--ktype', type=list, default=['time'])
    parser.add_argument('--qtype', type=list, default=['user'])
    parser.add_argument('--ktype', type=list, default=['item', 'time'])

    # Dataset
    parser.add_argument('--density', type=float, default=0.02)
    parser.add_argument('--num_users', type=int, default=99)
    parser.add_argument('--num_items', type=int, default=99)
    parser.add_argument('--num_times', type=int, default=688)
    parser.add_argument('--dataset', type=str, default='seattle')

    # Training
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # Setup Logger
    setup_logger(args, "LEFT")

    # Record Experiments Config
    logger.info(args)

    # Run Experiments
    RunExperiments(args=args)

