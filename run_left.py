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
from modules.LEFT import LEFT
from utils.metrics import RankMetrics


def RankPerf(model, dataModule, args, runId):
    qtype = args.qtype

    top20_metrics = RankMetrics(dataModule.fullset.tensor, topk=20, args=args)
    top50_metrics = RankMetrics(dataModule.fullset.tensor, topk=50, args=args)
    top100_metrics = RankMetrics(dataModule.fullset.tensor, topk=100, args=args)

    cnt = 0
    for i in range(eval(f'args.num_{qtype[0]}s')):
        for j in range(eval(f'args.num_{qtype[1]}s')):
            q_index = [t.tensor(i), t.tensor(j)]
            top20beam = model.beam_search(q_index, beam=100)[:20]
            top50beam = model.beam_search(q_index, beam=200)[:50]
            top100beam = model.beam_search(q_index, beam=400)[:100]

            top20_metrics.append(cnt, top20beam)
            top50_metrics.append(cnt, top50beam)
            top100_metrics.append(cnt, top100beam)

            cnt += 1

    top20_recall, top20_precision, top20_fmeasure = top20_metrics.compute()
    top50_recall, top50_precision, top50_fmeasure = top50_metrics.compute()
    top100_recall, top100_precision, top100_fmeasure = top100_metrics.compute()

    logger.info("***" * 22)
    logger.info(f"Round={runId} Tree Indexer Retrieval Performance")
    logger.info(f"Round={runId} Top20  Recall={top20_recall:.4f} Precision={top20_precision:.4f} F-Measure={top20_fmeasure:.4f}")
    logger.info(f"Round={runId} Top50  Recall={top50_recall:.4f} Precision={top50_precision:.4f} F-Measure={top50_fmeasure:.4f}")
    logger.info(f"Round={runId} Top100 Recall={top100_recall:.4f} Precision={top100_precision:.4f} F-Measure={top100_fmeasure:.4f}")
    logger.info("***" * 22)


def BruteForcePerf(model, dataModule, args, runId):
    qtype = args.qtype

    fullTensor = dataModule.fullset.tensor
    predTensor = model.meta_tcom.infer_full_tensor(dataModule.fullLoader())

    predTensor = einops.rearrange(predTensor, 'time user item -> (user item) (time)')

    top20_metrics = RankMetrics(fullTensor, topk=20, args=args)
    top50_metrics = RankMetrics(fullTensor, topk=50, args=args)
    top100_metrics = RankMetrics(fullTensor, topk=100, args=args)

    for i in range(eval(f'args.num_{qtype[0]}s') * eval(f'args.num_{qtype[1]}s')):
        top20beam = t.argsort(predTensor[i], descending=True)[:20]
        top50beam = t.argsort(predTensor[i], descending=True)[:50]
        top100beam = t.argsort(predTensor[i], descending=True)[:100]

        top20_metrics.append(i, top20beam)
        top50_metrics.append(i, top50beam)
        top100_metrics.append(i, top100beam)

    top20_recall, top20_precision, top20_fmeasure = top20_metrics.compute()
    top50_recall, top50_precision, top50_fmeasure = top50_metrics.compute()
    top100_recall, top100_precision, top100_fmeasure = top100_metrics.compute()

    logger.info("***" * 22)
    logger.info(f"Round={runId} Brute Force Retrieval Performance")
    logger.info(f"Round={runId} Top20  Recall={top20_recall:.4f} Precision={top20_precision:.4f} F-Measure={top20_fmeasure:.4f}")
    logger.info(f"Round={runId} Top50  Recall={top50_recall:.4f} Precision={top50_precision:.4f} F-Measure={top50_fmeasure:.4f}")
    logger.info(f"Round={runId} Top100 Recall={top100_recall:.4f} Precision={top100_precision:.4f} F-Measure={top100_fmeasure:.4f}")
    logger.info("***" * 22)


def RunOnce(args, runId, runHash):

    seed = runId + args.seed

    seed_everything(args.seed)

    # Initialize
    model = LEFT(args)
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

        logger.info(f"Round={runId} Epoch={epoch:02d} Loss={epoch_loss:.4f} vNRMSE={vNRMSE:.4f} vNMAE={vNMAE:.4f}")

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
        q_index = [t.randint(low=0, high=eval(f"args.num_{qtype}s"), size=(12, )) for qtype in args.qtype]


        # 上溯逐层学习, 使用Curriculum控制学习的层次大小
        sbs_loss = model.stochastic_beam_search_loss(q_index, beam=args.beam, curriculum=i // 60)
        pbs_loss, rank_loss, regret = model.beam_search_regret_loss(q_index, topk=args.beam // 2, beam=args.beam // 2)

        loss = sbs_loss
        loss.backward()

        # gradient clipping
        t.nn.utils.clip_grad_norm_(model.tree_embs.parameters(), 2.0)
        model.tree_opt.step()
        # model.opt_scheduler.step()

        if i % 20 == 0:
            print(f"Round={runId} Iter={i} sbs_loss={sbs_loss:.4f} pbs_loss={pbs_loss:.4f} rank_loss={rank_loss:.4f}, Regret={regret:.4f}")


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
    parser.add_argument('--model', type=str, default='LTP')

    # LEFT
    parser.add_argument('--narys', type=int, default=2)
    parser.add_argument('--beam', type=int, default=25)
    parser.add_argument('--qtype', type=list, default=['user', 'item'])
    parser.add_argument('--ktype', type=list, default=['time'])

    # Dataset
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--num_users', type=int, default=12)
    parser.add_argument('--num_items', type=int, default=12)
    parser.add_argument('--num_times', type=int, default=3000)
    parser.add_argument('--dataset', type=str, default='abilene')

    # Training
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # Setup Logger
    setup_logger(args, "LEFT")

    # Record Experiments Config
    logger.info(args)

    # Run Experiments
    RunExperiments(args=args)
