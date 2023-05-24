import json
import time
import argparse
import torch as t
from loguru import logger
from datasets.tensor_dataset import DataModule
from modules.tc import LTP, NTF, NTC, NCP
from utils.monitor import EarlyStopMonitor
from lightning import seed_everything
from utils.logger import setup_logger

def get_model(args):

    if args.model == 'LTP':
        model = LTP(args)
    elif args.model == 'NTF':
        model = NTF(args)
    elif args.model == 'NTC':
        model = NTC(args)
    elif args.model == 'NCP':
        model = NCP(args)
    else:
        raise NotImplementedError
    return model


def RunOnce(args, runId, runHash):

    seed = runId + args.seed

    seed_everything(args.seed)

    # Initialize
    model = get_model(args)
    model.setup_optimizer()
    dataModule = DataModule(args, seed=seed)
    monitor = EarlyStopMonitor(args.patience)

    for epoch in range(args.epochs):
        epoch_loss = model.train_one_epoch(dataModule.trainLoader())

        vNRMSE, vNMAE = model.valid_one_epoch(dataModule.validLoader())

        monitor.track(epoch, model.state_dict(), vNRMSE)

        logger.info(f"Round={runId} Epoch={epoch:02d} Loss={epoch_loss:.4f} "
                    f"vNRMSE={vNRMSE:.4f} vNMAE={vNMAE:.4f}")

        if monitor.early_stop():
            break


    # Test
    model.load_state_dict(monitor.params)
    tNRMSE, tNMAE = model.test_one_epoch(dataModule.testLoader())
    logger.info(f"Round={runId} tNRMSE={tNRMSE:.4f} tNMAE={tNMAE:.4f}")

    # Save
    t.save(model.state_dict(), f"./results/baselines/{runHash}.pt")

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

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LTP')
    parser.add_argument('--dataset', type=str, default='abilene')
    parser.add_argument('--num_users', type=int, default=12)
    parser.add_argument('--num_items', type=int, default=12)
    parser.add_argument('--num_times', type=int, default=3000)
    parser.add_argument('--density', type=float, default=0.02)
    parser.add_argument('--window', type=float, default=48)
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()

    # Setup Logger
    setup_logger(args, "LEFT")


    logger.info(args)

    # Run Experiments
    RunExperiments(args=args)
