r"""Hyperparameter search with Ray Tune: trials run in parallel across workers,
and the ASHA scheduler stops weak trials early.

Searches learning rate, hidden size, dropout and batch size for a small MLP on
FashionMNIST subsets. Each trial reserves --cpus-per-trial / --gpus-per-trial.

  ray job submit --working-dir jobs/tune --runtime-env jobs/runtime-envs/cpu.yaml \
    -- python tune_search.py --num-samples 8
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from filelock import FileLock
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ray import tune
from ray.tune.schedulers import ASHAScheduler


def make_loaders(data_dir, batch_size):
    """Return train/test loaders over 10,000 / 2,000-image FashionMNIST subsets (downloaded once into `data_dir`)."""
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    os.makedirs(data_dir, exist_ok=True)
    with FileLock(os.path.join(data_dir, ".lock")):
        train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
        test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
    # Small subsets keep each trial to a few seconds per epoch on CPU.
    return (
        DataLoader(Subset(train, range(10_000)), batch_size=batch_size, shuffle=True),
        DataLoader(Subset(test, range(2_000)), batch_size=512),
    )


def trainable(config):
    """One Tune trial: train an MLP with `config`'s hyperparameters and report test accuracy every epoch.

    ASHA uses the per-epoch reports to stop weak trials early.
    """
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, test_dl = make_loaders(config["data_dir"], config["batch_size"])
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, config["hidden"]),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config["hidden"], 10),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                correct += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
        tune.report({"accuracy": correct / len(test_dl.dataset), "epoch": epoch})


def main():
    """Define the search space, run the Tuner, and print the best trial."""
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--cpus-per-trial", type=float, default=1)
    p.add_argument("--gpus-per-trial", type=float, default=0)
    p.add_argument("--storage-path", default="/mnt/shared/ray_results")
    p.add_argument("--data-dir", default="/mnt/shared/data")
    args = p.parse_args()

    param_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "hidden": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.0, 0.5),
        "batch_size": tune.choice([64, 128]),
        "epochs": args.epochs,
        "data_dir": args.data_dir,
    }
    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            num_samples=args.num_samples,
            scheduler=ASHAScheduler(max_t=args.epochs, grace_period=1, time_attr="epoch"),
        ),
        run_config=tune.RunConfig(name="fashion-mnist-tune", storage_path=args.storage_path),
    )
    results = tuner.fit()
    best = results.get_best_result()
    print("Best config:", {k: v for k, v in best.config.items() if k != "data_dir"})
    print("Best accuracy:", round(best.metrics["accuracy"], 4))


if __name__ == "__main__":
    main()
