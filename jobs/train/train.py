r"""Distributed data-parallel training of a CNN on FashionMNIST with Ray Train.

Ray Train starts `--num-workers` PyTorch workers, sets up DDP between them, and
shards the data. Each worker reserves `--cpus-per-worker` CPUs (default 2, which
places one worker per CPU worker pod on the CPU cluster). Checkpoints go to
shared storage (/mnt/shared, a directory mounted into every Ray pod), so
re-running with the same --name resumes from the latest checkpoint.

CPU cluster:
  ray job submit --working-dir jobs/train --runtime-env jobs/runtime-envs/cpu.yaml \
    -- python train.py --num-workers 2
GPU cluster:
  ray job submit --working-dir jobs/train --runtime-env jobs/runtime-envs/gpu.yaml \
    -- python train.py --num-workers 1 --use-gpu
"""
import argparse
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from filelock import FileLock
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import ray.train
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, prepare_data_loader, prepare_model


class Net(nn.Module):
    """Two conv + max-pool blocks and a 2-layer classifier for 28x28 grayscale images (10 classes)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        return self.fc2(F.relu(self.fc1(x)))


def load_data(data_dir):
    """Return the normalized FashionMNIST train and test sets, downloading them into `data_dir` once."""
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    os.makedirs(data_dir, exist_ok=True)
    # All workers share data_dir, so only one of them downloads at a time.
    with FileLock(os.path.join(data_dir, ".lock")):
        train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
        test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
    return train, test


def train_func(config):
    """Per-worker training loop, run by Ray Train on every DDP worker.

    Resumes from the latest checkpoint if there is one, trains on this worker's
    shard, and reports loss/accuracy each epoch. Rank 0 saves the checkpoint.
    """
    ctx = ray.train.get_context()
    train_ds, test_ds = load_data(config["data_dir"])
    # prepare_data_loader adds a DistributedSampler and moves batches to the right device.
    train_dl = prepare_data_loader(DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True))
    test_dl = prepare_data_loader(DataLoader(test_ds, batch_size=512))

    net = Net()
    start_epoch = 0
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as d:
            state = torch.load(os.path.join(d, "state.pt"), map_location="cpu")
        net.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        print(f"[rank {ctx.get_world_rank()}] resuming from epoch {start_epoch}")

    model = prepare_model(net)  # wraps in DDP and moves to GPU when use_gpu=True
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(start_epoch, config["epochs"]):
        if ctx.get_world_size() > 1:
            train_dl.sampler.set_epoch(epoch)
        model.train()
        total_loss, batches = 0.0, 0
        for x, y in train_dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1

        model.eval()
        correct = seen = 0
        with torch.no_grad():
            for x, y in test_dl:
                correct += (model(x).argmax(1) == y).sum().item()
                seen += len(y)

        metrics = {"epoch": epoch, "train_loss": total_loss / batches, "test_accuracy": correct / seen}
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = None
            if ctx.get_world_rank() == 0:
                raw = model.module if hasattr(model, "module") else model
                torch.save({"model": raw.state_dict(), "epoch": epoch}, os.path.join(tmp, "state.pt"))
                ckpt = Checkpoint.from_directory(tmp)
            ray.train.report(metrics, checkpoint=ckpt)


def main():
    """Parse flags and run the TorchTrainer (job driver, on the Ray head)."""
    p = argparse.ArgumentParser()
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--cpus-per-worker", type=float, default=2,
                   help="2 = one training worker per Ray worker pod on the default cluster")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128, help="per worker")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--storage-path", default="/mnt/shared/ray_results")
    p.add_argument("--data-dir", default="/mnt/shared/data")
    p.add_argument("--name", default="fashion-mnist")
    args = p.parse_args()

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "data_dir": args.data_dir,
        },
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
            resources_per_worker={"CPU": args.cpus_per_worker},
        ),
        run_config=RunConfig(
            name=args.name,
            storage_path=args.storage_path,
            checkpoint_config=CheckpointConfig(num_to_keep=2),
        ),
    )
    result = trainer.fit()
    print("Final metrics:", result.metrics)
    print("Checkpoint:", result.checkpoint)


if __name__ == "__main__":
    main()
