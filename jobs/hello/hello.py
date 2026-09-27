"""Smoke test for the Ray cluster: tasks, actors, and (if present) GPUs.

  ray job submit --working-dir jobs/hello -- python hello.py
"""
import collections
import socket
import time

import ray

ray.init()
res = ray.cluster_resources()
print("Cluster resources:", {k: v for k, v in res.items() if not k.startswith("node:")})


# Tasks: stateless functions scheduled across the worker pods.
@ray.remote(num_cpus=1)
def where_am_i(i: int) -> str:
    """Return the hostname (Ray pod) this task ran on."""
    time.sleep(0.5)
    return socket.gethostname()


hosts = ray.get([where_am_i.remote(i) for i in range(16)])
print("16 tasks ran on:", dict(collections.Counter(hosts)))


# Actors: stateful workers that live on one pod.
@ray.remote(num_cpus=1)
class Counter:
    """Actor holding a running total; calls are processed one at a time in order."""

    def __init__(self):
        self.n = 0

    def incr(self, by: int) -> int:
        """Add `by` and return the new total."""
        self.n += by
        return self.n


counter = Counter.remote()
ray.get([counter.incr.remote(i) for i in range(10)])
print("Actor counter after 10 calls:", ray.get(counter.incr.remote(0)))


# GPU task: only scheduled if the cluster has a GPU worker (GPU=1 deploy).
if res.get("GPU", 0) >= 1:
    @ray.remote(num_gpus=1)
    def gpu_info() -> str:
        """Describe the GPU Ray assigned to this task: host, GPU ids, CUDA_VISIBLE_DEVICES and nvidia-smi -L."""
        import os
        import subprocess
        ids = ray.get_runtime_context().get_accelerator_ids()["GPU"]
        try:
            smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
        except FileNotFoundError:
            smi = "nvidia-smi not on PATH"
        return f"host={socket.gethostname()} gpu_ids={ids} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | {smi}"

    print("GPU task:", ray.get(gpu_info.remote()))
else:
    print("No GPUs in this cluster (deploy with GPU=1 to add one).")

print("Hello from Ray!")
