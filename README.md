# ray-distributed-training

Distributed training with [Ray](https://docs.ray.io) on a local [kind](https://kind.sigs.k8s.io) Kubernetes cluster. The [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) operator runs a long-lived Ray cluster, and you submit jobs to it from your machine with the Ray CLI. It works CPU-only or with a local NVIDIA GPU.

## Layout

```
infra/
  kind/           kind cluster configs (cluster.yaml, cluster-gpu.yaml)
  k8s/            RayCluster manifests (CPU / GPU) + NodePort service for the head
  monitoring/     Prometheus + Grafana values and Ray PodMonitors
  scripts/        deploy.sh, teardown.sh, setup-gpu.sh, setup-monitoring.sh, setup-venv.sh
jobs/
  hello/          tasks, actors and a GPU task: checks the cluster works
  train/          Ray Train: DDP training of a CNN on FashionMNIST, with checkpoints and resume
  tune/           Ray Tune: hyperparameter search with early stopping (ASHA)
  llm-summarize/  LoRA fine-tuning of Qwen2.5-0.5B-Instruct for dialogue summarization (GPU)
  runtime-envs/   per-job Python dependencies (CPU / CUDA 12.8 PyTorch / LLM)
requirements.txt  Ray CLI for job submission
shared/           job outputs: checkpoints, results, datasets, HF cache (gitignored, created by deploy.sh)
```

## Prerequisites

- Docker Desktop (running)
- `kind`, `kubectl`, `helm`, `curl`, Python 3.10+
- A bash shell (Git Bash works on Windows)
- About 8 CPUs and 16 GB RAM for Docker. The whole setup is capped at 16 GB; see [Memory budget](#memory-budget).
- For GPU mode: an NVIDIA GPU with a current Windows driver and Docker Desktop's WSL2 backend

## 1. Deploy the cluster

```bash
./infra/scripts/deploy.sh          # CPU: 1 head + 2 CPU workers (2 CPU / 3 GiB each)
GPU=1 ./infra/scripts/deploy.sh    # GPU: 1 head + 1 GPU worker (4 CPU / 7.5 GiB / 1 GPU)
```

The script:

1. Creates the kind cluster `ray`.
2. In GPU mode, runs [setup-gpu.sh](infra/scripts/setup-gpu.sh) to expose the GPU to Kubernetes (explained below).
3. Installs the KubeRay operator (Helm chart 1.7.1) into namespace `ray`.
4. Applies the RayCluster ([CPU](infra/k8s/raycluster-cpu.yaml) or [GPU](infra/k8s/raycluster-gpu.yaml)). All images are official Ray images. The CPU cluster uses `rayproject/ray:2.58.0-py312` (~0.9 GB). In the GPU cluster, the GPU worker uses `rayproject/ray:2.58.0-py312-cu128` (~6.8 GB, CUDA 12.8 toolkit and `gcc`), and the head uses the CPU image.
5. Installs Prometheus and Grafana and connects them to Ray (see [Metrics](#metrics-prometheus--grafana)).
6. Waits for the pods, then checks `http://127.0.0.1:8265/api/version`.

The Ray head runs with `num-cpus: 0`, so it only hosts job drivers and the dashboard, and all compute runs on the workers.

`GPU` only takes effect when the cluster is created. To switch modes, run `./infra/scripts/teardown.sh` first.

| Variable | Default | Purpose |
|---|---|---|
| `GPU` | `0` | `1` = GPU cluster |
| `CLUSTER_NAME` | `ray` | kind cluster name |
| `NAMESPACE` | `ray` | namespace for the operator and Ray |
| `KUBERAY_VERSION` | `1.7.1` | kuberay-operator chart version |
| `MONITORING` | `1` | `0` skips Prometheus/Grafana |
| `MEMORY_BUDGET_GB` | `16` | deploy fails if memory limits plus the system reserve exceed this |
| `WAIT_TIMEOUT` | `20m` | wait for Ray pods |

**Dashboard:** http://127.0.0.1:8265. The same port serves the job API. It has no authentication and is bound to localhost only.

## 2. Install the Ray CLI

```bash
./infra/scripts/setup-venv.sh      # creates .venv and installs requirements.txt
source .venv/Scripts/activate      # Git Bash   (Linux/macOS: .venv/bin/activate)
export RAY_ADDRESS=http://127.0.0.1:8265
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
$env:RAY_ADDRESS = "http://127.0.0.1:8265"
```

The Ray CLI runs natively on Windows. Keep `ray` in `requirements.txt` at the same version as the cluster image (2.58.0). The CLI talks to the cluster over the HTTP job API, so your local Python version doesn't have to match the cluster's 3.12.

## 3. Submit jobs

Run these from the repo root. `--working-dir` uploads that folder to the cluster, and `--runtime-env` installs the job's Python dependencies on each node it uses. The environment is cached, so only the first job that uses it waits for the PyTorch install.

```bash
# Smoke test: tasks, actors, GPU if present
ray job submit --working-dir jobs/hello -- python hello.py

# Distributed training on CPU: 2 DDP workers, one per worker pod
ray job submit --working-dir jobs/train --runtime-env jobs/runtime-envs/cpu.yaml \
  -- python train.py --num-workers 2 --epochs 3

# Distributed training on the GPU (GPU cluster)
ray job submit --working-dir jobs/train --runtime-env jobs/runtime-envs/gpu.yaml \
  -- python train.py --num-workers 1 --use-gpu --epochs 3

# Hyperparameter search: 8 trials in parallel, weak ones stopped early
ray job submit --working-dir jobs/tune --runtime-env jobs/runtime-envs/cpu.yaml \
  -- python tune_search.py --num-samples 8
```

### LLM fine-tuning: dialogue summarization (GPU)

[finetune.py](jobs/llm-summarize/finetune.py) fine-tunes [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) with LoRA. It is sized for an 8 GB laptop GPU: bf16 base weights, about 9M trainable adapter parameters, gradient checkpointing, and a batch of 4 with 4-step gradient accumulation.

```bash
ray job submit --working-dir jobs/llm-summarize --runtime-env jobs/runtime-envs/gpu-llm.yaml \
  -- python finetune.py                       # 2,000 training dialogues, 1 epoch (~15 min on an RTX 5060 Laptop)
```

The GPU runtime envs set `eager_install: false`, so the ~3 GB CUDA PyTorch install happens only on the nodes the job uses (the head for the driver, and the GPU worker), not on the CPU worker.

What the job does:

1. Scores ROUGE on 64 held-out test dialogues with the base model.
2. Trains the LoRA adapter.
3. Scores the fine-tuned model again and prints a few reference and predicted summaries.

The adapter (about 35 MB) and tokenizer are saved as a Ray Train checkpoint in `shared/ray_results/<name>` in this repo. The model and dataset are cached in `shared/hf-cache`. See [Shared storage](#shared-storage-and-checkpoints).

Useful flags: `--max-train-samples` (all 12,460 are available), `--epochs`, `--global-batch-size`, `--max-micro-batch`, `--lora-r`, `--max-len`, `--model` (any causal LM with a chat template), `--name`.

#### Scaling workers (distributed DDP)

`--num-workers` is the only flag you change. The job looks at the Ray cluster and divides it equally between the workers:

| | Rule | 1 GPU (this laptop) | 4 GPUs |
|---|---|---|---|
| GPU per worker | 1 if workers ≤ GPUs, otherwise GPUs ÷ workers | `--num-workers 2` → 0.5 each | `--num-workers 4` → 1 each; `8` → 0.5 each |
| CPU per worker | GPU node CPUs ÷ workers on that node | 4 CPUs ÷ 2 → 2 each | same rule per node |
| Gradient sync | NCCL with a whole GPU each, gloo when workers share a GPU | gloo | NCCL, or gloo for `8` |
| Micro batch | `--max-micro-batch` (4) × GPU share, so workers sharing a GPU also share its memory | 4 × 0.5 → 2 | 4 (or 2 with `8`) |
| Batch | `--global-batch-size` (16) stays fixed: micro batch × grad accumulation × workers | 2 × 4 × 2 = 16 | 4 × 1 × 4 = 16 |

```bash
ray job submit --working-dir jobs/llm-summarize --runtime-env jobs/runtime-envs/gpu-llm.yaml \
  -- python finetune.py --num-workers 2
```

The job prints the plan it chose, for example `Training with 2 worker(s): resources/worker={'CPU': 2, 'GPU': 0.5}, backend=gloo, micro batch 2 x grad accum 4 x 2 workers = global batch 16`. Each worker logs its rank, device and data shard.

Things to know:

- When workers outnumber GPUs, `--num-workers` must be a multiple of the GPU count so every GPU hosts the same number of workers. The job stops with a clear message otherwise.
- Because the global batch is fixed, changing the worker count doesn't change the training recipe, only how the work is split.
- GPU memory is divided equally too. Each worker caps PyTorch at its share of the card (`GPU share × 0.85`, so about 3.5 GB each with 2 workers on 8 GB). Without the cap, two workers oversubscribed the GPU, and WSL silently spilled GPU memory into system RAM. Training slowed from about 1 minute to more than 15 and never finished.
- During gradient accumulation, workers skip the gradient all-reduce except on the last micro-batch (`no_sync`), so they sync once per optimizer step.
- Every worker gets exactly the same number of examples. The sample is filtered first, trimmed to a multiple of the worker count, then dealt out round-robin. With uneven shards (968 vs 972 examples), the worker with the extra batch waited forever in a gradient all-reduce that the other worker never joined.
- DDP's start-up sync of the model is skipped (`init_sync=False`). Every worker loads the same frozen base weights anyway, so only the ~35 MB of LoRA weights are broadcast from rank 0. The default sync copied the whole ~1 GB model through pinned host memory on each worker, which ran the GPU pod out of RAM.
- On one GPU, 2 workers are not faster than 1: they share the same GPU. This is real DDP (separate processes, sharded data, gradients averaged every step), so it exercises the multi-GPU code path locally. On more GPUs, the same command gives a real speedup.
- **Laptop limit:** each worker holds its own copy of the model, so 2 workers is the practical maximum on an 8 GB GPU with a 7.5 GiB worker pod.

Managing jobs:

```bash
ray job list
ray job logs <submission-id> --follow
ray job status <submission-id>
ray job stop <submission-id>
```

Add `--no-wait` to `ray job submit` to return immediately instead of streaming logs.

### Adding your own job

Put a script in a new folder under `jobs/`. Call `ray.init()` in it with no arguments to connect to the cluster it's running on, and submit it with `--working-dir jobs/<folder>`. Put its dependencies in a runtime env YAML, or pass them inline with `--runtime-env-json '{"pip": ["pandas"]}'`.

### Shared storage and checkpoints

Everything the jobs produce is stored **inside this repo**, in the `shared/` folder, which is gitignored and created by `deploy.sh`. Every kind node mounts `shared/` at `/shared`, and each Ray pod mounts that at `/mnt/shared`. This gives all workers a common filesystem, which Ray Train needs for checkpoints, and you can open the results directly from Windows.

| In the pods | In the repo | What |
|---|---|---|
| `/mnt/shared/ray_results/<name>` | `shared/ray_results/<name>` | Ray Train/Tune results and checkpoints (e.g. the LoRA adapter) |
| `/mnt/shared/data` | `shared/data` | FashionMNIST |
| `/mnt/shared/hf-cache` | `shared/hf-cache` | Hugging Face model and dataset cache |

`deploy.sh` fills in the absolute path of `shared/` (the `__SHARED_DIR__` placeholder in the kind configs) when it creates the cluster. Set `SHARED_DIR=/some/other/folder` to use a different location. The mount is fixed at cluster creation, so changing it means recreating the cluster.

Tearing down or recreating the cluster doesn't touch `shared/`. Delete a run with `rm -rf shared/ray_results/<name>`, or delete `shared/hf-cache` to free about 1 GB (it re-downloads when needed). Running a job again with the same `--name` resumes from its latest checkpoint.

## Memory budget

The whole setup is designed to fit in **16 GB of RAM**:

- Every Ray pod sets its memory request equal to its limit, so Kubernetes never schedules more than the node can hold.
- Every add-on (Prometheus, Grafana, KubeRay, the NVIDIA device plugin) has a memory limit.
- At the end of `deploy.sh`, `check_memory_budget` adds up all container limits plus 1.5 GiB for the Kubernetes system pods, which have no limits. It fails if the total is over `MEMORY_BUDGET_GB`.

| Component | GPU mode | CPU mode | Measured peak (LLM fine-tune) |
|---|---|---|---|
| Ray head (job drivers, runtime-env installs, dashboard) | 3.5 GiB | 3.5 GiB | 2.4 GiB |
| Ray GPU worker (hosts every training worker) | 7.5 GiB | – | 4.7 GiB with 1 worker |
| Ray CPU worker(s) | 0 (group scaled to 0) | 2 × 3 GiB | – |
| Prometheus, Grafana, exporters, KubeRay, NVIDIA plugin | ~2.9 GiB | ~2.8 GiB | ~1.1 GiB |
| Kubernetes system pods (no limits; reserved) | 1.5 GiB | 1.5 GiB | ~1.1 GiB |
| **Total** | **~15.4 GiB** | **~13.8 GiB** | |

The head needs more room than its idle usage suggests. It runs each job's driver, and it installs the job's runtime env: pip unpacking the CUDA PyTorch wheels peaks at about 2.3 GiB. With a 2.5 GiB head, the LLM job was stopped by Ray's memory monitor.

Each Ray pod also has a fixed object store (512 MiB to 1 GiB) instead of Ray's default of 30% of pod memory. The object store lives in `/dev/shm`, which counts against the pod's limit.

If a task goes over its pod's memory, Ray's memory monitor kills the task and retries it, rather than the whole pod being OOM-killed. To give the GPU worker more room, lower something else by the same amount; the budget check tells you if the total goes over.

On Windows, Docker Desktop's WSL2 VM gets half the machine's RAM by default, which is 16 GB on a 32 GB laptop. To pin it explicitly, add this to `%UserProfile%\.wslconfig` and restart WSL (`wsl --shutdown`):

```ini
[wsl2]
memory=16GB
```

## Metrics: Prometheus + Grafana

`deploy.sh` runs [setup-monitoring.sh](infra/scripts/setup-monitoring.sh), which:

1. Installs `kube-prometheus-stack` into namespace `prometheus-system`, trimmed for kind ([values](infra/monitoring/values-kube-prometheus-stack.yaml)).
2. Applies [PodMonitors](infra/monitoring/ray-podmonitors.yaml) that scrape the Ray head and workers: node, task, actor and Train metrics on port 8080, autoscaler metrics on 44217, and dashboard metrics on 44227.
3. Copies the Grafana dashboards that the running Ray head generates (default, Train, Data, Serve and others) into ConfigMaps. Grafana therefore loads exactly the dashboard versions that the Ray dashboard embeds.

| URL | What |
|---|---|
| http://127.0.0.1:8265 | Ray dashboard. The **Metrics** tab and the Overview charts are embedded Grafana panels. |
| http://127.0.0.1:3000 | Grafana: anonymous read-only access; `admin` / `prom-operator` to edit |
| http://127.0.0.1:9090 | Prometheus (query `ray_` metrics; `/targets` shows scrape status) |

The Ray head gets four environment variables in the RayCluster manifests:

- `RAY_PROMETHEUS_HOST` and `RAY_GRAFANA_HOST`: in-cluster addresses the dashboard uses to query Prometheus and Grafana.
- `RAY_GRAFANA_IFRAME_HOST`: the Grafana address your browser loads the embedded panels from (`http://127.0.0.1:3000`).
- `RAY_PROMETHEUS_NAME`: the name of the Prometheus datasource in Grafana.

## GPU mode: how it works

1. The Windows driver exposes the GPU to WSL2 as `/dev/dxg`, plus user-mode libraries in `/usr/lib/wsl`. Docker Desktop's VM can see both.
2. [cluster-gpu.yaml](infra/kind/cluster-gpu.yaml) labels the worker node `nvidia.com/gpu.present=true` and mounts `/usr/lib/wsl` into it read-only.
3. [setup-gpu.sh](infra/scripts/setup-gpu.sh):
   - Installs the NVIDIA container toolkit inside that node and makes the NVIDIA runtime containerd's default.
   - Labels the node with its GPU model.
   - Installs the NVIDIA device plugin, which advertises `nvidia.com/gpu`.
4. The GPU worker pod requests `nvidia.com/gpu: 1`, and KubeRay starts Ray on it with `num-gpus=1`.

The GPU worker runs Ray's official CUDA image, `rayproject/ray:2.58.0-py312-cu128`. It includes the CUDA 12.8 toolkit and `gcc`. Recent PyTorch uses Triton kernels on the GPU (as does `torch.compile`), and Triton compiles a small C launcher the first time it runs; with the plain CPU image, jobs failed with `RuntimeError: Failed to find C compiler`. The image is about 6.8 GB and is pulled once per new cluster.

Ray's images don't include PyTorch. [gpu.yaml](jobs/runtime-envs/gpu.yaml) installs the CUDA 12.8 build (RTX 50-series cards need CUDA 12.8 or newer), which downloads about 3 GB the first time a job uses it on a node. `rayproject/ray-ml`, which bundled ML libraries, was discontinued after Ray 2.30, so there's no official 2.58 image with PyTorch.

## Tear down

```bash
./infra/scripts/teardown.sh
```
