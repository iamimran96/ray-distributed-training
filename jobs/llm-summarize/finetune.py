r"""LoRA fine-tuning of a small instruction LLM for dialogue summarization, on Ray Train.

Model:   Qwen/Qwen2.5-0.5B-Instruct (fits an 8 GB GPU in bf16 with LoRA)
Dataset: knkarthick/dialogsum (dialogue -> summary of one or two sentences)

The job scores ROUGE on held-out test dialogues with the base model (skipped
when resuming), fine-tunes LoRA adapters, then scores again. The adapter and
tokenizer are checkpointed to shared storage after every epoch; re-running with
the same --name resumes from the latest checkpoint.

  ray job submit --working-dir jobs/llm-summarize --runtime-env jobs/runtime-envs/gpu-llm.yaml \
    -- python finetune.py

Scaling: pass only --num-workers. `plan_workers` divides the cluster equally
between the DDP workers:
  - GPUs: one each if there are enough, otherwise GPUs / workers of a GPU each,
    and each worker caps its PyTorch GPU memory at that share;
  - CPUs: the GPU nodes' CPUs split evenly;
  - backend: NCCL with a whole GPU per worker, gloo when workers share a GPU;
  - batch: --global-batch-size stays fixed; each worker's micro batch (scaled by
    its GPU share) and gradient accumulation are derived from it.

    -- python finetune.py --num-workers 2     # on 1 GPU: 0.5 GPU each, gloo
"""
import argparse
import contextlib
import json
import logging
import math
import os
import tempfile
import time

import torch
# Import the function, not the module: if train_func referenced `torch.distributed`,
# Ray's cloudpickle would also try to capture torch.distributed.config (because
# train_func uses `.config`), which can't be pickled.
from torch.distributed import broadcast
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from rouge_score import rouge_scorer
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import ray.train
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer, get_device, prepare_model

INSTRUCTION = "Summarize the following conversation in one or two sentences.\n\n"


def prompt_messages(dialogue):
    """Chat messages asking the model to summarize `dialogue`."""
    return [{"role": "user", "content": INSTRUCTION + dialogue}]


def build_example(tokenizer, dialogue, summary, max_len):
    """Tokenize one (dialogue, summary) pair for causal-LM training.

    Returns input_ids (chat prompt + summary + EOS) and labels with the prompt
    positions set to -100, so the loss only covers the summary. Returns None if
    the pair is longer than `max_len` tokens.
    """
    prompt = tokenizer.apply_chat_template(prompt_messages(dialogue), tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(summary + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(target_ids) > max_len:
        return None  # skip over-long dialogues instead of cutting off the summary
    # Only the summary tokens contribute to the loss.
    return {"input_ids": prompt_ids + target_ids, "labels": [-100] * len(prompt_ids) + target_ids}


def collate(batch, pad_id):
    """Right-pad a list of examples to the longest one: input_ids, labels (-100 padding), attention_mask."""
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@torch.no_grad()
def rouge_eval(model, tokenizer, samples, device, max_new_tokens=80):
    """Greedy-decode a summary for each sample (batches of 8, left-padded).

    Returns (mean ROUGE-1/2/L F1 over `samples`, the first 3 reference/prediction pairs).
    """
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    examples = []
    tokenizer.padding_side = "left"
    for i in range(0, len(samples), 8):
        chunk = samples[i : i + 8]
        prompts = [
            tokenizer.apply_chat_template(prompt_messages(s["dialogue"]), tokenize=False, add_generation_prompt=True)
            for s in chunk
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for s, seq in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
            pred = tokenizer.decode(seq, skip_special_tokens=True).strip()
            for k, v in scorer.score(s["summary"], pred).items():
                totals[k] += v.fmeasure
            if len(examples) < 3:
                examples.append({"reference": s["summary"], "prediction": pred})
    tokenizer.padding_side = "right"
    return {k: round(v / len(samples), 4) for k, v in totals.items()}, examples


def train_func(cfg):
    """Per-worker training loop, run by Ray Train on every DDP worker.

    Each worker caps its GPU memory at its share, loads the base model onto its
    device, wraps it with LoRA adapters (fp32) and DDP, and trains on its own
    shard of the data. Gradients are all-reduced once per optimizer step
    (`no_sync` on the other micro-batches). Rank 0 alone runs the ROUGE
    evaluations and writes the checkpoint; every rank calls `ray.train.report`.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide per-file Hugging Face download requests
    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    device = get_device()
    if device.type == "cuda":
        # Enforce this worker's equal share of GPU memory. Without a cap, workers that
        # share a GPU can oversubscribe it; on WSL the overflow silently spills into
        # system RAM and training slows to a crawl. 0.85 leaves room for CUDA contexts.
        torch.cuda.set_per_process_memory_fraction(cfg["gpu_share"] * 0.85, device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load weights straight onto this worker's GPU: skips a full copy in host RAM,
    # which matters when several workers share one node's memory.
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16,
        device_map={"": str(device)} if device.type == "cuda" else None,
    )
    model.config.use_cache = False
    if cfg["gradient_checkpointing"]:
        # Recompute activations in the backward pass: much less GPU memory for ~20% more compute.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    torch.manual_seed(0)  # identical LoRA initialisation on every rank
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=2 * cfg["lora_r"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ))
    # Train the adapters in fp32 for stable optimizer updates; the frozen base stays bf16.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if rank == 0:
        model.print_trainable_parameters()

    start_epoch = 0
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as d:
            set_peft_model_state_dict(model, load_file(os.path.join(d, "adapter_model.safetensors")))
            start_epoch = json.load(open(os.path.join(d, "progress.json")))["epoch"] + 1
        print(f"[rank {rank}] resumed adapters, starting at epoch {start_epoch}")

    model.to(device)
    test = load_dataset(cfg["dataset"], split="test").shuffle(seed=0).select(range(cfg["eval_samples"]))
    test = [{"dialogue": r["dialogue"], "summary": r["summary"]} for r in test]
    base_scores = None
    if rank == 0 and start_epoch == 0:
        with model.disable_adapter():
            base_scores, _ = rouge_eval(model, tokenizer, test, device)
        print(f"Base model ROUGE on {len(test)} test dialogues: {base_scores}", flush=True)

    # Every rank must get exactly the same number of examples: with DDP, a rank
    # that has one more batch runs one more gradient all-reduce than the others
    # and waits for them forever. So filter the whole sample first (same shuffle
    # on every rank), trim it to a multiple of the world size, then deal it out.
    train = load_dataset(cfg["dataset"], split="train").shuffle(seed=0)
    train = train.select(range(min(cfg["max_train_samples"], len(train))))
    examples = [build_example(tokenizer, r["dialogue"], r["summary"], cfg["max_len"]) for r in train]
    examples = [e for e in examples if e is not None]
    examples = examples[: len(examples) // world * world][rank::world]
    print(f"[rank {rank}/{world}] device={device} pid={os.getpid()} "
          f"training on {len(examples)} examples (its shard of {cfg['max_train_samples']})", flush=True)
    loader = torch.utils.data.DataLoader(
        examples, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    # DDP wrapper when num_workers > 1. By default DDP broadcasts *every* parameter
    # from rank 0 at start-up, including the ~1 GB of frozen base weights, which
    # every rank already loaded from the same checkpoint. With gloo that copy is
    # staged through pinned host memory (~1.1 GB per worker that PyTorch keeps),
    # which pushed the shared GPU pod out of RAM. Skip it and broadcast only the
    # trainable LoRA weights (~35 MB).
    model = prepare_model(model, move_to_device=False, parallel_strategy_kwargs={"init_sync": False})
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1:
        for p in params:
            broadcast(p.data, src=0)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    steps_per_epoch = math.ceil(len(loader) / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    for _ in range(start_epoch * steps_per_epoch):
        sched.step()

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t0, running, micro = time.time(), 0.0, 0
        opt.zero_grad()
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            sync_step = (i + 1) % cfg["grad_accum"] == 0 or i + 1 == len(loader)
            # With DDP, only all-reduce gradients on the last micro-batch of each accumulation window.
            sync_ctx = model.no_sync() if world > 1 and not sync_step else contextlib.nullcontext()
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = model(**batch).loss / cfg["grad_accum"]
                loss.backward()
            running += loss.item()
            micro += 1
            if sync_step:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step = (i + 1) // cfg["grad_accum"]
                if rank == 0 and step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{steps_per_epoch} "
                          f"loss {running * cfg['grad_accum'] / micro:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)

        metrics = {"epoch": epoch, "train_loss": running * cfg["grad_accum"] / max(1, micro),
                   "epoch_minutes": round((time.time() - t0) / 60, 2)}
        if device.type == "cuda":
            metrics["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

        raw = model.module if hasattr(model, "module") else model
        if rank == 0 and epoch == cfg["epochs"] - 1:
            scores, samples = rouge_eval(raw, tokenizer, test, device)
            metrics.update({f"ft_{k}": v for k, v in scores.items()})
            if base_scores:
                metrics.update({f"base_{k}": v for k, v in base_scores.items()})
            print(f"Fine-tuned ROUGE: {scores}", flush=True)
            for s in samples:
                print(f"\nREFERENCE:  {s['reference']}\nPREDICTION: {s['prediction']}", flush=True)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = None
            if rank == 0:
                raw.save_pretrained(tmp)  # LoRA adapter only (~35 MB)
                tokenizer.save_pretrained(tmp)
                json.dump({"epoch": epoch, "base_model": cfg["model"]}, open(os.path.join(tmp, "progress.json"), "w"))
                ckpt = Checkpoint.from_directory(tmp)
            ray.train.report(metrics, checkpoint=ckpt)


def plan_workers(num_workers, use_gpu, global_batch_size, max_micro_batch):
    """Split the cluster's GPUs and CPUs equally across `num_workers` DDP workers.

    GPU mode:
      - num_workers <= GPUs: one whole GPU per worker, NCCL backend.
      - num_workers >  GPUs: GPUs / num_workers of a GPU each, gloo backend (NCCL
        refuses two ranks on one GPU). num_workers must then be a multiple of the
        GPU count, so every GPU hosts the same number of workers.
      - CPUs of the GPU nodes are split evenly between the workers they host.
      - The micro batch cap is scaled by the GPU share, since workers sharing a
        GPU also share its memory (train_func enforces the memory share).
    CPU mode (--cpu): all CPUs split evenly, gloo backend.

    The global batch stays fixed, so changing num_workers doesn't change the
    optimisation: micro batch x grad accumulation x num_workers = global_batch_size.

    Returns (resources_per_worker, backend, micro_batch, grad_accum).
    Exits with a message if the cluster has no GPUs (GPU mode) or the workers
    can't be split equally.
    """
    ray.init(ignore_reinit_error=True)
    nodes = [n["Resources"] for n in ray.nodes() if n["Alive"]]
    if use_gpu:
        gpu_nodes = [r for r in nodes if r.get("GPU", 0) > 0]
        total_gpus = int(sum(r["GPU"] for r in gpu_nodes))
        if total_gpus == 0:
            raise SystemExit("No GPUs in the Ray cluster. Deploy with GPU=1, or pass --cpu.")
        if num_workers > total_gpus and num_workers % total_gpus:
            raise SystemExit(f"--num-workers {num_workers} can't be split equally over {total_gpus} GPU(s); "
                             f"use a multiple of {total_gpus}.")
        gpus_per_worker = 1.0 if num_workers <= total_gpus else total_gpus / num_workers
        workers_per_gpu = max(1, round(1 / gpus_per_worker))
        # Each GPU node's CPUs are shared by the workers placed on its GPUs.
        cpus_per_worker = max(1, int(min(r.get("CPU", 0) / (r["GPU"] * workers_per_gpu) for r in gpu_nodes)))
        resources = {"CPU": cpus_per_worker, "GPU": gpus_per_worker}
        backend = "nccl" if gpus_per_worker == 1.0 else "gloo"
    else:
        cpu_nodes = [r for r in nodes if r.get("CPU", 0) > 0]
        total_cpus = sum(r["CPU"] for r in cpu_nodes)
        resources = {"CPU": max(1, int(total_cpus // num_workers))}
        backend = "gloo"

    # Workers sharing a GPU also share its memory: scale the micro batch cap by the GPU share.
    if use_gpu:
        max_micro_batch = max(1, int(max_micro_batch * resources["GPU"]))

    if global_batch_size % num_workers:
        raise SystemExit(f"--global-batch-size {global_batch_size} must be divisible by --num-workers {num_workers}")
    per_worker = global_batch_size // num_workers
    micro = max(d for d in range(1, min(max_micro_batch, per_worker) + 1) if per_worker % d == 0)
    return resources, backend, micro, per_worker // micro


def main():
    """Parse flags, plan the worker split, and run the TorchTrainer (job driver, on the Ray head)."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="knkarthick/dialogsum")
    p.add_argument("--num-workers", type=int, default=1,
                   help="DDP workers; the cluster's GPUs and CPUs are divided equally between them")
    p.add_argument("--cpu", action="store_true", help="train without a GPU (very slow)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-train-samples", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=16,
                   help="examples per optimizer step across all workers (kept fixed when scaling)")
    p.add_argument("--max-micro-batch", type=int, default=4,
                   help="largest forward/backward batch for a worker with a whole GPU (scaled down by the "
                        "worker's GPU share); the rest of the per-worker batch is gradient accumulation")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--storage-path", default="/mnt/shared/ray_results")
    p.add_argument("--name", default="qwen-dialogsum-lora")
    args = p.parse_args()

    resources, backend, micro, accum = plan_workers(
        args.num_workers, not args.cpu, args.global_batch_size, args.max_micro_batch)
    print(f"Training with {args.num_workers} worker(s): resources/worker={resources}, backend={backend}, "
          f"micro batch {micro} x grad accum {accum} x {args.num_workers} workers = global batch {args.global_batch_size}")

    config = {k: v for k, v in vars(args).items()
              if k not in ("num_workers", "cpu", "global_batch_size", "max_micro_batch", "storage_path", "name")}
    config.update(batch_size=micro, grad_accum=accum, gpu_share=resources.get("GPU", 0.0))
    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=not args.cpu,
            resources_per_worker=resources,
        ),
        torch_config=TorchConfig(backend=backend),
        run_config=RunConfig(
            name=args.name,
            storage_path=args.storage_path,
            checkpoint_config=CheckpointConfig(num_to_keep=2),
        ),
    )
    result = trainer.fit()
    print("\nFinal metrics:", json.dumps(result.metrics, indent=2))
    print("Adapter checkpoint:", result.checkpoint.path)


if __name__ == "__main__":
    main()
