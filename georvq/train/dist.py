"""Single-node data parallelism for the pyramid trainer.

Deliberately NOT `DistributedDataParallel`. DDP only synchronises
gradients for work that flows through the wrapper's `__call__`, and the
training step calls `encode_levels` -> `quantize` -> `decode` as three
separate methods; using DDP would mean collapsing the step into one
`forward()` and then fighting `grad_ckpt` (which needs `static_graph`).
An explicit all-reduce after `backward()` is a dozen lines, composes with
anything, and costs almost nothing here: with the encoder and codebooks
frozen only the decoder has gradients, so a step reduces ~222 MiB, tens
of milliseconds against a ~2.5 s step.

Correctness notes, all of which are easy to get silently wrong:
  * ranks must start from IDENTICAL weights -- a warm start leaves some
    modules freshly initialised (the depth `cond_mlp`), so we broadcast
    rank 0's parameters AND buffers rather than trusting equal seeds;
  * ranks must then draw DIFFERENT data, or the effective batch is one
    rank's batch and the whole exercise is wasted;
  * gradients are averaged, not summed, so the LR keeps its single-GPU
    meaning;
  * frozen codebooks are what make this simple. EMA books live in
    buffers, not gradients, so each rank would update them from its own
    shard and they would silently diverge. `vq.freeze_books` removes the
    problem; anything else needs buffer sync too, hence the assertion in
    `setup`.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def world_size():
    return dist.get_world_size() if is_distributed() else 1


def rank():
    return dist.get_rank() if is_distributed() else 0


def is_main():
    return rank() == 0


def setup(cfg):
    """Initialise from torchrun's env vars. Returns (device, rank, world).

    No-op when not launched under torchrun, so the same loop runs
    single-GPU unchanged."""
    if "LOCAL_RANK" not in os.environ:
        return None, 0, 1
    lr = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(lr)
        dist.init_process_group(backend="nccl")
    else:                                  # CPU path exists for the tests
        dist.init_process_group(backend="gloo")
    r, w = dist.get_rank(), dist.get_world_size()
    if w > 1 and cfg.model.vq is not None and not cfg.model.vq.freeze_books:
        raise RuntimeError(
            "multi-GPU with live EMA codebooks would let each rank's books "
            "drift apart (they are buffers, not gradients, so the gradient "
            "all-reduce does not touch them). Set model.vq.freeze_books, or "
            "add an explicit buffer all-reduce for the EMA state.")
    return torch.device(f"cuda:{lr}"), r, w


def broadcast_state(model, src=0):
    """Force every rank to rank 0's exact weights and buffers."""
    if not is_distributed():
        return 0
    n = 0
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=src); n += 1
        for b in model.buffers():
            if b.is_floating_point() or b.dtype in (torch.long, torch.bool):
                dist.broadcast(b.data, src=src); n += 1
    return n


def average_grads(params):
    """Average gradients across ranks, in place.

    Flattened into one buffer: a single 222 MiB all-reduce beats hundreds
    of small ones, which are latency-bound."""
    if not is_distributed():
        return
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size())
    for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(s)


def all_reduce_mean(x, device):
    """Mean of a scalar across ranks (for logging a true global value)."""
    if not is_distributed():
        return float(x)
    t = torch.tensor([float(x)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size())


def barrier():
    if is_distributed():
        dist.barrier()


def cleanup():
    if is_distributed():
        dist.destroy_process_group()
