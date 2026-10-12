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
  * EMA books live in buffers, not gradients, so each rank updates them
    from its own shard and they would diverge. `vq.freeze_books` removes
    the problem; live books are synced after every step by `BookSync`.
"""
from __future__ import annotations

import os
from datetime import timedelta

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
    # NCCL's default collective timeout is 10 minutes, which this trainer
    # blows through before it ever reaches a training step: each rank
    # prefills its own multi-hundred-GiB CPU chip pool from the same
    # array, so whichever rank wins the disk can arrive at the first
    # collective 10+ minutes ahead of the other and the weight broadcast
    # dies. Rank skew here is expected and benign, so allow for it.
    timeout = timedelta(minutes=int(os.environ.get("GEORVQ_NCCL_TIMEOUT_MIN",
                                                   "90")))
    if torch.cuda.is_available():
        torch.cuda.set_device(lr)
        # Bind the rank to its device EXPLICITLY. Without this NCCL infers
        # the device from the global rank on the first collective and
        # warns that it "can cause a hang" -- which is exactly what a
        # device-less barrier did here: both ranks present, neither
        # returning. Also lets NCCL eagerly initialise the communicator.
        try:
            dist.init_process_group(backend="nccl", timeout=timeout,
                                    device_id=torch.device(f"cuda:{lr}"))
        except TypeError:            # older torch without device_id
            dist.init_process_group(backend="nccl", timeout=timeout)
    else:                                  # CPU path exists for the tests
        dist.init_process_group(backend="gloo", timeout=timeout)
    r, w = dist.get_rank(), dist.get_world_size()
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


class BookSync:
    """Keep live EMA codebooks identical across ranks.

    Each rank's forward applies `ema <- d*ema + (1-d)*stat_r` from its own
    shard. The update is linear, so from the pre-step state `old` the
    single-GPU update on the whole global batch is recovered exactly as
    `d*old + sum_r (new_r - d*old)`. Statistics are SUMMED, not averaged:
    `dead_threshold` is an absolute count, so averaged counts would sit
    world-size times lower and change which codes are reset.

    A level that skipped the step on one rank (no active rows under
    `skip_masked`) contributes a zero delta there and is counted as
    stepped if any rank stepped, which keeps `step` (and so the reset
    schedule) aligned. Dead-code resets, splits and the lazy data init
    draw from rank-local data, so on a step where any of them ran the
    level takes rank 0's state outright; that drops one step of the other
    ranks' statistics, (1-d) of one update.

    The codebook is re-derived from the synced buffers with the formula
    in `VectorQuantizer.forward`, then projected for b-bit books."""

    _BUFS = ("ema_count", "ema_sum", "ema_sse")

    def __init__(self, model):
        self.levels = [lev for rvq in model.rvqs for lev in rvq.levels
                       if not getattr(lev, "grad_codebook", False)]
        self._old = None

    def _bufs(self, lev):
        return [b for b in (getattr(lev, n, None) for n in self._BUFS)
                if b is not None]

    @torch.no_grad()
    def snapshot(self):
        if not is_distributed():
            return
        self._old = [(int(lev.step), bool(lev.initialized),
                      [b.clone() for b in self._bufs(lev)])
                     for lev in self.levels]

    @torch.no_grad()
    def sync(self):
        """Call after the step's quantize; returns the number of levels
        that took rank 0's state."""
        if not is_distributed() or self._old is None:
            return 0
        dev = self.levels[0].ema_count.device
        stepped = torch.tensor([int(lev.step) - o[0] for lev, o
                                in zip(self.levels, self._old)],
                               device=dev, dtype=torch.float32)
        dist.all_reduce(stepped, op=dist.ReduceOp.MAX)
        deltas = []
        for lev, (s0, _, olds) in zip(self.levels, self._old):
            d = lev.decay if int(lev.step) > s0 else 1.0
            deltas += [b - d * o for b, o in zip(self._bufs(lev), olds)]
        flat = torch._utils._flatten_dense_tensors(deltas)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        summed = torch._utils._unflatten_dense_tensors(flat, deltas)
        k, n_bcast = 0, 0
        for lev, (s0, init0, olds), st in zip(self.levels, self._old,
                                              stepped.tolist()):
            bufs = self._bufs(lev)
            lev.step.fill_(s0 + int(st > 0))
            ri = lev.reset_interval
            special = (not init0) or (
                st > 0 and ri > 0 and (s0 + 1) % ri == 0)
            if special:
                k += len(bufs)
                for t in bufs + [lev.codebook.weight.data]:
                    dist.broadcast(t, src=0)
                flag = lev.initialized.to(torch.uint8)
                dist.broadcast(flag, src=0)
                lev.initialized.copy_(flag.to(lev.initialized.dtype))
                n_bcast += 1
                continue
            d = lev.decay if st > 0 else 1.0
            for b, o in zip(bufs, olds):
                b.copy_(d * o + summed[k]); k += 1
            n = lev.ema_count.sum()
            cs = (lev.ema_count + lev.eps) \
                / (n + lev.num_codes * lev.eps) * n
            lev.codebook.weight.data.copy_(lev.ema_sum / cs.unsqueeze(1))
            if hasattr(lev, "project_book"):
                lev.project_book()
        self._old = None
        return n_bcast


def state_checksum(model):
    """Max across ranks of |local checksum - rank 0's|, over every
    parameter and floating buffer. Zero when the ranks agree."""
    if not is_distributed():
        return 0.0
    dev = next(model.parameters()).device
    with torch.no_grad():
        t = torch.stack([x.detach().double().sum() for x in
                         list(model.parameters()) + [b for b in model.buffers()
                                                     if b.is_floating_point()]])
    ref = t.clone()
    dist.broadcast(ref, src=0)
    diff = (t - ref).abs().max().reshape(1).float()
    dist.all_reduce(diff, op=dist.ReduceOp.MAX)
    return float(diff.item())


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
