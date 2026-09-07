"""Instability trap: a per-step flight recorder and a gradient tripwire.

The tessera ladder runs fail through a fast transition: whole-model grad
norm sits near 0.05 for thousands of steps, ramps 1000x within ~100
steps, and in the terminal case one large step flips the model into a
zero-gradient state (run 202: one row at norm 92, then 5e-4 forever).
The 50-step scalar cadence straddles the event and whole-model norms say
nothing about where in the model it happens.

This module records, EVERY step, per-module-group gradient norms and
per-group Adam update ratios (max |exp_avg| / sqrt(exp_avg_sq + eps),
the direct probe of the small-denominator failure mode) into a ring
buffer, and trips when the pre-clip gradient norm crosses a threshold.
On a trip it saves the model, gradients, optimizer state, batch and RNG
states BEFORE opt.step() runs, so the saved weights are the ones the
exploding gradient was computed at and the step can be replayed exactly.
Optionally it then holds the process open on a localhost pdb socket for
live inspection (`nc 127.0.0.1 <port>` from the same machine).
"""
from __future__ import annotations

import os
import random
import sys
from collections import deque

import numpy as np
import torch


def group_of(name: str) -> str:
    """Map a parameter name to a coarse module group. Granular inside
    the decoder (where the failures live), one bucket per top-level
    module elsewhere."""
    if not name.startswith("decoder."):
        return name.split(".")[0]
    r = name[len("decoder."):]
    if r.startswith("stages."):
        stage = "stage" + str(int(r.split(".")[1]) + 1)
        for tag in ("ada", "attn", "mlp"):
            if f".{tag}." in r:
                return f"dec.{stage}.{tag}"
        return f"dec.{stage}.other"
    if r.startswith("stem."):
        return "dec.stem.ada" if ".ada." in r else "dec.stem.mlp"
    if r.startswith("heads."):
        return "dec.head" + r.split(".")[1]
    if r.startswith("cond_mlps"):
        return "dec.cond"
    if r.startswith(("projs", "pos")):
        return "dec.in"
    return "dec.other"


def rng_states() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _cpu_batch(batch: dict) -> dict:
    out = {}
    for k, t in batch.items():
        if torch.is_tensor(t):
            t = t.detach().cpu()
            if t.dtype is torch.float32 and k == "x":
                t = t.to(torch.bfloat16)   # halve the history footprint
        out[k] = t
    return out


class Trap:
    def __init__(self, cfg, model, out_dir):
        self.cfg = cfg
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.groups: dict[str, list] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.groups.setdefault(group_of(n), []).append((n, p))
        self.flight: deque[dict] = deque(maxlen=cfg.flight_maxlen)
        self.ema = None
        self.dumps = 0
        self.last_dump_step = -10**9
        self.snap = None                    # rolling pre-history snapshot
        self.batches: deque[dict] = deque()
        self._batch_bytes = 0

    # ---- per-step -------------------------------------------------------

    def _group_stats(self, opt) -> dict:
        state = opt.state
        keys, accs = [], []
        for g, ps in self.groups.items():
            gn2, upmax = None, None
            for _, p in ps:
                if p.grad is not None:
                    s = p.grad.detach().float().pow(2).sum()
                    gn2 = s if gn2 is None else gn2 + s
                st = state.get(p)
                if st and "exp_avg_sq" in st:
                    r = (st["exp_avg"].detach().abs() /
                         st["exp_avg_sq"].detach().sqrt().add(1e-8)).max()
                    upmax = r if upmax is None else torch.maximum(upmax, r)
            keys.append(g)
            accs.append((gn2, upmax))
        dev = next((a.device for pair in accs for a in pair
                    if a is not None), "cpu")
        z = torch.zeros((), device=dev)
        flat = torch.stack([a if a is not None else z
                            for pair in accs for a in pair])
        flat = flat.cpu()                # ONE device->host sync per step
        row = {}
        for i, g in enumerate(keys):
            row[f"g_{g}"] = float(flat[2 * i]) ** 0.5
            row[f"u_{g}"] = float(flat[2 * i + 1])
        return row

    def observe(self, step, gnorm, opt, scalars) -> bool:
        """Record one step; return True when the tripwire fires (call
        before opt.step(), while grads and pre-step weights are live)."""
        row = {"step": step, "gnorm": gnorm, **scalars,
               **self._group_stats(opt)}
        self.flight.append(row)
        c = self.cfg
        finite = np.isfinite(gnorm)
        trip = (step >= c.min_step and self.dumps < c.n_dumps and
                step >= self.last_dump_step + c.cooldown and
                (not finite or
                 (gnorm > c.gnorm_abs and
                  (self.ema is None or gnorm > c.gnorm_ratio * self.ema))))
        if finite and (self.ema is None or gnorm < 5 * max(self.ema, 1e-9)):
            self.ema = gnorm if self.ema is None \
                else 0.99 * self.ema + 0.01 * gnorm
        if step % 1000 == 0:
            self.flush_flight()
        return trip

    def after_step(self, step, model, opt, batch):
        c = self.cfg
        if c.snapshot_every and step % c.snapshot_every == 0:
            self.snap = {
                "step": step,
                "model": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()},
                "opt": opt.state_dict(),
                "rng": rng_states(),
            }
        if c.batch_history_bytes:
            b = _cpu_batch(batch)
            b["step"] = step
            nb = sum(t.numel() * t.element_size()
                     for t in b.values() if torch.is_tensor(t))
            self.batches.append(b)
            self._batch_bytes += nb
            while self._batch_bytes > c.batch_history_bytes \
                    and len(self.batches) > 1:
                old = self.batches.popleft()
                self._batch_bytes -= sum(
                    t.numel() * t.element_size()
                    for t in old.values() if torch.is_tensor(t))

    # ---- trip actions ---------------------------------------------------

    def flush_flight(self):
        if not self.flight:
            return
        keys = sorted({k for r in self.flight for k in r})
        arr = {k: np.array([r.get(k, np.nan) for r in self.flight])
               for k in keys}
        np.savez(os.path.join(self.dir, "trap_flight.npz"), **arr)

    def dump(self, step, model, opt, sched, batch, scalars):
        self.dumps += 1
        self.last_dump_step = step
        d = os.path.join(self.dir, f"trap_step{step}")
        os.makedirs(d, exist_ok=True)
        self.flush_flight()
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "grads": {n: p.grad.detach().cpu()
                      for n, p in model.named_parameters()
                      if p.grad is not None},
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "rng": rng_states(),
            "batch": _cpu_batch(batch),
            "scalars": scalars,
        }, os.path.join(d, "bundle.pt"))
        if self.snap is not None:
            torch.save(self.snap, os.path.join(d, "snapshot_pre.pt"))
        if self.batches:
            torch.save(list(self.batches),
                       os.path.join(d, "batch_history.pt"))
        print(f"step {step}: TRAP TRIPPED (dump {self.dumps}/"
              f"{self.cfg.n_dumps}) gnorm={scalars.get('gnorm', float('nan')):.3g}"
              f" -> {d}/bundle.pt (pre-step weights, grads, opt, batch,"
              f" rng)", flush=True)

    def hold(self, ns: dict):
        """Serve a pdb prompt on a localhost socket so the live process
        (GPU tensors, grads, optimizer) can be inspected in place.
        Continues training if nobody connects within hold_timeout."""
        import pdb
        import socket
        port = self.cfg.hold_port
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        srv.settimeout(self.cfg.hold_timeout)
        print(f"TRAP HOLD: attach with  nc 127.0.0.1 {port}  (same host; "
              f"waiting up to {self.cfg.hold_timeout}s). Locals of the "
              f"training step are on the frame above this one; the trap "
              f"object is `self`, extras in `ns`.", flush=True)
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            print("TRAP HOLD: no connection, resuming training",
                  flush=True)
            srv.close()
            return
        fh = conn.makefile("rw", buffering=1)
        dbg = pdb.Pdb(stdin=fh, stdout=fh)
        try:
            # interaction() blocks HERE on the caller's frame and returns
            # when the session ends ('c'); set_trace() would return at
            # once and only break at the next trace event, after the
            # socket teardown below had already run
            dbg.interaction(sys._getframe(1), None)
        finally:
            try:
                fh.close()
                conn.close()
            except OSError:
                pass
            srv.close()
            print("TRAP HOLD: session closed, resuming training",
                  flush=True)
