"""ONNX export and an onnxruntime codec for models with a transmitted mean.

A `stats_cond` model subtracts the per-window stats before the encoder,
conditions encoder and decoder on them, and adds them back after the
decoder, after which a source may constrain the field
(`finalize_uncentered`, e.g. AE renormalises each pixel). The work is
split as follows:

- `encoder.onnx`: (x, stats, valid) -> per-tier continuous latents z0..
  The centring and the encoder's stats embedding are inside the graph.
- `decoder.onnx`: (zq0.., Ls, stats, valid) -> the decoded field, with
  the stats added back, invalid pixels zeroed and the source's
  post-decode step applied, as `PyramidCodec.decode` returns it.
- `codebooks.npz`: residual assignment and the codebook sums run in
  numpy (`rvq_assign_np`, `rvq_lookup_np`), so the indices that define
  the bitstream come from the same arithmetic in every backend.

The stats are whatever the release transmits, already decoded to floats
(for a mean channel, `channel.decode(fields)`). Coding them is the
release layer's job, and so is any further state that defines the
bitstream, which it passes to `export_stats_onnx` as `extra_id` so the
export carries the release's bitstream id.

Exporting needs torch. `OnnxStatsCodec` and the numpy RVQ functions need
only numpy and onnxruntime, and importing this module does not import
torch.

meta.json records `graph_opsets`, the default-domain opset each graph
was written at, and `opset`, the highest of them, which is the opset a
runtime must support. Exports made before this rule recorded the
requested opset as `opset`; the runtime does not read either field.
"""
from __future__ import annotations

import json
import os

import numpy as np

FORMAT = "georvq-onnx-stats/1"
META_NAME = "meta.json"
CODEBOOKS_NAME = "codebooks.npz"
ENCODER_NAME = "encoder.onnx"
DECODER_NAME = "decoder.onnx"


# --------------------------------------------------------------------------
# numpy RVQ
# --------------------------------------------------------------------------

def rvq_assign_np(z, books, return_residual=False):
    """z: (B, cells, dim) fp32; books: list of (K, dim) codebooks.
    Greedy residual assignment -> (B, cells, depth) int64 indices
    (+ the final residual when return_residual)."""
    residual = np.ascontiguousarray(z, dtype=np.float32)
    idxs = []
    for cb in books:
        d = ((residual ** 2).sum(-1, keepdims=True)
             - 2.0 * residual @ cb.T + (cb ** 2).sum(-1))
        idx = d.argmin(-1)
        residual = residual - cb[idx]
        idxs.append(idx)
    out = np.stack(idxs, -1)
    return (out, residual) if return_residual else out


def rvq_lookup_np(indices, books, L=None):
    """(B, cells, depth) indices -> (B, cells, dim) summed vectors at depth
    L (default: full)."""
    L = indices.shape[-1] if L is None else L
    zq = np.zeros((*indices.shape[:2], books[0].shape[1]), np.float32)
    for lev in range(L):
        zq += books[lev][indices[..., lev]]
    return zq


# --------------------------------------------------------------------------
# export (torch side)
# --------------------------------------------------------------------------

def export_stats_onnx(model, out_dir, extra_id=(), meta_extra=None,
                      opset=18):
    """Export a stats_cond PyramidCodec to `out_dir` (encoder.onnx,
    decoder.onnx, codebooks.npz, meta.json). `extra_id` is the release's
    bitstream-defining state, hashed into the recorded bitstream id;
    `meta_extra` is merged into meta.json. `opset` is the opset asked of
    the exporter, which writes a decoder with ReduceL2 (AE's per-pixel
    renormalisation) at 18 even when 17 is asked for. Returns the meta
    dict."""
    import torch

    if model.stats_cond is None:
        raise ValueError("ONNX export supports stats_cond models only")
    if model.conditioner is not None or model.selector is not None:
        raise NotImplementedError(
            "ONNX export covers unconditioned codecs only "
            "(cascade/selector models interleave priors with assignment)")

    os.makedirs(out_dir, exist_ok=True)
    model = model.float().eval()
    specs = model.specs
    sc = model.stats_cond
    width = model.bands * (2 if sc.use_std else 1)

    class _Enc(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, stats, valid):
            return tuple(self.m.encode_levels(x, stats, valid))

    class _Dec(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, zqs, Ls, stats, valid):
            return self.m.decode(list(zqs), Ls, stats, valid)

    B, W, C = 2, model.window, model.bands
    dev = next(model.parameters()).device
    batch = torch.export.Dim("batch", min=1)
    g = torch.Generator().manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(B, C, W, W, generator=g),
                                      dim=1).to(dev)
    valid = torch.ones(B, W, W, dtype=torch.bool, device=dev)
    valid[0, : W // 4, : W // 3] = False
    x = x * valid[:, None]
    stats = model.patch_stats(x, valid).float()
    z_names = [f"z{i}" for i in range(len(specs))]
    torch.onnx.export(
        _Enc(model).eval(), (x, stats, valid),
        os.path.join(out_dir, ENCODER_NAME),
        input_names=["x", "stats", "valid"], output_names=z_names,
        dynamic_shapes=({0: batch}, {0: batch}, {0: batch}),
        opset_version=opset, dynamo=True)

    with torch.no_grad():
        zs = model.encode_levels(x, stats, valid)
    zq_names = [f"zq{i}" for i in range(len(specs))]
    Ls = torch.tensor([[sp.depth for sp in specs]] * B, dtype=torch.long,
                      device=dev)
    torch.onnx.export(
        _Dec(model).eval(), (tuple(z.float() for z in zs), Ls, stats, valid),
        os.path.join(out_dir, DECODER_NAME),
        input_names=zq_names + ["Ls", "stats", "valid"],
        output_names=["recon"],
        dynamic_shapes=(tuple({0: batch} for _ in specs), {0: batch},
                        {0: batch}, {0: batch}),
        opset_version=opset, dynamo=True)

    books = {f"cb_t{i}_l{j}": lev.codebook.weight.detach()
             .float().cpu().numpy()
             for i, rvq in enumerate(model.rvqs)
             for j, lev in enumerate(rvq.levels)}
    np.savez(os.path.join(out_dir, CODEBOOKS_NAME), **books)

    from georvq.inference.release import bitstream_id, decoder_id
    graph_opsets = {n: _graph_opset(os.path.join(out_dir, f))
                    for n, f in (("encoder", ENCODER_NAME),
                                 ("decoder", DECODER_NAME))}
    meta = {
        "format": FORMAT,
        "window": model.window,
        "bands": model.bands,
        "stats": {"width": width, "use_std": bool(sc.use_std),
                  "subtract_mean": bool(sc.subtract_mean)},
        "tiers": [{"grid": sp.grid, "dim": sp.dim, "codes": sp.codes,
                   "depth": sp.depth} for sp in specs],
        "norm_latents": bool(getattr(model, "norm_latents", False)),
        "bitstream_id": bitstream_id(model, extra=extra_id),
        "decoder_id": decoder_id(model),
        "opset": max((v for v in graph_opsets.values() if v is not None),
                     default=None),
        "graph_opsets": graph_opsets,
    }
    meta.update(meta_extra or {})
    with open(os.path.join(out_dir, META_NAME), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def _graph_opset(path):
    """The default-domain opset a graph was written at. The exporter can
    leave a graph above the requested opset when it cannot convert it
    down (seen for the decoder: 18 where 17 was asked for)."""
    import onnx
    m = onnx.load(path, load_external_data=False)
    return next((int(o.version) for o in m.opset_import
                 if o.domain in ("", "ai.onnx")), None)


# --------------------------------------------------------------------------
# runtime (numpy + onnxruntime)
# --------------------------------------------------------------------------

class OnnxStatsCodec:
    """Encode and decode through the exported graphs, with the RVQ in
    numpy. Arrays in and out are numpy; stats are the decoded float
    vectors (B, width) both ends condition on."""

    def __init__(self, export_dir, device="cpu", threads=None):
        import onnxruntime as ort
        with open(os.path.join(export_dir, META_NAME)) as f:
            self.meta = json.load(f)
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{export_dir}: format {self.meta.get('format')!r}"
                             f", expected {FORMAT}")
        self.window = int(self.meta["window"])
        self.bands = int(self.meta["bands"])
        self.tiers = self.meta["tiers"]
        z = np.load(os.path.join(export_dir, CODEBOOKS_NAME))
        self.books = [[z[f"cb_t{i}_l{j}"] for j in range(t["depth"])]
                      for i, t in enumerate(self.tiers)]
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(
                    "onnxruntime has no CUDA provider in this install; "
                    "use device='cpu' or install onnxruntime-gpu")
            providers = ["CUDAExecutionProvider"] + providers
        opts = ort.SessionOptions()
        if not threads and hasattr(os, "sched_getaffinity"):
            # onnxruntime's default is one thread per machine core, pinned;
            # under a Slurm or cgroup CPU limit the pinning fails, so the
            # default is the CPUs this process may run on
            threads = len(os.sched_getaffinity(0))
        if threads:
            opts.intra_op_num_threads = threads
        self.enc = ort.InferenceSession(
            os.path.join(export_dir, ENCODER_NAME), opts, providers=providers)
        self.dec = ort.InferenceSession(
            os.path.join(export_dir, DECODER_NAME), opts, providers=providers)
        # the exporter may drop a graph input the model does not read
        self._enc_in = {i.name for i in self.enc.get_inputs()}
        self._dec_in = {i.name for i in self.dec.get_inputs()}

    @staticmethod
    def _feeds(names, **arrays):
        return {k: v for k, v in arrays.items() if k in names}

    def latents(self, x, stats, valid):
        """(B, C, W, W) prepared input -> per-tier (B, cells, dim) latents."""
        return self.enc.run(None, self._feeds(
            self._enc_in, x=np.ascontiguousarray(x, dtype=np.float32),
            stats=np.ascontiguousarray(stats, dtype=np.float32),
            valid=np.ascontiguousarray(valid, dtype=bool)))

    def encode(self, x, stats, valid):
        """-> per tier (B, cells, depth) int64 full-depth indices."""
        return [rvq_assign_np(z, books)
                for z, books in zip(self.latents(x, stats, valid), self.books)]

    def decode_latents(self, zqs, depths, stats, valid):
        """Quantized latents at a depth tuple -> (B, C, W, W) float32."""
        B = zqs[0].shape[0]
        feeds = {f"zq{i}": np.ascontiguousarray(z, dtype=np.float32)
                 for i, z in enumerate(zqs)}
        feeds.update(self._feeds(
            self._dec_in,
            Ls=np.tile(np.asarray(depths, np.int64), (B, 1)),
            stats=np.ascontiguousarray(stats, dtype=np.float32),
            valid=np.ascontiguousarray(valid, dtype=bool)))
        return self.dec.run(None, feeds)[0]

    def decode(self, indices, depths, stats, valid):
        """indices: per tier (B, cells, >= L) codes, or None where the
        tier's depth is 0. -> (B, C, W, W) float32."""
        B = len(stats)
        zqs = []
        for t, books, idx, L in zip(self.tiers, self.books, indices, depths):
            if int(L) == 0:
                zqs.append(np.zeros((B, t["grid"] ** 2, t["dim"]), np.float32))
            else:
                zqs.append(rvq_lookup_np(np.asarray(idx), books, int(L)))
        return self.decode_latents(zqs, depths, stats, valid)
