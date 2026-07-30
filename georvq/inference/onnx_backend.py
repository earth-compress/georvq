"""ONNX runtime for the deployment codec: export + a torch-free backend.

`export_onnx` splits a trained `PyramidCodec` into the two neural halves —
encoder (stem+tokenizer -> per-tier latents) and decoder (quantized latents
+ depth tuple -> pixels) — and exports each as an ONNX graph with a dynamic
batch axis. The RVQ itself (nearest-code assignment and codebook-sum
reconstruction) is plain linear algebra and runs in numpy against the
exported `codebooks.npz`, so the bitstream-defining math is identical bytes
in every backend.

`OnnxCodec` mirrors the `georvq.inference.api.Codec` encode/decode surface
(numpy in, `Packet`s out) with onnxruntime sessions — CPU by default, CUDA
when requested and available — and needs no torch at inference time.

Precision note: the torch path may assign codes with bf16 scores
(`vq.bf16_scores`); the numpy path always scores in fp32. Both are valid
encoders for the same bitstream (each stream decodes exactly), but a rare
boundary vector may map to a different (equidistant-ish) code, so archives
are backend-consistent, not backend-identical. Decoding any given packet is
deterministic in all backends.

Models with conditioning submodules (cascade / selector) interleave neural
priors with assignment and are not exported yet — `export_onnx` refuses
them rather than silently producing a wrong bitstream.
"""
from __future__ import annotations

import json
import os

import numpy as np

from georvq.inference.api import Packet

META_NAME = "meta.json"
CODEBOOKS_NAME = "codebooks.npz"
ENCODER_NAME = "encoder.onnx"
DECODER_NAME = "decoder.onnx"
DEBLOCK_NAME = "deblock.onnx"


def export_deblock_onnx(net, out_dir, entry, deblock_decoder_id,
                        opset=17):
    """Export a deblock post-processor (georvq.ae.deblock.AEDeblocker or
    compatible) as deblock.onnx with dynamic batch/height/width, and
    record its recipe + identity in the export dir's meta.json.

    The seam mask is a graph INPUT (B,1,H,W) rather than computed
    in-graph: it is two lines of numpy for the client and keeps the
    graph free of shape-dependent index arithmetic. Height/width are
    exported as 4*Dim (the down/up path needs divisibility by 4; every
    window-aligned frame satisfies it). Returns the meta fragment."""
    import torch

    net = net.float().eval()

    class _Deb(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x, mask, Ls):
            return self.n(x, Ls, mask=mask)

    B, C = 2, net.bands
    x = torch.randn(B, C, 320, 384)          # H != W: keep dims distinct
    from georvq.ae.deblock import boundary_mask
    mask = boundary_mask(320, 384, net.win, x.device) \
        .expand(B, -1, -1, -1).contiguous()
    Ls = torch.tensor([[8, 8, 8]] * B, dtype=torch.long)
    b = torch.export.Dim("batch", min=1)
    h4 = torch.export.Dim("h4", min=16)
    w4 = torch.export.Dim("w4", min=16)
    sp = {0: b, 2: 4 * h4, 3: 4 * w4}
    torch.onnx.export(
        _Deb(net).eval(), (x, mask, Ls),
        os.path.join(out_dir, DEBLOCK_NAME),
        input_names=["x", "mask", "Ls"], output_names=["deblocked"],
        dynamic_shapes=(sp, sp, {0: b}), opset_version=opset)

    frag = {"recipe": entry["recipe"],
            "trained_against": entry["trained_against"],
            "decoder_id": deblock_decoder_id, "win": net.win}
    mpath = os.path.join(out_dir, META_NAME)
    with open(mpath) as f:
        meta = json.load(f)
    meta["deblock"] = frag
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=1)
    return frag


# --------------------------------------------------------------------------
# export (torch side)
# --------------------------------------------------------------------------

def export_onnx(model, out_dir, opset=17):
    """Export a PyramidCodec to `out_dir` (encoder.onnx, decoder.onnx,
    codebooks.npz, meta.json). Returns the meta dict."""
    import torch

    if model.conditioner is not None or model.selector is not None:
        raise NotImplementedError(
            "ONNX export currently covers unconditioned codecs only "
            "(cascade/selector models interleave priors with assignment)")

    os.makedirs(out_dir, exist_ok=True)
    model = model.float().eval()
    specs = model.specs

    class _Enc(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return tuple(self.m.encode_levels(x))

    class _Dec(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, zqs, Ls):
            return self.m.decode(list(zqs), Ls)

    B, W, C = 2, model.window, model.bands
    batch = torch.export.Dim("batch", min=1)
    x = torch.randn(B, C, W, W)
    z_names = [f"z{i}" for i in range(len(specs))]
    torch.onnx.export(
        _Enc(model).eval(), (x,), os.path.join(out_dir, ENCODER_NAME),
        input_names=["x"], output_names=z_names,
        dynamic_shapes=({0: batch},), opset_version=opset)

    with torch.no_grad():
        zs = model.encode_levels(x)
    zq_names = [f"zq{i}" for i in range(len(specs))]
    Ls = torch.tensor([[sp.depth for sp in specs]] * B, dtype=torch.long)
    torch.onnx.export(
        _Dec(model).eval(), (tuple(z.float() for z in zs), Ls),
        os.path.join(out_dir, DECODER_NAME),
        input_names=zq_names + ["Ls"], output_names=["recon"],
        dynamic_shapes=(tuple({0: batch} for _ in specs), {0: batch}),
        opset_version=opset)

    books = {f"cb_t{i}_l{j}": lev.codebook.weight.detach()
             .float().cpu().numpy()
             for i, rvq in enumerate(model.rvqs)
             for j, lev in enumerate(rvq.levels)}
    np.savez(os.path.join(out_dir, CODEBOOKS_NAME), **books)

    from georvq.inference.release import bitstream_id, decoder_id
    meta = {
        "format": "georvq-onnx/1",
        "window": model.window,
        "bands": model.bands,
        "tiers": [{"grid": sp.grid, "dim": sp.dim, "codes": sp.codes,
                   "depth": sp.depth} for sp in specs],
        "norm_latents": bool(getattr(model, "norm_latents", False)),
        "bitstream_id": bitstream_id(model),
        "decoder_id": decoder_id(model),
        "opset": opset,
    }
    with open(os.path.join(out_dir, META_NAME), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


# --------------------------------------------------------------------------
# numpy RVQ (backend-shared math)
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
# runtime (torch-free)
# --------------------------------------------------------------------------

class OnnxCodec:
    """Codec-shaped encode/decode over onnxruntime sessions + numpy RVQ.

    Accepts and returns numpy arrays; packets are interchangeable with the
    torch `Codec`'s (same `Packet` type, same bitstream)."""

    def __init__(self, export_dir, device="cpu", threads=None):
        import onnxruntime as ort
        with open(os.path.join(export_dir, META_NAME)) as f:
            self.meta = json.load(f)
        self.window = self.meta["window"]
        self.bands = self.meta["bands"]
        self.tiers = self.meta["tiers"]
        z = np.load(os.path.join(export_dir, CODEBOOKS_NAME))
        self.books = [[z[f"cb_t{i}_l{j}"] for j in range(t["depth"])]
                      for i, t in enumerate(self.tiers)]
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(
                    "onnxruntime has no CUDA provider in this install; "
                    "use --device cpu or pip install onnxruntime-gpu")
            providers = ["CUDAExecutionProvider"] + providers
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.enc = ort.InferenceSession(
            os.path.join(export_dir, ENCODER_NAME), opts, providers=providers)
        self.dec = ort.InferenceSession(
            os.path.join(export_dir, DECODER_NAME), opts, providers=providers)

    def encode(self, x, coarse_residual=False) -> list[Packet]:
        """x: (B, C, W, W) float32 prepared input -> full-depth Packets
        (optionally with the tier-0 post-quantization residual, 8*)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        zs = self.enc.run(None, {"x": x})
        stacked, res0 = [], None
        for t, (z, books) in enumerate(zip(zs, self.books)):
            if t == 0 and coarse_residual:
                idx, res0 = rvq_assign_np(z, books, return_residual=True)
            else:
                idx = rvq_assign_np(z, books)
            stacked.append(idx)
        return [Packet(indices=[t[b] for t in stacked],
                       coarse_residual=res0[b] if res0 is not None else None)
                for b in range(x.shape[0])]

    def decode(self, packets: list[Packet], depths=None):
        """Packets -> (B, C, W, W) float32 reconstruction at a depth tuple
        (default: full)."""
        depths = depths or [t["depth"] for t in self.tiers]
        feeds = {}
        for i, (books, L) in enumerate(zip(self.books, depths)):
            idx = np.stack([np.asarray(p.indices[i]) for p in packets])
            feeds[f"zq{i}"] = rvq_lookup_np(idx, books, L)
        if depths[0] == self.tiers[0]["depth"] \
                and packets[0].coarse_residual is not None:
            feeds["zq0"] = feeds["zq0"] + np.stack(
                [p.coarse_residual for p in packets]).astype(np.float32)
        feeds["Ls"] = np.tile(np.asarray(depths, np.int64),
                              (len(packets), 1))
        return self.dec.run(None, feeds)[0]

    def packet_bits(self, depths=None) -> float:
        depths = depths or [t["depth"] for t in self.tiers]
        return float(sum(t["grid"] ** 2 * L * np.log2(t["codes"])
                         for t, L in zip(self.tiers, depths)))
