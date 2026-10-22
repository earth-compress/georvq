"""ONNX export of the S2 v1 decoder, and an onnxruntime codec that reads
S2 v1 stores without torch.

The export holds the decode side only, which is all that reading a store
needs (encoding stays with the torch codec):

- `decoder.onnx`: (zq0, zq1, Ls, stats) -> (dn_10m, dn_20m, dn_60m), the
  decoder's three grids denormalised by the decoded statistics (B, 26)
  and scaled to DN, as `S2V1Codec._forward` computes them;
- `codebooks.npz`: the RVQ codebooks, looked up in numpy
  (georvq.inference.onnx_stats.rvq_lookup_np);
- `mean_channel.npz`: the statistics channel's tables;
- `meta.json`: format, ids, tiers, window, rungs and the channel's hash.

Importing this module does not import torch; exporting does.
"""
from __future__ import annotations

import json
import os

import numpy as np

FORMAT = "georvq-s2-onnx/1"
META_NAME = "meta.json"
DECODER_NAME = "decoder.onnx"
CODEBOOKS_NAME = "codebooks.npz"
CHANNEL_NAME = "mean_channel.npz"


def export_s2_onnx(codec, out_dir, opset=18) -> dict:
    """Export the decode side of an S2V1Codec to `out_dir`. Returns the
    meta dict."""
    import torch

    from georvq.inference.store_codec import channel_sha
    from georvq.s2.v1 import GRIDS, MAX_VALUE
    os.makedirs(out_dir, exist_ok=True)
    model = codec.model.float().eval()
    specs = codec.specs
    nb = 13

    class _Dec(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, zq0, zq1, Ls, stats):
            out = self.m.decode([zq0, zq1], Ls, stats=None, valid=None)
            res, i = [], 0
            for g, _, names in GRIDS:
                k = len(names)
                mu = stats[:, i:i + k, None, None]
                sd = stats[:, nb + i:nb + i + k, None, None]
                res.append((out[g].float() * sd + mu) * MAX_VALUE)
                i += k
            return tuple(res)

    B = 2
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(0)
    zq = [torch.randn(B, sp.cells, sp.dim, generator=g).to(dev)
          for sp in specs]
    Ls = torch.tensor([[sp.depth for sp in specs]] * B, dtype=torch.long,
                      device=dev)
    stats = torch.rand(B, 2 * nb, generator=g).to(dev) * 0.1 + 0.01
    batch = torch.export.Dim("batch", min=1)
    torch.onnx.export(
        _Dec(model).eval(), (zq[0], zq[1], Ls, stats),
        os.path.join(out_dir, DECODER_NAME),
        input_names=["zq0", "zq1", "Ls", "stats"],
        output_names=[f"dn_{g}" for g, _, _ in GRIDS],
        dynamic_shapes=({0: batch}, {0: batch}, {0: batch}, {0: batch}),
        opset_version=opset, dynamo=True)
    books = {f"cb_t{i}_l{j}": lev.codebook.weight.detach().float().cpu()
             .numpy() for i, rvq in enumerate(model.rvqs)
             for j, lev in enumerate(rvq.levels)}
    np.savez(os.path.join(out_dir, CODEBOOKS_NAME), **books)
    codec.channel.save(os.path.join(out_dir, CHANNEL_NAME))
    meta = {"format": FORMAT, "codec_format": codec.spec.codec_format,
            "window": codec.window, "bands": codec.bands,
            "tiers": [{"grid": sp.grid, "dim": sp.dim, "codes": sp.codes,
                       "depth": sp.depth} for sp in specs],
            "bitstream_id": codec.bitstream_id,
            "decoder_id": codec.decoder_id,
            "rungs": [list(r) for r in codec.rungs],
            "mean_channel": CHANNEL_NAME,
            "mean_channel_sha": channel_sha(codec.channel), "opset": opset}
    with open(os.path.join(out_dir, META_NAME), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


class S2V1OnnxCodec:
    """Decode side of an S2 v1 codec over an `export_s2_onnx` directory,
    with numpy and onnxruntime only. Decoding runs in fp32, so on CPU it
    agrees with the torch fp32 decode up to float rounding in the graph."""

    spec = None

    def __init__(self, onnx_dir, device="cpu", threads=None, spec=None):
        import onnxruntime as ort

        from georvq.config import TierSpec
        from georvq.inference.mean_codec import load_channel
        from georvq.inference.records import tier_names
        from georvq.inference.store_codec import channel_sha
        from georvq.s2 import v1
        self.spec = spec or v1.S2_V1
        with open(os.path.join(onnx_dir, META_NAME)) as f:
            meta = self.meta = json.load(f)
        if meta.get("format") != FORMAT or \
                meta.get("codec_format") != self.spec.codec_format:
            raise ValueError(f"{onnx_dir}: not an ONNX export of "
                             f"{self.spec.codec_format}")
        self.channel = load_channel(os.path.join(onnx_dir, meta["mean_channel"]))
        if channel_sha(self.channel) != meta["mean_channel_sha"]:
            raise ValueError(f"{onnx_dir}: statistics channel tables do not "
                             "match the export's record of them")
        self.specs = [TierSpec(**t) for t in meta["tiers"]]
        self.window = int(meta["window"])
        self.bands = int(meta["bands"])
        self.full = tuple(sp.depth for sp in self.specs)
        self.rungs = [tuple(r) for r in meta["rungs"]]
        self.names = tier_names(len(self.specs))
        self.n_mean = len(self.channel.field_widths())
        self.bitstream_id = meta["bitstream_id"]
        self.decoder_id = meta["decoder_id"]
        self.bf16 = False
        z = np.load(os.path.join(onnx_dir, CODEBOOKS_NAME))
        self.books = [[z[f"cb_t{i}_l{j}"] for j in range(sp.depth)]
                      for i, sp in enumerate(self.specs)]
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider"] + providers
        opts = ort.SessionOptions()
        if not threads and hasattr(os, "sched_getaffinity"):
            threads = len(os.sched_getaffinity(0))
        if threads:
            opts.intra_op_num_threads = threads
        self.dec = ort.InferenceSession(os.path.join(onnx_dir, DECODER_NAME),
                                        opts, providers=providers)
        self._dec_in = {i.name for i in self.dec.get_inputs()}

    def decode_field(self, base, tiers, depths):
        """{grid: (n, C, w, w)} float32 DN field from records."""
        from georvq.inference.onnx_stats import rvq_lookup_np
        from georvq.s2.v1 import GRIDS
        n = base.shape[0]
        sp0 = self.specs[0]
        stats = self.channel.decode(base[:, :self.n_mean])
        coarse = base[:, self.n_mean:].reshape(n, sp0.depth, sp0.cells)
        idx = [coarse.transpose(0, 2, 1)] + [
            None if t is None else t.transpose(0, 2, 1) for t in tiers]
        zqs = []
        for sp, books, i, L in zip(self.specs, self.books, idx, depths):
            if int(L) == 0:
                zqs.append(np.zeros((n, sp.cells, sp.dim), np.float32))
            else:
                zqs.append(rvq_lookup_np(np.asarray(i), books, int(L)))
        feeds = {"zq0": zqs[0], "zq1": zqs[1],
                 "Ls": np.tile(np.asarray(depths, np.int64), (n, 1)),
                 "stats": np.ascontiguousarray(stats, np.float32)}
        out = self.dec.run(None, {k: v for k, v in feeds.items()
                                  if k in self._dec_in})
        return {g: o for (g, _, _), o in zip(GRIDS, out)}

    def decode_records(self, base, tiers, depths, valid, trace=None,
                       **options):
        """As StoreCodec.decode_records: {grid: (n, C, w, w)} uint16."""
        from georvq.inference.records import span
        with span(trace, "decode", "lookup + decoder (onnx)"):
            field = self.decode_field(base, tiers, depths)
        opts = self.spec.options(options)
        with span(trace, "decode", "postprocess"):
            return self.spec.postprocess_np(field, np.asarray(valid, bool),
                                            **opts)
