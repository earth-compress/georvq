"""Encode/decode throughput by decoder kind and tier prefix.

Untrained weights (throughput only). Decode is timed from quantized
latents to pixels at each fill-order prefix; encode is tokenize +
full-depth quantization. MB/s counts the reconstructed model-space
field (37 x 64 x 64 fp32 = 0.578 MiB per window).
"""
import sys, time

sys.path.insert(0, "/home/ali/projects/georvq/.claude/worktrees/era5-model")
import torch
from georvq.cli import load_config
from georvq.factory import build_model

CONFIGS = {
    "attn": "configs/era5/era5_t64_statscond_std.yaml",
    "tiersum2": "configs/era5/era5_t64_tiersum2.yaml",
    "ladder": "configs/era5/era5_t64_ladder.yaml",
}
B, ITERS, WARM = 256, 20, 3
MIB = 37 * 64 * 64 * 4 / 2 ** 20


def timed(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS


dev = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")
x = torch.randn(B, 37, 64, 64, device=dev) * 5 + 250

print(f"batch {B}, bf16 autocast, {MIB:.3f} MiB/window (fp32 field)\n")
print(f"{'model':>9} {'op':>18} {'ms/batch':>9} {'win/s':>9} {'MiB/s':>9}")
for name, path in CONFIGS.items():
    cfg = load_config(path)
    m = build_model(cfg, dev).eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        xn, v = m.source.prep(x, dev)
        st = m._stats_of(xn, v)
        zs = m.encode_levels(xn, st, v)
        full = torch.tensor([[sp.depth for sp in m.specs]] * B, device=dev)
        zqs, _, _ = m.quantize(zs, full)

        def enc():
            z = m.encode_levels(xn, st, v)
            m.quantize(z, full)
        t = timed(enc)
        print(f"{name:>9} {'encode(full)':>18} {t*1e3:>9.1f} "
              f"{B/t:>9.0f} {B*MIB/t:>9.0f}")
        for tup in ([20, 0, 0], [20, 20, 0], [20, 20, 20]):
            Ls = torch.tensor([tup] * B, device=dev)
            zq_t = [q if L else torch.zeros_like(q)
                    for q, L in zip(zqs, tup)]

            def dec():
                m.decode(zq_t, Ls, st, v)
            t = timed(dec)
            lab = "decode " + "-".join(map(str, tup))
            print(f"{name:>9} {lab:>18} {t*1e3:>9.1f} "
                  f"{B/t:>9.0f} {B*MIB/t:>9.0f}")
    del m
    torch.cuda.empty_cache()
    print()
