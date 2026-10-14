"""Golden encode/decode gate on the released Tessera bundle
(tessera-beta-v1).

The gate needs the bundle (GEORVQ_TESSERA_BUNDLE, default
/data/georvq-releases/tessera-beta-v1/bundle) and skips when it is
absent. Its inputs are three 64 px windows of Tessera validation chips
(golden/tessera_val_windows.npz: int8 codes and the float16 scale plane,
as geotessera hands them out; the third window is partly nodata).
golden/tessera_golden.npz holds the indices and mean-channel codes the
bundle's encoder produced for them on CPU, the sha256 of the packed
refinement, and sha256 hashes of the float32 field and of the int8
requantized field decoded at four rungs.

Everything is compared exactly. The hashes were recorded on CPU in fp32
(cirrus, x86-64); a CPU with other vector kernels, or another torch
version, could produce different float bits, in which case the test
reports the relative L2 error next to the recorded one so the size of
the difference is visible. Re-record with

    python -m tests.test_tessera_bundle record

and rebuild the inputs from the chip store with

    python -m tests.test_tessera_bundle inputs
"""
import hashlib
import os
import sys

import numpy as np
import pytest
import torch

from georvq.inference import refine as _refine
from georvq.tessera import load, prep, requantize

HERE = os.path.dirname(__file__)
BUNDLE = os.environ.get("GEORVQ_TESSERA_BUNDLE",
                        "/data/georvq-releases/tessera-beta-v1/bundle")
STORE = "/data/tessera-samples/chips_512"
INPUTS = os.path.join(HERE, "golden/tessera_val_windows.npz")
GOLDEN = os.path.join(HERE, "golden/tessera_golden.npz")
# (depths, refine): the ladder's ends and a fine-tier prefix, and the
# refined full depth
GOLDEN_RUNGS = (((20, 0, 0), False), ((20, 20, 5), False),
                ((20, 20, 20), False), ((20, 20, 20), True))
W = 64


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _rel_l2(xhat, x, valid):
    e = (xhat - x).pow(2).sum(1).sqrt() / x.pow(2).sum(1).sqrt().clamp_min(
        1e-12)
    return float(e[valid].mean())


def _code(R, emb, scale):
    """Packets without and with the refinement, the prepared input, and
    {rung index: (float32 field, int8 requantized field)}."""
    x, valid = prep(emb, scale)
    plain = R.encode(x, valid)
    refined = R.encode(x, valid, refine=True)
    out = {}
    for i, (depths, ref) in enumerate(GOLDEN_RUNGS):
        f = R.decode(refined if ref else plain, depths=list(depths),
                     valid=valid)
        out[i] = (f.numpy(), requantize(f, scale)[1].numpy())
    return plain, refined, x, valid, out


def _packed_refinement(R, packets):
    t = R.refine.tier
    res = np.stack([p.refinements[t] for p in packets])
    codes, _ = _refine.pack(res, R.refine, R.refine_table)  # lloyd: no step
    return codes


@pytest.fixture(scope="module")
def released():
    if not os.path.exists(os.path.join(BUNDLE, "bundle.json")):
        pytest.skip("Tessera bundle not present (set GEORVQ_TESSERA_BUNDLE)")
    torch.set_grad_enabled(False)
    R = load(BUNDLE, "cpu")
    z = np.load(INPUTS)
    emb, scale = z["embedding"], z["scale"]
    yield (R, emb, scale) + _code(R, emb, scale)
    torch.set_grad_enabled(True)


def test_golden_codes(released):
    R, _, _, plain, refined, _, _, _ = released
    g = np.load(GOLDEN)
    assert R.bundle["bitstream_id"] == str(g["bitstream_id"])
    assert R.bundle["decoder_id"] == str(g["decoder_id"])
    for t in range(len(R.specs)):
        got = np.stack([p.indices[t] for p in plain])
        assert np.array_equal(got, g[f"idx_t{t}"]), t
        assert all(np.array_equal(p.indices[t], q.indices[t])
                   for p, q in zip(plain, refined)), t
    mu = R.mean_channel.encode(np.stack([p.mean for p in plain]))
    assert np.array_equal(mu, g["mean_codes"])
    assert _sha(_packed_refinement(R, refined)) == str(g["sha_refine"])


def test_golden_decode(released):
    R, _, _, _, _, x, valid, dec = released
    g = np.load(GOLDEN)
    for i, (depths, ref) in enumerate(GOLDEN_RUNGS):
        f, q = dec[i]
        name = "-".join(map(str, depths)) + ("+r" if ref else "")
        assert _sha(f) == str(g["sha_field"][i]), (
            f"{name}: float field differs, rel L2 "
            f"{_rel_l2(torch.from_numpy(f), x, valid):.9f} vs recorded "
            f"{float(g['rel_l2'][i]):.9f}")
        assert _sha(q) == str(g["sha_int8"][i]), name


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------

def make_inputs():
    """Three windows of geographic-split validation chips: two whole
    windows at the chip origin and one window with 20-80% nodata."""
    import pandas as pd
    from georvq.data.store import geographic_split
    index = pd.read_parquet(os.path.join(STORE, "index.parquet"))
    _, val = geographic_split(index, 0.04)

    def chip(row):
        a = np.load(os.path.join(STORE, f"shard_{row.shard:04d}.npy"),
                    mmap_mode="r")[row.slot]
        s = np.load(os.path.join(STORE, f"scales_{row.shard:04d}.npy"),
                    mmap_mode="r")[row.slot]
        return a, s

    whole = val[val.nodata_frac == 0].iloc[[0, 2]]
    embs, scales, ids = [], [], []
    for row in whole.itertuples():
        a, s = chip(row)
        embs.append(np.array(a[:, :W, :W]))
        scales.append(np.array(s[:W, :W]))
        ids.append((row.chip_id, 0, 0))
    for row in val[val.nodata_frac > 0.05].itertuples():
        a, s = chip(row)
        hit = None
        for r0 in range(0, a.shape[1] - W + 1, W):
            for c0 in range(0, a.shape[2] - W + 1, W):
                nod = (np.asarray(s[r0:r0 + W, c0:c0 + W]) == 0).mean()
                if 0.2 < nod < 0.8:
                    hit = (r0, c0)
                    break
            if hit:
                break
        if hit:
            r0, c0 = hit
            embs.append(np.array(a[:, r0:r0 + W, c0:c0 + W]))
            scales.append(np.array(s[r0:r0 + W, c0:c0 + W]))
            ids.append((row.chip_id, r0, c0))
            break
    np.savez_compressed(INPUTS, embedding=np.stack(embs),
                        scale=np.stack(scales),
                        chip_row_col=np.array(ids, dtype=np.int64))
    print(INPUTS, os.path.getsize(INPUTS), ids)


def record():
    torch.set_grad_enabled(False)
    R = load(BUNDLE, "cpu")
    z = np.load(INPUTS)
    emb, scale = z["embedding"], z["scale"]
    plain, refined, x, valid, dec = _code(R, emb, scale)
    out = {f"idx_t{t}": np.stack([p.indices[t] for p in plain])
           for t in range(len(R.specs))}
    out.update(
        bitstream_id=np.array(R.bundle["bitstream_id"]),
        decoder_id=np.array(R.bundle["decoder_id"]),
        mean_codes=R.mean_channel.encode(np.stack([p.mean for p in plain])),
        sha_refine=np.array(_sha(_packed_refinement(R, refined))),
        rungs=np.array([list(d) + [int(r)] for d, r in GOLDEN_RUNGS]),
        sha_field=np.array([_sha(dec[i][0]) for i in dec]),
        sha_int8=np.array([_sha(dec[i][1]) for i in dec]),
        rel_l2=np.array([_rel_l2(torch.from_numpy(dec[i][0]), x, valid)
                         for i in dec]))
    np.savez_compressed(GOLDEN, **out)
    for (d, r), e in zip(GOLDEN_RUNGS, out["rel_l2"]):
        print("-".join(map(str, d)) + ("+r" if r else ""), f"rel L2 {e:.6f}")
    print(GOLDEN, os.path.getsize(GOLDEN))


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        record()
    elif sys.argv[1:] == ["inputs"]:
        make_inputs()
