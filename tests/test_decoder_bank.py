"""Resolution-matched decoders: routing, geometry, and equivalence.

A rung carrying only the coarse tier does not need the compute that
decodes the whole token sequence to full resolution, so each tier prefix
gets its own decoder emitting at its own size. Two things have to hold
for that to be safe: a depth tuple must reach the member that can
actually serve it, and grouping a mixed batch must give the same answer
as decoding its rows one at a time.
"""
import pytest
import torch

from georvq.config import Config
from georvq.factory import build_model
from georvq.model.decoder import active_tiers


def _cfg(bank=True, pyramid=True):
    raw = {
        "data": {"source": "s2", "store_dir": "/tmp/georvq-nonexistent"},
        "model": {
            "window": 96, "bands": 13, "ff_ratio": 2.0,
            "tiers": ["1:16:32:2", "4:32:32:2", "16:32:32:2"],
            "min_depths": [1, 0, 0],
            "tokenizer": {"dim": 32, "depth": 1, "stem_pyramid": pyramid},
            "decoder": {"dim": 32, "depth": 1, "heads": 4, "head_feat": 8},
            "vq": {},
        },
        "train": {"batch_windows": 4, "steps": 10, "amp": False,
                  "tf32": False},
        "log": {"run_tag": "test", "wandb": False},
    }
    if bank:
        raw["model"]["decoder_bank"] = [
            {"tiers": 1, "out_px": 12, "dim": 16, "depth": 1, "heads": 4,
             "head_feat": 8},
            {"tiers": 2, "out_px": 48, "dim": 24, "depth": 1, "heads": 4,
             "head_feat": 8},
            {"tiers": 3, "out_px": 96, "dim": 32, "depth": 1, "heads": 4,
             "head_feat": 8},
        ]
    return Config.model_validate(raw)


def _latents(model):
    specs = model.specs
    torch.manual_seed(0)
    return [torch.randn(4, sp.cells, sp.dim) for sp in specs]


@pytest.mark.parametrize("depths,want", [
    ([2, 0, 0], 0),
    ([1, 0, 0], 0),
    ([2, 2, 0], 1),
    ([2, 1, 0], 1),
    ([2, 2, 2], 2),
    ([2, 2, 1], 2),
])
def test_routing_follows_the_active_prefix(depths, want):
    assert active_tiers(depths) - 1 == want


def test_a_gap_in_the_prefix_falls_back_to_the_full_decoder():
    """Tuples saturate a tier before opening the next, so a zero below a
    nonzero has no prefix decoder that could serve it. Routing it to the
    full decoder is wrong-but-correct; routing it to a prefix would drop
    the codes it does carry."""
    assert active_tiers([2, 0, 2]) == 3
    assert active_tiers([0, 0, 2]) == 3


def test_every_rung_decodes_to_the_window():
    model = build_model(_cfg())
    zqs = _latents(model)
    for depths in ([2, 0, 0], [2, 2, 0], [2, 2, 2]):
        Ls = torch.tensor(depths)[None].expand(4, -1)
        out = model.decode(zqs, Ls)
        assert out["10m"].shape == (4, 4, 96, 96)
        assert out["20m"].shape == (4, 6, 48, 48)
        assert out["60m"].shape == (4, 3, 16, 16)


def test_native_output_is_the_members_own_grid():
    model = build_model(_cfg())
    zqs = _latents(model)
    Ls = torch.tensor([2, 0, 0])[None].expand(4, -1)
    out = model.decoder(zqs, Ls, native=True)
    # 12 px on the 10m plane, and each coarser group keeps its own ratio
    assert out["10m"].shape[-1] == 12
    assert out["20m"].shape[-1] == 6
    assert out["60m"].shape[-1] == 2


def test_bucketing_equals_decoding_rows_separately():
    model = build_model(_cfg())
    zqs = _latents(model)
    mixed = torch.tensor([[2, 0, 0], [2, 2, 2], [2, 2, 0], [2, 0, 0]])
    out = model.decode(zqs, mixed)
    assert model.decoder.last_counts == {0: 2, 1: 1, 2: 1}
    # not bit-exact: a group of 2 rows and a single row take different
    # GEMM paths, which is worth ~1e-7 here against outputs of order
    # 1e-1. The invariant is that grouping does not change the answer,
    # not that float32 is associative.
    for i in range(4):
        one = model.decode([z[i:i + 1] for z in zqs], mixed[i:i + 1])
        for g in out:
            d = float((out[g][i] - one[g][0]).abs().max())
            assert d < 1e-5, f"{g}: max|diff| {d:.3e}"


def test_native_refuses_a_mixed_batch():
    """A mixed batch has no single native resolution to return."""
    model = build_model(_cfg())
    zqs = _latents(model)
    mixed = torch.tensor([[2, 0, 0], [2, 2, 2], [2, 2, 2], [2, 2, 2]])
    with pytest.raises(ValueError, match="native"):
        model.decoder(zqs, mixed, native=True)


def test_bank_is_off_by_default():
    """Every existing preset and checkpoint uses the single decoder."""
    model = build_model(_cfg(bank=False, pyramid=False))
    assert getattr(model.decoder, "members", None) is None
    zqs = _latents(model)
    Ls = torch.tensor([2, 2, 2])[None].expand(4, -1)
    assert model.decode(zqs, Ls)["10m"].shape == (4, 4, 96, 96)


@pytest.mark.parametrize("bad,match", [
    ([{"tiers": 1, "out_px": 12}, {"tiers": 3, "out_px": 96}], "tier prefix"),
    ([{"tiers": 1, "out_px": 12}, {"tiers": 2, "out_px": 48},
      {"tiers": 3, "out_px": 48}], "out_px must be the window"),
    ([{"tiers": 1, "out_px": 10}, {"tiers": 2, "out_px": 48},
      {"tiers": 3, "out_px": 96}], "divisible"),
])
def test_the_schema_rejects_a_malformed_bank(bad, match):
    raw = _cfg(bank=False).model_dump()
    raw["model"]["decoder_bank"] = bad
    with pytest.raises(Exception, match=match):
        Config.model_validate(raw)
