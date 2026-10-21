"""model.coarse_decoder: a conv decoder for coarse-only rows, routed inside
PyramidCodec.decode, and train.coarse_decoder_only, which trains it alone."""
import os

import numpy as np
import pytest
import torch

from georvq.config import Config, load_config
from georvq.data.build import ShardWriter
from georvq.factory import build_model


def _cfg(coarse=True, **model):
    m = {"window": 24, "bands": 13, "tiers": ["1:16:32:3", "4:16:32:3"],
         "min_depths": [3, 0],
         "tokenizer": {"kind": "source", "dim": 16},
         "decoder": {"dim": 32, "depth": 1, "heads": 4, "head_feat": 8,
                     "cls_bridge": False}, **model}
    if coarse:
        m["coarse_decoder"] = {"widths": [24, 16, 8], "blocks": 1}
    return Config.model_validate({
        "data": {"source": "s2", "store_dir": "/tmp/unused"}, "model": m})


def _zqs(B=6):
    g = torch.Generator().manual_seed(1)
    return [torch.randn(B, 1, 16, generator=g),
            torch.randn(B, 16, 16, generator=g)]


def test_coarse_decoder_shapes_and_no_conditioning():
    m = build_model(_cfg()).eval()
    cd = m.coarse_decoder
    names = [n for n, _ in cd.named_parameters()]
    assert names and all(n.split(".")[0] in {
        "fc", "stage_60m", "up_20m", "stage_20m", "up_10m", "stage_10m",
        "heads"} for n in names)
    assert not any(w in n.lower() for n in names
                   for w in ("cond", "ada", "depth", "therm", "emb"))
    out = cd(torch.randn(5, 16))
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        "10m": (5, 4, 24, 24), "20m": (5, 6, 12, 12), "60m": (5, 3, 4, 4)}


def test_preset_size():
    """configs/s2/beta_arch_w36_coarse.yaml: the coarse decoder's size,
    and the frozen model's tensors keep their names."""
    m = build_model(load_config("configs/s2/beta_arch_w36_coarse.yaml"))
    n = sum(p.numel() for p in m.coarse_decoder.parameters())
    assert 1.0e6 < n < 3.0e6
    base = build_model(load_config("configs/s2/beta_arch_w36.yaml"))
    extra = set(m.state_dict()) - set(base.state_dict())
    assert set(base.state_dict()) <= set(m.state_dict())
    assert extra and all(k.startswith("coarse_decoder.") for k in extra)


def test_routing():
    """Rows whose fine tier is at depth 0 come from the coarse decoder and
    the others from the main decoder, each equal to a direct call on those
    rows."""
    m = build_model(_cfg()).eval()
    zqs = _zqs()
    cond = torch.tensor([[3, 0], [3, 2], [2, 0], [3, 3], [1, 1], [3, 0]])
    with torch.no_grad():
        out = m.decode(zqs, cond)
        ci, fi = torch.tensor([0, 2, 5]), torch.tensor([1, 3, 4])
        oc = m.coarse_decoder(zqs[0][ci, 0])
        of = m.decoder([z[fi] for z in zqs], cond[fi])
    for g in out:
        assert torch.equal(out[g][ci], oc[g])
        assert torch.equal(out[g][fi], of[g])


def test_off_is_unchanged():
    """With the option off nothing is built and decode is the main
    decoder. With it on, the other modules draw the same weights from the
    seed and a batch with no coarse-only row decodes to the same bits."""
    torch.manual_seed(0)
    off = build_model(_cfg(coarse=False)).eval()
    torch.manual_seed(0)
    on = build_model(_cfg()).eval()
    assert off.coarse_decoder is None
    so, sn = off.state_dict(), on.state_dict()
    assert set(so) <= set(sn)
    assert all(k.startswith("coarse_decoder.") for k in set(sn) - set(so))
    assert all(torch.equal(so[k], sn[k]) for k in so)
    zqs = _zqs()
    cond = torch.tensor([[3, 1], [3, 2], [2, 2], [3, 3], [1, 1], [3, 1]])
    with torch.no_grad():
        a, b = off.decode(zqs, cond), on.decode(zqs, cond)
        d = off.decoder(zqs, cond)
    for g in a:
        assert torch.equal(a[g], b[g]) and torch.equal(a[g], d[g])


def test_config_checks():
    with pytest.raises(ValueError, match="single attention decoder"):
        _cfg(decoder_bank=[{"tiers": 1, "out_px": 24, "dim": 32, "depth": 1},
                           {"tiers": 2, "out_px": 24, "dim": 32, "depth": 1}])
    base = _cfg(coarse=False).model_dump(mode="json")
    base["train"] = {"coarse_decoder_only": True, "init_from": "/tmp/x"}
    base["model"]["vq"]["warmup_steps"] = 0
    with pytest.raises(ValueError, match="needs model.coarse_decoder"):
        Config.model_validate(base)


GROUPS = {"10m": (4, 48), "20m": (6, 24), "60m": (3, 8)}


@pytest.fixture
def s2_store(tmp_path):
    rng = np.random.default_rng(0)
    w = ShardWriter(tmp_path / "s2", shard_size=4, dtype=np.uint16,
                    groups=GROUPS)
    for i in range(6):
        chip = {g: rng.integers(1, 10000, size=(b, px, px), dtype=np.uint16)
                for g, (b, px) in GROUPS.items()}
        w.add(chip, {"scene_id": f"s{i//4}", "patch_row": i, "patch_col": 0,
                     "crs": "EPSG:32601", "lon": float(i), "lat": 0.0,
                     "year": 0}, log_every=0)
    w.finalize("s2", nodata=0, grid_period=6)
    return tmp_path / "s2"


def test_coarse_decoder_only_trains_only_the_coarse_decoder(s2_store,
                                                            tmp_path):
    """After a few steps with a validation in between, every tensor of
    the starting checkpoint, the main decoder's included, is
    bit-identical, and the coarse decoder has been added."""
    from georvq.repro import load_checkpoint
    from georvq.train.loop import train

    def cfg(**over):
        d = _cfg(coarse=False).model_dump(mode="json")
        d["data"] = {"source": "s2", "store_dir": str(s2_store),
                     "val_frac": 0.0,
                     "pool": {"size": 6, "workers": 0, "device": "cpu"}}
        d["model"]["vq"].update({"warmup_steps": 2, "init_mode": "tree",
                                 "init_per_code": 4})
        d["train"] = {"steps": 4, "batch_windows": 8, "amp": False,
                      "val_interval": 4, "fig_interval": 100,
                      "save_interval": 4}
        d["log"] = {"run_tag": "src", "wandb": False, "out_dir": str(tmp_path)}
        for k, v in over.items():
            sec, key = k.split("__")
            if isinstance(v, dict) and isinstance(d[sec].get(key), dict):
                d[sec][key].update(v)
            else:
                d[sec][key] = v
        return Config.model_validate(d)

    src = os.path.join(train(cfg(), gpu=-1), "model.pt")
    run = train(cfg(model__coarse_decoder={"widths": [24, 16, 8],
                                           "blocks": 1},
                    model__vq={"warmup_steps": 0},
                    train__init_from=src, train__coarse_decoder_only=True,
                    train__steps=6, train__val_interval=3,
                    train__save_interval=6, log__run_tag="coarse"),
                gpu=-1)
    a = load_checkpoint(src)["model"]
    b = load_checkpoint(os.path.join(run, "model.pt"))["model"]
    assert set(a) <= set(b)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(k.startswith("coarse_decoder.") for k in b)
