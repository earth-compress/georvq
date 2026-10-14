"""The seam-removal post-decoder (georvq.model.deblock).

These pin down what holds for any weights: identity at zero init, the
seam mask on rectangular frames, the locality of the tiled application
(bitwise), and that the post-processor enters the decoder_id.
"""
import torch
import torch.nn.functional as F

from georvq.model.deblock import (RECIPE, Deblocker, boundary_mask,
                                  deblock_mosaic, frame_support,
                                  load_deblocker)


def _unit(b, c, h, w, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(b, c, h, w, generator=g), dim=1,
                       eps=1e-6)


def _tiny_net(perturb=0.0, seed=0, unit_norm=True):
    torch.manual_seed(seed)
    net = Deblocker(bands=8, width=16, cond_dim=8, groups=4,
                    unit_norm=unit_norm).eval()
    if perturb:
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * perturb)
    return net


def test_identity_at_init():
    """Zero-init head + zero-init FiLM -> the untrained deblocker is an
    exact pass-through (up to the final renorm of already-unit input)."""
    net = _tiny_net()
    x = _unit(2, 8, 128, 192)
    y = net(x, torch.tensor([[8, 8, 8], [8, 1, 0]]))
    assert torch.allclose(y, x, atol=1e-5)


def test_boundary_mask_rectangular():
    m = boundary_mask(128, 192, 64, torch.device("cpu"))[0, 0]
    assert m.shape == (128, 192)
    assert m[64].all() and m[:, 64].all() and m[:, 128].all()
    assert m[0, :64].sum() == 0 and m[1, 1] == 0
    # square case matches the research repo's formula
    sq = boundary_mask(128, 128, 64, torch.device("cpu"))[0, 0]
    idx = torch.arange(128)
    on = ((idx % 64 == 0) & (idx > 0)).float()
    assert torch.equal(sq, (on[None, :] + on[:, None]).clamp(max=1.0))


def test_locality_contract():
    """A frame's core output depends on its support windows and nothing
    else — BITWISE. This is the property that makes the canonical recipe
    scheduler-independent (per-frame, slab, or viewport decodes agree)."""
    net = _tiny_net(perturb=0.05)
    Ls = torch.tensor([[8, 8, 8]])
    x = _unit(1, 8, 256, 256)
    y = deblock_mosaic(net, x, Ls, frame=128, halo=64)
    # perturb strictly outside frame (0,0)'s support (its core is
    # [0:128), support [0:192) on both axes)
    x2 = x.clone()
    x2[:, :, 192:, :] = _unit(1, 8, 64, 256, seed=9)
    x2[:, :, :, 192:] = _unit(1, 8, 256, 64, seed=10)
    y2 = deblock_mosaic(net, x2, Ls, frame=128, halo=64)
    assert torch.equal(y[:, :, :128, :128], y2[:, :, :128, :128])
    # ... and the perturbation did change things elsewhere
    assert not torch.allclose(y[:, :, 128:, 128:], y2[:, :, 128:, 128:])


def test_tiling_partitions_exactly():
    """Hard crop covers every output pixel exactly once; output stays
    unit-norm (odd-sized trailing frames included)."""
    net = _tiny_net(perturb=0.05)
    x = _unit(1, 8, 320, 192)
    y = deblock_mosaic(net, x, torch.tensor([[8, 4, 2]]),
                       frame=128, halo=64)
    assert y.shape == x.shape
    assert torch.allclose(y.norm(dim=1), torch.ones(1, 320, 192),
                          atol=1e-4)


def test_frame_support_grid():
    sup = list(frame_support(24, 24, 64))       # 512/64 = 8-window frames
    assert len(sup) == 9
    (r, c, ((sr0, sr1), (sc0, sc1))) = sup[0]
    assert r == (0, 8) and c == (0, 8) and (sr0, sr1) == (0, 9)
    (r, c, ((sr0, sr1), (sc0, sc1))) = sup[-1]
    assert r == (16, 24) and (sr0, sr1) == (15, 24)


def test_decoder_id_variants(tiny_raw_cfg):
    from georvq.config import Config
    from georvq.factory import build_model
    from georvq.inference.release import decoder_id
    torch.manual_seed(0)
    model = build_model(Config.model_validate(tiny_raw_cfg)).eval()
    net = _tiny_net()
    base = decoder_id(model)
    with_post = decoder_id(model, extra_state=net.state_dict(),
                           recipe=RECIPE)
    other_recipe = decoder_id(model, extra_state=net.state_dict(),
                              recipe={**RECIPE, "frame": 256})
    assert len({base, with_post, other_recipe}) == 3
    # stable across calls
    assert with_post == decoder_id(model, extra_state=net.state_dict(),
                                   recipe=RECIPE)


def test_identity_at_init_without_renorm():
    """For physical fields the output is not renormalised, so the
    untrained network passes any input through unchanged."""
    net = _tiny_net(unit_norm=False)
    x = torch.randn(2, 8, 64, 128) * 50.0
    y = net(x, torch.tensor([[8, 8, 8], [2, 0, 0]]))
    assert torch.allclose(y, x, atol=1e-4)


def test_save_and_load_round_trip(tmp_path):
    net = _tiny_net(perturb=0.05)
    path = tmp_path / "deblock.pt"
    torch.save({"arch": {"bands": 8, "width": 16, "cond_dim": 8,
                         "groups": 4},
                "model": net.state_dict(), "recipe": RECIPE,
                "trained_against": {"bitstream_id": "b", "decoder_id": "d"},
                "provenance": {}}, path)
    net2, entry = load_deblocker(str(path))
    x = _unit(1, 8, 128, 128)
    Ls = torch.tensor([[8, 4, 0]])
    assert torch.equal(net(x, Ls), net2(x, Ls))
    assert entry["recipe"] == RECIPE and net2.unit_norm
