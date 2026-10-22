"""The S2 seam deblocker (georvq.sources.s2_deblock) and the multi-grid
form of the tiled application (georvq.model.deblock.deblock_mosaic)."""
import torch
import torch.nn.functional as F

from georvq.sources.s2_deblock import (RECIPE, SCALES, S2BoundaryDeblocker,
                                       deblock_s2_mosaic, from_windows,
                                       load_s2_deblocker, residual_loss,
                                       seam_masks, to_windows)


def _mosaic(b, h, w, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {k: torch.rand(b, n, h // r, w // r, generator=g) * 0.3
            for k, (n, r) in (("10m", (4, 1)), ("20m", (6, 2)),
                              ("60m", (3, 6)))}


def _net(perturb=0.0, seed=0, width=8):
    torch.manual_seed(seed)
    net = S2BoundaryDeblocker(width=width, cond_dim=8).eval()
    if perturb:
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * perturb)
    return net


def test_parameter_count_matches_beta_v4():
    """At width 32 and cond_dim 64 the network has beta_v4's 139,309
    parameters (ring-space releases/beta_v4/deblocker.pt)."""
    net = S2BoundaryDeblocker(width=32, cond_dim=64)
    assert sum(p.numel() for p in net.parameters()) == 139309


def test_identity_at_init():
    net = _net()
    x = _mosaic(2, 252, 216)
    y = net(x, torch.tensor([[8, 0], [8, 8]]))
    for g in x:
        assert torch.allclose(y[g], x[g], atol=1e-5)


def test_window_mean_and_std_preserved():
    net = _net(perturb=0.1)
    x = _mosaic(2, 252, 252)
    y = net(x, torch.tensor([[8, 2], [8, 7]]))
    for g, r in SCALES.items():
        b = 36 // r
        assert not torch.allclose(y[g], x[g], atol=1e-3)
        assert torch.allclose(F.avg_pool2d(y[g], b), F.avg_pool2d(x[g], b),
                              atol=1e-5)
        sd = lambda t: (F.avg_pool2d(t * t, b)  # noqa: E731
                        - F.avg_pool2d(t, b) ** 2).clamp_min(0).sqrt()
        assert torch.allclose(sd(y[g]), sd(x[g]), atol=1e-4)


def test_seam_masks():
    m = seam_masks(72, 108, 36, 6, torch.device("cpu"))[0]
    win, cel = m
    assert win[36].all() and win[:, 72].all() and win[0, 1] == 0
    assert cel[6, 1] == 1 and cel[36, 1] == 0 and cel[1, 1] == 0
    assert ((win > 0) & (cel > 0)).sum() == 0


def test_windows_round_trip():
    x = _mosaic(2, 252, 252)
    w = to_windows(x, 36)
    assert w["10m"].shape == (98, 4, 36, 36) and w["60m"].shape == (98, 3, 6, 6)
    assert torch.equal(w["20m"][8], x["20m"][0, :, 18:36, 18:36])
    y = from_windows(w, 7, 7, 36)
    for g in x:
        assert torch.equal(y[g], x[g])


def test_tiled_locality_bitwise():
    """A frame's core depends only on its frame plus one window of halo,
    bitwise, on every grid."""
    net = _net(perturb=0.05)
    Ls = torch.tensor([[8, 4]])
    x = _mosaic(1, 432, 432)
    y = deblock_s2_mosaic(net, x, Ls)
    x2 = {g: t.clone() for g, t in x.items()}
    edge = RECIPE["frame"] + RECIPE["halo"]           # 216
    for g, r in SCALES.items():
        x2[g][:, :, edge // r:, :] = 0.7
        x2[g][:, :, :, edge // r:] = 0.7
    y2 = deblock_s2_mosaic(net, x2, Ls)
    for g, r in SCALES.items():
        c = RECIPE["frame"] // r
        assert torch.equal(y[g][..., :c, :c], y2[g][..., :c, :c])
        assert not torch.allclose(y[g][..., c:, c:], y2[g][..., c:, c:])


def test_tiled_batched_matches_single():
    net = _net(perturb=0.05)
    Ls = torch.tensor([[8, 1]])
    x = _mosaic(1, 540, 396)
    a = deblock_s2_mosaic(net, x, Ls)
    b = deblock_s2_mosaic(net, x, Ls, batch=4)
    for g in x:
        assert a[g].shape == x[g].shape
        assert torch.allclose(a[g], b[g], atol=1e-5)


def test_loss_and_gradient():
    net = _net(perturb=0.01).train()
    rec = _mosaic(2, 252, 252, seed=1)
    tgt = _mosaic(2, 252, 252, seed=2)
    valid = {g: torch.ones_like(t[:, :1], dtype=torch.bool)
             for g, t in rec.items()}
    valid["10m"][:, :, :40] = False
    loss, parts, _ = residual_loss(net, rec, tgt, torch.tensor([[8, 0],
                                                                [8, 3]]),
                                   valid)
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == set(SCALES)
    assert net.head_10m.weight.grad.abs().sum() > 0


def test_save_and_load_round_trip(tmp_path):
    net = _net(perturb=0.05)
    path = tmp_path / "deblocker.pt"
    torch.save({"arch": {"width": 8, "cond_dim": 8, "depth": 8,
                         "window": 36, "cell": 6},
                "model": net.state_dict(), "recipe": dict(RECIPE),
                "trained_against": {"bitstream_id": "b", "decoder_id": "d"},
                "provenance": {"step": 1}}, path)
    net2, entry = load_s2_deblocker(str(path))
    x = _mosaic(1, 252, 252)
    Ls = torch.tensor([[8, 5]])
    a, b = net(x, Ls), net2(x, Ls)
    for g in x:
        assert torch.equal(a[g], b[g])
    assert entry["recipe"] == RECIPE
