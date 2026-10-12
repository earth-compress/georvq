"""`vq.reset_mode="split"`: revive dead codes by LBG splitting.

The property that matters is WHERE a revived code lands. Random reseeding
draws a uniform point, which by construction sits in a dense region --
usually one already well covered -- so the revived code contributes
little and often dies again. Splitting targets the cell with the largest
distortion contribution, which is count x spread, not count alone.
"""
import torch

from georvq.config import load_config
from georvq.factory import build_model
from georvq.model.vq import VectorQuantizer


def _vq(mode, K=8, D=4, **kw):
    return VectorQuantizer(num_codes=K, dim=D, decay=0.9, kmeans_init=False,
                           reset_interval=0, dead_threshold=0.01,
                           reset_mode=mode, **kw)


def test_default_mode_is_random_and_has_no_extra_state():
    """State-dict layout must be untouched, or every existing checkpoint
    stops loading strict."""
    a, b = _vq("random"), _vq("split")
    assert set(a.state_dict()) == set(b.state_dict())
    assert not hasattr(a, "ema_sse") and hasattr(b, "ema_sse")
    assert load_config("configs/smoke.yaml", []).model.vq.reset_mode \
        == "random"


def test_sse_ema_tracks_distortion_contribution():
    """A cell holding many badly-fit points must outrank a cell holding
    many well-fit ones -- count alone is the wrong signal."""
    torch.manual_seed(0)
    vq = _vq("split", K=4, D=2)
    vq.codebook.weight.data.copy_(torch.tensor(
        [[0., 0.], [10., 0.], [0., 10.], [10., 10.]]))
    vq.initialized.fill_(True)
    vq.train()
    # code 0: 200 points scattered widely around it (high spread)
    wide = torch.randn(200, 2) * 1.5
    # code 1: 200 points sitting almost exactly on it (tight)
    tight = torch.tensor([[10., 0.]]) + torch.randn(200, 2) * 0.01
    for _ in range(5):
        vq(torch.cat([wide, tight]))
    assert vq.ema_sse[0] > 10 * vq.ema_sse[1], vq.ema_sse[:2]


def test_split_pair_straddles_the_donor_centroid():
    """LBG geometry: donor and revived code end up on OPPOSITE sides of
    the donor's old centroid, both at a typical member offset.

    The old behaviour teleported the revived code onto the donor's
    worst-fitting point -- an outlier, which served almost nobody and died
    again. Measured on tessera_vq8_split, ~100% of each reset's residual
    gain evaporated before the next reset, every reset, for 23k steps."""
    torch.manual_seed(0)
    vq = _vq("split", K=3, D=2)
    vq.codebook.weight.data.copy_(torch.tensor([[0., 0.], [50., 0.],
                                                [99., 99.]]))
    vq.initialized.fill_(True)
    vq.train()
    z = torch.cat([torch.randn(400, 2), torch.tensor([[9., 9.]])])  # outlier
    for _ in range(4):
        vq(z)
    donor = int(vq.ema_sse.argmax())
    c_old = vq.codebook.weight.data[donor].clone()
    vq.ema_count[2] = 0.0                       # declare code 2 dead
    assert vq._reset_dead(z, *_assign(vq, z)) == 1

    d_new = vq.codebook.weight.data[donor] - c_old
    v_new = vq.codebook.weight.data[2] - c_old
    assert torch.dot(d_new, v_new) < 0, "pair did not straddle the centroid"
    assert torch.allclose(d_new, -v_new, atol=1e-4), "pair is asymmetric"
    # and it must NOT have landed on the outlier
    assert (vq.codebook.weight.data[2]
            - torch.tensor([9., 9.])).norm() > 1.0


def _assign(vq, z):
    d = (z.pow(2).sum(1, keepdim=True) - 2 * z @ vq.codebook.weight.t()
         + vq.codebook.weight.pow(2).sum(1)[None])
    idx = d.argmin(1)
    return idx, vq.codebook.weight[idx]


def test_split_halves_the_source_mass():
    """The split pair shares the old cell, and one hot cell must not win
    every free slot in a single reset round."""
    torch.manual_seed(0)
    vq = _vq("split", K=4, D=2)
    vq.initialized.fill_(True)
    vq.train()
    z = torch.randn(256, 2) * 3
    for _ in range(4):
        vq(z)
    src = int(vq.ema_sse.argmax())
    before_c, before_s = float(vq.ema_count[src]), float(vq.ema_sse[src])
    vq.ema_count[(src + 1) % 4] = 0.0
    vq._reset_dead(z, *_assign(vq, z))
    assert float(vq.ema_count[src]) ==  before_c / 2
    assert float(vq.ema_sse[src]) == before_s / 2


def test_falls_back_to_random_when_no_live_cell_is_usable():
    """Degrade to the old behaviour rather than failing."""
    torch.manual_seed(0)
    vq = _vq("split", K=4, D=2)
    vq.initialized.fill_(True)
    vq.train()
    z = torch.randn(64, 2)
    vq(z)
    vq.ema_sse.zero_()                    # no distortion signal at all
    vq.ema_count[1] = 0.0
    n = vq._reset_dead(z, *_assign(vq, z))
    assert n == 1 and torch.isfinite(vq.codebook.weight.data).all()


def test_end_to_end_through_the_model():
    cfg = load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:3','4:8:32:3']",
        "model.vq={warmup_steps: 0, reset_mode: split, dead_threshold: 0.5}"])
    m = build_model(cfg, "cpu")
    m.train()
    x = torch.randn(4, cfg.model.bands, cfg.model.window, cfg.model.window)
    for _ in range(3):
        for rvq in m.rvqs:             # reset every step, as the trainer would
            for l in rvq.levels:
                l.reset_pending = True
        out, commit, _ = m.forward_train(x, torch.tensor([[3, 3]] * 4))
    assert torch.isfinite(out).all() and torch.isfinite(commit)
    for rvq in m.rvqs:
        for l in rvq.levels:
            assert torch.isfinite(l.codebook.weight).all()


def test_split_is_live_on_every_quantizer_path():
    """QuantizedBookVQ overrides forward() with its own copy of the EMA
    block. If a copy omits the SSE tracker, reset_mode reports "split"
    and silently behaves as "random" -- no error anywhere. Any future
    quantizer variant belongs in this list.
    """
    for extra in ("book_bits: 8",):
        cfg = load_config("configs/smoke.yaml", [
            "model.tiers=['1:16:32:3','4:8:32:3']",
            f"model.vq={{warmup_steps: 0, reset_mode: split, {extra}}}"])
        m = build_model(cfg, "cpu")
        m.train()
        x = torch.randn(8, cfg.model.bands, cfg.model.window,
                        cfg.model.window)
        for _ in range(3):
            m.forward_train(x, torch.tensor([[3, 3]] * 8))
        tot = sum(float(l.ema_sse.sum())
                  for rvq in m.rvqs for l in rvq.levels)
        assert tot > 0, f"SSE tracker never filled with {extra}"


def test_healthy_level_still_reports_diagnostics():
    """A level with nothing dead must still emit before/after stats. If it
    stays silent, the reset figure cannot distinguish 'perfectly healthy'
    from 'no data', and the healthiest levels look like broken ones."""
    torch.manual_seed(0)
    vq = _vq("split", K=4, D=2)
    vq.initialized.fill_(True)
    vq.train()
    z = torch.randn(256, 2)
    for _ in range(4):
        vq(z)
    vq.ema_count.fill_(1.0)                # nothing is dead
    n = vq._reset_dead(z, *_assign(vq, z))
    assert n == 0
    st = vq.last_reset
    assert st is not None and st["n_reset"] == 0
    assert st["perp_after"] == st["perp_before"] > 0
    assert st["resid_gain"] == 0.0
