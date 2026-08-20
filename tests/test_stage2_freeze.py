"""Stage-2 freeze semantics: frozen books really are frozen, and a
temporarily-frozen param really does train once released."""
import torch

from georvq.config import load_config
from georvq.factory import build_model
from georvq.train.warmstart import (decoder_tail_names, freeze_books,
                                    set_params_frozen, trainable_params)


def _smoke_model(overrides=()):
    cfg = load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:3','4:8:32:3']",
        "model.vq={warmup_steps: 0, dead_threshold: 0.01, reset_freeze_frac: 0}",
        *overrides])
    return cfg, build_model(cfg, "cpu")


def _fit_books(m):
    """Put the books in a fitted, initialized state (as init_from would)."""
    torch.manual_seed(0)
    for rvq in m.rvqs:
        for lev in rvq.levels:
            lev.codebook.weight.data.normal_()
            lev.initialized.fill_(True)


def test_freeze_books_stops_ema_and_resets():
    cfg, m = _smoke_model()
    _fit_books(m)
    freeze_books(m)
    before = [l.codebook.weight.data.clone()
              for rvq in m.rvqs for l in rvq.levels]
    m.train()                       # the surrounding model trains...
    freeze_books(m)                 # ...but the books must be re-asserted
    W, B = cfg.model.window, 2
    x = torch.randn(B, cfg.model.bands, W, W)
    Ls = torch.tensor([[3, 3]] * B)
    for _ in range(3):
        out, commit, _ = m.forward_train(x, Ls)
        (out.pow(2).mean() + commit).backward()
    after = [l.codebook.weight.data
             for rvq in m.rvqs for l in rvq.levels]
    for a, b in zip(before, after):
        assert torch.equal(a, b), "codebook moved under freeze_books"
    assert all(l.codebook.weight.grad is None
               for rvq in m.rvqs for l in rvq.levels)


def test_books_move_without_the_freeze():
    """Guard against the test above passing for the wrong reason."""
    cfg, m = _smoke_model()
    _fit_books(m)
    before = m.rvqs[0].levels[0].codebook.weight.data.clone()
    m.train()
    x = torch.randn(2, cfg.model.bands, cfg.model.window, cfg.model.window)
    m.forward_train(x, torch.tensor([[3, 3]] * 2))
    assert not torch.equal(before, m.rvqs[0].levels[0].codebook.weight.data)


def test_frozen_books_excluded_from_optimizer_but_tail_is_not():
    """The whole point of `trainable_params`: a param frozen for the first
    N steps must still be owned by the optimizer, or releasing it later is
    a silent no-op."""
    cfg, m = _smoke_model()
    freeze_books(m)
    tail = decoder_tail_names(m, 2)
    set_params_frozen(m, tail, True)

    def pred(n, p):
        return not n.startswith("rvqs.") and (p.requires_grad or n in tail)
    owned = {n for n, _ in trainable_params(m, pred)}
    assert tail and tail <= owned, "frozen tail missing from optimizer"
    assert not any(n.startswith("rvqs.") for n in owned)


def test_released_tail_param_actually_updates():
    cfg, m = _smoke_model()
    tail = decoder_tail_names(m, 2)
    set_params_frozen(m, tail, True)
    ps = [p for n, p in m.named_parameters() if not n.startswith("rvqs.")]
    opt = torch.optim.AdamW(ps, lr=1e-2)          # owns the frozen tail too
    name = sorted(tail)[0]
    watched = dict(m.named_parameters())[name]
    x = torch.randn(2, cfg.model.bands, cfg.model.window, cfg.model.window)
    Ls = torch.tensor([[3, 3]] * 2)

    m.train()
    before = watched.data.clone()
    out, commit, _ = m.forward_train(x, Ls)
    (out.pow(2).mean() + commit).backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    assert torch.equal(before, watched.data), "frozen tail param moved"

    set_params_frozen(m, tail, False)
    out, commit, _ = m.forward_train(x, Ls)
    (out.pow(2).mean() + commit).backward()
    opt.step()
    assert not torch.equal(before, watched.data), \
        "released tail param never trained (not owned by the optimizer)"


def test_decoder_tail_names_picks_the_last_blocks():
    _, m = _smoke_model()
    import re
    idx = sorted({int(mm.group(1)) for n, _ in m.named_parameters()
                  if (mm := re.match(r"decoder\.blocks\.(\d+)\.", n))})
    got = decoder_tail_names(m, 2)
    blocks = {int(re.match(r"decoder\.blocks\.(\d+)\.", n).group(1))
              for n in got if n.startswith("decoder.blocks.")}
    assert blocks == set(idx[-2:])
    assert any(n.startswith("decoder.head.") for n in got)


def test_loaded_cb_ignores_scalar_bookkeeping_buffers():
    """A model whose codebooks share NOTHING with the source checkpoint
    must not be reported as having loaded codebooks.

    Each VQ level carries `step` and `initialized` as 0-dim buffers, which
    shape-match any checkpoint. Keying on `startswith("rvqs")` therefore
    returned True on those alone, and mature_codebooks() then disabled
    resets and forced decay to 0.9995 -- silently overriding the config on
    a run whose whole point was live books with splitting."""
    import tempfile, os, torch
    from georvq.repro import save_checkpoint
    from georvq.train.warmstart import load_partial
    src_cfg, src = _smoke_model(["model.tiers=['1:16:32:3','4:8:32:3']"])
    dst_cfg, dst = _smoke_model(["model.tiers=['1:16:8:5','4:8:8:5']"])  # new K, new depth
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "src.pt")
        save_checkpoint(p, src, src_cfg, 0)
        _, loaded_cb = load_partial(dst, p, "cpu")
    assert loaded_cb is False, "scalar buffers made a K-mismatch look loaded"
    for rvq in dst.rvqs:
        for l in rvq.levels:
            assert l.reset_interval < 1 << 20, "books were matured anyway"


def test_grad_clip_is_on_by_default_and_bounds_the_update():
    """Absence of clipping is a live failure mode, not a tuning choice: an
    unclipped batch blew up a run at step 1900 and Adam never recovered."""
    from georvq.config import load_config
    cfg = load_config("configs/smoke.yaml", [])
    assert cfg.model.vq.clip_norm == 1.0, "clipping must default ON"

    ps = [torch.nn.Parameter(torch.zeros(4))]
    ps[0].grad = torch.full((4,), 1e6)          # a pathological gradient
    n = float(torch.nn.utils.clip_grad_norm_(ps, cfg.model.vq.clip_norm))
    assert n > 1e6, "clip_grad_norm_ must report the PRE-clip norm"
    assert abs(float(ps[0].grad.norm()) - 1.0) < 1e-4, "grad not clipped"
