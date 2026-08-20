"""fp16 release weights: half the file, identical arithmetic.

The claim being pinned is narrow and worth stating precisely. Storing
weights at fp16 changes the VALUES a release carries (by one fp16
rounding) but not the PRECISION anything computes in, because
`load_state_dict` copies into an fp32 module and upcasts. So an fp16
release decodes with fp32 arithmetic over slightly-rounded weights --
not with fp16 arithmetic, which would be a different proposal carrying a
different risk (a narrower accumulate makes near-tie RVQ assignments
hardware-dependent, and the encoder's assignments are the bitstream).
"""
import os

import torch

from georvq.config import load_config
from georvq.factory import build_model
from georvq.inference.release import halve_weights
from georvq.repro import save_checkpoint


def _tiny(tmp_path):
    cfg = load_config("configs/smoke.yaml", [
        "model.tiers=['1:16:32:3','4:8:32:3']"])
    m = build_model(cfg, "cpu").eval()
    p = os.path.join(tmp_path, "model.pt")
    save_checkpoint(p, m, cfg, 0)
    return cfg, m, p


def test_halving_makes_the_file_smaller_and_upcasts_on_load(tmp_path):
    cfg, m, p = _tiny(tmp_path)
    before = os.path.getsize(p)
    after = halve_weights(p)
    assert after < before * 0.75, "fp16 should roughly halve the weights"

    from georvq.factory import from_checkpoint
    m2 = from_checkpoint(p, "cpu")
    # loaded back as fp32 -- the arithmetic is untouched ...
    assert all(q.dtype == torch.float32 for q in m2.parameters())
    # ... over values that went through one fp16 rounding
    for (n, a), (_, b) in zip(m.named_parameters(), m2.named_parameters()):
        assert torch.equal(b.data, a.data.half().float()), n


def test_fp16_weights_do_not_move_the_reconstruction(tmp_path):
    """The measurement this flag rests on, in miniature: rounding every
    weight to fp16 must not move what the codec decodes."""
    cfg, m, p = _tiny(tmp_path)
    torch.manual_seed(0)
    x = torch.randn(2, cfg.model.bands, cfg.model.window, cfg.model.window)
    Ls = torch.tensor([[3, 3]] * 2)
    with torch.no_grad():
        zs = m.encode_levels(x)
        ref = m.decode(m.quantize(zs, Ls)[0], Ls)

    halve_weights(p)
    from georvq.factory import from_checkpoint
    m2 = from_checkpoint(p, "cpu")
    with torch.no_grad():
        zs2 = m2.encode_levels(x)
        got = m2.decode(m2.quantize(zs2, Ls)[0], Ls)

    rel = (got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-9)
    assert rel < 1e-2, f"fp16 storage moved the output by {rel:.2e}"


def test_the_flag_is_off_by_default(tmp_path):
    """The reference release must be exactly what was validated; fp16 is
    an opt-in size win, not a silent substitution."""
    import inspect

    from georvq.inference.release import save_release
    from georvq.tessera.release import build_bundle
    assert inspect.signature(save_release).parameters["fp16"].default is False
    assert inspect.signature(build_bundle).parameters["fp16"].default is False
