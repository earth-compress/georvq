"""Package a trained AE deblocker checkpoint into a release bundle.

Takes the research-repo training checkpoint (ring-space
`scripts/train_ae_deblock.py` output), verifies it loads into the
weight-compatible georvq port, stamps it against the target bundle's
bitstream_id + base decoder_id (a deblocker is trained on one base
decoder's outputs — mismatches must be refusable at load time), and
writes `<bundle>/decoders/deblock.pt` plus a `decoders` section in
bundle.json.

    python scripts/package_ae_deblocker.py \
        --ckpt ~/projects/ring-space/checkpoints/ae_deblock_deblock_big/model.pt \
        --bundle /data/georvq-releases/ae-beta-v1/bundle
"""
import argparse
import json
import os

import torch

from georvq.ae.deblock import RECIPE, AEDeblocker, load_deblocker
from georvq.inference.release import bitstream_id, decoder_id, load_release


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="research training checkpoint (model + args)")
    ap.add_argument("--bundle", required=True, help="release bundle dir")
    ap.add_argument("--provenance", default=None,
                    help="free-text origin note (default: derived from "
                         "the checkpoint's own metadata)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    arch = {"width": ta["width"], "cond_dim": ta["cond_dim"],
            "blocks": ta["blocks"]}
    net = AEDeblocker(**arch)
    net.load_state_dict(ck["model"])     # port parity: names must match

    codec = load_release(args.bundle)
    bid = bitstream_id(codec.model)
    base_did = decoder_id(codec.model)
    did = decoder_id(codec.model, extra_state=net.state_dict(),
                     recipe=RECIPE)
    payload = {
        "model": net.state_dict(),
        "arch": arch,
        "recipe": RECIPE,
        "trained_against": {"bitstream_id": bid, "decoder_id": base_did},
        "provenance": args.provenance or (
            f"ring-space {ta.get('run_tag')} step {ck.get('step')} "
            f"(trained on {ta.get('cache')}, codec {ck.get('codec_ckpt')})"),
    }
    out = os.path.join(args.bundle, "decoders")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "deblock.pt")
    torch.save(payload, path)
    net2, entry = load_deblocker(path)   # round-trip check
    assert all(torch.equal(a, b) for a, b in
               zip(net.state_dict().values(), net2.state_dict().values()))

    bj_path = os.path.join(args.bundle, "bundle.json")
    with open(bj_path) as f:
        bj = json.load(f)
    bj.setdefault("decoders", {})["deblock"] = {
        "file": "decoders/deblock.pt", "decoder_id": did,
        "recipe": RECIPE, "provenance": payload["provenance"]}
    with open(bj_path, "w") as f:
        json.dump(bj, f, indent=1)
    print(f"packaged {path}\n  bitstream_id  {bid}\n  base decoder  "
          f"{base_did}\n  deblock decoder {did}\n  recipe {RECIPE}")


if __name__ == "__main__":
    main()
