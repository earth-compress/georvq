"""Command-line entry points: `-c CONFIG [key.path=value ...] [--gpu N]`."""
from __future__ import annotations

import argparse

from georvq.config import load_config


def _parse(argv=None, description=""):
    """Shared argument handling for every `georvq-train*` entry point:
    a preset, a device, and trailing `key.path=value` overrides. Returns
    the fully resolved config, so overrides are validated by the same
    schema as the YAML rather than applied later.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("-c", "--config", required=True, help="YAML preset")
    ap.add_argument("--gpu", type=int, default=0,
                    help="CUDA device index (-1 = CPU)")
    ap.add_argument("overrides", nargs="*",
                    help="config overrides: key.path=value")
    a = ap.parse_args(argv)
    return load_config(a.config, a.overrides), a.gpu


def train_pyramid(argv=None):
    cfg, gpu = _parse(argv, "Train a pyramid RVQ codec")
    from georvq.train.loop import train
    train(cfg, gpu)                 # console scripts sys.exit() the return


def train_stage1(argv=None):
    cfg, gpu = _parse(argv, "Train a stage-1 continuous sigma-conditioned AE")
    from georvq.train.stage1 import train
    train(cfg, gpu)


# release commands of `georvq <release> ...`, by name: the module whose
# main(argv) takes the rest of the command line
RELEASE_COMMANDS = {"s2": "georvq.s2.v1"}


def main(argv=None):
    """`georvq <release> <command> ...`, e.g. `georvq s2 reconstruct
    STORE OUT_DIR --quality 8-4`."""
    import importlib
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in RELEASE_COMMANDS:
        print(f"usage: georvq {{{','.join(RELEASE_COMMANDS)}}} COMMAND ...",
              file=sys.stderr)
        return 2
    return importlib.import_module(RELEASE_COMMANDS[argv[0]]).main(argv[1:])
