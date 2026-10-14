"""Build `aef_georvq_index.parquet` for a tree of AE v2 files.

Usage:
  python scripts/build_ae_index.py <dir> --source-index aef_index.parquet \
      [--url-prefix https://.../aef/v2] [--out aef_georvq_index.parquet]

See georvq.ae.index for the schema.
"""
from __future__ import annotations

import argparse
import os
import sys

from georvq.ae.index import INDEX_NAME, build_index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="directory of *.georvq.parquet files")
    ap.add_argument("--source-index", required=True,
                    help="the source aef_index.parquet")
    ap.add_argument("--url-prefix", default=None,
                    help="published location of <root>; `path` is this joined "
                         "to each file's path under root (default: local paths)")
    ap.add_argument("--out", default=INDEX_NAME)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    if a.url_prefix is None:
        print("warning: no --url-prefix, so `path` holds absolute local paths "
              f"under {os.path.abspath(a.root)}; pass the published location "
              "of <root> (a URL, or a path relative to the index file) for an "
              "index others can use", file=sys.stderr)
    t = build_index(a.root, a.source_index, out_path=a.out,
                    url_prefix=a.url_prefix, workers=a.workers)
    print(f"{t.num_rows} tiles -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
