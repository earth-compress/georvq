"""Sentinel-2 L1C SAFE products: reading the band images, keeping the rest
of the product, and writing a product back from decoded bands.

An L1C product (`<name>.SAFE`) holds one 100 km tile as 13 JPEG 2000
band images under `GRANULE/<granule>/IMG_DATA`, at 10 m (B02 B03 B04
B08, 10980 px), 20 m (B05 B06 B07 B8A B11 B12, 5490 px) and 60 m (B01
B09 B10, 1830 px), each georeferenced by a GML box in the same UTM CRS
with the same upper-left corner. Everything else (the true-colour image
TCI, the preview PVI, the quality masks, the metadata XML,
`manifest.safe`, the HTML and schema files) is kept by a georvq S2
store as one zip, `safe_meta_zip`, and restored by `reconstruct_safe`
around band images decoded from the store.

This module needs numpy and rasterio (whose GDAL reads and writes the
JPEG 2000 files through OpenJPEG); it does not import torch.
"""
from __future__ import annotations

import glob
import hashlib
import io
import os
import re
import struct
import zipfile

import numpy as np

BANDS_10M = ("B02", "B03", "B04", "B08")
BANDS_20M = ("B05", "B06", "B07", "B8A", "B11", "B12")
BANDS_60M = ("B01", "B09", "B10")
GRIDS = (("10m", 1, BANDS_10M), ("20m", 2, BANDS_20M), ("60m", 6, BANDS_60M))
BANDS = BANDS_10M + BANDS_20M + BANDS_60M          # the model's order
S2_ORDER = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
            "B09", "B10", "B11", "B12")
TILE_PX = 10980
_BAND_FILE = re.compile(r"^GRANULE/[^/]+/IMG_DATA/[^/]+_(B0[1-9]|B1[0-2]|B8A)"
                        r"\.jp2$")
# files whose bytes are already compressed are stored in the zip as they are
_STORED = (".jp2", ".png", ".jpg", ".jpeg", ".gif")
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
MARKER = "GEORVQ_RECONSTRUCTION.json"
_GDAL_ENV = {"GDAL_NUM_THREADS": "ALL_CPUS"}
_PROGRESSIONS = ("LRCP", "RLCP", "RPCL", "PCRL", "CPRL")


# --------------------------------------------------------------------------
# finding and reading the bands
# --------------------------------------------------------------------------

def safe_name(safe_dir) -> str:
    """The product's name, its directory name without `.SAFE`."""
    n = os.path.basename(os.fspath(safe_dir).rstrip("/"))
    return n[:-5] if n.endswith(".SAFE") else n


def band_files(safe_dir) -> dict:
    """{band: path relative to the SAFE root} of the 13 band images."""
    safe_dir = os.fspath(safe_dir)
    gran = sorted(glob.glob(os.path.join(safe_dir, "GRANULE", "*", "IMG_DATA")))
    if len(gran) != 1:
        raise ValueError(f"{safe_dir}: expected one granule with IMG_DATA, "
                         f"found {len(gran)}")
    out = {}
    for b in BANDS:
        hits = glob.glob(os.path.join(gran[0], f"*_{b}.jp2"))
        if len(hits) != 1:
            raise ValueError(f"{gran[0]}: {len(hits)} images of band {b}")
        out[b] = os.path.relpath(hits[0], safe_dir).replace(os.sep, "/")
    return out


def read_bands(safe_dir, files=None, threads=13) -> dict:
    """{band: (uint16 array, rasterio profile facts)} of the 13 bands at
    their native resolution."""
    from concurrent.futures import ThreadPoolExecutor

    import rasterio
    files = files or band_files(safe_dir)

    def one(b):
        with rasterio.open(os.path.join(safe_dir, files[b])) as ds:
            if ds.count != 1 or ds.dtypes[0] != "uint16":
                raise ValueError(f"{files[b]}: {ds.count} x {ds.dtypes[0]}, "
                                 "expected one uint16 band")
            return b, (ds.read(1), {"crs": ds.crs, "transform": ds.transform,
                                    "shape": (ds.height, ds.width)})

    with rasterio.Env(**_GDAL_ENV), ThreadPoolExecutor(threads) as ex:
        return dict(ex.map(one, BANDS))


def check_geometry(info) -> tuple:
    """(epsg, 10 m transform, (height, width)) of the bands' facts
    {band: {"crs", "transform", "shape"}}, after checking that they share
    one CRS and one upper-left corner, and that each band's pixel size and
    shape are its grid's (10, 20 or 60 m on a 10 m grid divisible by 6)."""
    from affine import Affine
    t10 = info[BANDS_10M[0]]["transform"]
    h10, w10 = info[BANDS_10M[0]]["shape"]
    crs = info[BANDS_10M[0]]["crs"]
    if h10 % 6 or w10 % 6:
        raise ValueError(f"10 m grid {h10} x {w10} is not divisible by 6")
    for _, f, bands in GRIDS:
        for b in bands:
            i = info[b]
            if i["crs"] != crs:
                raise ValueError(f"{b}: CRS {i['crs']} differs from {crs}")
            want = t10 * Affine.scale(f)
            if not i["transform"].almost_equals(want, precision=1e-6):
                raise ValueError(f"{b}: transform {tuple(i['transform'])[:6]}"
                                 f", expected {tuple(want)[:6]}")
            if tuple(i["shape"]) != (h10 // f, w10 // f):
                raise ValueError(f"{b}: shape {i['shape']}, expected "
                                 f"{(h10 // f, w10 // f)}")
    epsg = crs.to_epsg()
    if epsg is None:
        raise ValueError(f"CRS {crs} has no EPSG code")
    return epsg, tuple(t10)[:6], (h10, w10)


def product_metadata(safe_dir) -> dict:
    """Facts from MTD_MSIL1C.xml: processing baseline, spacecraft, sensing
    start, tile id, the quantification value and each band's radiometric
    offset (absent before processing baseline 04.00)."""
    path = os.path.join(safe_dir, "MTD_MSIL1C.xml")
    if not os.path.exists(path):
        return {}
    x = open(path, encoding="utf-8").read()

    def tag(name):
        m = re.search(rf"<{name}[^>]*>([^<]*)</{name}>", x)
        return m.group(1).strip() if m else None

    ids = {m.group(1): m.group(2) for m in re.finditer(
        r'<Spectral_Information bandId="(\d+)" physicalBand="([^"]+)"', x)}
    offs = {ids.get(m.group(1), m.group(1)): int(m.group(2)) for m in
            re.finditer(r'<RADIO_ADD_OFFSET band_id="(\d+)">([-0-9]+)<', x)}
    q = tag("QUANTIFICATION_VALUE")
    return {"processing_baseline": tag("PROCESSING_BASELINE"),
            "spacecraft": tag("SPACECRAFT_NAME"),
            "sensing_start": tag("PRODUCT_START_TIME"),
            "product_uri": tag("PRODUCT_URI"),
            "quantification_value": float(q) if q else None,
            "radio_add_offset": {b: offs[b] for b in S2_ORDER if b in offs}
            or None}


# --------------------------------------------------------------------------
# JPEG 2000 codestream parameters
# --------------------------------------------------------------------------

def jp2_params(path, nbytes=1 << 20) -> dict:
    """Coding parameters of a JPEG 2000 file's main header: tile size,
    sample precision, decomposition levels, progression order, quality
    layers, code-block size, precinct sizes and wavelet, and whether it
    carries a GML box. Reads the first `nbytes` of the file."""
    with open(path, "rb") as f:
        b = f.read(nbytes)
    out = {"gml": False}
    off = 0
    cs = None
    while off + 8 <= len(b):
        L, T = struct.unpack(">I4s", b[off:off + 8])
        h = 8
        if L == 1:
            L = struct.unpack(">Q", b[off + 8:off + 16])[0]
            h = 16
        if T == b"asoc" and b"gml.root-instance" in b[off:off + min(L, 256)]:
            out["gml"] = True
        if T == b"jp2c":
            cs = b[off + h:]
            break
        if L == 0:
            break
        off += L
    if cs is None or cs[:2] != b"\xff\x4f":
        raise ValueError(f"{path}: no JPEG 2000 codestream in the first "
                         f"{nbytes} bytes")
    i = 2
    while i + 4 <= len(cs):
        m, L = struct.unpack(">HH", cs[i:i + 4])
        seg = cs[i + 4:i + 2 + L]
        if m == 0xFF51:                                   # SIZ
            v = struct.unpack(">HIIIIIIIIH", seg[:36])
            out.update(width=v[1] - v[3], height=v[2] - v[4],
                       tile_width=v[5], tile_height=v[6], components=v[9],
                       precision=(seg[36] & 0x7F) + 1,
                       signed=bool(seg[36] & 0x80))
        elif m == 0xFF52:                                 # COD
            levels = seg[5]
            out.update(progression=_PROGRESSIONS[seg[1]],
                       layers=struct.unpack(">H", seg[2:4])[0],
                       mct=seg[4], levels=levels,
                       codeblock=(1 << (seg[6] + 2), 1 << (seg[7] + 2)),
                       reversible=seg[9] == 1)
            if seg[0] & 1:
                pp = list(seg[10:10 + levels + 1])
                out["precincts"] = [(1 << (p & 0xF), 1 << (p >> 4))
                                    for p in pp]
            else:
                out["precincts"] = None
        elif m == 0xFF90:                                 # first tile part
            break
        i += 2 + L
    return out


def jp2_creation_options(p) -> dict:
    """GDAL JP2OpenJPEG creation options that write a lossless file with
    the coding parameters `p` (from `jp2_params`) and a GML box, as the
    L1C band images are written."""
    opts = {"CODEC": "JP2", "REVERSIBLE": "YES", "QUALITY": "100",
            "YCC": "NO", "NBITS": str(p["precision"]),
            "BLOCKXSIZE": str(p["tile_width"]),
            "BLOCKYSIZE": str(p["tile_height"]),
            "RESOLUTIONS": str(p["levels"] + 1),
            "PROGRESSION": p["progression"],
            "CODEBLOCK_WIDTH": str(p["codeblock"][0]),
            "CODEBLOCK_HEIGHT": str(p["codeblock"][1]),
            "GMLJP2": "YES" if p.get("gml", True) else "NO",
            "GeoJP2": "NO", "WRITE_METADATA": "NO", "INSPIRE_TG": "NO"}
    if p.get("precincts"):
        # GDAL lists precincts from the highest resolution level down,
        # the codestream from the lowest up
        opts["PRECINCTS"] = ",".join(f"{{{w},{h}}}" for w, h in
                                     reversed(p["precincts"]))
    return opts


def write_band_jp2(path, data, crs, transform, params) -> None:
    """Write one band image losslessly with GDAL's JP2OpenJPEG driver,
    with the coding parameters `params` (`jp2_params` of the original) and
    a GML box georeferencing it. No nodata value is declared, as in the
    L1C images."""
    import rasterio
    import rasterio.shutil
    from rasterio.io import MemoryFile
    data = np.asarray(data)
    if data.dtype != np.uint16 or data.ndim != 2:
        raise ValueError(f"expected a 2-D uint16 band, got {data.dtype} "
                         f"{data.shape}")
    top = (1 << params["precision"]) - 1
    if int(data.max(initial=0)) > top:
        raise ValueError(f"{path}: values above {top} do not fit the "
                         f"{params['precision']}-bit samples of the original")
    opts = jp2_creation_options(params)
    # GDAL refuses tiles under 32 px but cuts a tile larger than the image
    # to the image, so a single tile smaller than 32 px is asked for as 32
    for key, n in (("BLOCKXSIZE", data.shape[1]), ("BLOCKYSIZE", data.shape[0])):
        if int(opts[key]) < 32 and int(opts[key]) >= n:
            opts[key] = "32"
    with rasterio.Env(**_GDAL_ENV, GDAL_PAM_ENABLED="NO"):
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", width=data.shape[1],
                          height=data.shape[0], count=1, dtype="uint16",
                          crs=crs, transform=transform) as tmp:
                tmp.write(data, 1)
            with mem.open() as tmp:
                rasterio.shutil.copy(tmp, path, driver="JP2OpenJPEG",
                                     **opts)
    aux = path + ".aux.xml"
    if os.path.exists(aux):
        os.remove(aux)


# --------------------------------------------------------------------------
# the metadata zip
# --------------------------------------------------------------------------

def is_band_file(rel) -> bool:
    """Whether a path relative to the SAFE root is one of the 13 band
    images (the true-colour image TCI is not)."""
    return bool(_BAND_FILE.match(rel))


def safe_meta_zip(safe_dir) -> bytes:
    """Everything in the SAFE but the 13 band images, as zip bytes:
    every directory and file under the SAFE root, at its path relative to
    the root (the root itself not included), in sorted order, with fixed
    timestamps and permissions so that the same product always gives the
    same bytes (for a given zlib). XML and other text is deflated at level
    9; JPEG 2000, PNG and JPEG files are stored as they are. Symbolic
    links are refused."""
    safe_dir = os.fspath(safe_dir)
    entries = []
    n_bands = 0
    for root, dirs, files in os.walk(safe_dir):
        for d in dirs:
            p = os.path.join(root, d)
            if os.path.islink(p):
                raise ValueError(f"{p}: symbolic links are not kept")
            entries.append(os.path.relpath(p, safe_dir).replace(os.sep, "/")
                           + "/")
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p):
                raise ValueError(f"{p}: symbolic links are not kept")
            rel = os.path.relpath(p, safe_dir).replace(os.sep, "/")
            if is_band_file(rel):
                n_bands += 1
            else:
                entries.append(rel)
    if n_bands != len(BANDS):
        raise ValueError(f"{safe_dir}: {n_bands} band images, expected "
                         f"{len(BANDS)}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for rel in sorted(entries):
            zi = zipfile.ZipInfo(rel, date_time=_ZIP_TIME)
            zi.create_system = 3
            if rel.endswith("/"):
                zi.external_attr = (0o40755 << 16) | 0x10
                z.writestr(zi, b"")
                continue
            zi.external_attr = 0o100644 << 16
            stored = rel.lower().endswith(_STORED)
            zi.compress_type = (zipfile.ZIP_STORED if stored
                                else zipfile.ZIP_DEFLATED)
            with open(os.path.join(safe_dir, rel), "rb") as f:
                data = f.read()
            z.writestr(zi, data, compress_type=zi.compress_type,
                       compresslevel=None if stored else 9)
    return buf.getvalue()


def unzip_meta(data, out_dir) -> list:
    """Extract a `safe_meta_zip` into `out_dir` (created; it must not
    exist). Refuses entries that would land outside it. Returns the
    relative paths of the files written."""
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} exists")
    os.makedirs(out_dir)
    root = os.path.realpath(out_dir)
    written = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for zi in z.infolist():
            dest = os.path.realpath(os.path.join(root, zi.filename))
            if dest != root and not dest.startswith(root + os.sep):
                raise ValueError(f"zip entry {zi.filename!r} leaves the "
                                 "product directory")
            if zi.is_dir():
                os.makedirs(dest, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(zi) as src, open(dest, "wb") as dst:
                while True:
                    chunk = src.read(1 << 22)
                    if not chunk:
                        break
                    dst.write(chunk)
            written.append(zi.filename)
    return written


# --------------------------------------------------------------------------
# manifest.safe
# --------------------------------------------------------------------------

def file_checksum(path, name) -> str:
    """The checksum `name` (as manifest.safe names it: "SHA3-256" or
    "MD5") of a file, in upper-case hex as the manifest writes it."""
    algo = {"SHA3-256": "sha3_256", "MD5": "md5"}.get(name.upper())
    if algo is None:
        raise ValueError(f"checksum {name!r}: expected SHA3-256 or MD5")
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest().upper()


_BYTESTREAM = re.compile(
    r'(<byteStream\b[^>]*\bsize=")(\d+)("[^>]*>\s*<fileLocation\b[^>]*'
    r'\bhref="\.?/?([^"]+)"[^>]*/>\s*<checksum\b[^>]*\bchecksumName="'
    r'([^"]+)"[^>]*>)([0-9A-Fa-f]+)(</checksum>)', re.S)


def manifest_entries(text) -> dict:
    """{relative path: (size, checksum name, checksum)} of the byte
    streams manifest.safe lists."""
    return {m.group(4): (int(m.group(2)), m.group(5), m.group(6))
            for m in _BYTESTREAM.finditer(text)}


def rewrite_manifest(text, safe_dir, files, comment) -> str:
    """manifest.safe text with the size and checksum of each file of
    `files` (paths relative to the SAFE root) recomputed from the file in
    `safe_dir`, with the checksum algorithm the manifest names, and
    `comment` added as an XML comment after the XML declaration. Nothing
    else in the text changes. Raises if a file is not listed."""
    want = set(files)
    seen = set()

    def sub(m):
        rel = m.group(4)
        if rel not in want:
            return m.group(0)
        seen.add(rel)
        p = os.path.join(safe_dir, rel)
        return (m.group(1) + str(os.path.getsize(p)) + m.group(3)
                + file_checksum(p, m.group(5)) + m.group(7))

    out = _BYTESTREAM.sub(sub, text)
    if seen != want:
        raise ValueError(f"manifest.safe does not list {sorted(want - seen)}")
    if "--" in comment:
        raise ValueError("an XML comment cannot hold '--'")
    m = re.match(r"\s*<\?xml[^>]*\?>[ \t]*\r?\n?", out)
    at = m.end() if m else 0
    nl = "\r\n" if "\r\n" in out[:at + 2] else "\n"
    return out[:at] + f"<!-- {comment} -->{nl}" + out[at:]
