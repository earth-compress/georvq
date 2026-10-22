"""A small Sentinel-2 L1C SAFE product for tests, built from band arrays.

`make_safe` writes a directory with the layout of an L1C product of
processing baseline 05.xx: the 13 band images and the true-colour image
in GRANULE/<granule>/IMG_DATA, the preview and quality masks in QI_DATA,
the metadata XML files, AUX_DATA, DATASTRIP, HTML, rep_info, an empty
directory, and a manifest.safe listing every file with its size and
SHA3-256 checksum, as ESA's products do. The XML files are short
stand-ins written here, not copies of ESA's; the band images are JPEG
2000 with a GML box, as in the products.
"""
import hashlib
import os

import numpy as np

from georvq.s2.safe import BANDS, GRIDS, S2_ORDER, write_band_jp2

TILE = "T37UDB"
DATE = "20260503T084641"
NAME = f"S2C_MSIL1C_{DATE}_N0512_R107_{TILE}_20260503T104912"
GRANULE = f"L1C_{TILE}_A003456_20260503T085142"
DATASTRIP = "DS_2CPS_20260503T104912_S20260503T085142"
BAND_ID = {b: i for i, b in enumerate(S2_ORDER)}


def _params(shape, levels=3):
    t = min(shape)
    return {"gml": True, "precision": 15, "tile_width": t, "tile_height": t,
            "levels": levels, "progression": "LRCP", "codeblock": (16, 16),
            "precincts": None}


def _write_rgb_jp2(path, rgb, crs, transform):
    import rasterio
    with rasterio.open(path, "w", driver="JP2OpenJPEG", width=rgb.shape[2],
                       height=rgb.shape[1], count=rgb.shape[0], dtype="uint8",
                       crs=crs, transform=transform, QUALITY="100",
                       REVERSIBLE="YES", RESOLUTIONS="2") as ds:
        ds.write(rgb)


def _mtd(offset=-1000):
    spec = "\n".join(
        f'      <Spectral_Information bandId="{i}" physicalBand="{b}"/>'
        for b, i in BAND_ID.items())
    offs = "\n".join(f'      <RADIO_ADD_OFFSET band_id="{i}">{offset}'
                     f"</RADIO_ADD_OFFSET>" for i in BAND_ID.values())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-1C_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-1C.xsd">
  <n1:General_Info>
    <Product_Info>
      <PRODUCT_START_TIME>2026-05-03T08:46:41.024Z</PRODUCT_START_TIME>
      <PRODUCT_URI>{NAME}.SAFE</PRODUCT_URI>
      <PROCESSING_BASELINE>05.12</PROCESSING_BASELINE>
      <Datatake>
        <SPACECRAFT_NAME>Sentinel-2C</SPACECRAFT_NAME>
      </Datatake>
    </Product_Info>
    <Product_Image_Characteristics>
      <QUANTIFICATION_VALUE unit="none">10000</QUANTIFICATION_VALUE>
{offs}
{spec}
    </Product_Image_Characteristics>
  </n1:General_Info>
</n1:Level-1C_User_Product>
"""


def _manifest(root, files):
    objs = []
    for i, rel in enumerate(files):
        p = os.path.join(root, rel)
        h = hashlib.sha3_256(open(p, "rb").read()).hexdigest().upper()
        objs.append(f"""        <dataObject ID="obj_{i}">
            <byteStream mimeType="application/octet-stream" size="{os.path.getsize(p)}">
                <fileLocation href="./{rel}" locatorType="URL"/>
                <checksum checksumName="SHA3-256">{h}</checksum>
            </byteStream>
        </dataObject>""")
    return ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<xfdu:XFDU xmlns:xfdu=\"urn:ccsds:schema:xfdu:1\" "
            "version=\"esa/safe/sentinel/1.1/sentinel-2/msi/archive_l1c_user_product\">\n"
            "    <dataObjectSection>\n" + "\n".join(objs) + "\n"
            "    </dataObjectSection>\n</xfdu:XFDU>\n")


def make_safe(root, arrays, transform, epsg=32637, levels=3) -> str:
    """Write a SAFE product of the bands `arrays` ((10 m, 20 m, 60 m)
    uint16 arrays in the model's band order) at 10 m `transform` under
    `root`. Returns the product directory."""
    from affine import Affine
    from rasterio.crs import CRS
    safe = os.path.join(root, NAME + ".SAFE")
    g = os.path.join(safe, "GRANULE", GRANULE)
    img, qi = os.path.join(g, "IMG_DATA"), os.path.join(g, "QI_DATA")
    ds = os.path.join(safe, "DATASTRIP", DATASTRIP)
    for d in (img, qi, os.path.join(g, "AUX_DATA"), os.path.join(ds, "QI_DATA"),
              os.path.join(safe, "HTML"), os.path.join(safe, "rep_info"),
              os.path.join(safe, "AUX_DATA")):
        os.makedirs(d, exist_ok=True)
    crs = CRS.from_epsg(epsg)
    t10 = Affine(*transform[:6])
    stem = f"{TILE}_{DATE}"
    for a, (_, f, names) in zip(arrays, GRIDS):
        for k, b in enumerate(names):
            write_band_jp2(os.path.join(img, f"{stem}_{b}.jp2"), a[k], crs,
                           t10 * Affine.scale(f),
                           _params(a.shape[1:], levels))
    rgb = np.clip(arrays[0][[2, 1, 0]] // 16, 0, 255).astype(np.uint8)
    _write_rgb_jp2(os.path.join(img, f"{stem}_TCI.jp2"), rgb, crs, t10)
    _write_rgb_jp2(os.path.join(qi, f"{stem}_PVI.jp2"), rgb[:, ::4, ::4],
                   crs, t10 * Affine.scale(4))
    _write_rgb_jp2(os.path.join(qi, "MSK_CLASSI_B00.jp2"),
                   (arrays[2][:3] == 0).astype(np.uint8), crs,
                   t10 * Affine.scale(6))
    texts = {
        "MTD_MSIL1C.xml": _mtd(),
        "INSPIRE.xml": "<?xml version=\"1.0\"?>\n<gmd:MD_Metadata/>\n",
        f"GRANULE/{GRANULE}/MTD_TL.xml": "<?xml version=\"1.0\"?>\n<tile/>\n",
        f"GRANULE/{GRANULE}/AUX_DATA/AUX_ECMWFT": "aux " * 200,
        f"GRANULE/{GRANULE}/QI_DATA/GENERAL_QUALITY.xml":
            "<?xml version=\"1.0\"?>\n<quality/>\n",
        f"DATASTRIP/{DATASTRIP}/MTD_DS.xml":
            "<?xml version=\"1.0\"?>\n<ds>" + "<x>1</x>" * 2000 + "</ds>\n",
        f"DATASTRIP/{DATASTRIP}/QI_DATA/GENERAL_QUALITY.xml":
            "<?xml version=\"1.0\"?>\n<quality/>\n",
        "HTML/UserProduct_index.html": "<html><body>L1C</body></html>\n",
        "rep_info/S2_User_Product_Level-1C_Metadata.xsd":
            "<?xml version=\"1.0\"?>\n<xs:schema/>\n",
    }
    for rel, text in texts.items():
        with open(os.path.join(safe, rel), "w") as fh:
            fh.write(text)
    files = []
    for r, _, fs in os.walk(safe):
        for fn in fs:
            files.append(os.path.relpath(os.path.join(r, fn), safe)
                         .replace(os.sep, "/"))
    with open(os.path.join(safe, "manifest.safe"), "w") as fh:
        fh.write(_manifest(safe, sorted(files)))
    return safe


def synthetic_bands(n=144, seed=0, nodata=True):
    """(10 m, 20 m, 60 m) uint16 arrays of an n x n px 10 m crop, smooth
    fields around L1C-like levels (with the +1000 offset), with a nodata
    corner as at a swath edge."""
    rng = np.random.default_rng(seed)
    out = []
    for _, f, names in GRIDS:
        m = n // f
        yy, xx = np.mgrid[0:m, 0:m] * f
        a = np.stack([1000 + 800 + 300 * k + 200 * np.sin(xx / 17.0 + k)
                      + 150 * np.cos(yy / 23.0) + rng.normal(0, 40, (m, m))
                      for k in range(len(names))])
        a = np.clip(a, 1, 32767).astype(np.uint16)
        if nodata:
            cut = (yy + xx) < n // 2
            a[:, cut] = 0
        out.append(a)
    return tuple(out)


