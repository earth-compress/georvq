"""Benchmark registry: per-task coverage, probe type, and label attachment.

Each benchmark turns a table of held-out target pixels (lon/lat + provenance,
from extract.build_windows) into the same table plus an integer/float `label`
column, dropping pixels with no valid label. Coverage bboxes let the data-prep
pipeline restrict tile sampling to where labels exist -- so LUCAS/LCMAP already
sample the right regions even though their label join is still a stub.

Implemented now: ASTER GED surface-emissivity regression (a global raster, so it
runs end-to-end on data we already reach -- the lowest-setup AEF benchmark).
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


class Benchmark:
    name: str
    task: str           # "classification" | "regression"
    metric: str         # "balanced_accuracy" | "r2"
    mode = "raster"     # "raster" (sample tiles, look up label) | "points"
    coverage_bbox = None  # (w, s, e, n) WGS84, or None for global
    years = None          # restrict AEF tile years, or None
    survey_year = None    # for point benchmarks: AEF year to extract at

    def attach_labels(self, targets: pd.DataFrame, cache_dir: str) -> pd.DataFrame:
        """raster mode: add a `label` column to sampled targets."""
        raise NotImplementedError

    def load_points(self, cache_dir: str) -> pd.DataFrame:
        """points mode: return survey points with columns lon, lat, label."""
        raise NotImplementedError


def _download(url, fp, session=None):
    import requests
    if os.path.exists(fp) and os.path.getsize(fp) > 1e5:
        return fp
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    session = session or requests.Session()
    r = session.get(url, stream=True, timeout=600, allow_redirects=True)
    r.raise_for_status()
    tmp = fp + ".tmp"
    with open(tmp, "wb") as f:
        for c in r.iter_content(1 << 20):
            f.write(c)
    os.replace(tmp, fp)
    return fp


def _drop_rare(pts, min_count=50):
    """Drop classes too rare to score a stable balanced accuracy."""
    vc = pts.label.value_counts()
    return pts[pts.label.isin(vc[vc >= min_count].index)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  ASTER Global Emissivity Database (AG100 v3) -- regression, R^2              #
# --------------------------------------------------------------------------- #
class AsterGED(Benchmark):
    name, task, metric, mode = "aster_ged", "regression", "r2", "raster"
    coverage_bbox = (-180.0, -60.0, 180.0, 82.0)  # land; skip Antarctic ice
    EMIS_SCALE = 1e-3                            # int16 -> emissivity
    GRID = 1000                                  # px per 1-deg tile
    BASE = ("https://data.lpdaac.earthdatacloud.nasa.gov/"
            "lp-prod-protected/AG100.003")
    CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"

    def __init__(self, band="mean"):
        # band: "mean" (avg of 5 TIR bands) or an index 0..4 (ASTER 10..14)
        self.band = band

    @staticmethod
    def _nw_corner(lon, lat):
        return int(math.ceil(lat)), int(math.floor(lon))

    @staticmethod
    def _fmt(v, width):                          # zero-padded, signed
        return f"-{abs(v):0{width}d}" if v < 0 else f"{v:0{width}d}"

    def _granule(self, nwlat, nwlon):            # e.g. 9,-5 -> ...09.-005...
        return f"AG100.v003.{self._fmt(nwlat, 2)}.{self._fmt(nwlon, 3)}.0001"

    def _download(self, nwlat, nwlon, cache_dir, session):
        import requests
        name = self._granule(nwlat, nwlon)
        fp = os.path.join(cache_dir, name + ".h5")
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            return fp
        urls = [f"{self.BASE}/{name}/{name}.h5"]
        try:                                     # CMR fallback (point query)
            r = session.get(self.CMR, params=dict(
                short_name="AG100", version="003",
                bounding_box=f"{nwlon+0.5},{nwlat-0.5},{nwlon+0.5},{nwlat-0.5}",
                page_size=1), timeout=60)
            for e in r.json().get("feed", {}).get("entry", []):
                for L in e.get("links", []):
                    h = L.get("href", "")
                    if h.endswith(".h5") and h.startswith("http"):
                        urls.append(h)
        except Exception:                        # noqa: BLE001
            pass
        for url in urls:
            try:
                r = session.get(url, stream=True, timeout=180,
                                allow_redirects=True)
                if r.status_code != 200:
                    continue
                tmp = fp + ".tmp"
                with open(tmp, "wb") as f:
                    for c in r.iter_content(1 << 20):
                        f.write(c)
                os.replace(tmp, fp)
                return fp
            except Exception:                    # noqa: BLE001
                continue
        return None                              # ocean / no clear-sky data

    def _sample_tile(self, fp, sub):
        """Read emissivity for all targets in one AG100 tile -> (label array,
        valid mask) aligned to `sub` rows."""
        import h5py
        nwlat = int(math.ceil(sub.lat.iloc[0]))     # tile NW corner
        nwlon = int(math.floor(sub.lon.iloc[0]))
        with h5py.File(fp, "r") as h:
            emis = h["Emissivity/Mean"][()]          # (5, 1000, 1000) int16
            nobs = h["Observations/NumObs"][()]      # (1000, 1000)
        row = np.clip(((nwlat - sub.lat.to_numpy()) * self.GRID).astype(int),
                      0, self.GRID - 1)
        col = np.clip(((sub.lon.to_numpy() - nwlon) * self.GRID).astype(int),
                      0, self.GRID - 1)
        bands = emis[:, row, col].astype(np.float32)  # (5, n)
        valid = (bands > 0).all(0) & (nobs[row, col] > 0)
        if self.band == "mean":
            lab = bands.mean(0) * self.EMIS_SCALE
        else:
            lab = bands[int(self.band)] * self.EMIS_SCALE
        return lab, valid

    def attach_labels(self, targets, cache_dir):
        import requests
        os.makedirs(cache_dir, exist_ok=True)
        nw = np.array([self._nw_corner(lo, la)
                       for lo, la in zip(targets.lon, targets.lat)])
        targets = targets.copy()
        targets["_nwlat"], targets["_nwlon"] = nw[:, 0], nw[:, 1]
        uniq = targets[["_nwlat", "_nwlon"]].drop_duplicates().to_numpy()
        print(f"  ASTER GED: {len(targets)} targets across {len(uniq)} "
              f"1-deg tiles -> downloading", flush=True)

        session = requests.Session()
        with ThreadPoolExecutor(max_workers=8) as ex:
            paths = dict(zip(map(tuple, uniq), ex.map(
                lambda t: self._download(int(t[0]), int(t[1]), cache_dir,
                                         session), uniq)))
        miss = sum(p is None for p in paths.values())
        if miss:
            print(f"  ASTER GED: {miss}/{len(uniq)} tiles unavailable "
                  f"(ocean / no data) -- their targets dropped", flush=True)

        label = np.full(len(targets), np.nan, np.float32)
        for key, sub in targets.groupby(["_nwlat", "_nwlon"]):
            fp = paths.get(key)
            if fp is None:
                continue
            lab, valid = self._sample_tile(fp, sub)
            idx = sub.index.to_numpy()
            label[idx] = np.where(valid, lab, np.nan)
        targets["label"] = label
        kept = targets.dropna(subset=["label"]).drop(
            columns=["_nwlat", "_nwlon"]).reset_index(drop=True)
        print(f"  ASTER GED: kept {len(kept)}/{len(targets)} labelled targets "
              f"(emissivity {kept.label.min():.3f}-{kept.label.max():.3f})",
              flush=True)
        return kept


# --------------------------------------------------------------------------- #
#  LUCAS land cover (EU) -- classification, balanced accuracy                  #
# --------------------------------------------------------------------------- #
class LucasLandCover(Benchmark):
    """d'Andrimont et al. 2020 harmonised LUCAS (Figshare 9962765). Uses the
    2018 survey; label = LUCAS level-1 land cover (first letter of lc1)."""
    name, task, metric, mode = "lucas_lc", "classification", "balanced_accuracy", "points"
    coverage_bbox = (-12.0, 34.0, 45.0, 72.0)    # Europe (mainland + Nordics)
    survey_year = 2018
    ZIP_URL = "https://ndownloader.figshare.com/files/24265361"
    LC1 = {"A": "Artificial", "B": "Cropland", "C": "Woodland",
           "D": "Shrubland", "E": "Grassland", "F": "Bareland",
           "G": "Water", "H": "Wetland"}
    # land USE (lu1 -> level-1 group); the paper's separate "function" axis
    LU = {"U11": "Agriculture", "U13": "Agriculture", "U12": "Forestry",
          "U41": "Unused", "U42": "Unused", "U37": "Residential",
          "U31": "Transport", "U34": "Services", "U35": "Services",
          "U36": "Services", "U14": "Industry", "U21": "Industry",
          "U22": "Industry", "U32": "Industry", "U33": "Industry"}

    def load_points(self, cache_dir):
        import io
        import zipfile
        fp = _download(self.ZIP_URL, os.path.join(cache_dir,
                                                  "LUCAS_harmonised.zip"))
        with zipfile.ZipFile(fp) as z:
            inner = z.read("LUCAS_harmonised/1_table/lucas_harmo_uf_2018.zip")
        with zipfile.ZipFile(io.BytesIO(inner)) as z2:
            name = next(n for n in z2.namelist() if n.endswith(".csv"))
            with z2.open(name) as f:
                df = pd.read_csv(f, usecols=["th_lat", "th_long", "lc1", "lu1"])
        df = df.dropna(subset=["th_lat", "th_long", "lc1"])
        out = pd.DataFrame({
            "lon": df.th_long.astype(float), "lat": df.th_lat.astype(float),
            "label": df.lc1.astype(str).str[0].map(self.LC1).to_numpy(),
            "label_lu": df.lu1.astype(str).str[:3].map(self.LU).to_numpy()})
        out = _drop_rare(out.dropna(subset=["label"]))   # primary = land cover
        print(f"  LUCAS 2018: {len(out)} points | LC {sorted(out.label.unique())}"
              f" | LU {sorted(out.label_lu.dropna().unique())}", flush=True)
        return out


# --------------------------------------------------------------------------- #
#  LCMAP land cover (CONUS) -- classification, balanced accuracy              #
# --------------------------------------------------------------------------- #
class LcmapLandCover(Benchmark):
    """LCMAP CONUS Reference Data 1984-2018 v1.2 (ScienceBase 5e42e54b). Uses
    2018; label = LCMAP annual land cover class. x/y are CONUS Albers -> WGS84."""
    name, task, metric, mode = "lcmap_lc", "classification", "balanced_accuracy", "points"
    coverage_bbox = (-125.0, 24.0, -66.0, 50.0)  # CONUS
    survey_year = 2018
    ZIP_URL = ("https://www.sciencebase.gov/catalog/file/get/"
               "5e42e54be4b0edb47be84535?f=__disk__71%2Fa5%2F30%2F"
               "71a5309afe8c927921f45da77d5dad96b8ea9630")

    def load_points(self, cache_dir):
        import glob
        import zipfile

        from rasterio.crs import CRS
        from rasterio.warp import transform as warp_transform
        fp = _download(self.ZIP_URL, os.path.join(cache_dir, "LCMAP_REF.zip"))
        ddir = os.path.join(cache_dir, "lcmap_ref")
        if not glob.glob(os.path.join(ddir, "*", "*REF.csv")):
            with zipfile.ZipFile(fp) as z:
                z.extractall(ddir)
        csv = glob.glob(os.path.join(ddir, "*", "*REF.csv"))[0]
        prj = glob.glob(os.path.join(ddir, "*", "*.prj"))[0]
        df = pd.read_csv(csv, usecols=["x", "y", "image_year", "LCMAP",
                                       "dominant_landuse"])
        df = df[(df.image_year == self.survey_year)].dropna(
            subset=["x", "y", "LCMAP"])
        src = CRS.from_wkt(open(prj).read())
        lon, lat = warp_transform(src, "EPSG:4326", df.x.tolist(),
                                  df.y.tolist())
        out = pd.DataFrame({"lon": lon, "lat": lat,
                            "label": df.LCMAP.astype(str).to_numpy(),
                            "label_lu": df.dominant_landuse.astype(str).to_numpy()})
        out = _drop_rare(out)
        print(f"  LCMAP {self.survey_year}: {len(out)} points, classes "
              f"{sorted(out.label.unique())}", flush=True)
        return out


BENCHMARKS = {b.name: b for b in
              [AsterGED(), LucasLandCover(), LcmapLandCover()]}
