"""Chip loaders for the ClimateBenchPress-related stores, in the
(x, valid, q) form that data_probe.iter_chips yields, so data_probe and
pyramid_sizing run on them unchanged.

  cmip6:<var>[:<model>]   one monthly field per chip from
                          /data/CMIP6-samples/<model>_{historical,ssp585}_<var>.zarr;
                          ta carries its 19 pressure levels as bands. A pixel
                          is valid when every band is finite (ta is NaN below
                          topography, tos over land). Default model
                          ACCESS-ESM1-5.
  nextgems:<var>          one random 512 px crop of one random 3-hourly step
                          from the train blocks of /data/NextGEMS-samples.
  agb                     one random 1024 px crop of the 100 m ESA CCI map,
                          kept when at least half of it is valid and at
                          least a tenth carries biomass.
  cams                    one 3-hourly step of the benchmark hej6 no2 file,
                          25 hybrid levels as bands, analysed as log10(no2)
                          because the values span 1e-14..1e-7 and the
                          benchmark bounds them relatively.

None of these files carries an integer packing of its own except the
biomass map (int16 Mg/ha), so the lossless unit is a stated quantum per
variable (LOSSLESS_UNIT below). Precipitation and no2 span orders of
magnitude and the benchmark bounds them relatively, so their quantum is
relative: 1 % steps in the log domain. Temperatures use 0.01 K, OLR
0.1 W m-2. These choices are printed with the numbers and are the only
part of the measurement that is a judgement.
"""
from __future__ import annotations

import glob
import os

import numpy as np

LOSSLESS_UNIT = {
    "cmip6:ta": "0.01 K", "cmip6:tos": "0.01 degC",
    "nextgems:pr": "1 % relative (log domain), zeros kept exact",
    "nextgems:rlut": "0.1 W m-2",
    "agb": "int16 Mg/ha as stored",
    "cams": "1 % relative (log domain); field analysed as log10(no2)",
}


def _quant(x, source):
    if source == "cams":                  # x is already log10(no2)
        return np.round(x * np.log(10) / np.log(1.01)).astype(np.int32)
    if source == "nextgems:pr":
        # zeros stay a distinct symbol below every positive code
        q = np.full(x.shape, -1, dtype=np.int32)
        pos = x > 0
        q[pos] = np.round(np.log(x[pos]) / np.log(1.01)).astype(np.int32)
        return q
    step = {"cmip6:ta": 0.01, "cmip6:tos": 0.01, "nextgems:rlut": 0.1}[source]
    return np.round(x / step).astype(np.int32)


def _finish(x, source, q=None):
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x).all(0)
    x = np.where(valid[None], x, 0.0).astype(np.float32)
    if q is None:
        q = _quant(x, source)
    return x, valid, q


def iter_chips(source, path, n_chips, seed=0):
    import xarray as xr
    rng = np.random.default_rng(seed)
    kind = source.split(":")[0]

    if kind == "cmip6":
        parts = source.split(":")
        var = parts[1]
        model = parts[2] if len(parts) > 2 else "ACCESS-ESM1-5"
        path = path or "/data/CMIP6-samples"
        stores = [xr.open_zarr(p) for p in sorted(glob.glob(
            os.path.join(path, f"{model}_*_{var}.zarr")))
            if not p.endswith("_2020.zarr")]
        n_per = [int(s.time.size) for s in stores]
        picks = rng.integers(0, sum(n_per), n_chips)
        for k in picks:
            k = int(k)
            for s, n in zip(stores, n_per):
                if k < n:
                    da = s[var].isel(time=k)
                    break
                k -= n
            x = da.values
            if x.ndim == 2:
                x = x[None]
            yield _finish(x, f"cmip6:{var}")
        return

    if kind == "nextgems":
        var = source.split(":")[1]
        path = path or "/data/NextGEMS-samples"
        stores = [xr.open_zarr(p) for p in sorted(glob.glob(
            os.path.join(path, "train_block*.zarr")))]
        crop = 512
        for _ in range(n_chips):
            s = stores[int(rng.integers(len(stores)))]
            t = int(rng.integers(s.time.size))
            r = int(rng.integers(0, s.lat.size - crop))
            c = int(rng.integers(0, s.lon.size - crop))
            x = s[var].isel(time=t, lat=slice(r, r + crop),
                            lon=slice(c, c + crop)).values[None]
            yield _finish(x, source)
        return

    if kind == "agb":
        path = path or ("/data/ESA-CCI-biomass/"
                        "ESACCI-BIOMASS-L4-AGB-MERGED-100m-2020-fv5.01.nc")
        ds = xr.open_dataset(path, mask_and_scale=False)
        a = ds["agb"]
        fill = a.attrs.get("_FillValue", -32768)
        crop = 1024
        got = 0
        while got < n_chips:
            r = int(rng.integers(0, a.lat.size - crop))
            c = int(rng.integers(0, a.lon.size - crop))
            q = a.isel(time=0, lat=slice(r, r + crop),
                       lon=slice(c, c + crop)).values.astype(np.int32)
            valid = q != fill
            # ocean and bare ground are stored as 0, not fill, so an
            # all-zero crop passes the fill test; require some biomass
            if valid.mean() < 0.5 or (q > 0).mean() < 0.1:
                continue
            x = np.where(valid, q, 0).astype(np.float32)[None]
            got += 1
            yield x, valid, q[None]
        return

    if kind == "cams":
        path = path or "/data/CAMS-samples/cams_no2_hej6.nc"
        ds = xr.open_dataset(path)
        n = ds.step.size
        for k in rng.permutation(n)[:n_chips]:
            x = ds["no2"].isel(step=int(k)).values     # (hybrid, lat, lon)
            x = np.log10(np.clip(x, 1e-30, None))
            yield _finish(x, "cams")
        return

    raise ValueError(f"unknown source {source}")
