"""georvq: pyramidal residual-VQ compression for geospatial datasets."""

__version__ = "0.1.0"


def build_model(cfg, device="cpu"):
    from georvq.factory import build_model as _build
    return _build(cfg, device=device)


def from_checkpoint(path, device="cpu"):
    from georvq.factory import from_checkpoint as _from_ck
    return _from_ck(path, device=device)


def open(store, **kw):
    """Open a georvq store (a path, a URL or a zarr.Group) as a
    rasterio-like dataset, finding its release and decoder from the
    store itself; see georvq.releases.open."""
    from georvq.releases import open as _open
    return _open(store, **kw)
