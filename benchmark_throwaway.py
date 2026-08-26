import os
import time
import shutil
import numpy as np
import h5py
import zarr
import rasterio
from rasterio.transform import from_origin
from zarr.codecs import BytesCodec, GzipCodec  # Pure Gzip codec for Zarr v3

# Shape from previous run: (Time, Levels, Lat, Lon) -> ~124.4 MB raw array data
SHAPE = (24, 5, 360, 720)
DTYPE = np.float32
NUM_RUNS = 3

def generate_datasets():
    print(f"Generating synthetic datasets (Array Shape: {SHAPE}, Size: ~124.4 MB)...")
    white_noise = np.random.rand(*SHAPE).astype(DTYPE)

    t, l, y, x = np.meshgrid(
        np.linspace(0, 1, SHAPE[0]),
        np.linspace(0, 1, SHAPE[1]),
        np.linspace(0, 1, SHAPE[2]),
        np.linspace(0, 1, SHAPE[3]),
        indexing='ij'
    )
    smooth_gradient = (t + l + y + x).astype(DTYPE)
    identical_values = np.zeros(SHAPE, dtype=DTYPE)

    return {
        "White Noise": white_noise,
        "Smooth Gradient": smooth_gradient,
        "Identical Values": identical_values
    }

def get_on_disk_size(path):
    if not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)

    total_bytes = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return total_bytes / (1024 * 1024)

def test_zarr(data, path, compress=True):
    if os.path.exists(path):
        shutil.rmtree(path)

    # Using GzipCodec instead of Blosc to match HDF5 and GeoTIFF perfectly
    codecs = [
        BytesCodec(),
        GzipCodec(level=3)
    ] if compress else [BytesCodec()]

    start = time.time()
    z = zarr.open(
        path,
        mode='w',
        shape=SHAPE,
        chunks=(1, 1, SHAPE[2], SHAPE[3]),
        dtype=DTYPE,
        codecs=codecs
    )
    z[:] = data
    return time.time() - start

def test_hdf5(data, path, compress=True):
    if os.path.exists(path):
        os.remove(path)

    compression = "gzip" if compress else None
    compression_opts = 3 if compress else None

    start = time.time()
    with h5py.File(path, 'w') as f:
        f.create_dataset(
            'data',
            data=data,
            chunks=(1, 1, SHAPE[2], SHAPE[3]),
            compression=compression,
            compression_opts=compression_opts
        )
    return time.time() - start

def test_geotiff(data, dir_path, compress=True):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    transform = from_origin(-180, 90, 1, 1)
    meta = {
        'driver': 'GTiff',
        'height': SHAPE[2],
        'width': SHAPE[3],
        'count': 1,
        'dtype': DTYPE,
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'DEFLATE' if compress else 'NONE',
        'zlevel': 3 if compress else None,
        'num_threads': 'ALL_CPUS'
    }

    start = time.time()
    for t_idx in range(SHAPE[0]):
        for l_idx in range(SHAPE[1]):
            file_path = os.path.join(dir_path, f"t_{t_idx}_l_{l_idx}.tif")
            raster_slice = data[t_idx, l_idx, :, :]

            with rasterio.open(file_path, 'w', **meta) as dst:
                dst.write(raster_slice, 1)

    return time.time() - start

def run_benchmark_suite(test_func, data, path, compress):
    times = []
    for _ in range(NUM_RUNS):
        elapsed = test_func(data, path, compress=compress)
        times.append(elapsed)
    size_mb = get_on_disk_size(path)
    return np.mean(times), np.std(times), size_mb

def main():
    datasets = generate_datasets()
    out_dir = "./benchmark_identical_codec"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nRunning write benchmark ({NUM_RUNS}x averaged) with IDENTICAL codecs...")
    print(f"{'Data Profile':<18} | {'Format & Codec (Level 3)':<35} | {'Mean Time':<12} | {'Std Dev':<9} | {'Disk Size':<10}")
    print("-" * 96)

    for name, data in datasets.items():
        # --- UNCOMPRESSED RUNS ---
        m, s, sz = run_benchmark_suite(test_zarr, data, os.path.join(out_dir, "test.zarr"), compress=False)
        print(f"{name:<18} | Zarr (No Compression)               | {m:>7.4f}s    | ±{s:.3f}s   | {sz:>7.2f} MB")

        m, s, sz = run_benchmark_suite(test_hdf5, data, os.path.join(out_dir, "test.h5"), compress=False)
        print(f"{name:<18} | HDF5 (No Compression)               | {m:>7.4f}s    | ±{s:.3f}s   | {sz:>7.2f} MB")

        m, s, sz = run_benchmark_suite(test_geotiff, data, os.path.join(out_dir, "tiffs"), compress=False)
        print(f"{name:<18} | GeoTIFF (No Compression, Multi)     | {m:>7.4f}s    | ±{s:.3f}s   | {sz:>7.2f} MB")

        # --- IDENTICAL COMPRESSION RUNS ---
        m, s, sz = run_benchmark_suite(test_zarr, data, os.path.join(out_dir, "test.zarr"), compress=True)
        print(f"{name:<18} | Zarr (Gzip / DEFLATE)               | {m:>7.4f}s    | ±{s:.3f}s   | {sz:>7.2f} MB")

        m, s, sz = run_benchmark_suite(test_hdf5, data, os.path.join(out_dir, "test.h5"), compress=True)
        print(f"{name:<18} | HDF5 (Gzip / DEFLATE)               | {m:>7.4f}s    | ±{s:.3f}s   | {sz:>7.2f} MB")

        m, s, sz = run_benchmark_suite(test_geotiff, data, os.path.join(out_dir, "tiffs"), compress=True)
        print(f"{name:<18} | GeoTIFF (Gzip / DEFLATE, Multi)     | {m:>7.4f}s    | ±{s:.3f}s   | {sz:>7.2f} MB")

        print("-" * 96)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

if __name__ == "__main__":
    main()
