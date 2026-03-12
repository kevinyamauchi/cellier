"""Generate sample zarr data for the multiscale chunked rendering viewer.

Creates three independent zarr arrays — one per resolution level:

    multiscale_blobs_s0.zarr   — 1024³, chunks (64, 64, 64), float32  (~268 MB)
    multiscale_blobs_s1.zarr   —  512³, chunks (64, 64, 64), float32  ( ~34 MB)
    multiscale_blobs_s2.zarr   —  256³, chunks (64, 64, 64), float32  (  ~4 MB)

The files are written next to this script so that ``viewer.py`` can find them
with relative paths.

Usage
-----
    python make_zarr_sample.py

Requirements
------------
    pip install zarr scikit-image

Notes
-----
Generating the 1024³ blob volume requires roughly 1 GB of RAM.  If memory is
tight you can reduce ``SHAPE`` to ``(512, 512, 512)``; the viewer still
demonstrates multiscale behaviour.
"""

import pathlib

import numpy as np
import zarr
from skimage.data import binary_blobs
from skimage.transform import downscale_local_mean

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SHAPE: tuple[int, int, int] = (1024, 1024, 1024)
CHUNK: tuple[int, int, int] = (64, 64, 64)

OUT_DIR = pathlib.Path(__file__).parent

# (downscale factor, output name)
LEVELS = [(1, "s0"), (2, "s1"), (4, "s2")]

# ---------------------------------------------------------------------------
# Generate data at full resolution
# ---------------------------------------------------------------------------
print("Generating 3D blob volume at 1024³ …")
blobs = binary_blobs(
    length=SHAPE[0],
    n_dim=3,
    blob_size_fraction=0.05,
    volume_fraction=0.15,
).astype(np.float32)

# ---------------------------------------------------------------------------
# Write each scale level
# ---------------------------------------------------------------------------
for factor, name in LEVELS:
    out_path = OUT_DIR / f"multiscale_blobs_{name}.zarr"

    if factor == 1:
        data = blobs
    else:
        print(f"Downscaling by {factor}x ...")
        data = downscale_local_mean(blobs, (factor, factor, factor)).astype(np.float32)

    print(f"Writing {name}: shape={data.shape} → {out_path} …")
    z = zarr.open(
        str(out_path),
        mode="w",
        shape=data.shape,
        chunks=CHUNK,
        dtype="f4",
    )
    z[:] = data

    print(f"  shape  : {z.shape}")
    print(f"  chunks : {z.chunks}")
    print(f"  dtype  : {z.dtype}")
    print(f"  size   : {data.nbytes / 1024 ** 2:.1f} MB")

print("Done.")
