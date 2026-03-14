# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "zarr",
#   "scikit-image",
# ]
# ///
"""Generate sample zarr data for the multiscale chunked rendering viewer.

Creates a single zarr group with one array per resolution level:

    multiscale_blobs.zarr/
        s0   — 1024³, chunks (32, 32, 32), float32  (~4 GB)
        s1   —  512³, chunks (32, 32, 32), float32  (~512 MB)
        s2   —  256³, chunks (32, 32, 32), float32  (~64 MB)

Usage
-----
    uv run make_zarr_sample.py

Notes
-----
Generating the 1024³ blob volume requires roughly 4 GB of RAM.
"""

import pathlib

import numpy as np
import zarr
from skimage.data import binary_blobs
from skimage.transform import downscale_local_mean

# ---------------------------------------------------------------------------
# Parameters — match lod_debug.py constants
# ---------------------------------------------------------------------------
SHAPE: tuple[int, int, int] = (1024, 1024, 1024)
CHUNK: tuple[int, int, int] = (32, 32, 32)

OUT_PATH = pathlib.Path(__file__).parent / "multiscale_blobs.zarr"

# (downscale factor, array name)
LEVELS = [(1, "s0"), (2, "s1"), (4, "s2")]

# ---------------------------------------------------------------------------
# Generate data at full resolution
# ---------------------------------------------------------------------------
print(f"Generating 3D blob volume at {SHAPE[0]}³ …")
blobs = binary_blobs(
    length=SHAPE[0],
    n_dim=3,
    blob_size_fraction=0.05,
    volume_fraction=0.15,
).astype(np.float32)

# ---------------------------------------------------------------------------
# Write each scale level into a single zarr group
# ---------------------------------------------------------------------------
store = zarr.open_group(str(OUT_PATH), mode="w")

for factor, name in LEVELS:
    if factor == 1:
        data = blobs
    else:
        print(f"Downscaling by {factor}x …")
        data = downscale_local_mean(blobs, (factor, factor, factor)).astype(
            np.float32
        )

    print(f"Writing {name}: shape={data.shape} → {OUT_PATH}/{name} …")
    arr = store.create_array(
        name,
        shape=data.shape,
        chunks=CHUNK,
        dtype="f4",
    )
    arr[:] = data

    print(f"  shape  : {arr.shape}")
    print(f"  chunks : {arr.chunks}")
    print(f"  dtype  : {arr.dtype}")
    print(f"  size   : {data.nbytes / 1024 ** 2:.1f} MB")

print(f"\nDone. Output: {OUT_PATH}")
