"""Generate sample zarr data for the chunked rendering viewer example.

Creates ``chunked_blobs.zarr`` — a single 256³ float32 array (chunk size 32³)
containing a synthetic blob image.  The file is written next to this script so
that ``viewer.py`` can find it with a relative path.

Usage
-----
    python make_zarr_sample.py

Requirements
------------
    pip install zarr scikit-image
"""

import pathlib

import numpy as np
import zarr
from skimage.data import binary_blobs

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SHAPE: tuple[int, int, int] = (256, 256, 256)
CHUNKS: tuple[int, int, int] = (32, 32, 32)
OUT_PATH = pathlib.Path(__file__).parent / "chunked_blobs.zarr"

# ---------------------------------------------------------------------------
# Generate data
# ---------------------------------------------------------------------------
print("Generating 3D blob volume …")
blobs = binary_blobs(
    length=SHAPE[0],
    n_dim=3,
    blob_size_fraction=0.1,
    volume_fraction=0.15,
    seed=42,
).astype(np.float32)

# ---------------------------------------------------------------------------
# Write to zarr
# ---------------------------------------------------------------------------
print(f"Writing to {OUT_PATH} …")
store = zarr.open(
    str(OUT_PATH),
    mode="w",
    shape=SHAPE,
    chunks=CHUNKS,
    dtype="f4",
)
store[:] = blobs

print("Done.")
print(f"  path   : {OUT_PATH}")
print(f"  shape  : {store.shape}")
print(f"  chunks : {store.chunks}")
print(f"  dtype  : {store.dtype}")
print(f"  size   : {blobs.nbytes / 1024**2:.1f} MB")
