"""Example of rendering a zarr file with tiles."""

import numpy as np
import zarr
from numcodecs import Blosc
from skimage.data import binary_blobs
from skimage.transform import resize

base_length = 1024
chunk_shape = (32, 32, 32)


image_0 = binary_blobs(
    length=base_length, n_dim=3, blob_size_fraction=0.2, volume_fraction=0.2
).astype(np.float32)

multiscale_image = [image_0]
for level in range(1, 5):
    edge_length = int(base_length / (2**level))
    print(edge_length)
    multiscale_image.append(
        resize(image_0, (edge_length, edge_length, edge_length), order=0)
    )

# write the images
root = zarr.open("multiscale_blobs.zarr", mode="w")
compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
for level_index, image in enumerate(multiscale_image):
    name = f"level_{level_index}"
    print(name)
    array = root.zeros(name, shape=image.shape, chunks=chunk_shape, dtype="f4")
    array[:] = image
