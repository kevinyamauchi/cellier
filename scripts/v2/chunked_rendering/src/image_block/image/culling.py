"""Viewport culling for tiled 2D image rendering.

Performs a vectorised 2D AABB overlap test: for each tile, compute its
world-space bounding box and test against the viewport rectangle.
"""

from __future__ import annotations

import numpy as np

from image_block.core.block_key import BlockKey
from image_block.image.layout import BlockLayout2D


def viewport_cull(
    required: dict[BlockKey, int],
    base_layout: BlockLayout2D,
    camera_info: dict,
) -> tuple[dict[BlockKey, int], int]:
    """Remove tiles that lie entirely outside the viewport.

    Parameters
    ----------
    required : dict[BlockKey, int]
        Tile key -> level mapping (order-preserving).
    base_layout : BlockLayout2D
        Layout of the finest level.
    camera_info : dict
        Must contain ``view_min`` and ``view_max`` as ndarray shape (2,),
        representing the viewport AABB in world space ``(x, y)``.

    Returns
    -------
    culled : dict[BlockKey, int]
        Subset of ``required`` that overlaps the viewport.
    n_culled : int
        Number of tiles removed.
    """
    if not required:
        return required, 0

    view_min = np.asarray(camera_info["view_min"], dtype=np.float64)
    view_max = np.asarray(camera_info["view_max"], dtype=np.float64)
    bs = base_layout.block_size

    keys = list(required.keys())
    n = len(keys)

    # Build arrays of (level, gy, gx).
    levels = np.array([k.level for k in keys], dtype=np.int32)
    gy = np.array([k.gy for k in keys], dtype=np.float64)
    gx = np.array([k.gx for k in keys], dtype=np.float64)

    scale = (2.0 ** (levels - 1)).astype(np.float64)
    bw = bs * scale  # world-space tile width at this level

    # Tile AABB in world space: [gx*bw, gy*bw] to [(gx+1)*bw, (gy+1)*bw]
    tile_min_x = gx * bw
    tile_min_y = gy * bw
    tile_max_x = (gx + 1.0) * bw
    tile_max_y = (gy + 1.0) * bw

    # AABB overlap test: tile overlaps viewport iff
    #   tile_max > view_min AND tile_min < view_max  (per axis)
    visible = (
        (tile_max_x > view_min[0])
        & (tile_min_x < view_max[0])
        & (tile_max_y > view_min[1])
        & (tile_min_y < view_max[1])
    )

    n_culled = n - int(np.sum(visible))
    if n_culled == 0:
        return required, 0

    culled = {}
    for i, keep in enumerate(visible):
        if keep:
            k = keys[i]
            culled[k] = required[k]

    return culled, n_culled


if __name__ == "__main__":
    layout = BlockLayout2D.from_shape((256, 256), block_size=32, overlap=1)
    # Create some tiles at level 1 (finest, 8x8 grid).
    required = {}
    for gy in range(8):
        for gx in range(8):
            required[BlockKey(1, gy, gx)] = 1

    # Viewport covers roughly the center quarter.
    camera_info = {
        "view_min": np.array([64.0, 64.0]),
        "view_max": np.array([192.0, 192.0]),
    }
    culled, n_culled = viewport_cull(required, layout, camera_info)
    print(f"Total: {len(required)}, visible: {len(culled)}, culled: {n_culled}")
    print("Culling OK")
