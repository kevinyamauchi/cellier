"""LUT (look-up table) texture for 2D tiled image rendering.

The LUT maps each tile in the finest-level grid to its current slot
in the GPU cache.  Shape is ``(gH, gW, 4)`` with dtype ``float32``.

Channel layout: ``(cache_tile_x, cache_tile_y, level, 0)``.
Slot ``(0, 0)`` is the reserved empty slot (zero data, black).

IMPORTANT: dtype MUST be float32, not uint8.  pygfx maps ``np.uint8``
to ``rgba8unorm``, which normalises slot indices (e.g. 3) to
``3/255 ~ 0.01``, collapsing all tile origins to zero (all-black).
float32 stores slot indices as exact integers (0.0, 1.0, 2.0, ...)
with no precision loss.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx

from image_block.core.block_key import BlockKey
from image_block.core.tile_manager import TileManager
from image_block.image.layout import BlockLayout2D


def build_lut_texture(
    grid_dims: tuple[int, int],
) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate CPU lut_data array and a gfx.Texture.

    Parameters
    ----------
    grid_dims : tuple[int, int]
        ``(gH, gW)`` -- finest level grid dimensions.

    Returns
    -------
    lut_data : np.ndarray
        Shape ``(gH, gW, 4)``, dtype float32.  Initialised to zeros
        (all tiles point to the reserved empty slot).
    lut_tex : gfx.Texture
        pygfx 2D texture wrapping ``lut_data``.
    """
    gh, gw = grid_dims
    lut_data = np.zeros((gh, gw, 4), dtype=np.float32)
    lut_tex = gfx.Texture(lut_data, dim=2)
    return lut_data, lut_tex


def rebuild_lut(
    base_layout: BlockLayout2D,
    tile_manager: TileManager,
    n_levels: int,
    lut_data: np.ndarray,
    lut_tex: gfx.Texture,
) -> None:
    """Rewrite lut_data from tile_manager.tilemap and upload to GPU.

    For each position in the finest-level grid, walks from finest to
    coarsest level and writes the first resident tile's cache slot
    coordinates into the LUT.

    Called once per commit batch (not per tile).

    Parameters
    ----------
    base_layout : BlockLayout2D
        Layout of the finest level.
    tile_manager : TileManager
        Current tile manager with resident tiles.
    n_levels : int
        Total number of LOAD levels.
    lut_data : np.ndarray
        Backing float32 array ``(gH, gW, 4)`` to overwrite.
    lut_tex : gfx.Texture
        The LUT texture to schedule for GPU upload.
    """
    gh, gw = base_layout.grid_dims
    tilemap = tile_manager.tilemap

    lut_data[:] = 0.0  # Reset everything to empty slot.

    for gy in range(gh):
        for gx in range(gw):
            # Walk from finest to coarsest.
            for level in range(1, n_levels + 1):
                scale = 2 ** (level - 1)
                key = BlockKey(
                    level=level,
                    gz=0,
                    gy=gy // scale,
                    gx=gx // scale,
                )
                if key in tilemap:
                    slot = tilemap[key]
                    sy, sx = slot.grid_pos
                    lut_data[gy, gx, 0] = float(sx)
                    lut_data[gy, gx, 1] = float(sy)
                    lut_data[gy, gx, 2] = float(level)
                    # Channel 3 unused, stays 0.0
                    break

    lut_tex.update_full()


if __name__ == "__main__":
    grid_dims = (4, 4)
    lut_data, lut_tex = build_lut_texture(grid_dims)
    print(f"LUT: shape={lut_data.shape}, dtype={lut_data.dtype}")
    print(f"  all zeros: {np.all(lut_data == 0.0)}")
    print("LUT OK")
