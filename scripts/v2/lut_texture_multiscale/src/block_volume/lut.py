"""LUT (lookup texture) construction for bricked volume rendering.

Phase 1: the LUT is rebuilt each frame from the TileManager state.
Each entry falls back to the finest cached ancestor; missing bricks
render black (level 0).

The LUT is an RGBA8UI 3D texture where each voxel encodes
``(tile_x, tile_y, tile_z, level)`` as uint8 values.

Notes
-----
Channel assignments follow the axis convention:

- ``lut[d, h, w, 0]`` = tile_x = sx (cache W axis)
- ``lut[d, h, w, 1]`` = tile_y = sy (cache H axis)
- ``lut[d, h, w, 2]`` = tile_z = sz (cache D axis)
- ``lut[d, h, w, 3]`` = level  (1 = finest; 0 = out-of-bounds)
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx

from block_volume.layout import BlockLayout
from block_volume.tile_manager import BrickKey, TileManager


def build_lut_texture(base_layout: BlockLayout) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate the LUT texture (zeroed = all out-of-bounds).

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest (level 1) resolution.

    Returns
    -------
    lut_data : np.ndarray
        Backing uint8 array of shape ``(gD, gH, gW, 4)``.
    lut_tex : gfx.Texture
        RGBA8UI 3D texture.
    """
    gd, gh, gw = base_layout.grid_dims
    lut_data = np.zeros((gd, gh, gw, 4), dtype=np.uint8)
    lut_tex = gfx.Texture(lut_data, dim=3, format="rgba8uint")
    return lut_data, lut_tex


def rebuild_lut(
    base_layout: BlockLayout,
    tile_manager: TileManager,
    n_levels: int,
    lut_data: np.ndarray,
    lut_tex: gfx.Texture,
) -> None:
    """Rebuild the full LUT from the current tile manager state.

    For each brick in the base grid, walk from the finest level up to
    the coarsest.  The first level that has a cached tile is used.  If
    nothing is cached, the entry stays ``(0, 0, 0, 0)`` (black).

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest (level 1) resolution.
    tile_manager : TileManager
        Current tile manager with resident bricks.
    n_levels : int
        Total number of LOD levels.
    lut_data : np.ndarray
        Backing uint8 array ``(gD, gH, gW, 4)`` to overwrite.
    lut_tex : gfx.Texture
        The LUT texture to schedule for GPU upload.
    """
    gd, gh, gw = base_layout.grid_dims
    tilemap = tile_manager.tilemap

    lut_data[:] = 0  # Reset everything to out-of-bounds.

    for gz in range(gd):
        for gy in range(gh):
            for gx in range(gw):
                # Walk from finest to coarsest.
                for level in range(1, n_levels + 1):
                    scale = 2 ** (level - 1)
                    key = BrickKey(
                        level=level,
                        gz=gz // scale,
                        gy=gy // scale,
                        gx=gx // scale,
                    )
                    if key in tilemap:
                        slot = tilemap[key]
                        sz, sy, sx = slot.grid_pos
                        lut_data[gz, gy, gx, 0] = sx
                        lut_data[gz, gy, gx, 1] = sy
                        lut_data[gz, gy, gx, 2] = sz
                        lut_data[gz, gy, gx, 3] = level
                        break

    lut_tex.update_full()
