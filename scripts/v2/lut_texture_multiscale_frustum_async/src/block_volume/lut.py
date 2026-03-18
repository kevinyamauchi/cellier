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
from block_volume.tile_manager import TileManager


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

    Strategy: work **coarsest-to-finest**.  Each level writes its slot
    data into all base-grid cells it covers via a numpy slice assignment
    (one C-speed fill per resident brick).  Because finer levels are
    written last, they naturally overwrite the coarser fallback —
    exactly the same result as the original finest-first walk, but
    without the triple-nested Python loop over all grid cells.

    Complexity: O(n_resident_bricks) Python iterations, each doing a
    constant-time numpy slice write.  For a 32³ grid with 3 levels and
    a 3375-slot cache this is ~100-1000× faster than the original loop.

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

    lut_data[:] = 0  # Reset everything to out-of-bounds (level 0 = black).

    # Group resident bricks by level so we can iterate each level once.
    by_level: dict[int, list] = {}
    for key, slot in tile_manager.tilemap.items():
        if key.level > 0:
            by_level.setdefault(key.level, []).append((key, slot))

    # Write coarsest first so finer levels overwrite where both are resident.
    for level in range(n_levels, 0, -1):
        if level not in by_level:
            continue
        scale = 2 ** (level - 1)
        for key, slot in by_level[level]:
            sz, sy, sx = slot.grid_pos
            # Base-grid slice covered by this coarse brick, clamped to grid.
            gz0 = key.gz * scale;  gz1 = min(gz0 + scale, gd)
            gy0 = key.gy * scale;  gy1 = min(gy0 + scale, gh)
            gx0 = key.gx * scale;  gx1 = min(gx0 + scale, gw)
            # Single numpy slice assignment — fills the block at C speed.
            lut_data[gz0:gz1, gy0:gy1, gx0:gx1] = (sx, sy, sz, level)

    lut_tex.update_range((0, 0, 0), lut_tex.size)
