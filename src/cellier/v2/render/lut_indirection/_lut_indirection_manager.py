"""LUT-based indirection manager for bricked volume rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

if TYPE_CHECKING:
    from cellier.v2.render.block_cache import TileManager
    from cellier.v2.render.lut_indirection._layout import BlockLayout3D


class LutIndirectionManager3D:
    """LUT indirection table for a bricked 3-D volume.

    Maintains a CPU-side uint8 array and a matching GPU RGBA8UI texture.
    Each voxel of the LUT corresponds to one brick position in the
    finest-level grid and encodes ``(sx, sy, sz, level)`` — the cache
    slot where that brick (or its best coarser fallback) currently lives.

    Notes
    -----
    Channel assignments follow the axis convention:

    - ``lut[d, h, w, 0]`` = tile_x = sx (cache W axis)
    - ``lut[d, h, w, 1]`` = tile_y = sy (cache H axis)
    - ``lut[d, h, w, 2]`` = tile_z = sz (cache D axis)
    - ``lut[d, h, w, 3]`` = level  (1 = finest; 0 = out-of-bounds)

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest LOAD level.  Determines the LUT grid
        dimensions ``(gD, gH, gW)``.
    n_levels : int
        Total number of level of detail levels.  Used by ``rebuild()`` for the
        coarse-to-fine fallback sweep.

    Attributes
    ----------
    lut_data : np.ndarray
        CPU backing array, shape ``(gD, gH, gW, 4)``, dtype uint8.
        Channel layout: ``(tile_x=sx, tile_y=sy, tile_z=sz, level)``.
        All zeros = out-of-bounds (level 0, renders the null color).
    lut_tex : gfx.Texture
        GPU RGBA8UI 3-D texture wrapping ``lut_data``.
    """

    def __init__(self, base_layout: BlockLayout3D, n_levels: int) -> None:
        self._base_layout = base_layout
        self._n_levels = n_levels
        self.lut_data, self.lut_tex = build_lut_texture(base_layout)

    # ------------------------------------------------------------------
    # GPU writes
    # ------------------------------------------------------------------

    def rebuild(self, tile_manager: TileManager) -> None:
        """Rewrite ``lut_data`` from current tilemap state and schedule GPU upload.

        Strategy: sweep coarsest-to-finest through all resident bricks.
        Each brick slice-writes the base-grid cells it covers, so finer
        writes naturally overwrite coarser fallbacks.

        The GPU upload is deferred until the next ``renderer.render()``
        call (see the pygfx texture update_range behavior).

        Parameters
        ----------
        tile_manager : TileManager
            Current tile manager holding the resident brick mapping.
        """
        rebuild_lut(
            self._base_layout,
            tile_manager,
            self._n_levels,
            self.lut_data,
            self.lut_tex,
        )


def build_lut_texture(base_layout: BlockLayout3D) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate the LUT texture (zeroed = all out-of-bounds).

    Notes
    -----
    Channel assignments follow the axis convention:

    - ``lut[d, h, w, 0]`` = tile_x = sx (cache W axis)
    - ``lut[d, h, w, 1]`` = tile_y = sy (cache H axis)
    - ``lut[d, h, w, 2]`` = tile_z = sz (cache D axis)
    - ``lut[d, h, w, 3]`` = level  (1 = finest; 0 = out-of-bounds)

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
    base_layout: BlockLayout3D,
    tile_manager: TileManager,
    n_levels: int,
    lut_data: np.ndarray,
    lut_tex: gfx.Texture,
) -> None:
    """Rebuild the full LUT from the current tile manager state.

    This works coarsest-to-finest.  Each level writes its slot
    data into all base-grid cells it covers via a numpy slice assignment
    (one C-speed fill per resident brick).  Because finer levels are
    written last, they naturally overwrite the coarser fallback.

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest (level 1) resolution.
    tile_manager : TileManager
        Current tile manager with resident bricks.
    n_levels : int
        Total number of LOAD levels.
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
            gz0 = key.gz * scale
            gz1 = min(gz0 + scale, gd)
            gy0 = key.gy * scale
            gy1 = min(gy0 + scale, gh)
            gx0 = key.gx * scale
            gx1 = min(gx0 + scale, gw)
            # Single numpy slice assignment — fills the block at C speed.
            lut_data[gz0:gz1, gy0:gy1, gx0:gx1] = (sx, sy, sz, level)

    lut_tex.update_range((0, 0, 0), lut_tex.size)
