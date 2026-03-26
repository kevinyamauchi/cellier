from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

if TYPE_CHECKING:
    from cellier.v2.render.block_cache._tile_manager_2d import TileManager2D
    from cellier.v2.render.lut_indirection._layout_2d import BlockLayout2D


class LutIndirectionManager2D:
    """Manages the 2D LUT indirection texture.

    The LUT maps each finest-level grid cell ``(gy, gx)`` to a cache
    slot ``(sx, sy)`` and a level indicator.  ``rebuild()`` rewrites
    the LUT from the current tile manager state.

    Parameters
    ----------
    base_layout : BlockLayout2D
        Layout of the finest (level 1) resolution.
    n_levels : int
        Total number of LOD levels.
    """

    def __init__(self, base_layout: BlockLayout2D, n_levels: int) -> None:
        self._base_layout = base_layout
        self._n_levels = n_levels
        self.lut_data, self.lut_tex = build_lut_texture_2d(base_layout.grid_dims)

    def rebuild(self, tile_manager: TileManager2D) -> None:
        """Rewrite ``lut_data`` from current tilemap state and schedule GPU upload.

        Sweeps coarsest-to-finest so finer tiles naturally overwrite
        coarser fallbacks in the base grid.

        Parameters
        ----------
        tile_manager : TileManager2D
            Current tile manager holding the resident tile mapping.
        """
        rebuild_lut_2d(
            self._base_layout,
            tile_manager,
            self._n_levels,
            self.lut_data,
            self.lut_tex,
        )


def build_lut_texture_2d(
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


def rebuild_lut_2d(
    base_layout: BlockLayout2D,
    tile_manager: TileManager2D,
    n_levels: int,
    lut_data: np.ndarray,
    lut_tex: gfx.Texture,
) -> None:
    """Rebuild the full 2D LUT from the current tile manager state.

    Works coarsest-to-finest.  Each level writes its cache slot
    coordinates into all base-grid cells it covers.  Finer levels
    overwrite coarser fallbacks.

    Channel assignments:

    - ``lut[gy, gx, 0]`` = sx (cache grid X)
    - ``lut[gy, gx, 1]`` = sy (cache grid Y)
    - ``lut[gy, gx, 2]`` = 0  (unused, reserved)
    - ``lut[gy, gx, 3]`` = level (1 = finest; 0 = out-of-bounds)

    Parameters
    ----------
    base_layout : BlockLayout2D
        Layout of the finest resolution.
    tile_manager : TileManager2D
        Current tile manager with resident tiles.
    n_levels : int
        Total number of LOD levels.
    lut_data : np.ndarray
        Backing float32 array ``(gH, gW, 4)`` to overwrite.
    lut_tex : gfx.Texture
        The LUT texture to schedule for GPU upload.
    """
    gh, gw = base_layout.grid_dims

    lut_data[:] = 0  # Reset to out-of-bounds (level 0).

    # Group resident tiles by level.
    by_level: dict[int, list] = {}
    for key, slot in tile_manager.tilemap.items():
        if key.level > 0:
            by_level.setdefault(key.level, []).append((key, slot))

    # Write coarsest first so finer levels overwrite.
    for level in range(n_levels, 0, -1):
        if level not in by_level:
            continue
        scale = 2 ** (level - 1)
        for key, slot in by_level[level]:
            sy, sx = slot.grid_pos
            # Base-grid slice covered by this coarse tile, clamped.
            gy0 = key.gy * scale
            gy1 = min(gy0 + scale, gh)
            gx0 = key.gx * scale
            gx1 = min(gx0 + scale, gw)
            lut_data[gy0:gy1, gx0:gx1] = (sx, sy, level, 0)

    lut_tex.update_range((0, 0, 0), lut_tex.size)
