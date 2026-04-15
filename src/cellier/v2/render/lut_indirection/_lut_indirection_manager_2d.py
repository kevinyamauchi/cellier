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
    scale_vecs_data : list[np.ndarray] or None
        Per-level scale vectors in data-axis order ``(sy, sx)`` — i.e.
        ``(scale_row, scale_col)`` for the tile grid.  When provided the
        LUT fill uses the actual per-axis downsampling factor instead of
        assuming a uniform ``2^(level-1)`` factor.  Pass
        ``ImageGeometry2D._scale_vecs_data`` here.
    """

    def __init__(
        self,
        base_layout: BlockLayout2D,
        n_levels: int,
        scale_vecs_data: list[np.ndarray] | None = None,
    ) -> None:
        self._base_layout = base_layout
        self._n_levels = n_levels
        self._scale_vecs_data = scale_vecs_data
        self.lut_data, self.lut_tex = build_lut_texture_2d(base_layout.grid_dims)

    def rebuild(
        self,
        tile_manager: TileManager2D,
        current_slice_coord: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        """Rewrite ``lut_data`` from current tilemap state and schedule GPU upload.

        Uses a two-phase sweep when ``current_slice_coord`` is provided:
        old-slice tiles are written first (background), then current-slice
        tiles overwrite them (foreground).  This keeps the previous image
        visible while new tiles stream in.

        Parameters
        ----------
        tile_manager : TileManager2D
            Current tile manager holding the resident tile mapping.
        current_slice_coord : tuple of (axis_index, world_value) pairs or None
            The slice coordinate set at the start of the most-recent
            ``build_slice_request_2d`` call.  When ``None`` all tiles are
            treated as foreground (backward-compatible single-phase sweep).
        """
        rebuild_lut_2d(
            self._base_layout,
            tile_manager,
            self._n_levels,
            self.lut_data,
            self.lut_tex,
            scale_vecs_data=self._scale_vecs_data,
            current_slice_coord=current_slice_coord,
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
    scale_vecs_data: list[np.ndarray] | None = None,
    current_slice_coord: tuple[tuple[int, int], ...] | None = None,
) -> None:
    """Rebuild the full 2D LUT from the current tile manager state.

    When ``current_slice_coord`` is provided, uses a two-phase sweep:

    - **Phase 1 (background):** Write tiles whose ``slice_coord`` differs from
      ``current_slice_coord``, coarsest-to-finest.  These are old-slice tiles
      that serve as a visual placeholder while new data loads.
    - **Phase 2 (foreground):** Write tiles whose ``slice_coord`` matches
      ``current_slice_coord``, coarsest-to-finest.  These overwrite the
      background wherever new data has arrived.

    When ``current_slice_coord`` is ``None``, all tiles are written in a single
    coarsest-to-finest sweep (backward-compatible behaviour).

    Channel assignments:

    - ``lut[gy, gx, 0]`` = sx (cache grid X)
    - ``lut[gy, gx, 1]`` = sy (cache grid Y)
    - ``lut[gy, gx, 2]`` = level (1 = finest; 0 = out-of-bounds)
    - ``lut[gy, gx, 3]`` = 0  (unused, reserved)

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
    scale_vecs_data : list[np.ndarray] or None
        Per-level scale vectors in data-axis order ``(sy, sx)`` mapping
        level-k tiles to base-grid coverage.  When provided the actual
        per-axis downsampling factor is used; otherwise a uniform
        ``2^(level-1)`` fallback is applied (correct only for isotropic
        power-of-2 multiscale pyramids).
    current_slice_coord : tuple of (axis_index, world_value) pairs or None
        The slice coordinate to use for phase separation.  ``None`` disables
        the two-phase sweep.
    """
    gh, gw = base_layout.grid_dims

    lut_data[:] = 0  # Reset to out-of-bounds (level 0).

    def _write_tiles(tiles_by_level: dict[int, list]) -> None:
        """Write one group of tiles coarsest-to-finest into lut_data."""
        for level in range(n_levels, 0, -1):
            if level not in tiles_by_level:
                continue
            level_idx = level - 1  # 0-indexed into scale_vecs_data
            if scale_vecs_data is not None and level_idx < len(scale_vecs_data):
                sv = scale_vecs_data[level_idx]
                scale_y = max(1, int(round(float(sv[0]))))
                scale_x = max(1, int(round(float(sv[1]))))
            else:
                iso_scale = 2 ** (level - 1)
                scale_y = iso_scale
                scale_x = iso_scale
            for key, slot in tiles_by_level[level]:
                sy, sx = slot.grid_pos
                # Base-grid slice covered by this coarse tile, clamped.
                gy0 = key.gy * scale_y
                gy1 = min(gy0 + scale_y, gh)
                gx0 = key.gx * scale_x
                gx1 = min(gx0 + scale_x, gw)
                lut_data[gy0:gy1, gx0:gx1] = (sx, sy, level, 0)

    if current_slice_coord is None:
        # Single-phase: all tiles are treated as foreground.
        by_level: dict[int, list] = {}
        for key, slot in tile_manager.tilemap.items():
            if key.level > 0:
                by_level.setdefault(key.level, []).append((key, slot))
        _write_tiles(by_level)
    else:
        # Two-phase: background (old-slice) first, then foreground (current-slice).
        bg_by_level: dict[int, list] = {}
        fg_by_level: dict[int, list] = {}
        for key, slot in tile_manager.tilemap.items():
            if key.level <= 0:
                continue
            if key.slice_coord == current_slice_coord:
                fg_by_level.setdefault(key.level, []).append((key, slot))
            else:
                bg_by_level.setdefault(key.level, []).append((key, slot))
        _write_tiles(bg_by_level)
        _write_tiles(fg_by_level)

    lut_tex.update_range((0, 0, 0), lut_tex.size)
