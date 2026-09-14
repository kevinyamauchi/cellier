"""LUT-based indirection manager for bricked volume rendering."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

from cellier.logging import _GPU_LOGGER
from cellier.render.lut_indirection._cell_brick_rule import (
    brick_rule_issues,
    cell_range,
    level_brick_counts,
    level_cell_spans,
)

if TYPE_CHECKING:
    from cellier.render.block_cache import TileManager3D
    from cellier.render.lut_indirection._layout_3d import BlockLayout3D


class LutIndirectionManager3D:
    """LUT indirection table for a bricked 3-D volume.

    Maintains a CPU-side uint8 array and a matching GPU RGBA8UI texture.
    Each voxel of the LUT corresponds to one brick position in the
    finest-level grid and encodes ``(sx, sy, sz, level)`` — the cache
    slot where that brick (or its best coarser fallback) currently lives.

    A companion ``brick_max_tex`` (R32Float, same grid dimensions) stores
    the maximum intensity per brick for MIP early-out.

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
    level_scale_vecs_data : list of ndarray, optional
        Per-level scale vectors in data order ``(sz, sy, sx)``.  Entry ``k``
        is the downscale factor of level ``k`` relative to the finest level.
        When provided, ``rebuild()`` uses the actual per-axis scales instead
        of the uniform ``2^(level-1)`` assumption.  Required for datasets
        where axes are downsampled at different rates (e.g. z-anisotropic
        microscopy data where z is never downsampled).
    level_shapes : list of tuple of int, optional
        Per-level voxel shapes in the same order.  Gives each level's brick
        count, so the last brick owns the grid's tail cells (see
        ``_cell_brick_rule``).  Without it the count is inferred from the grid.
    border : float, optional
        The brick cache's padding in voxels.  When given together with the
        scales and shapes, levels whose cell -> brick rule needs more padding
        than this are logged as warnings.

    Attributes
    ----------
    lut_data : np.ndarray
        CPU backing array, shape ``(gD, gH, gW, 4)``, dtype uint8.
        Channel layout: ``(tile_x=sx, tile_y=sy, tile_z=sz, level)``.
        All zeros = out-of-bounds (level 0, renders the null color).
    lut_tex : gfx.Texture
        GPU RGBA8UI 3-D texture wrapping ``lut_data``.
    brick_max_data : np.ndarray
        CPU backing array, shape ``(gD, gH, gW)``, dtype float32.
        Stores per-brick maximum intensity for MIP early-out.
    brick_max_tex : gfx.Texture
        GPU R32Float 3-D texture wrapping ``brick_max_data``.
    """

    def __init__(
        self,
        base_layout: BlockLayout3D,
        n_levels: int,
        level_scale_vecs_data: list | None = None,
        level_shapes: list | None = None,
        border: float | None = None,
    ) -> None:
        self._base_layout = base_layout
        self._n_levels = n_levels
        self._level_scale_vecs_data = level_scale_vecs_data
        self._level_shapes = level_shapes
        if border is not None:
            for issue in brick_rule_issues(
                level_scale_vecs_data, level_shapes, base_layout.block_size, border
            ):
                _GPU_LOGGER.warning("brick_rule_padding  3d  %s", issue.describe())
        self.lut_data, self.lut_tex = build_lut_texture(base_layout)
        self.brick_max_data, self.brick_max_tex = build_brick_max_texture(base_layout)

    # ------------------------------------------------------------------
    # GPU writes
    # ------------------------------------------------------------------

    def rebuild(
        self,
        tile_manager: TileManager3D,
        current_slice_coord: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        """Rewrite ``lut_data`` from current tilemap state and schedule GPU upload.

        Strategy: sweep coarsest-to-finest through all resident bricks.
        Each brick slice-writes the base-grid cells it covers, so finer
        writes naturally overwrite coarser fallbacks.

        When ``current_slice_coord`` is provided, uses a two-phase sweep:

        - **Phase 1 (background):** bricks whose ``slice_coord`` differs from
          ``current_slice_coord`` are written coarsest-to-finest as LOD
          placeholders visible while new-slice data is still loading.
        - **Phase 2 (foreground):** bricks whose ``slice_coord`` matches
          ``current_slice_coord`` are written coarsest-to-finest, overwriting
          wherever current-slice data is resident.

        The GPU upload is deferred until the next ``renderer.render()``
        call (see the pygfx texture update_range behavior).

        Parameters
        ----------
        tile_manager : TileManager3D
            Current tile manager holding the resident brick mapping.
        current_slice_coord : tuple of (axis_index, world_value) pairs or None
            Non-displayed axis positions for the current frame.  ``None``
            disables the two-phase sweep (single-phase, backward-compatible).
        """
        rebuild_lut(
            self._base_layout,
            tile_manager,
            self._n_levels,
            self.lut_data,
            self.lut_tex,
            self.brick_max_data,
            self.brick_max_tex,
            level_scale_vecs_data=self._level_scale_vecs_data,
            current_slice_coord=current_slice_coord,
            level_shapes=self._level_shapes,
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


def build_brick_max_texture(
    base_layout: BlockLayout3D,
) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate the per-brick max intensity texture (zeroed).

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest (level 1) resolution.

    Returns
    -------
    brick_max_data : np.ndarray
        Backing float32 array of shape ``(gD, gH, gW)``.
    brick_max_tex : gfx.Texture
        R32Float 3D texture.
    """
    gd, gh, gw = base_layout.grid_dims
    brick_max_data = np.zeros((gd, gh, gw), dtype=np.float32)
    brick_max_tex = gfx.Texture(brick_max_data, dim=3, format="r32float")
    return brick_max_data, brick_max_tex


def rebuild_lut(
    base_layout: BlockLayout3D,
    tile_manager: TileManager3D,
    n_levels: int,
    lut_data: np.ndarray,
    lut_tex: gfx.Texture,
    brick_max_data: np.ndarray,
    brick_max_tex: gfx.Texture,
    level_scale_vecs_data: list | None = None,
    current_slice_coord: tuple[tuple[int, int], ...] | None = None,
    level_shapes: list | None = None,
) -> None:
    """Rebuild the full LUT from the current tile manager state.

    This works coarsest-to-finest.  Each level writes its slot
    data into all base-grid cells it covers via a numpy slice assignment
    (one C-speed fill per resident brick).  Because finer levels are
    written last, they naturally overwrite the coarser fallback.

    When ``current_slice_coord`` is provided, uses a two-phase sweep:

    - **Phase 1 (background):** bricks whose ``slice_coord`` differs from
      ``current_slice_coord`` are written coarsest-to-finest.  These are
      old-slice bricks that serve as LOD placeholders while new data loads.
    - **Phase 2 (foreground):** bricks whose ``slice_coord`` matches
      ``current_slice_coord`` are written coarsest-to-finest, overwriting
      the background wherever current-slice data is resident.

    When ``current_slice_coord`` is ``None``, all bricks are written in a
    single coarsest-to-finest sweep (backward-compatible behaviour).

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest (level 1) resolution.
    tile_manager : TileManager3D
        Current tile manager with resident bricks.
    n_levels : int
        Total number of LOAD levels.
    lut_data : np.ndarray
        Backing uint8 array ``(gD, gH, gW, 4)`` to overwrite.
    lut_tex : gfx.Texture
        The LUT texture to schedule for GPU upload.
    brick_max_data : np.ndarray
        Backing float32 array ``(gD, gH, gW)`` to overwrite.
    brick_max_tex : gfx.Texture
        The brick-max texture to schedule for GPU upload.
    level_scale_vecs_data : list of ndarray, optional
        Per-level scale vectors in data order ``(sz, sy, sx)``.  Entry ``k``
        is the downscale factor of level ``k+1`` (1-indexed) relative to
        the finest level.  When ``None``, falls back to the uniform
        ``2^(level-1)`` assumption (correct for isotropic power-of-2
        pyramids only).
    current_slice_coord : tuple of (axis_index, selection) pairs or None
        The block-cache slice key of the current frame.  ``None``
        disables the two-phase sweep.
    level_shapes : list of tuple of int, optional
        Per-level voxel shapes in data order.  Gives each level's brick count
        for the cell -> brick rule; without it the count is inferred from the
        grid.
    """
    gd, gh, gw = base_layout.grid_dims

    lut_data[:] = 0  # Reset everything to out-of-bounds (level 0 = black).
    brick_max_data[:] = 0.0

    # The one cell -> brick rule, shared with the shaders through the
    # block-scales buffer.  See _cell_brick_rule.
    spans = level_cell_spans(n_levels, 3, level_scale_vecs_data)
    counts = level_brick_counts(
        spans, (gd, gh, gw), base_layout.block_size, level_shapes
    )

    def _write_bricks(by_level: dict[int, list]) -> None:
        """Write one group of bricks coarsest-to-finest into lut_data."""
        for level in range(n_levels, 0, -1):
            if level not in by_level:
                continue
            span_z, span_y, span_x = spans[level - 1]
            count_z, count_y, count_x = counts[level - 1]
            for key, slot in by_level[level]:
                sz, sy, sx = slot.grid_pos
                gz0, gz1 = cell_range(key.g0, span_z, count_z, gd)
                gy0, gy1 = cell_range(key.g1, span_y, count_y, gh)
                gx0, gx1 = cell_range(key.g2, span_x, count_x, gw)
                if gz0 >= gz1 or gy0 >= gy1 or gx0 >= gx1:
                    continue  # A brick that owns no cell.
                lut_data[gz0:gz1, gy0:gy1, gx0:gx1] = (sx, sy, sz, level)
                brick_max_data[gz0:gz1, gy0:gy1, gx0:gx1] = slot.brick_max

    if current_slice_coord is None:
        # Single-phase: all bricks treated as foreground.
        by_level: dict[int, list] = {}
        for key, slot in tile_manager.tilemap.items():
            if key.level > 0:
                by_level.setdefault(key.level, []).append((key, slot))
        _write_bricks(by_level)
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
        _write_bricks(bg_by_level)
        _write_bricks(fg_by_level)
        by_level = {**bg_by_level, **fg_by_level}

    # Log per-level brick counts resident in LUT (np.unique scan is deferred
    # behind the level check to avoid scanning the full array every batch).
    if _GPU_LOGGER.isEnabledFor(logging.INFO):
        lut_by_level = {level: len(bricks) for level, bricks in by_level.items()}
        lut_level_vals, lut_level_counts = np.unique(
            lut_data[:, :, :, 3], return_counts=True
        )
        lut_cells_by_level = {
            int(lv): int(cnt)
            for lv, cnt in zip(lut_level_vals, lut_level_counts)
            if lv > 0
        }
        _GPU_LOGGER.info(
            "rebuild_lut  resident_bricks_by_level=%s  lut_cells_by_level=%s",
            lut_by_level,
            lut_cells_by_level,
        )

    lut_tex.update_range((0, 0, 0), lut_tex.size)
    brick_max_tex.update_range((0, 0, 0), brick_max_tex.size)
