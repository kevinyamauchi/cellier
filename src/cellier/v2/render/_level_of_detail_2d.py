"""LOD level selection for tiled 2D image rendering.

The 2D LOD selector uses zoom level (pixels-per-world-unit) to choose
the coarsest level where each tile still has sufficient resolution.

For orthographic cameras, all tiles are at the same effective "distance",
so a single global LOD level is selected based on the zoom ratio.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from cellier.v2.render.block_cache._tile_manager_2d import BlockKey2D

if TYPE_CHECKING:
    from cellier.v2.render.lut_indirection._layout_2d import BlockLayout2D


def build_tile_grids_2d(
    base_layout: BlockLayout2D,
    n_levels: int,
) -> list[dict]:
    """Precompute static per-level coarse tile grids.  Called once at startup.

    For level k (1-indexed):
    - The coarse grid has dims ``ceil(gH / 2^(k-1))`` per axis.
    - Every coarse tile in that grid is enumerated and its world-space
      centre is computed.

    Parameters
    ----------
    base_layout : BlockLayout2D
        Layout of the finest (level 1) resolution.
    n_levels : int
        Total number of LOD levels.

    Returns
    -------
    grids : list[dict]
        One dict per level (index 0 = level 1).  Each dict contains:

        ``arr`` : ndarray, shape (M_k, 3), dtype int32
            ``[level, gy_c, gx_c]`` for every coarse tile.
        ``centres`` : ndarray, shape (M_k, 2), dtype float64
            World-space ``(x, y)`` centre of each coarse tile.
    """
    bs = base_layout.block_size
    gh, gw = base_layout.grid_dims

    grids = []
    for level in range(1, n_levels + 1):
        scale = 1 << (level - 1)
        cgh = (gh + scale - 1) // scale
        cgw = (gw + scale - 1) // scale

        gy_c, gx_c = np.meshgrid(
            np.arange(cgh, dtype=np.int32),
            np.arange(cgw, dtype=np.int32),
            indexing="ij",
        )
        gy_c = gy_c.ravel()
        gx_c = gx_c.ravel()

        lvl_col = np.full(len(gy_c), level, dtype=np.int32)
        arr = np.stack([lvl_col, gy_c, gx_c], axis=1)  # (M_k, 3)

        bw = float(bs * scale)
        centres = np.empty((len(gy_c), 2), dtype=np.float64)
        centres[:, 0] = (gx_c + 0.5) * bw  # x = W axis
        centres[:, 1] = (gy_c + 0.5) * bw  # y = H axis

        grids.append({"arr": arr, "centres": centres})

    return grids


def select_lod_2d(
    level_grids: list[dict],
    n_levels: int,
    viewport_width_px: float,
    world_width: float,
    lod_bias: float = 1.0,
    force_level: int | None = None,
) -> np.ndarray:
    """Select LOD level based on the camera zoom.

    Uses a single global LOD level for all tiles based on the current
    zoom ratio (standard mipmapping with multiplicative bias).

    Parameters
    ----------
    level_grids : list[dict]
        Precomputed output of ``build_tile_grids_2d``.
    n_levels : int
        Number of LOD levels.
    viewport_width_px : float
        Viewport width in logical pixels.
    world_width : float
        Visible world width in world units.
    lod_bias : float
        Multiplicative bias. 1.0 = neutral (default).
    force_level : int or None
        If set, bypass zoom selection and use this level.

    Returns
    -------
    arr : ndarray, shape (M, 3), dtype int32
        ``[level, gy_c, gx_c]`` rows for all selected tiles.
    """
    if force_level is not None:
        level = min(max(force_level, 1), n_levels)
        return level_grids[level - 1]["arr"].copy()

    if world_width <= 0 or viewport_width_px <= 0:
        return level_grids[0]["arr"].copy()

    # World units per screen pixel.
    screen_pixel_size = world_width / viewport_width_px

    # Apply multiplicative bias.
    biased = screen_pixel_size * max(lod_bias, 1e-6)

    # Continuous LOD level: 1.0 at 1:1, 2.0 when each screen px = 2 data px.
    ideal = 1.0 + math.log2(max(biased, 1e-6))

    # Round to nearest integer level, clamped to valid range.
    selected_level = max(1, min(n_levels, round(ideal)))

    return level_grids[selected_level - 1]["arr"].copy()


def sort_tiles_by_distance_2d(
    arr: np.ndarray,
    camera_pos: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """Sort tiles nearest-to-camera-center first.

    Parameters
    ----------
    arr : ndarray, shape (M, 3), dtype int32
        ``[level, gy_c, gx_c]`` rows.
    camera_pos : ndarray, shape (3,)
        Camera world-space position ``(x, y, z)``.
    block_size : int
        Tile side length in pixels.

    Returns
    -------
    sorted_arr : ndarray
        Same shape, sorted by Euclidean distance from camera center.
    """
    if len(arr) == 0:
        return arr

    levels = arr[:, 0]
    gy = arr[:, 1].astype(np.float64)
    gx = arr[:, 2].astype(np.float64)

    scale = (2.0 ** (levels - 1)).astype(np.float64)
    bw = block_size * scale

    cx = (gx + 0.5) * bw
    cy = (gy + 0.5) * bw

    # Camera x, y (ignoring z for 2D).
    vx, vy = float(camera_pos[0]), float(camera_pos[1])
    dx = cx - vx
    dy = cy - vy
    dist_sq = dx * dx + dy * dy

    order = np.argsort(dist_sq)
    return arr[order]


def viewport_cull_2d(
    required: dict[BlockKey2D, int],
    block_size: int,
    view_min: np.ndarray,
    view_max: np.ndarray,
) -> tuple[dict[BlockKey2D, int], int]:
    """Remove tiles that lie entirely outside the viewport.

    Parameters
    ----------
    required : dict[BlockKey2D, int]
        Tile key -> level mapping (order-preserving).
    block_size : int
        Tile side length in data pixels at finest level.
    view_min : ndarray, shape (2,)
        Viewport AABB minimum ``(x, y)`` in world space.
    view_max : ndarray, shape (2,)
        Viewport AABB maximum ``(x, y)`` in world space.

    Returns
    -------
    culled : dict[BlockKey2D, int]
        Subset of ``required`` that overlaps the viewport.
    n_culled : int
        Number of tiles removed.
    """
    if not required:
        return required, 0

    keys = list(required.keys())
    n = len(keys)

    levels = np.array([k.level for k in keys], dtype=np.int32)
    gy = np.array([k.gy for k in keys], dtype=np.float64)
    gx = np.array([k.gx for k in keys], dtype=np.float64)

    scale = (2.0 ** (levels - 1)).astype(np.float64)
    bw = block_size * scale

    tile_min_x = gx * bw
    tile_min_y = gy * bw
    tile_max_x = (gx + 1.0) * bw
    tile_max_y = (gy + 1.0) * bw

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


def arr_to_block_keys_2d(arr: np.ndarray) -> dict[BlockKey2D, int]:
    """Convert array rows to a BlockKey2D dict.

    Parameters
    ----------
    arr : ndarray
        Array of shape ``(M, 3)`` with columns ``(level, gy, gx)``.

    Returns
    -------
    required : dict[BlockKey2D, int]
        ``{BlockKey2D: level}`` preserving row order.
    """
    required: dict[BlockKey2D, int] = {}
    for row in arr:
        level = int(row[0])
        key = BlockKey2D(level=level, gy=int(row[1]), gx=int(row[2]))
        required[key] = level
    return required
