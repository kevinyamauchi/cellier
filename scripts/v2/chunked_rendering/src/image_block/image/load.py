"""LOAD level selection for tiled 2D image rendering.

The 2D LOAD selector uses zoom level (pixels-per-world-unit) to choose
the coarsest level where each tile still has sufficient resolution.

Optimisations match the 3D Phase 3 approach:
- Per-level coarse grids are precomputed at startup.
- The hot path does no allocation, only masking on cached arrays.
"""

from __future__ import annotations

import math

import numpy as np

from image_block.core.block_key import BlockKey
from image_block.image.layout import BlockLayout2D


# ---------------------------------------------------------------------------
# Startup: build per-level coarse grid cache
# ---------------------------------------------------------------------------


def build_tile_grids(
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
        Total number of LOAD levels.

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


# ---------------------------------------------------------------------------
# Hot path: per-Update LOAD selection
# ---------------------------------------------------------------------------


def select_load_2d(
    level_grids: list[dict],
    n_levels: int,
    camera_info: dict,
    load_bias: float = 1.0,
    force_level: int | None = None,
) -> np.ndarray:
    """Select LOAD levels based on the camera zoom.

    The 2D selector uses a single global LOAD level for all tiles based
    on the current zoom ratio.  This is the appropriate strategy for an
    orthographic 2D view where all tiles are at the same "distance".

    Algorithm (standard mipmapping with multiplicative bias)
    --------------------------------------------------------
    Compute the continuous LOAD level from the ratio of screen resolution
    to data resolution:

        screen_pixel_size = world_width / viewport_width   (world units per screen pixel)

    At level 1 (finest), each data pixel = 1 world unit.
    At level k, each data pixel = 2^(k-1) world units.

    The ideal continuous level is:

        ideal_level = 1 + log2(screen_pixel_size * load_bias)

    This means:
        screen_pixel_size = 1.0  -> ideal = 1.0  (1:1 mapping, use L1)
        screen_pixel_size = 2.0  -> ideal = 2.0  (each screen px = 2 data px, use L2)
        screen_pixel_size = 4.0  -> ideal = 3.0  (each screen px = 4 data px, use L3)

    We round to the nearest integer level and clamp to [1, n_levels].

    ``load_bias`` is multiplicative, matching the 3D volume viewer:
        - load_bias = 1.0  standard mipmapping (default, neutral)
        - load_bias > 1.0  prefer coarser levels (acts as if screen pixels
          were larger than they really are)
        - load_bias < 1.0  prefer finer levels (acts as if screen pixels
          were smaller)

    Doubling the bias shifts every transition by one full LOAD level.

    Parameters
    ----------
    level_grids : list[dict]
        Precomputed output of ``build_tile_grids``.
    n_levels : int
        Number of LOAD levels.
    camera_info : dict
        Must contain ``viewport_width`` (pixels) and ``world_width`` (float).
    load_bias : float
        Multiplicative bias. 1.0 = neutral (default).
        Same semantics as the 3D volume viewer.
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

    viewport_width = camera_info["viewport_width"]
    world_width = camera_info["world_width"]
    if world_width <= 0 or viewport_width <= 0:
        return level_grids[0]["arr"].copy()

    # World units per screen pixel.
    screen_pixel_size = world_width / viewport_width

    # Apply multiplicative bias: load_bias > 1 makes the screen pixel
    # appear larger, pushing toward coarser levels.
    biased = screen_pixel_size * max(load_bias, 1e-6)

    # Continuous LOAD level: 1.0 at 1:1, 2.0 when each screen px = 2 data px.
    ideal = 1.0 + math.log2(max(biased, 1e-6))

    # Round to nearest integer level, clamped to valid range.
    selected_level = max(1, min(n_levels, round(ideal)))

    return level_grids[selected_level - 1]["arr"].copy()


def sort_tiles_by_distance(
    arr: np.ndarray,
    camera_info: dict,
    block_size: int,
) -> np.ndarray:
    """Sort tiles nearest-to-camera-center first.

    Parameters
    ----------
    arr : ndarray, shape (M, 3), dtype int32
        ``[level, gy_c, gx_c]`` rows.
    camera_info : dict
        Must contain ``view_center`` as ``(x, y)`` in world space.
    block_size : int
        Tile side length in pixels.

    Returns
    -------
    sorted_arr : ndarray
        Same shape, sorted by Euclidean distance from view centre.
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

    vx, vy = camera_info["view_center"]
    dx = cx - vx
    dy = cy - vy
    dist_sq = dx * dx + dy * dy

    order = np.argsort(dist_sq)
    return arr[order]


def arr_to_tile_keys(arr: np.ndarray) -> dict[BlockKey, int]:
    """Convert array rows to a BlockKey dict.

    Parameters
    ----------
    arr : ndarray, shape (M, 3), dtype int32

    Returns
    -------
    required : dict[BlockKey, int]
        ``{BlockKey: level}`` preserving row order.
    """
    required: dict[BlockKey, int] = {}
    for row in arr:
        level = int(row[0])
        key = BlockKey(level=level, gz=0, gy=int(row[1]), gx=int(row[2]))
        required[key] = level
    return required


if __name__ == "__main__":
    layout = BlockLayout2D.from_shape((1024, 1024), block_size=32, overlap=1)
    grids = build_tile_grids(layout, n_levels=3)
    for i, g in enumerate(grids):
        print(f"  Level {i+1}: {len(g['arr'])} tiles")

    camera_info = {
        "viewport_width": 800,
        "world_width": 1024.0,
        "view_center": (512.0, 512.0),
    }
    arr = select_load_2d(grids, 3, camera_info, load_bias=1.0)
    print(f"Selected {len(arr)} tiles at level {arr[0, 0] if len(arr) else '?'}")

    sorted_arr = sort_tiles_by_distance(arr, camera_info, 32)
    keys = arr_to_tile_keys(sorted_arr)
    print(f"Sorted: {len(keys)} unique tile keys")
    print("LOAD OK")
