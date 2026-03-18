"""LOD level selection for bricked volume rendering.

Phase 3: replaces the triple-nested Python loop with a single vectorised
numpy pass.  The public ``select_levels`` API is identical to Phase 2.

Speedup target: < 5 ms for a 32³-grid volume (was ~80 ms in Python loops).
"""

from __future__ import annotations

import itertools

import numpy as np

from block_volume.layout import BlockLayout
from block_volume.tile_manager import BrickKey

# Pre-computed (8, 3) offset table for AABB corner construction.
# Reused by the vectorised frustum helper in frustum.py.
CORNER_OFFSETS = np.array(
    list(itertools.product([0.0, 1.0], repeat=3)), dtype=np.float64
)  # (8, 3)


def select_levels(
    base_layout: BlockLayout,
    n_levels: int,
    camera_pos: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[BrickKey, int]:
    """Assign a desired LOD level to every brick in the base grid.

    Vectorised implementation: builds the full base-grid index array
    once in numpy, computes distances and LOD assignments without any
    Python loop over individual cells.  A single O(N) loop at the end
    converts the index arrays to ``BrickKey`` objects.

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest (level 1) resolution.
    n_levels : int
        Total number of LOD levels available (e.g. 3).
    camera_pos : np.ndarray
        Camera position in world space ``(x, y, z)``.
    thresholds : list[float] or None
        Distance cutoffs for level transitions.  ``thresholds[i]`` is
        the distance beyond which level ``i + 2`` is selected instead
        of ``i + 1``.  Length must be ``n_levels - 1``.  If ``None``,
        defaults are computed from the volume diagonal.

    Returns
    -------
    required : dict[BrickKey, int]
        ``{BrickKey(level, gz, gy, gx): level}`` for every brick.
    """
    gd, gh, gw = base_layout.grid_dims
    bs = base_layout.block_size

    if thresholds is None:
        diag = np.sqrt(sum(s**2 for s in base_layout.volume_shape))
        thresholds = [diag * (i + 1) for i in range(n_levels - 1)]

    # ── Build base-grid index arrays ──────────────────────────────────
    # gz_arr, gy_arr, gx_arr: each shape (gd*gh*gw,)  [i.e. (N,)]
    gz_arr, gy_arr, gx_arr = np.meshgrid(
        np.arange(gd, dtype=np.int32),
        np.arange(gh, dtype=np.int32),
        np.arange(gw, dtype=np.int32),
        indexing="ij",
    )
    gz_arr = gz_arr.ravel()
    gy_arr = gy_arr.ravel()
    gx_arr = gx_arr.ravel()

    # ── Brick centres in world space (x=W, y=H, z=D) — shape (N, 3) ──
    centres = np.stack(
        [
            (gx_arr + 0.5) * bs,
            (gy_arr + 0.5) * bs,
            (gz_arr + 0.5) * bs,
        ],
        axis=1,
        dtype=np.float64,
    )  # (N, 3)

    # ── Camera distances — shape (N,) ─────────────────────────────────
    cam = np.asarray(camera_pos, dtype=np.float64)
    distances = np.linalg.norm(centres - cam, axis=1)  # (N,)

    # ── LOD assignment — vectorised threshold comparison ──────────────
    # Start everyone at level 1; bump up for each exceeded threshold.
    levels_assigned = np.ones(len(centres), dtype=np.int32)
    for k, thr in enumerate(thresholds):
        levels_assigned[distances > thr] = k + 2
    levels_assigned = np.clip(levels_assigned, 1, n_levels)

    # ── Build BrickKey dict (dedup coarse keys via dict assignment) ────
    required: dict[BrickKey, int] = {}
    for i in range(len(gz_arr)):
        level = int(levels_assigned[i])
        scale = 2 ** (level - 1)
        key = BrickKey(
            level=level,
            gz=int(gz_arr[i]) // scale,
            gy=int(gy_arr[i]) // scale,
            gx=int(gx_arr[i]) // scale,
        )
        required[key] = level

    return required


def sort_by_distance(
    required: dict[BrickKey, int],
    camera_pos: np.ndarray,
    block_size: int,
) -> dict[BrickKey, int]:
    """Return ``required`` sorted nearest-to-camera first.

    Vectorised: converts all brick-centre coordinates to a numpy array,
    computes distances in one shot, then does a single argsort.

    Parameters
    ----------
    required : dict[BrickKey, int]
        Unsorted brick dict from ``select_levels``.
    camera_pos : np.ndarray
        Camera world-space position ``(x, y, z)``.
    block_size : int
        Level-1 brick side length in voxels.

    Returns
    -------
    sorted_required : dict[BrickKey, int]
        Same bricks, insertion-order sorted nearest-first.
    """
    if not required:
        return required

    keys = list(required.keys())
    bs = block_size

    # Brick centres in world space — vectorised over all keys.
    centres = np.empty((len(keys), 3), dtype=np.float64)
    for i, key in enumerate(keys):
        scale = 2 ** (key.level - 1)
        centres[i, 0] = (key.gx + 0.5) * bs * scale   # x
        centres[i, 1] = (key.gy + 0.5) * bs * scale   # y
        centres[i, 2] = (key.gz + 0.5) * bs * scale   # z

    cam = np.asarray(camera_pos, dtype=np.float64)
    distances = np.linalg.norm(centres - cam, axis=1)
    order = np.argsort(distances)

    return {keys[i]: required[keys[i]] for i in order}
