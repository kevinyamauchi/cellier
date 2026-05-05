"""LOD level selection for bricked volume rendering.

Two-level speedup over a naive enumeration approach.

Optimisation 1 — enumerate coarse grids directly, no deduplication
-------------------------------------------------------------------
For each LOD level k enumerate *only that level's coarse grid*
(e.g. 512 bricks for L3 on a 32³ base grid), filter by its distance
band, and concatenate.  No two levels can produce the same coarse
brick, so no deduplication is ever needed.

Optimisation 2 — precomputed, cached grid data
-----------------------------------------------
The coarse grid index arrays and world-space brick centres depend only
on ``block_size`` and ``grid_dims``, which are fixed at startup.  They
are computed once by ``build_level_grids()`` and cached.  The hot path
(``select_levels_from_cache``) does no allocation — only distance
computation and boolean masking on the cached arrays.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from cellier.v2.render.block_cache import BlockKey3D

if TYPE_CHECKING:
    from cellier.v2.render.lut_indirection import BlockLayout3D

# Pre-computed (8, 3) offset table for AABB corner construction.
# Reused by the vectorised frustum helper in _frustum.py.
CORNER_OFFSETS = np.array(
    list(itertools.product([0.0, 1.0], repeat=3)), dtype=np.float64
)  # (8, 3)


# ---------------------------------------------------------------------------
# Startup: build per-level coarse grid cache
# ---------------------------------------------------------------------------


def build_level_grids(
    base_layout: BlockLayout3D,
    n_levels: int,
    level_shapes: list[tuple[int, ...]] | None = None,
    scale_vecs_shader: list[np.ndarray] | None = None,
    translation_vecs_shader: list[np.ndarray] | None = None,
) -> list[dict]:
    """Precompute static per-level coarse grid arrays.  Called once at startup.

    For level k (1-indexed):

    - The coarse grid dims are computed from ``level_shapes`` (actual
      voxel counts per level) instead of a power-of-2 assumption.
    - World-space brick centres incorporate per-axis scale and
      translation from the level transforms.

    Parameters
    ----------
    base_layout : BlockLayout3D
        Layout of the finest (level 1) resolution.
    n_levels : int
        Total number of LOD levels.
    level_shapes : list[tuple[int, ...]] or None
        ``(D, H, W)`` shape per level (data order).  When ``None``,
        falls back to power-of-2 derivation from ``base_layout``.
    scale_vecs_shader : list[np.ndarray] or None
        ``(3,)`` per level in shader order ``(x=W, y=H, z=D)``.
    translation_vecs_shader : list[np.ndarray] or None
        ``(3,)`` per level in shader order ``(x=W, y=H, z=D)``.

    Returns
    -------
    grids : list[dict]
        One dict per level (index 0 = level 1).  Each dict contains:

        ``arr`` : ndarray, shape (M_k, 4), dtype int32
            ``[level, gz_c, gy_c, gx_c]`` for every coarse brick.
        ``centres`` : ndarray, shape (M_k, 3), dtype float64
            World-space ``(x, y, z)`` centre of each coarse brick.
    """
    bs = base_layout.block_size
    gd, gh, gw = base_layout.grid_dims

    grids = []
    for level in range(1, n_levels + 1):
        k = level - 1  # 0-indexed

        # Coarse grid dimensions from actual level shapes.
        if level_shapes is not None:
            # data order: (D=axis0, H=axis1, W=axis2)
            d_k, h_k, w_k = level_shapes[k]
            cgd = (d_k + bs - 1) // bs  # z grid dim
            cgh = (h_k + bs - 1) // bs  # y grid dim
            cgw = (w_k + bs - 1) // bs  # x grid dim
        else:
            scale = 1 << k
            cgd = (gd + scale - 1) // scale
            cgh = (gh + scale - 1) // scale
            cgw = (gw + scale - 1) // scale

        gz_c, gy_c, gx_c = np.meshgrid(
            np.arange(cgd, dtype=np.int32),
            np.arange(cgh, dtype=np.int32),
            np.arange(cgw, dtype=np.int32),
            indexing="ij",
        )
        gz_c = gz_c.ravel()
        gy_c = gy_c.ravel()
        gx_c = gx_c.ravel()

        lvl_col = np.full(len(gz_c), level, dtype=np.int32)
        arr = np.stack([lvl_col, gz_c, gy_c, gx_c], axis=1)  # (M_k, 4)

        centres = np.empty((len(gz_c), 3), dtype=np.float64)
        if scale_vecs_shader is not None and translation_vecs_shader is not None:
            sv = scale_vecs_shader[k]  # (sx, sy, sz) = (W, H, D)
            tv = translation_vecs_shader[k]  # (tx, ty, tz)
            bw_x = float(bs * sv[0])  # W axis
            bw_y = float(bs * sv[1])  # H axis
            bw_z = float(bs * sv[2])  # D axis
            centres[:, 0] = (gx_c + 0.5) * bw_x + tv[0]  # x = W
            centres[:, 1] = (gy_c + 0.5) * bw_y + tv[1]  # y = H
            centres[:, 2] = (gz_c + 0.5) * bw_z + tv[2]  # z = D
        else:
            scale = 1 << k
            bw = float(bs * scale)
            centres[:, 0] = (gx_c + 0.5) * bw  # x = W axis
            centres[:, 1] = (gy_c + 0.5) * bw  # y = H axis
            centres[:, 2] = (gz_c + 0.5) * bw  # z = D axis

        grids.append({"arr": arr, "centres": centres})

    return grids


# ---------------------------------------------------------------------------
# Hot path: per-update LOD selection using the precomputed cache
# ---------------------------------------------------------------------------


def select_levels_from_cache(
    level_grids: list[dict],
    n_levels: int,
    camera_pos: np.ndarray,
    thresholds: list[float] | None = None,
    base_layout: BlockLayout3D | None = None,
) -> np.ndarray:
    """Select LOD levels using precomputed coarse grid data.

    For each level k, compute distances from the camera to the
    precomputed coarse brick centres, apply the distance band that
    belongs to level k, and collect survivors.  No new coordinate
    arrays are allocated; only distance computation and boolean masking
    on the cached arrays.

    Because the bands partition the distance axis, no brick can appear
    in more than one level — no deduplication needed.

    Parameters
    ----------
    level_grids : list[dict]
        Precomputed output of ``build_level_grids``.
    n_levels : int
        Number of LOD levels.
    camera_pos : np.ndarray
        Camera world-space position ``(x, y, z)``.
    thresholds : list[float] or None
        LOD cutoff distances.  ``thresholds[i]`` is the distance beyond
        which level ``i+2`` is preferred over level ``i+1``.
        If None, all bricks are assigned level 1.
    base_layout : BlockLayout3D or None
        Only used to compute default thresholds when ``thresholds`` is
        None.

    Returns
    -------
    arr : ndarray, shape (M, 4), dtype int32
        ``[level, gz_c, gy_c, gx_c]`` rows for all selected bricks,
        not yet sorted — call ``sort_arr_by_distance`` next.
    """
    cam = np.asarray(camera_pos, dtype=np.float64)

    if thresholds is None:
        if base_layout is not None:
            diag = np.sqrt(sum(s**2 for s in base_layout.volume_shape))
            thresholds = [diag * (i + 1) for i in range(n_levels - 1)]
        else:
            thresholds = []

    parts: list[np.ndarray] = []

    for level in range(1, n_levels + 1):
        grid = level_grids[level - 1]
        centres = grid["centres"]  # (M_k, 3) — precomputed, no alloc
        arr_k = grid["arr"]  # (M_k, 4)

        diff = centres - cam
        dist = np.sqrt((diff * diff).sum(axis=1))

        if level == 1:
            mask = (
                dist < thresholds[0] if thresholds else np.ones(len(dist), dtype=bool)
            )
        elif level == n_levels:
            mask = dist >= thresholds[level - 2]
        else:
            mask = (dist >= thresholds[level - 2]) & (dist < thresholds[level - 1])

        if mask.any():
            parts.append(arr_k[mask])

    if not parts:
        return np.empty((0, 4), dtype=np.int32)
    return np.concatenate(parts, axis=0)


def select_levels_arr_forced(
    base_layout: BlockLayout3D,
    force_level: int,
    level_grids: list[dict] | None = None,
) -> np.ndarray:
    """Return the full coarse grid for a forced single LOD level.

    Uses the precomputed cache when available; falls back to computing
    from scratch otherwise.

    Parameters
    ----------
    base_layout : BlockLayout3D
        The block layout of the finest (level 1) resolution.
    force_level : int
        1-indexed level to force.
    level_grids : list[dict] or None
        If provided, returns ``level_grids[force_level - 1]["arr"]``
        directly (zero-copy view).
    """
    if level_grids is not None:
        return level_grids[force_level - 1]["arr"]

    gd, gh, gw = base_layout.grid_dims
    scale = 1 << (force_level - 1)
    cgd = (gd + scale - 1) // scale
    cgh = (gh + scale - 1) // scale
    cgw = (gw + scale - 1) // scale

    gz_c, gy_c, gx_c = np.meshgrid(
        np.arange(cgd, dtype=np.int32),
        np.arange(cgh, dtype=np.int32),
        np.arange(cgw, dtype=np.int32),
        indexing="ij",
    )
    gz_c = gz_c.ravel()
    gy_c = gy_c.ravel()
    gx_c = gx_c.ravel()
    lvl = np.full(len(gz_c), force_level, dtype=np.int32)
    return np.stack([lvl, gz_c, gy_c, gx_c], axis=1)


# ---------------------------------------------------------------------------
# Sort
# ---------------------------------------------------------------------------


def sort_arr_by_distance(
    arr: np.ndarray,
    camera_pos: np.ndarray,
    block_size: int,
    scale_vecs_shader: np.ndarray | list[np.ndarray] | None = None,
    translation_vecs_shader: np.ndarray | list[np.ndarray] | None = None,
) -> np.ndarray:
    """Sort brick rows nearest-to-camera first.

    Brick centres are computed in world XYZ (shader) space using the
    per-level scale and translation vectors so that anisotropic datasets
    sort correctly.  ``camera_pos`` must be in the same world XYZ space.

    Falls back to the isotropic power-of-2 approximation when
    ``scale_vecs_shader`` / ``translation_vecs_shader`` are not supplied,
    but that path does not handle anisotropic voxel spacing correctly.

    Parameters
    ----------
    arr : ndarray, shape (M, 4), dtype int32
        The array of bricks to sort — columns ``[level, gz, gy, gx]``.
    camera_pos : np.ndarray
        Camera position in world XYZ (shader) space, shape ``(3,)``.
    block_size : int
        Brick edge length in finest-level voxels.
    scale_vecs_shader : list of ndarray, optional
        Per-level scale vectors in shader order ``(sx, sy, sz)``.
        Index 0 = level 1 (finest).
    translation_vecs_shader : list of ndarray, optional
        Per-level translation vectors in shader order ``(tx, ty, tz)``.
        Index 0 = level 1 (finest).

    Returns
    -------
    sorted_arr : ndarray, shape (M, 4)
    """
    if len(arr) == 0:
        return arr

    cam = np.asarray(camera_pos, dtype=np.float64)
    levels = arr[:, 0]
    gz_c = arr[:, 1]
    gy_c = arr[:, 2]
    gx_c = arr[:, 3]

    cx = np.empty(len(arr), dtype=np.float64)
    cy = np.empty(len(arr), dtype=np.float64)
    cz = np.empty(len(arr), dtype=np.float64)

    if scale_vecs_shader is not None and translation_vecs_shader is not None:
        # World-space centres using actual per-level physical scale/translation.
        # Mirrors the formula in build_level_grids so centres match exactly.
        for level in np.unique(levels):
            mask = levels == level
            k = int(level) - 1
            sv = np.asarray(scale_vecs_shader[k], dtype=np.float64)
            tv = np.asarray(translation_vecs_shader[k], dtype=np.float64)
            bw_x = block_size * sv[0]
            bw_y = block_size * sv[1]
            bw_z = block_size * sv[2]
            cx[mask] = (gx_c[mask].astype(np.float64) + 0.5) * bw_x + tv[0]
            cy[mask] = (gy_c[mask].astype(np.float64) + 0.5) * bw_y + tv[1]
            cz[mask] = (gz_c[mask].astype(np.float64) + 0.5) * bw_z + tv[2]
    else:
        # Fallback: isotropic power-of-2 approximation (ignores physical scale).
        scales = np.left_shift(1, (levels - 1)).astype(np.float64)
        bw = float(block_size) * scales
        cx = (gx_c.astype(np.float64) + 0.5) * bw
        cy = (gy_c.astype(np.float64) + 0.5) * bw
        cz = (gz_c.astype(np.float64) + 0.5) * bw

    distances = np.sqrt((cx - cam[0]) ** 2 + (cy - cam[1]) ** 2 + (cz - cam[2]) ** 2)
    order = np.argsort(distances, kind="stable")
    return arr[order]


# ---------------------------------------------------------------------------
# Convert to BrickKey dict
# ---------------------------------------------------------------------------


def arr_to_brick_keys(arr: np.ndarray) -> dict[BlockKey3D, int]:
    """Convert a brick array to ``dict[BlockKey3D, int]``.

    Parameters
    ----------
    arr : ndarray, shape (M, 4), dtype int32
        Columns: ``[level, gz_c, gy_c, gx_c]``.

    Returns
    -------
    required : dict[BlockKey3D, int]
        ``{BlockKey3D: level}`` preserving row order.
    """
    required: dict[BlockKey3D, int] = {}
    for row in arr:
        level = int(row[0])
        key = BlockKey3D(level=level, g0=int(row[1]), g1=int(row[2]), g2=int(row[3]))
        required[key] = level
    return required
