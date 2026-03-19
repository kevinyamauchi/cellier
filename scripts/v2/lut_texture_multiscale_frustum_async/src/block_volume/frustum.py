"""Frustum geometry helpers and AABB visibility test.

Phase 3: ``bricks_in_frustum`` now uses the pre-computed ``CORNER_OFFSETS``
from ``lod.py`` and the vectorised corner construction from plan section 2.2,
eliminating the per-brick Python loop.

Functions
---------
get_frustum_corners_world   Extract (2, 4, 3) corners from a pygfx camera.
get_view_direction_world     Return the normalised forward vector.
get_camera_position_world    Return the camera world position.
frustum_planes_from_corners  Convert corners to 6 half-space planes (6, 4).
compute_brick_aabb_corners   Convert a BrickKey to 8 world-space AABB corners.
bricks_in_frustum            Conservative AABB test; returns (visible_keys, timings).

Axis-order note
---------------
BrickKey stores (gz, gy, gx) in DHW / numpy order, but world coordinates
are (x, y, z).  The conversion is:

    world_x = gx * block_world
    world_y = gy * block_world
    world_z = gz * block_world
"""

from __future__ import annotations

import time

import numpy as np
import pygfx as gfx

from block_volume.layout import BlockLayout
from block_volume.lod import CORNER_OFFSETS
from block_volume.tile_manager import BrickKey


# ---------------------------------------------------------------------------
# Camera extraction
# ---------------------------------------------------------------------------


def get_frustum_corners_world(camera: gfx.PerspectiveCamera) -> np.ndarray:
    """Extract world-space frustum corners from a pygfx camera.

    Returns
    -------
    corners : ndarray, shape (2, 4, 3)
        ``corners[0]`` = near plane, ``corners[1]`` = far plane.
        Within each plane: (left-bottom, right-bottom, right-top, left-top).
    """
    return np.asarray(camera.frustum, dtype=np.float64)


def get_view_direction_world(camera: gfx.PerspectiveCamera) -> np.ndarray:
    """Return the normalised view direction in world space as ``(3,)``."""
    mat = np.asarray(camera.world.matrix, dtype=np.float64)
    forward = -mat[:3, 2]
    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        return np.array([0.0, 0.0, -1.0])
    return forward / norm


def get_camera_position_world(camera: gfx.PerspectiveCamera) -> np.ndarray:
    """Return the camera world position as ``(3,)``."""
    return np.array(camera.world.position, dtype=np.float64)


# ---------------------------------------------------------------------------
# Plane computation helpers
# ---------------------------------------------------------------------------


def _compute_plane_parameters(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    """Return (4,) plane coefficients [a, b, c, d].

    Normal points toward the side where the three points wind counter-clockwise.
    A point ``p`` is *inside* when ``dot(p, [a,b,c]) + d >= 0``.
    """
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 0.0])
    normal = normal / norm
    d = -np.dot(normal, p0)
    return np.array([normal[0], normal[1], normal[2], d])


def frustum_planes_from_corners(corners: np.ndarray) -> np.ndarray:
    """Compute the 6 frustum half-space planes from frustum corners.

    Parameters
    ----------
    corners : ndarray, shape (2, 4, 3)
        ``corners[0]`` = near plane, ``corners[1]`` = far plane.
        Within each plane: (left-bottom, right-bottom, right-top, left-top).

    Returns
    -------
    planes : ndarray, shape (6, 4)
        Plane coefficients ordered: near, far, left, right, top, bottom.
        A point ``p`` is *inside* the frustum when
        ``dot(plane[:3], p) + plane[3] >= 0`` for all 6 planes.
        Normals all point *inward*.
    """
    n = corners[0]  # near: lb, rb, rt, lt
    f = corners[1]  # far:  lb, rb, rt, lt

    planes = np.empty((6, 4), dtype=np.float64)

    # Near  — inward normal points from near toward far
    planes[0] = _compute_plane_parameters(n[0], n[2], n[1])
    # Far   — inward normal points from far toward near
    planes[1] = _compute_plane_parameters(f[0], f[1], f[2])
    # Left  — n_lt, n_lb, f_lb
    planes[2] = _compute_plane_parameters(n[3], n[0], f[0])
    # Right — n_rb, n_rt, f_rt
    planes[3] = _compute_plane_parameters(n[1], n[2], f[2])
    # Top   — n_rt, n_lt, f_lt
    planes[4] = _compute_plane_parameters(n[2], n[3], f[3])
    # Bottom — n_lb, n_rb, f_rb
    planes[5] = _compute_plane_parameters(n[0], n[1], f[1])

    return planes


# ---------------------------------------------------------------------------
# Wireframe helper
# ---------------------------------------------------------------------------


def _frustum_edges(corners: np.ndarray) -> np.ndarray:
    """Return the 12 frustum edges as (start, end) point pairs."""
    edge_indices = [
        # Near-plane ring
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((0, 3), (0, 0)),
        # Far-plane ring
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 2), (1, 3)),
        ((1, 3), (1, 0)),
        # Connecting edges
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((0, 2), (1, 2)),
        ((0, 3), (1, 3)),
    ]
    return np.array(
        [[corners[a], corners[b]] for (a, b) in edge_indices],
        dtype=np.float64,
    )


def make_frustum_wireframe(corners: np.ndarray, color: str = "#00cc44") -> gfx.Line:
    """Build a frustum wireframe as a ``gfx.Line`` with segment material."""
    edges = _frustum_edges(corners)  # (12, 2, 3)
    positions = edges.reshape(-1, 3).astype(np.float32)
    geometry = gfx.Geometry(positions=positions)
    material = gfx.LineSegmentMaterial(color=color, thickness=1.5)
    return gfx.Line(geometry, material)


# ---------------------------------------------------------------------------
# AABB helpers
# ---------------------------------------------------------------------------


def compute_brick_aabb_corners(
    brick_key: BrickKey,
    layout: BlockLayout,
    block_size: int,
) -> np.ndarray:
    """Return the 8 world-space AABB corners for a brick.

    Parameters
    ----------
    brick_key : BrickKey
        Brick identifier.
    layout : BlockLayout
        Layout of the *finest* (level 1) resolution (unused beyond type check).
    block_size : int
        Brick side length in voxels at level 1.

    Returns
    -------
    corners : ndarray, shape (8, 3)
    """
    scale = 2 ** (brick_key.level - 1)
    block_world = float(block_size * scale)
    min_corner = np.array(
        [brick_key.gx * block_world, brick_key.gy * block_world, brick_key.gz * block_world],
        dtype=np.float64,
    )
    return min_corner + CORNER_OFFSETS * block_world  # (8, 3)


# ---------------------------------------------------------------------------
# Frustum culling — vectorised AABB corner construction (Phase 3 plan §2.2)
# ---------------------------------------------------------------------------


def bricks_in_frustum(
    brick_keys: set[BrickKey] | dict[BrickKey, int],
    layout: BlockLayout,
    block_size: int,
    frustum_planes: np.ndarray,
) -> tuple[dict[BrickKey, int], dict]:
    """Conservative AABB frustum test over a set of brick keys.

    Phase 3 improvement: the ``(N, 8, 3)`` corners array is built with
    a single numpy broadcasting operation instead of a per-brick loop.

    A brick is *visible* if, for every frustum plane, at least one of its
    8 AABB corners has a non-negative signed distance (is inside or on the
    plane).

    Parameters
    ----------
    brick_keys : set[BrickKey] or dict[BrickKey, int]
        Candidate bricks.  If a dict, the values are preserved.
    layout : BlockLayout
        Finest-level layout.
    block_size : int
        Level-1 brick side length in voxels.
    frustum_planes : ndarray, shape (6, 4)
        Inward-pointing half-space planes.

    Returns
    -------
    visible : dict[BrickKey, int]
        Subset that passes the frustum test.
    timings : dict
        Wall-clock timings in milliseconds.
    """
    if isinstance(brick_keys, dict):
        keys_list = list(brick_keys.keys())
        values = brick_keys
    else:
        keys_list = list(brick_keys)
        values = {k: 0 for k in keys_list}

    n = len(keys_list)
    if n == 0:
        return {}, {"build_corners_ms": 0.0, "einsum_ms": 0.0, "mask_ms": 0.0}

    # ── 1. Vectorised (N, 8, 3) AABB corner array ─────────────────────
    # brick_mins: (N, 3) — world-space min corner for each brick.
    t0 = time.perf_counter()
    brick_mins = np.empty((n, 3), dtype=np.float64)
    for i, key in enumerate(keys_list):
        scale = 2 ** (key.level - 1)
        bw = float(block_size * scale)
        brick_mins[i, 0] = key.gx * bw   # x
        brick_mins[i, 1] = key.gy * bw   # y
        brick_mins[i, 2] = key.gz * bw   # z

    # Compute world-size per brick (varies by level).
    block_worlds = np.empty(n, dtype=np.float64)
    for i, key in enumerate(keys_list):
        block_worlds[i] = float(block_size * (2 ** (key.level - 1)))

    # Broadcasting: (N, 1, 3) + (1, 8, 3) * (N, 1, 1) → (N, 8, 3)
    all_corners = (
        brick_mins[:, np.newaxis, :]
        + CORNER_OFFSETS[np.newaxis, :, :] * block_worlds[:, np.newaxis, np.newaxis]
    )
    build_corners_ms = (time.perf_counter() - t0) * 1000

    # ── 2. Vectorised signed-distance test ────────────────────────────
    # dists[i, j, k] = dot(corner j of brick i, normal of plane k) + d_k
    # Shape: (N, 8, 6)
    t0 = time.perf_counter()
    dists = (
        np.einsum("ijk,lk->ijl", all_corners, frustum_planes[:, :3])
        + frustum_planes[:, 3]
    )
    einsum_ms = (time.perf_counter() - t0) * 1000

    # ── 3. Visibility mask ────────────────────────────────────────────
    # visible[i] = all planes have at least one inside corner.
    t0 = time.perf_counter()
    any_inside = dists.max(axis=1) >= 0.0   # (N, 6)
    visible_mask = any_inside.all(axis=1)    # (N,)
    mask_ms = (time.perf_counter() - t0) * 1000

    visible = {
        keys_list[i]: values[keys_list[i]]
        for i in range(n)
        if visible_mask[i]
    }
    timings = {
        "build_corners_ms": build_corners_ms,
        "einsum_ms": einsum_ms,
        "mask_ms": mask_ms,
    }
    return visible, timings


def bricks_in_frustum_arr(
    arr: np.ndarray,
    block_size: int,
    frustum_planes: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Conservative AABB frustum test over a brick array.

    Vectorised; operates entirely on numpy arrays — no BrickKey objects.

    Parameters
    ----------
    arr : ndarray, shape (M, 4), dtype int32
        Columns: ``[level, gz_c, gy_c, gx_c]``.
    block_size : int
        Level-1 brick side length in voxels.
    frustum_planes : ndarray, shape (6, 4)
        Inward-pointing half-space planes.

    Returns
    -------
    visible_arr : ndarray, shape (K, 4)
        Subset of rows that pass the frustum test.
    timings : dict
        Wall-clock timings in milliseconds.
    """
    import time as _time

    M = len(arr)
    if M == 0:
        return arr, {"build_corners_ms": 0.0, "einsum_ms": 0.0, "mask_ms": 0.0}

    levels = arr[:, 0]
    gz_c   = arr[:, 1]
    gy_c   = arr[:, 2]
    gx_c   = arr[:, 3]

    # ── Brick world sizes and min corners ─────────────────────────────
    t0 = _time.perf_counter()
    scales     = np.left_shift(1, (levels - 1)).astype(np.float64)  # (M,)
    bw         = float(block_size) * scales                          # (M,)
    brick_mins = np.stack(
        [gx_c.astype(np.float64) * bw,   # x
         gy_c.astype(np.float64) * bw,   # y
         gz_c.astype(np.float64) * bw],  # z
        axis=1,
    )  # (M, 3)

    # (M, 8, 3) via broadcasting
    all_corners = (
        brick_mins[:, np.newaxis, :]
        + CORNER_OFFSETS[np.newaxis, :, :] * bw[:, np.newaxis, np.newaxis]
    )
    build_corners_ms = (_time.perf_counter() - t0) * 1000

    # ── Signed-distance test, shape (M, 8, 6) ─────────────────────────
    t0 = _time.perf_counter()
    dists = (
        np.einsum("ijk,lk->ijl", all_corners, frustum_planes[:, :3])
        + frustum_planes[:, 3]
    )
    einsum_ms = (_time.perf_counter() - t0) * 1000

    # ── Visibility mask ────────────────────────────────────────────────
    t0 = _time.perf_counter()
    visible_mask = (dists.max(axis=1) >= 0.0).all(axis=1)  # (M,)
    mask_ms = (_time.perf_counter() - t0) * 1000

    timings = {
        "build_corners_ms": build_corners_ms,
        "einsum_ms": einsum_ms,
        "mask_ms": mask_ms,
    }
    return arr[visible_mask], timings
