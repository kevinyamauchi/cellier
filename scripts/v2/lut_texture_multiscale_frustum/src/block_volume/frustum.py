"""Frustum geometry helpers and AABB visibility test.

Phase 2: pure-numpy module — no pygfx scene objects.

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

See ``compute_brick_aabb_corners`` for the canonical implementation and
the ``__main__`` block at the bottom for an axis-order assertion test.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pygfx as gfx

from block_volume.layout import BlockLayout
from block_volume.tile_manager import BrickKey


# ---------------------------------------------------------------------------
# Camera extraction
# ---------------------------------------------------------------------------


def get_frustum_corners_world(camera: gfx.PerspectiveCamera) -> np.ndarray:
    """Extract world-space frustum corners from a pygfx camera.

    Parameters
    ----------
    camera : gfx.PerspectiveCamera

    Returns
    -------
    corners : ndarray, shape (2, 4, 3)
        ``corners[0]`` = near plane, ``corners[1]`` = far plane.
        Within each plane: (left-bottom, right-bottom, right-top, left-top).
    """
    return np.asarray(camera.frustum, dtype=np.float64)


def get_view_direction_world(camera: gfx.PerspectiveCamera) -> np.ndarray:
    """Return the normalised view direction in world space as ``(3,)``.

    PyGFX cameras look along local -z.  We derive the forward vector from
    the world matrix's third column (local z axis) and negate.
    """
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
    Convention: (x, y, z) — axis 0 = x, 1 = y, 2 = z.
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

    # Near  — inward normal points from near toward far (into the scene)
    planes[0] = _compute_plane_parameters(n[0], n[2], n[1])
    # Far   — inward normal points from far toward near (back toward camera)
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
# Wireframe helper (returns a pygfx scene object)
# ---------------------------------------------------------------------------


def _frustum_edges(corners: np.ndarray) -> np.ndarray:
    """Return the 12 frustum edges as (start, end) point pairs.

    Parameters
    ----------
    corners : ndarray, shape (2, 4, 3)

    Returns
    -------
    edges : ndarray, shape (12, 2, 3)
    """
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
    """Build a frustum wireframe as a ``gfx.Line`` with segment material.

    Parameters
    ----------
    corners : ndarray, shape (2, 4, 3)
        Frustum corners from ``get_frustum_corners_world``.
    color : str
        Hex colour string.

    Returns
    -------
    line : gfx.Line
    """
    edges = _frustum_edges(corners)  # (12, 2, 3)
    positions = edges.reshape(-1, 3).astype(np.float32)
    geometry = gfx.Geometry(positions=positions)
    material = gfx.LineSegmentMaterial(color=color, thickness=1.5)
    return gfx.Line(geometry, material)


# ---------------------------------------------------------------------------
# AABB helpers
# ---------------------------------------------------------------------------


# 8 corner offsets in itertools.product([0, 1], repeat=3) order.
# Index 0 = (0,0,0) = all-min corner, index 7 = (1,1,1) = all-max corner.
_CORNER_OFFSETS = np.array(
    list(itertools.product([0.0, 1.0], repeat=3)), dtype=np.float64
)  # (8, 3)


def compute_brick_aabb_corners(
    brick_key: BrickKey,
    layout: BlockLayout,
    block_size: int,
) -> np.ndarray:
    """Return the 8 world-space AABB corners for a brick.

    Parameters
    ----------
    brick_key : BrickKey
        Brick identifier.  ``level`` determines the voxel-size scale.
    layout : BlockLayout
        Layout of the *finest* (level 1) resolution — used only to anchor
        the coordinate system; the actual scale is derived from the level.
    block_size : int
        Brick side length in voxels at level 1.

    Returns
    -------
    corners : ndarray, shape (8, 3)
        World-space (x, y, z) corners in itertools.product order.

    Notes
    -----
    **Axis-order caution**: ``BrickKey`` stores ``(gz, gy, gx)`` (DHW order),
    but world coordinates are ``(x, y, z)``.  This function performs the
    required swap:

        world_x = gx * block_world
        world_y = gy * block_world
        world_z = gz * block_world
    """
    # Level k bricks are 2^(k-1) times larger than level 1 bricks.
    scale = 2 ** (brick_key.level - 1)
    block_world = float(block_size * scale)

    # NOTE: swap DHW → XYZ: gx→x, gy→y, gz→z
    min_corner = np.array(
        [brick_key.gx * block_world, brick_key.gy * block_world, brick_key.gz * block_world],
        dtype=np.float64,
    )
    return min_corner + _CORNER_OFFSETS * block_world  # (8, 3)


# ---------------------------------------------------------------------------
# Frustum culling
# ---------------------------------------------------------------------------


def bricks_in_frustum(
    brick_keys: set[BrickKey] | dict[BrickKey, int],
    layout: BlockLayout,
    block_size: int,
    frustum_planes: np.ndarray,
) -> tuple[dict[BrickKey, int], dict]:
    """Conservative AABB frustum test over a set of brick keys.

    A brick is *visible* if, for every frustum plane, at least one of its
    8 AABB corners has a non-negative signed distance (is inside or on the
    plane).  A brick is only culled if ALL 8 corners are on the wrong side
    of at least one plane.

    Parameters
    ----------
    brick_keys : set[BrickKey] or dict[BrickKey, int]
        Candidate bricks.  If a dict, the values are preserved.
    layout : BlockLayout
        Finest-level layout (used to anchor coordinates).
    block_size : int
        Brick side length in voxels at level 1.
    frustum_planes : ndarray, shape (6, 4)
        Inward-pointing half-space planes from ``frustum_planes_from_corners``.

    Returns
    -------
    visible : dict[BrickKey, int]
        Subset of ``brick_keys`` that pass the frustum test, with the
        original int values (0 if ``brick_keys`` was a plain set).
    timings : dict
        Wall-clock timings in milliseconds::

            {
                "build_corners_ms": float,
                "einsum_ms": float,
                "mask_ms": float,
            }
    """
    # Normalise input to a dict
    if isinstance(brick_keys, dict):
        keys_list = list(brick_keys.keys())
        values = brick_keys
    else:
        keys_list = list(brick_keys)
        values = {k: 0 for k in keys_list}

    n = len(keys_list)
    if n == 0:
        return {}, {"build_corners_ms": 0.0, "einsum_ms": 0.0, "mask_ms": 0.0}

    # 1. Build (N, 8, 3) array of AABB corners for all bricks
    t0 = time.perf_counter()
    all_corners = np.empty((n, 8, 3), dtype=np.float64)
    for i, key in enumerate(keys_list):
        all_corners[i] = compute_brick_aabb_corners(key, layout, block_size)
    build_corners_ms = (time.perf_counter() - t0) * 1000

    # 2. Vectorised signed-distance test:
    #    dists[i, j, k] = dot(corner j of brick i, normal of plane k) + d_k
    #    Shape: (N, 8, 6)
    t0 = time.perf_counter()
    dists = np.einsum("ijk,lk->ijl", all_corners, frustum_planes[:, :3]) + frustum_planes[:, 3]
    einsum_ms = (time.perf_counter() - t0) * 1000

    # 3. A brick is visible if for every plane at least one corner is inside.
    #    any_inside[i, k] = True  ⟺  at least one corner of brick i is
    #                                  on the positive side of plane k.
    #    visible[i] = True  ⟺  all 6 planes have at least one inside corner.
    t0 = time.perf_counter()
    any_inside = (dists >= 0).any(axis=1)   # (N, 6)
    visible_mask = any_inside.all(axis=1)   # (N,) bool
    mask_ms = (time.perf_counter() - t0) * 1000

    visible = {keys_list[i]: values[keys_list[i]] for i in range(n) if visible_mask[i]}

    timings = {
        "build_corners_ms": build_corners_ms,
        "einsum_ms": einsum_ms,
        "mask_ms": mask_ms,
    }
    return visible, timings


# ---------------------------------------------------------------------------
# Axis-order verification (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from block_volume.layout import BlockLayout
    from block_volume.tile_manager import BrickKey

    layout = BlockLayout.from_volume_shape((64, 64, 64), block_size=32)

    # BrickKey at gz=0, gy=0, gx=1 → world x in [32, 64], y/z in [0, 32]
    key = BrickKey(level=1, gz=0, gy=0, gx=1)
    corners = compute_brick_aabb_corners(key, layout, block_size=32)
    assert corners[:, 0].min() == 32.0, f"x_min wrong: {corners[:, 0].min()}"
    assert corners[:, 0].max() == 64.0, f"x_max wrong: {corners[:, 0].max()}"
    assert corners[:, 2].min() ==  0.0, f"z_min wrong: {corners[:, 2].min()}"

    # BrickKey at gz=1, gy=0, gx=0 → world z in [32, 64], x/y in [0, 32]
    key2 = BrickKey(level=1, gz=1, gy=0, gx=0)
    corners2 = compute_brick_aabb_corners(key2, layout, block_size=32)
    assert corners2[:, 2].min() == 32.0, f"z_min wrong: {corners2[:, 2].min()}"
    assert corners2[:, 2].max() == 64.0, f"z_max wrong: {corners2[:, 2].max()}"
    assert corners2[:, 0].min() ==  0.0, f"x_min wrong: {corners2[:, 0].min()}"

    print("axis-order check passed ✓")
