# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pygfx",
#   "PySide6",
#   "rendercanvas",
#   "tensorstore",
# ]
# ///
"""LOD debug visualiser — async tensorstore edition.

A self-contained script (runnable via ``uv run lod_debug_image_async.py``)
that implements the LOD scale-selection and texture-placement algorithms
from scratch, then visualises them in a PySide6 window with an embedded
PyGFX 3D scene.

Async pipeline
--------------
All zarr I/O goes through tensorstore.  Chunk loads run as a single
cancellable ``asyncio.Task`` driven by ``PySide6.QtAsyncio``, which bridges
Qt's event loop and asyncio in one thread — no cross-thread locking is
needed for pygfx scene mutations.  When the camera settles after a move,
any in-flight load task is cancelled and a fresh one is scheduled.

Coordinate convention
---------------------
This script uses **(x, y, z)** throughout — axis 0 = x, 1 = y, 2 = z.
This matches PyGFX's native convention and avoids constant transposition
when reading camera matrices. It differs from the main Cellier codebase
which uses (z, y, x).

Frustum corners are shaped ``(2, 4, 3)``:
- axis 0: near plane = index 0, far plane = index 1
- axis 1: corners within plane ordered (left-bottom, right-bottom,
  right-top, left-top)
- axis 2: (x, y, z) coordinates
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import pathlib
import time
from dataclasses import dataclass, field

import numpy as np
import pygfx as gfx
import PySide6.QtAsyncio as QtAsyncio
import tensorstore as ts
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from rendercanvas.qt import QRenderWidget

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DATASET_SHAPE = (1024, 1024, 1024)
CHUNK_SHAPE = (32, 32, 32)
TEXTURE_WIDTH = 256
AABB_MIN = np.array([0.0, 0.0, 0.0])
AABB_MAX = np.array([1024.0, 1024.0, 1024.0])

SCALES = [
    dict(shape=(1024, 1024, 1024), voxel_size=1.0),  # index 0 — finest
    dict(shape=(512, 512, 512), voxel_size=2.0),  # index 1
    dict(shape=(256, 256, 256), voxel_size=4.0),  # index 2 — coarsest
]

SCALE_COLORS = ["#4488ff", "#ffaa22", "#ff4444"]  # blue, orange, red
SCALE_COLORMAPS = ["viridis", "plasma", "inferno"]  # one per scale level
AABB_COLOR = "#ffffff"
FRUSTUM_COLOR = "#00cc44"

# Path to the zarr group produced by make_zarr_sample.py (expected next to
# this script).  The coarsest scale ("s2") is loaded for reference rendering.
ZARR_PATH = pathlib.Path(__file__).parent / "multiscale_blobs.zarr"
COARSEST_SCALE_NAME = "s2"
ZARR_SCALE_NAMES = ["s0", "s1", "s2"]  # one per scale level
CAMERA_DEBOUNCE_MS = 300  # ms to wait after camera stops before re-running LOD


def _zarr_spec(zarr_path: pathlib.Path, scale_name: str) -> dict:
    """Build a tensorstore zarr spec for the given scale name (read-only open).

    Parameters
    ----------
    zarr_path : pathlib.Path
        Root directory of the multiscale zarr store.
    scale_name : str
        Subdirectory name within ``zarr_path`` (e.g. ``"s0"``, ``"s1"``).

    Returns
    -------
    spec : dict
        Tensorstore spec dict suitable for ``ts.open()``.
    """
    return {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": str(zarr_path / scale_name),
        },
    }


# ---------------------------------------------------------------------------
# Section 1 — Data structures
# ---------------------------------------------------------------------------


# 8 corner offsets in itertools.product([0, 1], repeat=3) order.
# Index 0 = (0,0,0) = all-min, index 7 = (1,1,1) = all-max.
_CORNER_OFFSETS = np.array(list(itertools.product([0, 1], repeat=3)), dtype=np.float64)


@dataclass
class ScaleLevel:
    shape: tuple[int, int, int]
    chunk_shape: tuple[int, int, int]
    voxel_size: float
    chunk_corners: np.ndarray  # (n_chunks, 8, 3) — world-space corners
    n_chunks_per_axis: np.ndarray  # (3,) int — number of chunks along each axis
    chunk_world: np.ndarray  # (3,) — chunk size in world units


def precompute_chunk_corners(
    shape: tuple[int, int, int],
    chunk_shape: tuple[int, int, int],
    voxel_size: float,
) -> np.ndarray:
    """Pre-compute all chunk corners in world (= scale_0) space.

    Returns shape ``(n_chunks, 8, 3)`` using the
    ``itertools.product([min, max], repeat=3)`` corner ordering.
    """
    shape_arr = np.array(shape, dtype=np.float64)
    chunk_arr = np.array(chunk_shape, dtype=np.float64)
    chunk_world = chunk_arr * voxel_size  # (3,) world-space chunk size

    n_per_axis = (shape_arr / chunk_arr).astype(int)  # (3,)
    # Meshgrid of chunk indices → min-corner positions
    ix, iy, iz = (np.arange(n) for n in n_per_axis)
    grid = np.stack(np.meshgrid(ix, iy, iz, indexing="ij"), axis=-1)  # (nx, ny, nz, 3)
    mins = grid.reshape(-1, 3).astype(np.float64) * chunk_world  # (N, 3)

    # Expand to 8 corners: (N, 1, 3) + (1, 8, 3)
    offsets = _CORNER_OFFSETS[np.newaxis, :, :] * chunk_world  # (1, 8, 3)
    corners = mins[:, np.newaxis, :] + offsets  # (N, 8, 3)
    return corners


@dataclass
class MockDataset:
    scales: list[ScaleLevel]
    aabb_min: np.ndarray  # (3,)
    aabb_max: np.ndarray  # (3,)


@dataclass
class LodResult:
    scale_idx: int
    texture_min: np.ndarray  # (3,)
    texture_max: np.ndarray  # (3,)
    union_vertices: np.ndarray  # (N, 3)
    primary_axis: int
    visible_chunk_indices: np.ndarray  # (M,) — indices into ScaleLevel.chunk_corners
    camera_pos: np.ndarray  # (3,) — world-space camera position at time of LOD run
    timings: dict[str, float] = field(default_factory=dict)  # step name → seconds


@dataclass
class ChunkRequest:
    """Everything needed to load one chunk into a texture.

    Produced by the plan phase (pure arithmetic, main thread).
    Consumed by the execute phase (I/O, can move to a worker).
    """

    chunk_index: int
    scale_idx: int
    zarr_slice: tuple[slice, slice, slice]  # where to read (array zyx order)
    texture_slice: tuple[slice, slice, slice]  # where to write (array zyx order)


# ---------------------------------------------------------------------------
# Section 2 — Geometry utilities (local re-implementations)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Section 2 — Geometry utilities (local re-implementations)
# ---------------------------------------------------------------------------


def compute_plane_parameters(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    """Return (4,) plane coefficients [a, b, c, d].

    Normal points toward the side where the points are counter-clockwise.

    Convention: (x, y, z) — axis 0 = x, 1 = y, 2 = z.
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
        Frustum corners.  ``corners[0]`` = near plane, ``corners[1]`` = far
        plane.  Within each plane the order is (left-bottom, right-bottom,
        right-top, left-top).

    Returns
    -------
    planes : ndarray, shape (6, 4)
        Plane coefficients ordered: near, far, left, right, top, bottom.
        For each plane, a point ``p`` is *inside* the frustum when
        ``dot(plane[:3], p) + plane[3] >= 0``.
    """
    n = corners[0]  # near: lb, rb, rt, lt
    f = corners[1]  # far:  lb, rb, rt, lt

    planes = np.empty((6, 4), dtype=np.float64)

    # Near — inward normal points from near toward far (into the scene)
    planes[0] = compute_plane_parameters(n[0], n[2], n[1])
    # Far  — inward normal points from far toward near (back toward camera)
    planes[1] = compute_plane_parameters(f[0], f[1], f[2])
    # Left  — n_lt, n_lb, f_lb
    planes[2] = compute_plane_parameters(n[3], n[0], f[0])
    # Right — n_rb, n_rt, f_rt
    planes[3] = compute_plane_parameters(n[1], n[2], f[2])
    # Top   — n_rt, n_lt, f_lt
    planes[4] = compute_plane_parameters(n[2], n[3], f[3])
    # Bottom — n_lb, n_rb, f_rb
    planes[5] = compute_plane_parameters(n[0], n[1], f[1])

    return planes


def points_in_frustum(points: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """Test which points lie inside (or on) all half-spaces.

    Parameters
    ----------
    points : ndarray, shape (M, 3)
    planes : ndarray, shape (N, 4)

    Returns
    -------
    mask : ndarray, shape (M,), dtype bool
    """
    # distances: (M, N)
    distances = points @ planes[:, :3].T + planes[:, 3]
    return np.all(distances >= 0, axis=1)


def frustum_edges(corners: np.ndarray) -> np.ndarray:
    """Return the 12 edges of a frustum as (start, end) point pairs.

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
    edges = np.array(
        [[corners[a], corners[b]] for (a, b) in edge_indices],
        dtype=np.float64,
    )
    return edges


# ---------------------------------------------------------------------------
# Section 3 — LOD geometry functions
# ---------------------------------------------------------------------------


def select_primary_axis(view_direction: np.ndarray, aabb_extents: np.ndarray) -> int:
    """Select the primary viewing axis.

    Convention: (x, y, z) — axis 0 = x, 1 = y, 2 = z.
    """
    scores = np.abs(view_direction)
    candidates = np.where(scores == scores.max())[0]
    if len(candidates) == 1:
        return int(candidates[0])
    return int(candidates[np.argmax(aabb_extents[candidates])])


def get_aabb_faces(aabb_min: np.ndarray, aabb_max: np.ndarray) -> list[dict]:
    """Return the 6 faces of an AABB.

    Each face dict contains: axis, outward_normal, face_coord, corners (4,3),
    face_min (3,), face_max (3,).

    Convention: (x, y, z).
    """
    faces = []
    for a in range(3):
        q = (a + 1) % 3
        r = (a + 2) % 3

        for sign, coord in [(-1, aabb_min[a]), (1, aabb_max[a])]:
            normal = np.zeros(3)
            normal[a] = sign

            # 4 corners spanning the other two axes
            corners = np.empty((4, 3), dtype=np.float64)
            vals_q = [aabb_min[q], aabb_max[q]]
            vals_r = [aabb_min[r], aabb_max[r]]

            # Ordered: (min_q, min_r), (max_q, min_r),
            #          (max_q, max_r), (min_q, max_r)
            for i, (vq, vr) in enumerate(
                [
                    (vals_q[0], vals_r[0]),
                    (vals_q[1], vals_r[0]),
                    (vals_q[1], vals_r[1]),
                    (vals_q[0], vals_r[1]),
                ]
            ):
                corners[i, a] = coord
                corners[i, q] = vq
                corners[i, r] = vr

            face_min = corners.min(axis=0)
            face_max = corners.max(axis=0)

            faces.append(
                dict(
                    axis=a,
                    outward_normal=normal,
                    face_coord=coord,
                    corners=corners,
                    face_min=face_min,
                    face_max=face_max,
                )
            )
    return faces


def is_camera_inside(
    camera_pos: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> bool:
    return bool(np.all(camera_pos >= aabb_min) and np.all(camera_pos <= aabb_max))


def clip_polygon_against_plane(
    polygon: list[np.ndarray], plane: np.ndarray
) -> list[np.ndarray]:
    """Sutherland-Hodgman clipping of a convex polygon against one half-space.

    Points with ``dot(p, plane[:3]) + plane[3] >= 0`` are inside.
    """
    if len(polygon) == 0:
        return []

    output: list[np.ndarray] = []
    n = plane[:3]
    d = plane[3]

    for i in range(len(polygon)):
        current = polygon[i]
        nxt = polygon[(i + 1) % len(polygon)]
        d_cur = np.dot(current, n) + d
        d_nxt = np.dot(nxt, n) + d

        if d_cur >= 0:
            output.append(current)
        if (d_cur >= 0) != (d_nxt >= 0):
            t = d_cur / (d_cur - d_nxt)
            output.append(current + t * (nxt - current))

    return output


def clip_face_against_frustum_sides(
    face_corners: np.ndarray, frustum_planes: np.ndarray
) -> np.ndarray | None:
    """Clip face polygon against the 4 side planes of the frustum.

    Side planes are indices 2..5 (left, right, top, bottom).
    """
    polygon = [face_corners[i] for i in range(face_corners.shape[0])]

    for plane_idx in (2, 3, 4, 5):
        polygon = clip_polygon_against_plane(polygon, frustum_planes[plane_idx])
        if len(polygon) == 0:
            return None

    return np.array(polygon, dtype=np.float64)


def face_intersects_frustum(
    face: dict,
    frustum_corners: np.ndarray,
    frustum_planes: np.ndarray,
) -> bool:
    """Test whether an AABB face intersects the frustum.

    Case 1: any face corner inside the frustum.
    Case 2: any frustum edge crosses the face rectangle.
    """
    # Case 1
    mask = points_in_frustum(face["corners"], frustum_planes)
    if np.any(mask):
        return True

    # Case 2
    edges = frustum_edges(frustum_corners)  # (12, 2, 3)
    a = face["axis"]
    q = (a + 1) % 3
    r = (a + 2) % 3

    for edge in edges:
        start, end = edge[0], edge[1]
        denom = end[a] - start[a]
        if abs(denom) < 1e-10:
            continue
        t = (face["face_coord"] - start[a]) / denom
        if not (0.0 <= t <= 1.0):
            continue
        point = start + t * (end - start)
        if (
            face["face_min"][q] <= point[q] <= face["face_max"][q]
            and face["face_min"][r] <= point[r] <= face["face_max"][r]
        ):
            return True

    return False


def compute_near_visible_extent(
    frustum_corners: np.ndarray,
    frustum_planes: np.ndarray,
    camera_pos: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    view_direction: np.ndarray,
    primary_axis: int,
) -> tuple[np.ndarray, float, float, int] | None:
    """Compute the visible extent of the AABB in the frustum.

    Returns ``(union_vertices, extent_q, extent_r, actual_primary_axis)``
    or ``None``.

    Tries the primary-axis face first.  If it does not intersect the
    frustum (oblique views where the camera is beside the AABB), falls
    back to the remaining near-facing faces, picking the one with the
    strongest opposition to the view direction.
    """
    if is_camera_inside(camera_pos, aabb_min, aabb_max):
        # Code path A — camera inside volume
        q = (primary_axis + 1) % 3
        r = (primary_axis + 2) % 3
        near_corners = frustum_corners[0].copy()  # (4, 3)
        near_corners[:, q] = np.clip(near_corners[:, q], aabb_min[q], aabb_max[q])
        near_corners[:, r] = np.clip(near_corners[:, r], aabb_min[r], aabb_max[r])

        union_vertices = near_corners
        extent_q = np.max(union_vertices[:, q]) - np.min(union_vertices[:, q])
        extent_r = np.max(union_vertices[:, r]) - np.min(union_vertices[:, r])
        return union_vertices, extent_q, extent_r, primary_axis

    # Code path B — camera outside volume
    faces = get_aabb_faces(aabb_min, aabb_max)

    # Collect all near-facing faces (outward normal opposes view direction)
    near_faces = [f for f in faces if np.dot(f["outward_normal"], view_direction) < 0]

    # Sort: primary-axis face first, then by dot-product magnitude
    # (strongest opposition = most head-on to the camera)
    def sort_key(f: dict) -> tuple[int, float]:
        is_primary = 0 if f["axis"] == primary_axis else 1
        dot = np.dot(f["outward_normal"], view_direction)
        return (is_primary, dot)  # more negative dot = stronger opposition

    near_faces.sort(key=sort_key)

    for face in near_faces:
        if not face_intersects_frustum(face, frustum_corners, frustum_planes):
            continue
        clipped = clip_face_against_frustum_sides(face["corners"], frustum_planes)
        if clipped is None:
            continue

        actual_axis = face["axis"]
        q = (actual_axis + 1) % 3
        r = (actual_axis + 2) % 3
        union_vertices = clipped
        extent_q = np.max(union_vertices[:, q]) - np.min(union_vertices[:, q])
        extent_r = np.max(union_vertices[:, r]) - np.min(union_vertices[:, r])
        return union_vertices, extent_q, extent_r, actual_axis

    return None


def select_scale(
    dataset: MockDataset,
    extent_q: float,
    extent_r: float,
    primary_axis: int,
    texture_width: int,
) -> int:
    """Select the finest scale whose texture covers the visible extent."""
    for scale_idx, scale in enumerate(dataset.scales):
        T = texture_width * scale.voxel_size
        if T >= extent_q and T >= extent_r:
            return scale_idx
    raise ValueError("No scale has a texture large enough to cover the visible extent.")


# ---------------------------------------------------------------------------
# Section 4 — Texture placement
# ---------------------------------------------------------------------------


def place_texture(
    union_vertices: np.ndarray,
    view_direction: np.ndarray,
    primary_axis: int,
    scale: ScaleLevel,
    texture_width: int,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place the texture AABB in world space.

    Returns ``(texture_min, texture_max)`` each shape ``(3,)``.
    """
    q = (primary_axis + 1) % 3
    r = (primary_axis + 2) % 3
    p = primary_axis

    T = texture_width * scale.voxel_size
    chunk_world = np.array(scale.chunk_shape, dtype=float) * scale.voxel_size

    texture_min = np.zeros(3)
    texture_max = np.zeros(3)

    # Primary axis — anchor at near face of visible data
    if view_direction[p] >= 0:
        anchor = np.min(union_vertices[:, p])
        texture_min[p] = np.floor(anchor / chunk_world[p]) * chunk_world[p]
        texture_max[p] = texture_min[p] + T
    else:
        anchor = np.max(union_vertices[:, p])
        texture_max[p] = np.ceil(anchor / chunk_world[p]) * chunk_world[p]
        texture_min[p] = texture_max[p] - T

    # Transverse axes — centre on centroid, snap to chunk boundary
    for axis in [q, r]:
        centroid = np.mean(union_vertices[:, axis])
        raw_min = centroid - T / 2
        texture_min[axis] = np.floor(raw_min / chunk_world[axis]) * chunk_world[axis]
        texture_max[axis] = texture_min[axis] + T

    return texture_min, texture_max


def clamp_texture_to_aabb(
    texture_min: np.ndarray,
    texture_max: np.ndarray,
    primary_axis: int,
    scale: ScaleLevel,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift the texture window on the transverse axes so it lies within the data AABB.

    The texture has a fixed world-space size ``T`` on every axis.  When the
    centroid-based placement in ``place_texture`` puts ``texture_min`` or
    ``texture_max`` outside ``[aabb_min, aabb_max]``, this function shifts
    the window — preserving ``T`` — back inside.  The shift is always rounded
    up to the nearest chunk boundary so that ``texture_min`` stays
    chunk-aligned and voxel offsets from it remain whole-chunk multiples.

    Only the **transverse** axes (q, r) are adjusted.  The primary axis
    placement is anchored at the near data face and intentionally left
    unchanged.

    Parameters
    ----------
    texture_min, texture_max : ndarray, shape (3,)
        Texture AABB bounds from ``place_texture``.  Modified copies are
        returned; the originals are not mutated.
    primary_axis : int
        The primary viewing axis (0 = x, 1 = y, 2 = z).  Skipped during
        clamping.
    scale : ScaleLevel
        Provides ``chunk_world`` (world-space chunk size per axis).
    aabb_min, aabb_max : ndarray, shape (3,)
        Data bounding box.

    Returns
    -------
    texture_min, texture_max : ndarray, shape (3,)
        Shifted (and still fixed-size) texture bounds.
    """
    texture_min = texture_min.copy()
    texture_max = texture_max.copy()

    transverse_axes = [a for a in range(3) if a != primary_axis]

    for axis in transverse_axes:
        cw = scale.chunk_world[axis]

        # Low-side overhang: how far texture_min protrudes below aabb_min.
        low_overhang = max(0.0, aabb_min[axis] - texture_min[axis])
        if low_overhang > 0.0:
            # Round up to the next chunk boundary so texture_min stays aligned.
            shift = np.ceil(low_overhang / cw) * cw
            texture_min[axis] += shift
            texture_max[axis] += shift

        # High-side overhang: how far texture_max protrudes above aabb_max.
        # Re-evaluate after the low-side shift.
        high_overhang = max(0.0, texture_max[axis] - aabb_max[axis])
        if high_overhang > 0.0:
            shift = np.ceil(high_overhang / cw) * cw
            texture_min[axis] -= shift
            texture_max[axis] -= shift

    return texture_min, texture_max


def select_visible_chunks(
    scale: ScaleLevel,
    texture_min: np.ndarray,
    texture_max: np.ndarray,
    frustum_planes: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Select chunks that overlap the texture AABB, data AABB, and frustum.

    Parameters
    ----------
    scale : ScaleLevel
        The selected scale level (provides chunk grid geometry and
        pre-computed corners).
    texture_min, texture_max : ndarray, shape (3,)
        Texture AABB bounds.
    frustum_planes : ndarray, shape (6, 4)
        Inward-pointing frustum half-space planes.
    aabb_min, aabb_max : ndarray, shape (3,)
        Data AABB bounds.

    Returns
    -------
    indices : ndarray, shape (M,)
        Indices into ``scale.chunk_corners`` of the visible chunks.
    timings : dict[str, float]
        Sub-step timings in seconds.
    """
    timings: dict[str, float] = {}

    # Pass 1 — intersect texture AABB with data AABB, convert to chunk
    # index ranges.  Pure scalar arithmetic, replaces two full-array
    # broadcasts over all chunk corners.
    t0 = time.perf_counter()
    eff_min = np.maximum(texture_min, aabb_min)
    eff_max = np.minimum(texture_max, aabb_max)

    # Chunk index ranges (inclusive start, exclusive stop)
    idx_start = np.floor(eff_min / scale.chunk_world).astype(int)
    idx_stop = np.ceil(eff_max / scale.chunk_world).astype(int)
    idx_start = np.clip(idx_start, 0, scale.n_chunks_per_axis)
    idx_stop = np.clip(idx_stop, 0, scale.n_chunks_per_axis)

    # Meshgrid of candidate chunk indices → flat indices into chunk_corners
    ranges = [np.arange(idx_start[a], idx_stop[a]) for a in range(3)]
    if any(len(r) == 0 for r in ranges):
        timings["index_ranges"] = time.perf_counter() - t0
        return np.array([], dtype=int), timings

    gi, gj, gk = np.meshgrid(*ranges, indexing="ij")
    # Flat index = i * (ny * nz) + j * nz + k
    ny, nz = scale.n_chunks_per_axis[1], scale.n_chunks_per_axis[2]
    candidate_indices = (gi * ny * nz + gj * nz + gk).ravel()
    timings["index_ranges"] = time.perf_counter() - t0

    # Pass 2 — conservative frustum cull (AABB-vs-planes)
    t0 = time.perf_counter()
    cand_corners = scale.chunk_corners[candidate_indices]  # (n_cand, 8, 3)
    timings["gather_candidates"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dists = (
        np.einsum("ijk,lk->ijl", cand_corners, frustum_planes[:, :3])
        + frustum_planes[:, 3]
    )
    timings["signed_distances"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    any_inside = np.any(dists >= 0, axis=1)
    visible = np.all(any_inside, axis=1)
    timings["visibility_mask"] = time.perf_counter() - t0

    return candidate_indices[visible], timings


def plan_chunk_loads(
    result: LodResult,
    scale: ScaleLevel,
) -> tuple[list[ChunkRequest], dict[str, float]]:
    """Compute zarr/texture slices for each visible chunk, sorted nearest-first.

    Sorting is by Euclidean distance from the camera to each chunk's
    world-space centre.  This ensures the most perceptually important chunks
    arrive first during progressive async streaming.

    Parameters
    ----------
    result : LodResult
        Output from the LOD pipeline.  Provides ``texture_min``,
        ``scale_idx``, ``visible_chunk_indices``, and ``camera_pos``.
    scale : ScaleLevel
        The selected scale level.

    Returns
    -------
    requests : list[ChunkRequest]
        One entry per visible chunk, ordered nearest-camera-first.
    timings : dict[str, float]
        Sub-step timings in seconds: ``slice_build``, ``sort``, ``total``.
    """
    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    vs = scale.voxel_size
    cs = np.array(scale.chunk_shape, dtype=int)
    indices = result.visible_chunk_indices

    # ------------------------------------------------------------------
    # Sort indices by distance from camera to chunk centre (vectorised).
    # chunk_corners[:, 0] = world-space min corner (x, y, z).
    # chunk_corners[:, 7] = world-space max corner (x, y, z).
    # Centre = midpoint of min and max corners.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    chunk_mins = scale.chunk_corners[indices, 0]   # (M, 3)
    chunk_maxs = scale.chunk_corners[indices, 7]   # (M, 3)
    centres = (chunk_mins + chunk_maxs) * 0.5      # (M, 3)
    diffs = centres - result.camera_pos            # (M, 3)
    sq_dists = np.einsum("ij,ij->i", diffs, diffs) # (M,) squared distances
    sort_order = np.argsort(sq_dists)              # nearest first
    sorted_indices = indices[sort_order]
    timings["sort"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Build ChunkRequest objects in sorted order.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    requests: list[ChunkRequest] = []
    for idx in sorted_indices:
        chunk_world_min = scale.chunk_corners[idx, 0]

        # Source: zarr array coords.  World (x,y,z) → array (z,y,x).
        src_xyz = (chunk_world_min / vs).astype(int)
        src_start = np.array([src_xyz[2], src_xyz[1], src_xyz[0]])
        src_end = src_start + cs
        zarr_slice = (
            slice(src_start[0], src_end[0]),
            slice(src_start[1], src_end[1]),
            slice(src_start[2], src_end[2]),
        )

        # Destination: texture array coords.  Offset from texture_min,
        # then world (x,y,z) → array (z,y,x).
        # dst_end is clamped to TEXTURE_WIDTH: when T is not an integer
        # multiple of chunk_world the last chunk on an axis extends past the
        # texture boundary in world space, and we must not write beyond the
        # pre-allocated array.
        dst_xyz = ((chunk_world_min - result.texture_min) / vs).astype(int)
        dst_start = np.array([dst_xyz[2], dst_xyz[1], dst_xyz[0]])
        dst_end = np.minimum(dst_start + cs, TEXTURE_WIDTH)
        texture_slice = (
            slice(dst_start[0], dst_end[0]),
            slice(dst_start[1], dst_end[1]),
            slice(dst_start[2], dst_end[2]),
        )

        requests.append(
            ChunkRequest(
                chunk_index=idx,
                scale_idx=result.scale_idx,
                zarr_slice=zarr_slice,
                texture_slice=texture_slice,
            )
        )
    timings["slice_build"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_total

    return requests, timings


# ---------------------------------------------------------------------------
# Section 5 — Top-level LOD pipeline
# ---------------------------------------------------------------------------


def run_lod_check(
    dataset: MockDataset, camera: gfx.PerspectiveCamera
) -> LodResult | None:
    """Run the full LOD check pipeline.

    Raises ``ValueError`` if no scale can cover the visible extent.
    Returns ``None`` if no data is in view.
    """
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    frustum_corners = get_frustum_corners_world(camera)
    view_direction = get_view_direction_world(camera)
    camera_pos = get_camera_position_world(camera)
    timings["camera_extract"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    primary_axis = select_primary_axis(
        view_direction, dataset.aabb_max - dataset.aabb_min
    )
    timings["primary_axis"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    frustum_planes = frustum_planes_from_corners(frustum_corners)
    timings["frustum_planes"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = compute_near_visible_extent(
        frustum_corners,
        frustum_planes,
        camera_pos,
        dataset.aabb_min,
        dataset.aabb_max,
        view_direction,
        primary_axis,
    )
    timings["visible_extent"] = time.perf_counter() - t0

    if result is None:
        return None

    union_vertices, extent_q, extent_r, primary_axis = result

    t0 = time.perf_counter()
    scale_idx = select_scale(dataset, extent_q, extent_r, primary_axis, TEXTURE_WIDTH)
    scale = dataset.scales[scale_idx]
    timings["scale_selection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    texture_min, texture_max = place_texture(
        union_vertices,
        view_direction,
        primary_axis,
        scale,
        TEXTURE_WIDTH,
        dataset.aabb_min,
        dataset.aabb_max,
    )
    timings["texture_placement"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    texture_min, texture_max = clamp_texture_to_aabb(
        texture_min,
        texture_max,
        primary_axis,
        scale,
        dataset.aabb_min,
        dataset.aabb_max,
    )
    timings["texture_clamp"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    visible_chunk_indices, chunk_timings = select_visible_chunks(
        scale,
        texture_min,
        texture_max,
        frustum_planes,
        dataset.aabb_min,
        dataset.aabb_max,
    )
    timings["chunk_selection"] = time.perf_counter() - t0
    for key, val in chunk_timings.items():
        timings[f"  chunk.{key}"] = val

    return LodResult(
        scale_idx=scale_idx,
        texture_min=texture_min,
        texture_max=texture_max,
        union_vertices=union_vertices,
        primary_axis=primary_axis,
        visible_chunk_indices=visible_chunk_indices,
        camera_pos=camera_pos,
        timings=timings,
    )


# ---------------------------------------------------------------------------
# Section 6 — PyGFX camera extraction
# ---------------------------------------------------------------------------


def get_frustum_corners_world(camera: gfx.PerspectiveCamera) -> np.ndarray:
    """Extract world-space frustum corners from the camera.

    Returns shape ``(2, 4, 3)`` matching the (near, far) x (lb, rb, rt, lt)
    convention.

    Uses the camera's built-in ``.frustum`` property which handles NDC
    unprojection (depth range [0, 1]) internally.
    """
    return np.asarray(camera.frustum, dtype=np.float64)


def get_view_direction_world(camera: gfx.PerspectiveCamera) -> np.ndarray:
    """Return the normalised view direction in world space as ``(3,)``."""
    # PyGFX cameras look along local -z.  world.forward gives this in world
    # space, but the sign convention may vary — we derive from the world
    # matrix's third column (the local z axis) and negate.
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
# Section 7 — PyGFX wireframe helpers
# ---------------------------------------------------------------------------

# 12 edge index pairs for an AABB — pairs of the 8 corner indices (from
# itertools.product) that differ in exactly one bit.
_AABB_EDGE_INDICES = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),  # vary z
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),  # vary y
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # vary x
]


def make_box_wireframe(
    box_min: np.ndarray, box_max: np.ndarray, color: str
) -> gfx.Line:
    """Build a wireframe AABB as ``gfx.Line`` with segment material."""
    corners = np.array(
        list(
            itertools.product(
                [box_min[0], box_max[0]],
                [box_min[1], box_max[1]],
                [box_min[2], box_max[2]],
            )
        ),
        dtype=np.float32,
    )
    positions = np.array(
        [[corners[a], corners[b]] for a, b in _AABB_EDGE_INDICES],
        dtype=np.float32,
    ).reshape(-1, 3)

    geometry = gfx.Geometry(positions=positions)
    material = gfx.LineSegmentMaterial(color=color, thickness=1.5)
    return gfx.Line(geometry, material)


def make_frustum_wireframe(frustum_corners: np.ndarray, color: str) -> gfx.Line:
    """Build a wireframe frustum as ``gfx.Line`` with segment material."""
    edges = frustum_edges(frustum_corners)  # (12, 2, 3)
    positions = edges.reshape(-1, 3).astype(np.float32)

    geometry = gfx.Geometry(positions=positions)
    material = gfx.LineSegmentMaterial(color=color, thickness=1.5)
    return gfx.Line(geometry, material)


# ---------------------------------------------------------------------------
# Section 8 — Qt application
# ---------------------------------------------------------------------------


class LodDebugApp(QMainWindow):
    def __init__(self, zarr_path: pathlib.Path = ZARR_PATH) -> None:
        super().__init__()

        self._zarr_path = zarr_path

        self._dataset = MockDataset(
            scales=[
                ScaleLevel(
                    chunk_corners=precompute_chunk_corners(
                        s["shape"], CHUNK_SHAPE, s["voxel_size"]
                    ),
                    n_chunks_per_axis=(np.array(s["shape"]) // np.array(CHUNK_SHAPE)),
                    chunk_world=np.array(CHUNK_SHAPE, dtype=np.float64)
                    * s["voxel_size"],
                    chunk_shape=CHUNK_SHAPE,
                    **s,
                )
                for s in SCALES
            ],
            aabb_min=AABB_MIN.copy(),
            aabb_max=AABB_MAX.copy(),
        )
        self._frustum_line: gfx.Line | None = None
        self._texture_line: gfx.Line | None = None
        self._union_points: gfx.Points | None = None
        self._chunk_lines: list[gfx.Line] = []
        self._reference_volume: gfx.Volume | None = None
        self._scale_volumes: list[dict] | None = None  # per-scale rendering
        self._ts_stores: list[ts.TensorStore] | None = None  # per-scale tensorstore handles

        # Async chunk loading state — one cancellable task at a time
        self._load_task: asyncio.Task | None = None
        # Cache the most recent LOD result so the chunk wireframe toggle can
        # redraw lines without re-running the full pipeline.
        self._last_lod_result: LodResult | None = None

        self._setup_scene()
        self._load_reference_volume()
        self._setup_scale_volumes()
        self._setup_ui()

        # Camera debounce: re-run LOD pipeline after camera stops moving
        self._last_camera_matrix = np.array(
            self._camera.world.matrix, dtype=np.float64
        ).copy()
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(CAMERA_DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._on_camera_settled)

    # ---- scene setup ------------------------------------------------------

    def _setup_scene(self) -> None:
        # update_mode="continuous" → rendercanvas calls renderer.render() every
        # vsync via a Qt timer.  texture.update_range() marks regions dirty and
        # the actual GPU upload happens automatically on the next frame, so no
        # manual canvas.update() calls are needed anywhere.
        self._canvas = QRenderWidget(parent=self, update_mode="continuous")
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()

        self._camera = gfx.PerspectiveCamera(fov=50, aspect=1, depth_range=(1, 8000))
        self._camera.local.position = (512, 512, 3500)
        self._camera.show_pos((512, 512, 512))

        self._controller = gfx.OrbitController(
            camera=self._camera, register_events=self._renderer
        )

        # Permanent data AABB wireframe
        aabb_wire = make_box_wireframe(AABB_MIN, AABB_MAX, color=AABB_COLOR)
        self._scene.add(aabb_wire)

        self._canvas.request_draw(self._animate)

    # ---- reference volume ----------------------------------------------------

    def _load_reference_volume(self) -> None:
        """Load the coarsest scale from the zarr file and add as a Volume."""
        if not self._zarr_path.exists():
            print(
                f"WARNING: {self._zarr_path} not found — run with --make-file "
                f"first. Reference volume disabled."
            )
            return

        store = ts.open(_zarr_spec(self._zarr_path, COARSEST_SCALE_NAME)).result()
        data = np.asarray(store.read().result(), dtype=np.float32)
        print(
            f"Loaded reference volume '{COARSEST_SCALE_NAME}': "
            f"shape={data.shape}, "
            f"range=[{data.min():.3f}, {data.max():.3f}]"
        )

        # PyGFX Volume: array axis 0→z, 1→y, 2→x in world space.
        # Local space spans [-0.5, shape-0.5] per axis.
        # We need to map to world AABB [0, 1024].
        coarsest = self._dataset.scales[-1]
        vs = coarsest.voxel_size  # 4.0

        geometry = gfx.Geometry(grid=data)
        material = gfx.VolumeIsoMaterial(
            clim=(0, 1),
            threshold=0.5,
            map=gfx.utils.cm.pink,
        )

        self._reference_volume = gfx.Volume(geometry, material)
        # Scale: 1 voxel in local space = voxel_size in world space
        self._reference_volume.local.scale = (vs, vs, vs)
        # Position offset: compensate for the -0.5 voxel origin so that
        # the volume bounding box aligns with [0, 1024]
        offset = vs * 0.5
        self._reference_volume.local.position = (offset, offset, offset)

        self._reference_volume.visible = False
        self._scene.add(self._reference_volume)

    # ---- per-scale texture volumes -------------------------------------------

    def _setup_scale_volumes(self) -> None:
        """Pre-allocate a Volume + Texture per scale and open tensorstore handles."""
        if not self._zarr_path.exists():
            print("WARNING: zarr not found — chunk rendering disabled.")
            return

        # Open one tensorstore handle per scale and keep it alive.  These are
        # opened synchronously once at startup and reused across all async loads.
        self._ts_stores = [
            ts.open(_zarr_spec(self._zarr_path, name)).result()
            for name in ZARR_SCALE_NAMES
        ]

        tw = TEXTURE_WIDTH  # 256
        self._scale_volumes = []

        for scale_idx, scale in enumerate(self._dataset.scales):
            data = np.zeros((tw, tw, tw), dtype=np.float32)
            tex = gfx.Texture(data, dim=3, force_contiguous=True)
            cmap = getattr(gfx.utils.cm, SCALE_COLORMAPS[scale_idx])
            material = gfx.VolumeIsoMaterial(
                clim=(0, 1),
                threshold=0.5,
                map=cmap,
            )
            volume = gfx.Volume(gfx.Geometry(grid=tex), material)
            volume.visible = False
            self._scene.add(volume)

            self._scale_volumes.append(dict(volume=volume, texture=tex, data=data))

        print(f"Prepared {len(self._scale_volumes)} scale volumes ({tw}³ textures)")

    # ---- UI layout --------------------------------------------------------

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        btn = QPushButton("Run LOD check")
        btn.clicked.connect(self._on_button_clicked)
        toolbar.addWidget(btn)

        self._toggle_vol_btn = QPushButton("Show volume")
        self._toggle_vol_btn.setCheckable(True)
        self._toggle_vol_btn.setEnabled(self._reference_volume is not None)
        self._toggle_vol_btn.toggled.connect(self._on_toggle_volume)
        toolbar.addWidget(self._toggle_vol_btn)

        self._show_chunks_cb = QPushButton("Show chunk wireframes")
        self._show_chunks_cb.setCheckable(True)
        self._show_chunks_cb.setChecked(True)
        self._show_chunks_cb.toggled.connect(self._on_toggle_chunk_wireframes)
        toolbar.addWidget(self._show_chunks_cb)

        self._status_label = QLabel("")
        toolbar.addWidget(self._status_label)
        toolbar.addStretch()

        layout.addLayout(toolbar)
        layout.addWidget(self._canvas)

    # ---- button handler ---------------------------------------------------

    def _on_button_clicked(self) -> None:
        """Manual LOD check with full debug output."""
        # --- Camera state (always printed) ------------------------------------
        cam_pos = get_camera_position_world(self._camera)
        view_dir = get_view_direction_world(self._camera)
        frustum_corners = get_frustum_corners_world(self._camera)
        inside = is_camera_inside(
            cam_pos, self._dataset.aabb_min, self._dataset.aabb_max
        )
        primary = select_primary_axis(
            view_dir, self._dataset.aabb_max - self._dataset.aabb_min
        )
        print("-" * 70)
        print(f"camera_pos:      {cam_pos}")
        print(f"view_direction:  {view_dir}")
        print(f"camera_inside:   {inside}")
        print(f"primary_axis:    {primary} ({'xyz'[primary]})")
        print(f"near corners:\n{frustum_corners[0]}")
        print(f"far corners:\n{frustum_corners[1]}")
        print("-" * 70)
        # ----------------------------------------------------------------------
        self._run_lod_pipeline(verbose=True)

    def _on_camera_settled(self) -> None:
        """Called by debounce timer after camera stops moving."""
        self._run_lod_pipeline(verbose=False)

    def _run_lod_pipeline(self, verbose: bool = False) -> None:
        """Run the full LOD pipeline and update visuals.

        Parameters
        ----------
        verbose : bool
            If True, print detailed result and timing output.
        """
        try:
            result = run_lod_check(self._dataset, self._camera)
        except ValueError as e:
            self._status_label.setText(f"Error: {e}")
            self._clear_dynamic_visuals()
            return

        if result is None:
            self._status_label.setText("No data in view.")
            self._clear_dynamic_visuals()
            return

        self._clear_dynamic_visuals()

        if verbose:
            print("=" * 70)
            print(
                f"union_vertices ({result.union_vertices.shape[0]} pts):\n{result.union_vertices}"
            )
            centroid = np.mean(result.union_vertices, axis=0)
            print(f"union centroid:  {centroid}")
            scale = self._dataset.scales[result.scale_idx]
            T = TEXTURE_WIDTH * scale.voxel_size
            print(f"scale_idx={result.scale_idx}  voxel_size={scale.voxel_size}  T={T}")
            print(f"texture_min:     {result.texture_min}")
            print(f"texture_max:     {result.texture_max}")
            print(f"texture_size:    {result.texture_max - result.texture_min}")
            print(f"visible_chunks:  {len(result.visible_chunk_indices)}")
            print("=" * 70)

            total = sum(v for k, v in result.timings.items() if not k.startswith("  "))
            print("--- timing (ms) ---")
            for step, elapsed in result.timings.items():
                print(f"  {step + ':':<28s} {elapsed * 1000:>7.2f}")
            print(f"  {'total pipeline:':<28s} {total * 1000:>7.2f}")
            print("---")

        # Frustum wireframe
        self._frustum_line = make_frustum_wireframe(
            get_frustum_corners_world(self._camera), color=FRUSTUM_COLOR
        )
        self._scene.add(self._frustum_line)

        # Texture AABB wireframe
        color = SCALE_COLORS[result.scale_idx]
        self._texture_line = make_box_wireframe(
            result.texture_min, result.texture_max, color=color
        )
        self._scene.add(self._texture_line)

        # Union vertices as points
        self._union_points = gfx.Points(
            gfx.Geometry(positions=result.union_vertices.astype(np.float32)),
            gfx.PointsMaterial(color="#ff00ff", size=12),
        )
        self._scene.add(self._union_points)

        # Visible chunk wireframes (same color as texture)
        if self._show_chunks_cb.isChecked():
            scale = self._dataset.scales[result.scale_idx]
            vis_corners = scale.chunk_corners[result.visible_chunk_indices]
            for i in range(len(vis_corners)):
                c_min = vis_corners[i, 0]  # corner 0 = (min, min, min)
                c_max = vis_corners[i, 7]  # corner 7 = (max, max, max)
                line = make_box_wireframe(c_min, c_max, color=color)
                self._scene.add(line)
                self._chunk_lines.append(line)

        # Load chunks into texture and show volume
        self._last_lod_result = result
        self._load_chunks_to_texture(result)

        # Status message
        scale = self._dataset.scales[result.scale_idx]
        T = TEXTURE_WIDTH * scale.voxel_size
        self._status_label.setText(
            f"Scale {result.scale_idx}  shape={scale.shape}  "
            f"voxel_size={scale.voxel_size:.1f}  T={T:.0f} world units  "
            f"primary_axis={'xyz'[result.primary_axis]}  "
            f"chunks={len(result.visible_chunk_indices)}"
        )

    # ---- volume toggle -----------------------------------------------------

    def _on_toggle_volume(self, checked: bool) -> None:
        if self._reference_volume is not None:
            self._reference_volume.visible = checked
            self._toggle_vol_btn.setText("Hide volume" if checked else "Show volume")

    # ---- chunk wireframe toggle --------------------------------------------

    def _on_toggle_chunk_wireframes(self, checked: bool) -> None:
        """Show or hide chunk wireframes.

        When turning on, if no lines exist yet (e.g. the checkbox was off
        during the last LOD run), redraw them from the cached result so the
        view is always up to date without re-running the full pipeline.
        """
        self._show_chunks_cb.setText(
            "Hide chunk wireframes" if checked else "Show chunk wireframes"
        )
        if checked:
            if not self._chunk_lines and self._last_lod_result is not None:
                # Wireframes were skipped on the last run — draw them now.
                result = self._last_lod_result
                color = SCALE_COLORS[result.scale_idx]
                scale = self._dataset.scales[result.scale_idx]
                vis_corners = scale.chunk_corners[result.visible_chunk_indices]
                for i in range(len(vis_corners)):
                    c_min = vis_corners[i, 0]
                    c_max = vis_corners[i, 7]
                    line = make_box_wireframe(c_min, c_max, color=color)
                    self._scene.add(line)
                    self._chunk_lines.append(line)
            else:
                for line in self._chunk_lines:
                    line.visible = True
        else:
            for line in self._chunk_lines:
                line.visible = False

    # ---- chunk loading (async) -----------------------------------------------

    def _load_chunks_to_texture(self, result: LodResult) -> None:
        """Plan chunk loads and schedule the async load task.

        Everything up to ``asyncio.ensure_future`` is synchronous and runs on
        the main thread before yielding to the event loop, so the scene is
        always in a consistent state even if the previous task was cancelled
        mid-flight.
        """
        if self._scale_volumes is None or self._ts_stores is None:
            return

        scale_idx = result.scale_idx
        scale = self._dataset.scales[scale_idx]
        vs = scale.voxel_size

        # Plan phase — pure arithmetic, no I/O (includes distance sort)
        t0 = time.perf_counter()
        requests, plan_timings = plan_chunk_loads(result, scale)
        plan_ms = (time.perf_counter() - t0) * 1000

        # --- Synchronous scene-graph reset (no await → not cancellable) -------
        # Hide all scale volumes and zero the backing array for this scale so
        # the volume starts visually empty while chunks stream in.
        for entry in self._scale_volumes:
            entry["volume"].visible = False
        sv = self._scale_volumes[scale_idx]
        sv["data"][:] = 0
        sv["texture"].update_range((0, 0, 0), sv["texture"].size)

        # Position and show the (empty) volume immediately.
        volume = sv["volume"]
        volume.local.scale = (vs, vs, vs)
        offset = vs * 0.5
        volume.local.position = (
            result.texture_min[0] + offset,
            result.texture_min[1] + offset,
            result.texture_min[2] + offset,
        )
        volume.visible = True
        # -----------------------------------------------------------------------

        nearest = requests[0].chunk_index if requests else None
        nearest_world = (
            scale.chunk_corners[nearest, 0] + scale.chunk_world * 0.5
            if nearest is not None
            else None
        )
        print(
            f"Plan: {len(requests)} chunks  |  "
            f"sort {plan_timings['sort'] * 1000:.3f} ms  "
            f"slice_build {plan_timings['slice_build'] * 1000:.3f} ms  "
            f"total {plan_ms:.3f} ms"
        )
        if nearest_world is not None:
            nearest_dist = float(np.linalg.norm(nearest_world - result.camera_pos))
            print(
                f"  nearest chunk: index={nearest}  "
                f"centre={nearest_world.tolist()}  "
                f"dist={nearest_dist:.1f} world units"
            )

        # Cancel any previous load task, then schedule the new one.
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
        self._load_task = asyncio.ensure_future(
            self._do_load_chunks(result, requests, scale_idx)
        )

    async def _do_load_chunks(
        self,
        result: LodResult,
        requests: list[ChunkRequest],
        scale_idx: int,
    ) -> None:
        """Async task — stream chunk reads from tensorstore into the texture.

        Each ``await store[...].read()`` yields control to the Qt/asyncio event
        loop so the UI and renderer stay live between reads.  The task is fully
        cancellable at every chunk boundary; ``CancelledError`` is re-raised so
        asyncio marks the task cancelled correctly.

        ``texture.update_range`` marks the dirty region; the actual GPU memcpy
        is deferred until the next ``renderer.render()`` call inside
        ``_animate``, which fires every vsync via ``update_mode="continuous"``.
        """
        sv = self._scale_volumes[scale_idx]
        data_buf = sv["data"]
        texture = sv["texture"]
        store = self._ts_stores[scale_idx]
        t_start = time.perf_counter()

        try:
            for i, req in enumerate(requests):
                # Await one chunk read — yields to Qt/asyncio event loop.
                chunk = np.asarray(
                    await store[req.zarr_slice].read(), dtype=np.float32
                )

                # Write into the pre-allocated backing array in-place.
                data_buf[req.texture_slice] = chunk

                # Mark the dirty sub-region.  pygfx update_range uses
                # (x, y, z) order — the reverse of numpy's (z, y, x) layout.
                zs, ys, xs = req.texture_slice
                texture.update_range(
                    offset=(xs.start, ys.start, zs.start),
                    size=(xs.stop - xs.start, ys.stop - ys.start, zs.stop - zs.start),
                )

        except asyncio.CancelledError:
            raise  # mandatory: let asyncio mark the task cancelled

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        print(
            f"All {len(requests)} chunks loaded in {elapsed_ms:.1f} ms"
        )

    # ---- helpers ----------------------------------------------------------

    def _clear_dynamic_visuals(self) -> None:
        # Cancel any in-flight async load task
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()

        self._last_lod_result = None

        if self._frustum_line is not None:
            self._scene.remove(self._frustum_line)
            self._frustum_line = None
        if self._texture_line is not None:
            self._scene.remove(self._texture_line)
            self._texture_line = None
        if self._union_points is not None:
            self._scene.remove(self._union_points)
            self._union_points = None
        for line in self._chunk_lines:
            self._scene.remove(line)
        self._chunk_lines.clear()
        # Hide all scale volumes
        if self._scale_volumes is not None:
            for entry in self._scale_volumes:
                entry["volume"].visible = False

    def closeEvent(self, event) -> None:
        self._debounce_timer.stop()
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
        super().closeEvent(event)

    def _animate(self) -> None:
        self._renderer.render(self._scene, self._camera)

        # Detect camera movement: cancel any in-flight load immediately so
        # stale chunks stop arriving, then (re)start the debounce timer so
        # the LOD pipeline re-runs once the camera settles.
        current_matrix = np.asarray(self._camera.world.matrix, dtype=np.float64)
        if not np.array_equal(current_matrix, self._last_camera_matrix):
            self._last_camera_matrix = current_matrix.copy()
            if self._load_task is not None and not self._load_task.done():
                self._load_task.cancel()
            self._debounce_timer.start()  # restarts if already running


# ---------------------------------------------------------------------------
# Section 9 — Test-data generation
# ---------------------------------------------------------------------------


def make_test_zarr(zarr_path: pathlib.Path) -> None:
    """Generate the multiscale zarr store used by this script.

    Creates three scale levels matching ``ZARR_SCALE_NAMES`` / ``SCALES``:

    ============  ===========  ==========  ==================
    Name          Shape        voxel_size  Uncompressed size
    ============  ===========  ==========  ==================
    s0            1024³        1.0         4 GiB
    s1             512³        2.0         512 MiB
    s2             256³        4.0          64 MiB
    ============  ===========  ==========  ==================

    Data is 3-D binary blobs downsampled by 2× per level using
    nearest-neighbour interpolation.  Requires ``scikit-image``
    (``pip install scikit-image`` or add it to the uv script header).

    Parameters
    ----------
    zarr_path : pathlib.Path
        Destination directory for the zarr store.  Created if absent.
        Existing scale directories are overwritten.
    """
    try:
        from skimage.data import binary_blobs
        from skimage.transform import resize
    except ImportError:
        raise SystemExit(
            "scikit-image is required for --make-file.\n"
            "Install it with:  pip install scikit-image\n"
            "or add 'scikit-image' to the uv script dependencies header."
        )

    base_length = SCALES[0]["shape"][0]  # 1024

    print(f"Generating {base_length}³ binary blobs…")
    image_0 = binary_blobs(
        length=base_length,
        n_dim=3,
        blob_size_fraction=0.2,
        volume_fraction=0.2,
    ).astype("float32")

    for scale_name, scale_info in zip(ZARR_SCALE_NAMES, SCALES):
        edge = scale_info["shape"][0]
        if edge == base_length:
            data = image_0
        else:
            data = resize(image_0, (edge, edge, edge), order=0).astype("float32")

        spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(zarr_path / scale_name),
            },
            "metadata": {
                "dtype": "<f4",
                "shape": list(data.shape),
                "chunks": list(CHUNK_SHAPE),
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        store[...].write(data).result()
        print(
            f"  Wrote {scale_name}: shape={data.shape}  "
            f"({data.nbytes // 2 ** 20} MiB uncompressed)"
        )

    print(f"Done → {zarr_path}")


# ---------------------------------------------------------------------------
# Section 10 — Entry point
# ---------------------------------------------------------------------------


async def async_main(zarr_path: pathlib.Path) -> None:
    """Top-level coroutine driven by PySide6.QtAsyncio.

    Creates the window synchronously, then suspends until the application
    closes.  Chunk-load tasks are scheduled inside ``LodDebugApp`` via
    ``asyncio.ensure_future`` as the user interacts with the scene.

    Parameters
    ----------
    zarr_path : pathlib.Path
        Path to the multiscale zarr store.  Passed via ``--zarr-path``
        (default: ``multiscale_blobs.zarr`` next to this script).
    """
    app = QApplication.instance()
    window = LodDebugApp(zarr_path=zarr_path)
    window.resize(1200, 800)
    window.setWindowTitle("LOD Debug")
    window.show()

    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(
        description="LOD debug visualiser — async tensorstore edition.",
    )
    parser.add_argument(
        "zarr_path",
        nargs="?",
        default=None,
        type=pathlib.Path,
        metavar="ZARR_PATH",
        help=(
            "Path to the multiscale zarr store "
            f"(default: {ZARR_PATH.name} next to this script)."
        ),
    )
    parser.add_argument(
        "--make-file",
        action="store_true",
        help=(
            "Generate the test zarr store at ZARR_PATH using tensorstore "
            "and exit without launching the viewer.  "
            "Requires scikit-image."
        ),
    )
    args = parser.parse_args()

    # Resolve zarr path: CLI argument overrides the module-level default.
    zarr_path: pathlib.Path = args.zarr_path if args.zarr_path is not None else ZARR_PATH

    if args.make_file:
        make_test_zarr(zarr_path)
        sys.exit(0)

    if not zarr_path.exists():
        print(f"Error: zarr store not found at '{zarr_path}'")
        print("Run with --make-file to generate it:")
        print(f"    uv run --script {pathlib.Path(__file__).name} --make-file")
        sys.exit(1)

    # Pass only the program name to Qt — argparse flags must not reach Qt.
    app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(zarr_path), handle_sigint=True)