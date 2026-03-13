"""Tests for coverage-based scale selection in ChunkedImageStore._select_scale.

The criterion: return the finest scale i where

    texture_width * 2^i >= in_view_width * lod_bias

where ``in_view_width`` is the perpendicular extent of the intersection of the
near data face with the frustum (in scale_0 coordinates), and ``lod_bias``
shifts the threshold (>1 → coarser, <1 → finer).

Each frustum edge (near_corner[i] → far_corner[i]) is intersected individually
with the near-data-face plane; the bounding box of those four intersection
points, clamped to the data AABB, gives the width of data visible on the near
data face.  This avoids far-clipping-plane domination while correctly measuring
"how much data is the camera looking at?"
"""

import numpy as np
import pytest

from cellier.utils.chunked_image._chunk_selection import compute_in_view_aabb
from cellier.utils.chunked_image._data_classes import (
    TextureConfiguration,
    ViewParameters,
)
from cellier.utils.chunked_image._multiscale_image_model import MultiscaleImageModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A 1024^3 dataset: with texture_width=256, scale transitions are visible as
# the frustum covers increasing fractions of the data.
_DATA_CENTER = np.array([512.0, 512.0, 512.0])
_DATA_SHAPE = (1024, 1024, 1024)

# Data AABB in world space (default identity world_transform)
_DATA_BB_MIN = np.array([0.0, 0.0, 0.0])
_DATA_BB_MAX = np.array([1024.0, 1024.0, 1024.0])

FOV_DEG = 50.0
TEXTURE_WIDTH = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_perspective_frustum(
    cam_pos: np.ndarray,
    view_dir: np.ndarray,
    near: float = 1.0,
    far: float = 200.0,
    fov_deg: float = FOV_DEG,
) -> np.ndarray:
    """Return (2, 4, 3) frustum corners for a symmetric perspective camera.

    Corner order per plane: left-bottom, right-bottom, right-top, left-top.
    """
    vd = np.asarray(view_dir, dtype=np.float64)
    vd /= np.linalg.norm(vd)

    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(vd, world_up)) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(vd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, vd)
    up /= np.linalg.norm(up)

    def _plane(dist):
        half_h = dist * np.tan(np.radians(fov_deg / 2.0))
        half_w = half_h
        center = cam_pos + dist * vd
        lb = center - half_w * right - half_h * up
        rb = center + half_w * right - half_h * up
        rt = center + half_w * right + half_h * up
        lt = center - half_w * right + half_h * up
        return np.array([lb, rb, rt, lt], dtype=np.float32)

    return np.stack([_plane(near), _plane(far)])


def make_view_params_perspective(
    cam_pos: np.ndarray,
    view_dir: np.ndarray,
    near: float = 1.0,
    far: float = 200.0,
    fov_deg: float = FOV_DEG,
) -> ViewParameters:
    """Build ViewParameters from a perspective camera specification."""
    frustum = make_perspective_frustum(cam_pos, view_dir, near, far, fov_deg)
    near_corners = frustum[0]
    near_center = near_corners.mean(axis=0)
    far_center = frustum[1].mean(axis=0)
    raw_dir = far_center - near_center
    vd = raw_dir / np.linalg.norm(raw_dir)
    return ViewParameters(
        frustum_corners=frustum,
        view_direction=vd.astype(np.float32),
        near_plane_center=near_center,
    )


def make_multiscale(
    n_scales: int = 3, shape: tuple = _DATA_SHAPE
) -> MultiscaleImageModel:
    """Return an n-scale model: factors [1, 2, 4, ...], chunk 64^3."""
    factors = [2.0**i for i in range(n_scales)]
    return MultiscaleImageModel.from_shape_and_scales(
        shape=shape,
        chunk_shapes=[(64, 64, 64)] * n_scales,
        downscale_factors=factors,
    )


def make_store(lod_bias: float = 1.0, n_scales: int = 3, shape: tuple = _DATA_SHAPE):
    """Return a ChunkedImageStore with the given parameters."""
    from cellier.models.data_stores.chunked_image import ChunkedImageStore

    return ChunkedImageStore(
        multiscale_model=make_multiscale(n_scales, shape=shape),
        store_paths=["dummy"],
        texture_config=TextureConfiguration(texture_width=TEXTURE_WIDTH),
        lod_bias=lod_bias,
    )


def _compute_in_view_width(frustum_corners, data_bb_min, data_bb_max, view_dir):
    """Compute in_view_width via per-corner frustum-edge / near-data-face intersection.

    Mirrors the algorithm in ``ChunkedImageStore._select_scale`` so that tests
    can independently derive the expected width and compare against the
    implementation.

    Each frustum edge (near_corner[i] → far_corner[i]) is intersected with
    the near data face plane individually.  The perpendicular bounding box of
    those four intersection points, clamped to the data AABB, gives the width
    of data visible on the near data face.
    """
    # Quick overlap check — same guard used in _select_scale.
    if compute_in_view_aabb(frustum_corners, data_bb_min, data_bb_max) is None:
        return None

    view_dir = np.asarray(view_dir, dtype=float)
    primary_axis = int(np.argmax(np.abs(view_dir)))
    perp_axes = [ax for ax in range(3) if ax != primary_axis]

    near_corners = np.asarray(frustum_corners[0], dtype=float)
    far_corners = np.asarray(frustum_corners[1], dtype=float)
    edge_deltas = far_corners - near_corners  # shape (4, 3)

    data_face = (
        data_bb_min[primary_axis]
        if view_dir[primary_axis] >= 0
        else data_bb_max[primary_axis]
    )

    denoms = edge_deltas[:, primary_axis]
    ts = np.where(
        np.abs(denoms) < 1e-10,
        0.0,
        np.clip(
            (data_face - near_corners[:, primary_axis]) / denoms,
            0.0,
            1.0,
        ),
    )
    intersections = near_corners + ts[:, np.newaxis] * edge_deltas

    cmin = np.maximum(intersections.min(axis=0), data_bb_min)
    cmax = np.minimum(intersections.max(axis=0), data_bb_max)
    extents = cmax - cmin
    return float(max(extents[ax] for ax in perp_axes))


def _expected_scale_coverage(
    in_view_width: float,
    texture_width: int = TEXTURE_WIDTH,
    lod_bias: float = 1.0,
    n_scales: int = 3,
) -> int:
    """Expected scale for the coverage criterion (finest N satisfying coverage)."""
    for i in range(n_scales):
        if texture_width * 2.0**i >= in_view_width * lod_bias:
            return i
    return n_scales - 1


# ---------------------------------------------------------------------------
# Basic criterion tests with realistic perspective frustums
# ---------------------------------------------------------------------------


def test_fine_scale_when_close_to_data():
    """Narrow frustum (small far dist) → small in_view_width → finest scale."""
    # Camera 10 units outside data (at z=-10), far plane at z=10 (inside data).
    # far=20 → far plane at z=10, half_h ≈ 9.3 → in_view_width ≈ 18.7 → scale 0.
    view_dir = np.array([0.0, 0.0, 1.0])
    cam_pos = np.array([512.0, 512.0, -10.0])

    vp = make_view_params_perspective(cam_pos, view_dir, near=1.0, far=20.0)
    store = make_store()

    width = _compute_in_view_width(
        vp.frustum_corners, _DATA_BB_MIN, _DATA_BB_MAX, view_dir
    )
    assert width is not None, "Expected frustum to overlap data."
    expected = _expected_scale_coverage(width)
    assert store._select_scale(vp) == expected
    assert expected == 0, f"Expected finest scale 0, got {expected} (width={width:.1f})"


def test_coarse_scale_when_far_from_data():
    """Camera far from data face → large cross-section → coarsest scale.

    With the camera 2000 units outside the data the frustum cross-section at
    the data face is ~2*2000*tan(25 deg) ~ 1863 voxels, which exceeds the data
    extent (1024) and is clamped to it.  The coverage criterion then selects
    scale 2 (texture_width*4 = 1024 >= 1024).
    """
    view_dir = np.array([0.0, 0.0, 1.0])
    # Camera 2000 units outside the data face (z=0 → camera at z=-2000).
    cam_pos = np.array([512.0, 512.0, -2000.0])

    vp = make_view_params_perspective(cam_pos, view_dir, near=1.0, far=5000.0)
    store = make_store()

    width = _compute_in_view_width(
        vp.frustum_corners, _DATA_BB_MIN, _DATA_BB_MAX, view_dir
    )
    assert width is not None
    expected = _expected_scale_coverage(width)
    assert store._select_scale(vp) == expected
    assert (
        expected == 2
    ), f"Expected coarsest scale 2, got {expected} (width={width:.1f})"


def test_scale_transitions_with_increasing_distance():
    """Scale is non-decreasing as camera distance from the data increases.

    The frustum cross-section at the data face scales with camera distance, so
    a camera further from the data needs a coarser LOD.

    Distances [100, 400, 700, 2000] give cross-sections ≈ [93, 373, 653, 1024]
    → scales [0, 1, 2, 2] (non-decreasing), for texture_width=256, n_scales=3.
    """
    view_dir = np.array([0.0, 0.0, 1.0])
    store = make_store(n_scales=3)

    prev_scale = 0
    for dist in [100.0, 400.0, 700.0, 2000.0]:
        cam_pos = np.array([512.0, 512.0, -dist])
        vp = make_view_params_perspective(cam_pos, view_dir, near=1.0, far=5000.0)
        scale = store._select_scale(vp)
        assert scale >= prev_scale, (
            f"Scale should be non-decreasing with camera distance; "
            f"dist={dist} → scale={scale}, prev={prev_scale}"
        )
        prev_scale = scale


# ---------------------------------------------------------------------------
# lod_bias shifts the threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lod_bias", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_lod_bias_matches_expected_coverage(lod_bias):
    """_select_scale matches _expected_scale_coverage for any lod_bias."""
    view_dir = np.array([0.0, 0.0, 1.0])
    cam_pos = np.array([512.0, 512.0, -100.0])
    vp = make_view_params_perspective(cam_pos, view_dir, near=1.0, far=500.0)

    width = _compute_in_view_width(
        vp.frustum_corners, _DATA_BB_MIN, _DATA_BB_MAX, view_dir
    )
    assert width is not None
    expected = _expected_scale_coverage(width, lod_bias=lod_bias)

    store = make_store(lod_bias=lod_bias)
    assert store._select_scale(vp) == expected


def test_lower_lod_bias_selects_finer_or_equal():
    """lod_bias=0.5 selects same or finer scale than lod_bias=1.0."""
    view_dir = np.array([0.0, 0.0, 1.0])
    cam_pos = np.array([512.0, 512.0, -100.0])
    vp = make_view_params_perspective(cam_pos, view_dir, near=1.0, far=500.0)

    scale_neutral = make_store(lod_bias=1.0)._select_scale(vp)
    scale_fine = make_store(lod_bias=0.5)._select_scale(vp)
    assert scale_fine <= scale_neutral


def test_higher_lod_bias_selects_coarser_or_equal():
    """lod_bias=2.0 selects same or coarser scale than lod_bias=1.0."""
    view_dir = np.array([0.0, 0.0, 1.0])
    cam_pos = np.array([512.0, 512.0, -100.0])
    vp = make_view_params_perspective(cam_pos, view_dir, near=1.0, far=500.0)

    scale_neutral = make_store(lod_bias=1.0)._select_scale(vp)
    scale_coarse = make_store(lod_bias=2.0)._select_scale(vp)
    assert scale_coarse >= scale_neutral


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_degenerate_frustum_returns_coarsest():
    """All-zero frustum corners produce no in-view AABB → coarsest fallback."""
    store = make_store()
    vp = ViewParameters(
        frustum_corners=np.zeros((2, 4, 3), dtype=np.float32),
        view_direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        near_plane_center=np.zeros(3, dtype=np.float32),
    )
    assert store._select_scale(vp) == 2  # coarsest (n_scales - 1)


def test_frustum_outside_data_returns_coarsest():
    """Frustum entirely outside the data extent → coarsest fallback."""
    store = make_store()
    # Place a tiny frustum far beyond the data (data is z=[0,1024], frustum at z=5000+)
    view_dir = np.array([0.0, 0.0, 1.0])
    cam_pos = np.array([512.0, 512.0, 5000.0])
    vp = make_view_params_perspective(cam_pos, view_dir, near=1.0, far=10.0)
    # Frustum z=[5001,5010], data z=[0,1024] → no overlap
    assert store._select_scale(vp) == 2  # coarsest


def test_single_scale_always_returns_zero():
    """With only one scale, always return 0 regardless of frustum size."""
    store = make_store(n_scales=1)
    view_dir = np.array([0.0, 0.0, 1.0])
    # Even with a huge frustum covering the full data
    cam_pos = np.array([512.0, 512.0, -100.0])
    vp = make_view_params_perspective(cam_pos, view_dir, near=1.0, far=5000.0)
    assert store._select_scale(vp) == 0


def test_view_direction_invariance():
    """Same far distance along each axis gives the same in_view_width and scale."""
    store = make_store()
    far = 5000.0  # large enough to cover full data in all directions
    results = []
    for view_dir in [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]:
        cam_pos = _DATA_CENTER - 200.0 * view_dir
        vp = make_view_params_perspective(cam_pos, view_dir, near=1.0, far=far)
        results.append(store._select_scale(vp))
    assert (
        len(set(results)) == 1
    ), f"Expected same scale for all axis-aligned view directions, got {results}"
