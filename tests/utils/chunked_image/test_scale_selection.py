"""Tests for mip-map-based scale selection in ChunkedImageStore._select_scale.

The criterion: return the finest scale i where
    2^i >= world_per_pixel * lod_bias
where, for a perspective camera,
    world_per_pixel = (near_h * dist_to_data / near_dist) / canvas_height_px

For a perspective frustum:
  - near_h / far_h = near_dist / far_dist  (perspective ratio)
  - near_dist = d_near_far / (far_h/near_h - 1)
  - camera_pos = near_center - near_dist * view_dir

near_corners order: left-bottom, right-bottom, right-top, left-top.
"""

import numpy as np
import pytest

from cellier.utils.chunked_image._data_classes import (
    TextureConfiguration,
    ViewParameters,
)
from cellier.utils.chunked_image._multiscale_image_model import MultiscaleImageModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FOV_DEG = 50.0
NEAR_DIST = 0.1
FAR_DIST = 2000.0
CANVAS_H = 800
CANVAS_W = 1200


def _frustum_heights(dist: float) -> float:
    """World-space height of a perspective frustum plane at ``dist`` from camera."""
    return 2.0 * dist * np.tan(np.radians(FOV_DEG / 2.0))


def make_perspective_frustum(
    cam_pos: np.ndarray,
    view_dir: np.ndarray,
    near: float = NEAR_DIST,
    far: float = FAR_DIST,
    fov_deg: float = FOV_DEG,
) -> np.ndarray:
    """Return (2, 4, 3) frustum corners for a symmetric perspective camera.

    Corner order per plane: left-bottom, right-bottom, right-top, left-top.
    The 'height' axis is along an arbitrary perpendicular (world-y projected
    onto the plane perpendicular to view_dir); width axis is the cross-product.

    Parameters
    ----------
    cam_pos : array (3,)
    view_dir : array (3,)  unit vector
    near, far : float  clipping distances
    fov_deg : float  vertical field of view in degrees
    """
    vd = np.asarray(view_dir, dtype=np.float64)
    vd /= np.linalg.norm(vd)

    # Build a stable up-vector perpendicular to view_dir
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(vd, world_up)) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(vd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, vd)
    up /= np.linalg.norm(up)

    def _plane(dist):
        half_h = dist * np.tan(np.radians(fov_deg / 2.0))
        half_w = half_h  # square frustum for simplicity
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
    canvas_h: int = CANVAS_H,
    canvas_w: int = CANVAS_W,
    near: float = NEAR_DIST,
    far: float = FAR_DIST,
    fov_deg: float = FOV_DEG,
) -> ViewParameters:
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
        canvas_size=(canvas_w, canvas_h),
    )


def make_multiscale(
    n_scales: int = 3, shape: tuple = (256, 256, 256)
) -> MultiscaleImageModel:
    """Return an n-scale model: factors [1, 2, 4, ...], given shape, chunk 64^3."""
    factors = [2.0**i for i in range(n_scales)]
    return MultiscaleImageModel.from_shape_and_scales(
        shape=shape,
        chunk_shapes=[(64, 64, 64)] * n_scales,
        downscale_factors=factors,
    )


def make_store(lod_bias: float = 1.0, n_scales: int = 3, shape=(256, 256, 256)):
    """Return a ChunkedImageStore with a dummy store path."""
    from cellier.models.data_stores.chunked_image import ChunkedImageStore

    return ChunkedImageStore(
        multiscale_model=make_multiscale(n_scales, shape=shape),
        store_paths=["dummy"],
        texture_config=TextureConfiguration(texture_width=256),
        lod_bias=lod_bias,
    )


def _world_per_pixel(cam_pos, view_dir, data_center, fov_deg, canvas_h):
    """Expected world_per_pixel for a perspective camera, computed independently."""
    dist_to_data = float(np.dot(data_center - cam_pos, view_dir))
    world_h = 2.0 * dist_to_data * np.tan(np.radians(fov_deg / 2.0))
    return world_h / canvas_h


# ---------------------------------------------------------------------------
# Basic criterion tests with realistic perspective frustums
# ---------------------------------------------------------------------------

# Data center for the default 256^3 volume
_DATA_CENTER = np.array([128.0, 128.0, 128.0])


def _expected_scale(wpp, lod_bias=1.0, n_scales=3):
    """Return the expected scale index for given world_per_pixel."""
    for i in range(n_scales):
        if 2.0**i >= wpp * lod_bias:
            return i
    return n_scales - 1


def test_fine_scale_when_close_to_data():
    """Camera very close → world_per_pixel < 1 → scale 0 selected."""
    # Place camera 10 units in front of the data center
    view_dir = np.array([0.0, 0.0, 1.0])
    cam_pos = _DATA_CENTER - 10.0 * view_dir
    store = make_store()
    vp = make_view_params_perspective(cam_pos, view_dir)

    wpp = _world_per_pixel(cam_pos, view_dir, _DATA_CENTER, FOV_DEG, CANVAS_H)
    expected = _expected_scale(wpp)
    assert store._select_scale(vp) == expected


def test_coarse_scale_when_far_from_data():
    """Camera very far → world_per_pixel >> 1 → coarsest scale selected."""
    view_dir = np.array([0.0, 0.0, 1.0])
    cam_pos = _DATA_CENTER - 5000.0 * view_dir
    store = make_store()
    vp = make_view_params_perspective(cam_pos, view_dir)

    wpp = _world_per_pixel(cam_pos, view_dir, _DATA_CENTER, FOV_DEG, CANVAS_H)
    expected = _expected_scale(wpp)
    assert store._select_scale(vp) == expected


def test_scale_transitions_with_distance():
    """Scale index increases (coarser) as camera moves further away."""
    view_dir = np.array([0.0, 0.0, 1.0])
    store = make_store(n_scales=4)
    prev_scale = 0
    for dist in [5.0, 50.0, 500.0, 5000.0]:
        cam_pos = _DATA_CENTER - dist * view_dir
        vp = make_view_params_perspective(cam_pos, view_dir)
        scale = store._select_scale(vp)
        assert scale >= prev_scale, (
            f"Scale should be non-decreasing with distance; "
            f"dist={dist} → scale={scale}, prev={prev_scale}"
        )
        prev_scale = scale


# ---------------------------------------------------------------------------
# lod_bias shifts the threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lod_bias", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_lod_bias_shifts_scale(lod_bias):
    """Higher lod_bias picks a coarser (or equal) scale."""
    view_dir = np.array([0.0, 0.0, 1.0])
    cam_pos = _DATA_CENTER - 200.0 * view_dir
    vp = make_view_params_perspective(cam_pos, view_dir)

    wpp = _world_per_pixel(cam_pos, view_dir, _DATA_CENTER, FOV_DEG, CANVAS_H)
    expected = _expected_scale(wpp, lod_bias=lod_bias)

    store = make_store(lod_bias=lod_bias)
    assert store._select_scale(vp) == expected


def test_lower_lod_bias_selects_finer_or_equal():
    """lod_bias=0.5 selects same or finer scale than lod_bias=1.0."""
    view_dir = np.array([0.0, 0.0, 1.0])
    cam_pos = _DATA_CENTER - 200.0 * view_dir
    vp = make_view_params_perspective(cam_pos, view_dir)

    scale_neutral = make_store(lod_bias=1.0)._select_scale(vp)
    scale_fine = make_store(lod_bias=0.5)._select_scale(vp)
    assert scale_fine <= scale_neutral


def test_higher_lod_bias_selects_coarser_or_equal():
    """lod_bias=2.0 selects same or coarser scale than lod_bias=1.0."""
    view_dir = np.array([0.0, 0.0, 1.0])
    cam_pos = _DATA_CENTER - 200.0 * view_dir
    vp = make_view_params_perspective(cam_pos, view_dir)

    scale_neutral = make_store(lod_bias=1.0)._select_scale(vp)
    scale_coarse = make_store(lod_bias=2.0)._select_scale(vp)
    assert scale_coarse >= scale_neutral


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_default_canvas_size_returns_coarsest():
    """canvas_size=(1,1) placeholder → coarsest fallback."""
    store = make_store()
    vp = ViewParameters(
        frustum_corners=np.zeros((2, 4, 3), dtype=np.float32),
        view_direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        near_plane_center=np.zeros(3, dtype=np.float32),
        # canvas_size defaults to (1, 1)
    )
    assert store._select_scale(vp) == 2  # coarsest


def test_single_scale_always_returns_zero():
    """With only one scale, always return 0 regardless of zoom."""
    store = make_store(n_scales=1)
    view_dir = np.array([0.0, 0.0, 1.0])
    cam_pos = _DATA_CENTER - 10000.0 * view_dir
    vp = make_view_params_perspective(cam_pos, view_dir)
    assert store._select_scale(vp) == 0


def test_view_direction_invariance():
    """Scale selection should give the same result regardless of view direction."""
    store = make_store()
    dist = 200.0
    results = []
    for view_dir in [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]:
        cam_pos = _DATA_CENTER - dist * view_dir
        vp = make_view_params_perspective(cam_pos, view_dir)
        results.append(store._select_scale(vp))
    assert len(set(results)) == 1, f"Expected all the same scale, got {results}"
