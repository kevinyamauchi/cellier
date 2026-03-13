"""Tests for mip-map-based scale selection in ChunkedImageStore._select_scale.

The criterion: return the finest scale i where
    2^i >= world_per_pixel * lod_bias
where
    world_per_pixel = near_plane_height_world / canvas_height_px

near_plane_height_world is computed as norm(corners[3] - corners[0]) of the
near-plane face (corners order: left-bottom, right-bottom, right-top, left-top).
"""

import numpy as np
import pytest

from cellier.utils.chunked_image._data_classes import (
    TextureConfiguration,
    ViewParameters,
)
from cellier.utils.chunked_image._multiscale_image_model import MultiscaleImageModel


def make_multiscale(n_scales: int = 3) -> MultiscaleImageModel:
    """Return a 3-scale model: factors [1, 2, 4], shape 256^3, chunk 64^3."""
    factors = [2.0**i for i in range(n_scales)]
    return MultiscaleImageModel.from_shape_and_scales(
        shape=(256, 256, 256),
        chunk_shapes=[(64, 64, 64)] * n_scales,
        downscale_factors=factors,
    )


def make_store(lod_bias: float = 1.0, n_scales: int = 3):
    """Return a ChunkedImageStore with a dummy store path."""
    from cellier.models.data_stores.chunked_image import ChunkedImageStore

    return ChunkedImageStore(
        multiscale_model=make_multiscale(n_scales),
        store_paths=["dummy"],
        texture_config=TextureConfiguration(texture_width=256),
        lod_bias=lod_bias,
    )


def make_view_params(
    near_plane_height: float,
    canvas_height: int,
    canvas_width: int = 512,
) -> ViewParameters:
    """Construct ViewParameters with the given near-plane world height.

    The near plane is axis-aligned at z=0, spanning [0, near_plane_height]
    in y and [0, near_plane_height] in x.  Corner order matches the docstring:
    left-bottom, right-bottom, right-top, left-top.
    """
    h = near_plane_height
    near = np.array(
        [
            [0.0, 0.0, 0.0],  # left-bottom
            [0.0, 0.0, h],  # right-bottom
            [0.0, h, h],  # right-top
            [0.0, h, 0.0],  # left-top
        ],
        dtype=np.float32,
    )
    far = near.copy()
    far[:, 0] = 100.0  # shift far plane along z
    return ViewParameters(
        frustum_corners=np.stack([near, far]),
        view_direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        near_plane_center=near.mean(axis=0),
        canvas_size=(canvas_width, canvas_height),
    )


# ---------------------------------------------------------------------------
# Basic criterion: world_per_pixel = near_plane_h / canvas_h
# Scale i is selected when 2^i >= world_per_pixel * lod_bias
# ---------------------------------------------------------------------------


def test_fine_scale_when_zoomed_in():
    """world_per_pixel=1.0 → scale 0 (finest) selected."""
    store = make_store()
    vp = make_view_params(near_plane_height=256.0, canvas_height=256)
    # world_per_pixel = 256/256 = 1.0; 2^0=1 >= 1.0 → scale 0
    assert store._select_scale(vp) == 0


def test_medium_scale_at_medium_zoom():
    """world_per_pixel=2.0 → scale 1 selected."""
    store = make_store()
    vp = make_view_params(near_plane_height=512.0, canvas_height=256)
    # world_per_pixel = 512/256 = 2.0; 2^0=1 < 2.0, 2^1=2 >= 2.0 → scale 1
    assert store._select_scale(vp) == 1


def test_coarse_scale_when_zoomed_out():
    """world_per_pixel=4.0 → scale 2 (coarsest) selected."""
    store = make_store()
    vp = make_view_params(near_plane_height=1024.0, canvas_height=256)
    # world_per_pixel = 1024/256 = 4.0; 2^2=4 >= 4.0 → scale 2
    assert store._select_scale(vp) == 2


def test_fallback_to_coarsest_when_beyond_range():
    """world_per_pixel > max voxel_size → coarsest scale returned."""
    store = make_store()
    vp = make_view_params(near_plane_height=4096.0, canvas_height=256)
    # world_per_pixel = 16.0; no scale satisfies 2^i >= 16 for i in {0,1,2}
    assert store._select_scale(vp) == 2


# ---------------------------------------------------------------------------
# lod_bias shifts the selected scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lod_bias, expected_scale",
    [
        (0.5, 0),  # 2^i >= 2.0*0.5=1.0 → scale 0 (finer)
        (1.0, 1),  # 2^i >= 2.0*1.0=2.0 → scale 1 (neutral)
        (2.0, 2),  # 2^i >= 2.0*2.0=4.0 → scale 2 (coarser)
    ],
)
def test_lod_bias_shifts_scale(lod_bias, expected_scale):
    """lod_bias < 1 → finer; = 1 → neutral; > 1 → coarser."""
    store = make_store(lod_bias=lod_bias)
    vp = make_view_params(near_plane_height=512.0, canvas_height=256)
    # base world_per_pixel = 2.0
    assert store._select_scale(vp) == expected_scale


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
    vp = make_view_params(near_plane_height=10000.0, canvas_height=256)
    assert store._select_scale(vp) == 0
