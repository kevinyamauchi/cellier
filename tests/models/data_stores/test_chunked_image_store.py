"""Tests for ChunkedImageStore (Phase D).

All tests use an in-memory zarr array so no disk I/O occurs.  The shared
thread pool comes from a real AsynchronousDataSlicer (cheap to construct).
The ChunkManager cache is pre-populated where needed to make cache-hit tests
fully synchronous.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from cellier.models.data_stores.chunked_image import ChunkedImageStore
from cellier.slicer.slicer import AsynchronousDataSlicer
from cellier.transform import AffineTransform
from cellier.types import (
    ChunkedDataResponse,
    ChunkedSelectedRegion,
    TilingMethod,
)
from cellier.utils.chunked_image._data_classes import (
    TextureConfiguration,
    ViewParameters,
)
from cellier.utils.chunked_image._multiscale_image_model import MultiscaleImageModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHAPE = (128, 128, 128)
_CHUNK = (32, 32, 32)
_TEXTURE_WIDTH = 64
_VISUAL_ID = "visual_001"
_SCENE_ID = "scene_001"


def _make_frustum(
    z_near: float,
    z_far: float,
    y_min: float,
    y_max: float,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Return (2, 4, 3) frustum corners (z, y, x) for a +z-looking camera."""
    return np.array(
        [
            [
                [z_near, y_min, x_min],
                [z_near, y_min, x_max],
                [z_near, y_max, x_max],
                [z_near, y_max, x_min],
            ],
            [
                [z_far, y_min, x_min],
                [z_far, y_min, x_max],
                [z_far, y_max, x_max],
                [z_far, y_max, x_min],
            ],
        ],
        dtype=np.float64,
    )


def _make_view_params(frustum: np.ndarray) -> ViewParameters:
    """Build ViewParameters for a camera looking along +z."""
    near_plane_center = frustum[0].mean(axis=0)
    return ViewParameters(
        frustum_corners=frustum,
        view_direction=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        near_plane_center=near_plane_center,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zarr_array():
    """128^3 float32 zarr array backed by an in-memory store."""
    store = MemoryStore()
    z = zarr.open(
        store,
        mode="w",
        shape=_SHAPE,
        chunks=_CHUNK,
        dtype="f4",
    )
    rng = np.random.default_rng(0)
    z[:] = rng.random(_SHAPE, dtype=np.float32)
    return z


@pytest.fixture
def multiscale_model():
    """Single-scale 128^3 model with identity world transform."""
    return MultiscaleImageModel.from_shape_and_scales(
        shape=_SHAPE,
        chunk_shapes=[_CHUNK],
        downscale_factors=[1.0],
    )


@pytest.fixture
def chunked_store(zarr_array, multiscale_model):
    """ChunkedImageStore wired to a real slicer with the in-memory zarr array."""
    cs = ChunkedImageStore(
        multiscale_model=multiscale_model,
        store_path="memory://sentinel",  # actual array injected below
        texture_config=TextureConfiguration(texture_width=_TEXTURE_WIDTH),
    )
    slicer = AsynchronousDataSlicer(max_workers=2)
    cs.setup_chunk_manager(slicer, max_cache_bytes=64 * 1024 * 1024)
    # Inject the in-memory array so no filesystem access occurs
    cs._zarr_array = zarr_array
    return cs


# ---------------------------------------------------------------------------
# T-D-01  get_data_request returns requests when frustum intersects volume
# ---------------------------------------------------------------------------


def test_get_data_request_intersecting_frustum(chunked_store):
    """T-D-01: Frustum inside the volume → at least one ChunkRequest produced."""
    frustum = _make_frustum(
        z_near=0.0, z_far=128.0, y_min=0.0, y_max=128.0, x_min=0.0, x_max=128.0
    )
    view_params = _make_view_params(frustum)
    region = ChunkedSelectedRegion(
        view_parameters=view_params, visual_id=_VISUAL_ID, scene_id=_SCENE_ID
    )

    requests = chunked_store.get_data_request(
        region, TilingMethod.NONE, _VISUAL_ID, _SCENE_ID
    )

    assert len(requests) == 1
    assert len(requests[0].chunk_requests) > 0


# ---------------------------------------------------------------------------
# T-D-02  get_data_request returns empty list for out-of-view frustum
# ---------------------------------------------------------------------------


def test_get_data_request_outside_frustum(chunked_store):
    """T-D-02: Frustum entirely outside the volume → empty list returned."""
    # Place the frustum far beyond the volume (z > 1000)
    frustum = _make_frustum(
        z_near=1000.0,
        z_far=2000.0,
        y_min=0.0,
        y_max=128.0,
        x_min=0.0,
        x_max=128.0,
    )
    view_params = _make_view_params(frustum)
    region = ChunkedSelectedRegion(
        view_parameters=view_params, visual_id=_VISUAL_ID, scene_id=_SCENE_ID
    )

    requests = chunked_store.get_data_request(
        region, TilingMethod.NONE, _VISUAL_ID, _SCENE_ID
    )

    assert requests == []


# ---------------------------------------------------------------------------
# T-D-03  get_data returns cached chunks immediately
# ---------------------------------------------------------------------------


def test_get_data_returns_cached_chunks(chunked_store):
    """T-D-03: Pre-populate cache; get_data returns arrays in available_chunks."""
    # Build a real ChunkedDataRequest manually
    from cellier.transform import AffineTransform
    from cellier.types import ChunkedDataRequest, ChunkRequest
    from cellier.utils.chunked_image._data_classes import TextureConfiguration

    chunk_index = 0
    scale_index = 0
    expected_data = np.ones(_CHUNK, dtype=np.float32) * 42.0

    # Pre-populate the ChunkManager cache directly
    cache_key = (scale_index, chunk_index)
    chunked_store._chunk_manager._cache[cache_key] = expected_data
    chunked_store._chunk_manager._cache_bytes += expected_data.nbytes

    req = ChunkedDataRequest(
        scene_id=_SCENE_ID,
        visual_id=_VISUAL_ID,
        resolution_level=0,
        chunk_requests=[
            ChunkRequest(
                chunk_index=chunk_index,
                scale_index=scale_index,
                priority=0.0,
                visual_id=_VISUAL_ID,
                scene_id=_SCENE_ID,
            )
        ],
        texture_config=TextureConfiguration(texture_width=_TEXTURE_WIDTH),
        texture_to_world_transform=AffineTransform.from_translation((0.0, 0.0, 0.0)),
    )

    response = chunked_store.get_data(req)

    assert isinstance(response, ChunkedDataResponse)
    assert len(response.available_chunks) == 1
    assert response.pending_count == 0
    np.testing.assert_array_equal(response.available_chunks[0].data, expected_data)


# ---------------------------------------------------------------------------
# T-D-04  texture_to_world_transform passes through correctly
# ---------------------------------------------------------------------------


def test_texture_to_world_transform_consistency(chunked_store, multiscale_model):
    """T-D-04: Transform in ChunkedDataResponse equals what ChunkSelector produced."""
    frustum = _make_frustum(
        z_near=0.0, z_far=128.0, y_min=0.0, y_max=128.0, x_min=0.0, x_max=128.0
    )
    view_params = _make_view_params(frustum)
    region = ChunkedSelectedRegion(
        view_parameters=view_params, visual_id=_VISUAL_ID, scene_id=_SCENE_ID
    )

    requests = chunked_store.get_data_request(
        region, TilingMethod.NONE, _VISUAL_ID, _SCENE_ID
    )
    assert len(requests) == 1, "Expected at least 1 request for intersecting frustum"

    request = requests[0]
    response = chunked_store.get_data(request)

    # The transform in the response must equal the one in the request
    np.testing.assert_array_equal(
        response.texture_to_world_transform.matrix,
        request.texture_to_world_transform.matrix,
    )


# ---------------------------------------------------------------------------
# T-D-05  non-identity world transform selects the same physical region
# ---------------------------------------------------------------------------


def test_non_identity_world_transform(zarr_array):
    """T-D-05: Shifting world_transform and frustum together selects same chunks."""
    translation = (64.0, 0.0, 0.0)

    # Baseline store: identity world transform, frustum centred in the volume
    base_model = MultiscaleImageModel.from_shape_and_scales(
        shape=_SHAPE, chunk_shapes=[_CHUNK], downscale_factors=[1.0]
    )
    base_store = ChunkedImageStore(
        multiscale_model=base_model,
        store_path="memory://sentinel",
        texture_config=TextureConfiguration(texture_width=_TEXTURE_WIDTH),
    )
    slicer = AsynchronousDataSlicer(max_workers=1)
    base_store.setup_chunk_manager(slicer, max_cache_bytes=64 * 1024 * 1024)
    base_store._zarr_array = zarr_array

    base_frustum = _make_frustum(
        z_near=32.0, z_far=96.0, y_min=32.0, y_max=96.0, x_min=32.0, x_max=96.0
    )
    base_view = _make_view_params(base_frustum)
    base_region = ChunkedSelectedRegion(
        view_parameters=base_view, visual_id=_VISUAL_ID, scene_id=_SCENE_ID
    )
    base_requests = base_store.get_data_request(
        base_region, TilingMethod.NONE, _VISUAL_ID, _SCENE_ID
    )

    # Translated store: world_transform shifts scale_0 → world by (64, 0, 0)
    # so the same physical voxels appear at world-z in [32+64, 96+64]
    shifted_model = MultiscaleImageModel(
        scales=base_model.scales,
        world_transform=AffineTransform.from_scale_and_translation(
            (1.0, 1.0, 1.0), translation
        ),
    )
    shifted_store = ChunkedImageStore(
        multiscale_model=shifted_model,
        store_path="memory://sentinel",
        texture_config=TextureConfiguration(texture_width=_TEXTURE_WIDTH),
    )
    shifted_store.setup_chunk_manager(slicer, max_cache_bytes=64 * 1024 * 1024)
    shifted_store._zarr_array = zarr_array

    # Frustum shifted by the same translation in world space
    tz, ty, tx = translation
    shifted_frustum = _make_frustum(
        z_near=32.0 + tz,
        z_far=96.0 + tz,
        y_min=32.0 + ty,
        y_max=96.0 + ty,
        x_min=32.0 + tx,
        x_max=96.0 + tx,
    )
    shifted_view = _make_view_params(shifted_frustum)
    shifted_region = ChunkedSelectedRegion(
        view_parameters=shifted_view, visual_id=_VISUAL_ID, scene_id=_SCENE_ID
    )
    shifted_requests = shifted_store.get_data_request(
        shifted_region, TilingMethod.NONE, _VISUAL_ID, _SCENE_ID
    )

    # Both stores should select the same number of chunks
    n_base = len(base_requests[0].chunk_requests) if base_requests else 0
    n_shifted = len(shifted_requests[0].chunk_requests) if shifted_requests else 0
    assert n_base == n_shifted, (
        f"Identity and shifted stores selected different chunk counts: "
        f"{n_base} vs {n_shifted}"
    )


# ---------------------------------------------------------------------------
# T-D-06  chunks closer to the near plane have lower priority values
# ---------------------------------------------------------------------------


def test_priority_ordering_in_requests(chunked_store):
    """T-D-06: Chunks closer to near plane (lower z) have smaller priority values."""
    frustum = _make_frustum(
        z_near=0.0, z_far=128.0, y_min=0.0, y_max=128.0, x_min=0.0, x_max=128.0
    )
    view_params = _make_view_params(frustum)
    region = ChunkedSelectedRegion(
        view_parameters=view_params, visual_id=_VISUAL_ID, scene_id=_SCENE_ID
    )

    requests = chunked_store.get_data_request(
        region, TilingMethod.NONE, _VISUAL_ID, _SCENE_ID
    )
    assert len(requests) == 1

    chunk_requests = requests[0].chunk_requests
    assert len(chunk_requests) >= 2, "Need at least 2 chunks to verify ordering"

    scale_level = chunked_store.multiscale_model.scales[0]

    # For each request, compute the chunk's z-centre in scale coords
    def chunk_z_center(chunk_idx: int) -> float:
        corners = scale_level.chunk_corners_scale[chunk_idx]  # (8, 3)
        return float(corners[:, 0].mean())  # z is axis 0

    # Pair each chunk with its z centre; verify closer (lower z) → lower priority
    # (view direction is +z, near_plane_center z is 0)
    paired = [(chunk_z_center(cr.chunk_index), cr.priority) for cr in chunk_requests]
    paired.sort(key=lambda t: t[0])  # sort by z centre ascending

    z_centers_sorted = [p[0] for p in paired]
    priorities_sorted = [p[1] for p in paired]

    # Priorities should be non-decreasing as z increases
    for i in range(len(priorities_sorted) - 1):
        assert priorities_sorted[i] <= priorities_sorted[i + 1], (
            f"Priority not monotone: chunk at z={z_centers_sorted[i]:.1f} "
            f"has priority {priorities_sorted[i]:.2f} but next chunk at "
            f"z={z_centers_sorted[i + 1]:.1f} has priority "
            f"{priorities_sorted[i + 1]:.2f}"
        )
