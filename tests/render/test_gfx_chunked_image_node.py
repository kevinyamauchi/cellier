"""Tests for GFXChunkedImageNode (Phase E).

PyGFX GPU objects are mocked so the tests run without a GPU or display.

``GFXChunkedImageNode`` imports ``pygfx`` at module level (``import pygfx as gfx``),
so we patch the name ``gfx`` inside the ``cellier.render.chunked_image`` namespace.
The ``TextureAtlas`` that it constructs internally also needs the lazy ``pygfx``/
``wgpu`` imports patched in ``sys.modules`` so that ``TextureAtlas.__init__`` does
not attempt real GPU allocations.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np

from cellier.transform import AffineTransform
from cellier.types import ChunkData, ChunkedDataResponse
from cellier.utils.chunked_image._data_classes import TextureConfiguration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    texture_width: int = 64,
    n_chunks: int = 2,
    chunk_shape: tuple[int, int, int] = (32, 32, 32),
    scene_id: str = "s1",
    visual_id: str = "v1",
) -> ChunkedDataResponse:
    """Build a minimal ChunkedDataResponse for testing."""
    rng = np.random.default_rng(0)
    chunks = [
        ChunkData(
            chunk_index=i,
            scale_index=0,
            data=rng.random(chunk_shape, dtype=np.float32),
            texture_offset=(i * chunk_shape[0], 0, 0),
        )
        for i in range(n_chunks)
    ]
    return ChunkedDataResponse(
        id="resp_001",
        scene_id=scene_id,
        visual_id=visual_id,
        resolution_level=0,
        available_chunks=chunks,
        pending_count=0,
        texture_to_world_transform=AffineTransform.from_translation((0.0, 0.0, 0.0)),
        texture_config=TextureConfiguration(texture_width=texture_width),
    )


def _make_chunked_visual():
    """Build a minimal ChunkedImageVisual for node construction."""
    from cmap import Colormap

    from cellier.models.visuals.chunked_image import ChunkedImageVisual
    from cellier.models.visuals.image import ImageAppearance

    return ChunkedImageVisual(
        name="test_chunked",
        data_store_id="store_001",
        appearance=ImageAppearance(color_map=Colormap("grays")),
    )


def _make_mock_gfx():
    """Return a mock pygfx module with Group/Volume/Geometry/VolumeMipMaterial."""
    mock_gfx = MagicMock()
    mock_gfx.Group.return_value = MagicMock()
    mock_gfx.Volume.return_value = MagicMock()
    mock_gfx.Geometry.return_value = MagicMock()
    mock_gfx.VolumeMipMaterial.return_value = MagicMock()
    return mock_gfx


# ---------------------------------------------------------------------------
# Context manager that patches both the module-level `gfx` in chunked_image
# and the lazy pygfx/wgpu imports in _texture_atlas.__init__.
# ---------------------------------------------------------------------------


@contextmanager
def _mock_gfx_environment():
    """Patch pygfx references in both chunked_image.py and _texture_atlas.py."""
    mock_gfx = _make_mock_gfx()
    mock_wgpu = MagicMock()
    # Separate texture mock so send_data is trackable
    mock_texture = MagicMock()
    mock_gfx.Texture.return_value = mock_texture

    with (
        patch("cellier.render.chunked_image.gfx", mock_gfx),
        patch.dict(sys.modules, {"pygfx": mock_gfx, "wgpu": mock_wgpu}),
    ):
        yield mock_gfx, mock_texture


# ---------------------------------------------------------------------------
# T-E-10  GFXChunkedImageNode lazy init: atlas is None before first set_slice
# ---------------------------------------------------------------------------


def test_atlas_is_none_before_first_set_slice():
    """T-E-10: Atlas not created until the first set_slice() call."""
    from cellier.render.chunked_image import GFXChunkedImageNode

    with _mock_gfx_environment():
        visual = _make_chunked_visual()
        node = GFXChunkedImageNode(model=visual)

    assert node._atlas is None
    assert node._volume_node is None


# ---------------------------------------------------------------------------
# T-E-11  set_slice initialises atlas on first call
# ---------------------------------------------------------------------------


def test_set_slice_initialises_atlas():
    """T-E-11: First set_slice() call creates atlas and Volume sub-node."""
    from cellier.render.chunked_image import GFXChunkedImageNode

    with _mock_gfx_environment() as (mock_gfx, _):
        visual = _make_chunked_visual()
        node = GFXChunkedImageNode(model=visual)

        response = _make_response(texture_width=64)
        node.set_slice(response)

    assert node._atlas is not None
    assert node._volume_node is not None
    # Volume should have been added to the group node
    node.node.add.assert_called_once_with(node._volume_node)


# ---------------------------------------------------------------------------
# T-E-12  set_slice uploads all available_chunks to the atlas
# ---------------------------------------------------------------------------


def test_set_slice_uploads_available_chunks():
    """T-E-12: set_slice uploads every chunk in available_chunks to the atlas."""
    from cellier.render.chunked_image import GFXChunkedImageNode

    n_chunks = 3
    with _mock_gfx_environment() as (_, mock_texture):
        visual = _make_chunked_visual()
        node = GFXChunkedImageNode(model=visual)

        response = _make_response(n_chunks=n_chunks)
        node.set_slice(response)

    # The atlas texture's send_data should have been called once per chunk
    assert mock_texture.send_data.call_count == n_chunks


# ---------------------------------------------------------------------------
# T-E-13  atlas not re-created on second set_slice
# ---------------------------------------------------------------------------


def test_atlas_not_recreated_on_second_set_slice():
    """T-E-13: The atlas is created only once, even across multiple set_slice calls."""
    from cellier.render.chunked_image import GFXChunkedImageNode

    with _mock_gfx_environment() as (mock_gfx, _):
        visual = _make_chunked_visual()
        node = GFXChunkedImageNode(model=visual)

        response1 = _make_response(n_chunks=1)
        node.set_slice(response1)
        atlas_after_first = node._atlas

        response2 = _make_response(n_chunks=2)
        node.set_slice(response2)
        atlas_after_second = node._atlas

    assert atlas_after_first is atlas_after_second
    # node.add() is only called inside _initialize_atlas; it must have been called once.
    assert mock_gfx.Group.return_value.add.call_count == 1


# ---------------------------------------------------------------------------
# T-E-14  upload_chunk is silently ignored before atlas initialisation
# ---------------------------------------------------------------------------


def test_upload_chunk_before_atlas_init_is_safe():
    """T-E-14: upload_chunk() silently no-ops when called before set_slice."""
    from cellier.render.chunked_image import GFXChunkedImageNode

    with _mock_gfx_environment() as (_, mock_texture):
        visual = _make_chunked_visual()
        node = GFXChunkedImageNode(model=visual)

        # Should not raise even though atlas is None
        chunk = ChunkData(
            chunk_index=0,
            scale_index=0,
            data=np.zeros((32, 32, 32), dtype=np.float32),
            texture_offset=(0, 0, 0),
        )
        node.upload_chunk(chunk)

    # Atlas still None, no GPU calls made
    assert node._atlas is None
    mock_texture.send_data.assert_not_called()


# ---------------------------------------------------------------------------
# T-E-15  upload_chunk forwards chunk to atlas after initialisation
# ---------------------------------------------------------------------------


def test_upload_chunk_after_init_forwards_to_atlas():
    """T-E-15: upload_chunk() after set_slice forwards to atlas.upload_chunk."""
    from cellier.render.chunked_image import GFXChunkedImageNode

    with _mock_gfx_environment() as (_, mock_texture):
        visual = _make_chunked_visual()
        node = GFXChunkedImageNode(model=visual)

        # First call initialises the atlas (0 chunks in response)
        response = _make_response(n_chunks=0)
        node.set_slice(response)

        send_calls_before = mock_texture.send_data.call_count

        # Now upload a background chunk
        chunk = ChunkData(
            chunk_index=0,
            scale_index=0,
            data=np.ones((32, 32, 32), dtype=np.float32),
            texture_offset=(0, 0, 0),
        )
        node.upload_chunk(chunk)

    # send_data should have been called once more
    send_calls_after = mock_texture.send_data.call_count
    assert send_calls_after == send_calls_before + 1
