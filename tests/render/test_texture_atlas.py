"""Tests for TextureAtlas (Phase E).

PyGFX GPU objects are mocked so the tests run without a GPU or display.
``_texture_atlas.py`` imports pygfx/wgpu lazily inside ``TextureAtlas.__init__``,
so patching ``sys.modules`` for the duration of each test is sufficient — no
module reload is needed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from cellier.render._texture_atlas import TextureAtlas, TextureRegion
from cellier.types import ChunkData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk_data(
    chunk_index: int,
    scale_index: int,
    texture_offset: tuple[int, int, int] = (0, 0, 0),
    shape: tuple[int, int, int] = (32, 32, 32),
) -> ChunkData:
    rng = np.random.default_rng(chunk_index)
    return ChunkData(
        chunk_index=chunk_index,
        scale_index=scale_index,
        data=rng.random(shape, dtype=np.float32),
        texture_offset=texture_offset,
    )


def _make_mock_pygfx_and_wgpu():
    """Return (mock_gfx, mock_wgpu, mock_texture) ready for patching."""
    mock_texture = MagicMock()
    mock_gfx = MagicMock()
    mock_gfx.Texture.return_value = mock_texture
    mock_wgpu = MagicMock()
    return mock_gfx, mock_wgpu, mock_texture


# ---------------------------------------------------------------------------
# T-E-01  TextureAtlas construction creates one texture per scale level
# ---------------------------------------------------------------------------


def test_texture_atlas_construction_creates_textures():
    """T-E-01: Constructor pre-allocates one gfx.Texture per scale level."""
    mock_gfx, mock_wgpu, mock_texture = _make_mock_pygfx_and_wgpu()

    with patch.dict(sys.modules, {"pygfx": mock_gfx, "wgpu": mock_wgpu}):
        atlas = TextureAtlas(texture_width=64, n_scales=3)

    # One gfx.Texture constructed per scale
    assert mock_gfx.Texture.call_count == 3

    # texture_for_scale returns the mock objects
    for i in range(3):
        assert atlas.texture_for_scale(i) is mock_texture


# ---------------------------------------------------------------------------
# T-E-02  upload_chunk reverses (z, y, x) → (x, y, z) for send_data
# ---------------------------------------------------------------------------


def test_upload_chunk_reverses_offset_for_pygfx():
    """T-E-02: upload_chunk reverses (z, y, x) offset to (x, y, z) for send_data."""
    mock_gfx, mock_wgpu, mock_texture = _make_mock_pygfx_and_wgpu()

    with patch.dict(sys.modules, {"pygfx": mock_gfx, "wgpu": mock_wgpu}):
        atlas = TextureAtlas(texture_width=64, n_scales=1)

        chunk = _make_chunk_data(
            chunk_index=0,
            scale_index=0,
            texture_offset=(4, 8, 16),  # (z, y, x)
        )
        atlas.upload_chunk(chunk)

    # PyGFX should receive (x=16, y=8, z=4) — reversed axes
    mock_texture.send_data.assert_called_once()
    offset_arg = mock_texture.send_data.call_args[0][0]
    assert offset_arg == (16, 8, 4), f"Expected (16, 8, 4), got {offset_arg}"


# ---------------------------------------------------------------------------
# T-E-03  upload_chunk records the chunk in the residency map
# ---------------------------------------------------------------------------


def test_upload_chunk_records_residency():
    """T-E-03: After upload_chunk, has_chunk returns True for that chunk."""
    mock_gfx, mock_wgpu, _ = _make_mock_pygfx_and_wgpu()

    with patch.dict(sys.modules, {"pygfx": mock_gfx, "wgpu": mock_wgpu}):
        atlas = TextureAtlas(texture_width=64, n_scales=2)

        chunk_a = _make_chunk_data(
            chunk_index=5, scale_index=0, texture_offset=(0, 0, 0)
        )
        chunk_b = _make_chunk_data(
            chunk_index=3, scale_index=1, texture_offset=(32, 0, 0)
        )

        assert not atlas.has_chunk(0, 5)
        assert not atlas.has_chunk(1, 3)

        atlas.upload_chunk(chunk_a)
        assert atlas.has_chunk(0, 5)
        assert not atlas.has_chunk(1, 3)

        atlas.upload_chunk(chunk_b)
        assert atlas.has_chunk(0, 5)
        assert atlas.has_chunk(1, 3)


# ---------------------------------------------------------------------------
# T-E-04  evict_chunk removes the residency record
# ---------------------------------------------------------------------------


def test_evict_chunk_removes_residency():
    """T-E-04: evict_chunk removes the residency record; double-evict is safe."""
    mock_gfx, mock_wgpu, _ = _make_mock_pygfx_and_wgpu()

    with patch.dict(sys.modules, {"pygfx": mock_gfx, "wgpu": mock_wgpu}):
        atlas = TextureAtlas(texture_width=64, n_scales=1)
        chunk = _make_chunk_data(chunk_index=2, scale_index=0)
        atlas.upload_chunk(chunk)
        assert atlas.has_chunk(0, 2)

        atlas.evict_chunk(0, 2)
        assert not atlas.has_chunk(0, 2)

        # Double-evict must not raise
        atlas.evict_chunk(0, 2)


# ---------------------------------------------------------------------------
# T-E-05  upload_chunk stores correct TextureRegion metadata
# ---------------------------------------------------------------------------


def test_upload_chunk_stores_correct_region():
    """T-E-05: The TextureRegion stored matches the chunk's offset and shape."""
    mock_gfx, mock_wgpu, _ = _make_mock_pygfx_and_wgpu()

    with patch.dict(sys.modules, {"pygfx": mock_gfx, "wgpu": mock_wgpu}):
        atlas = TextureAtlas(texture_width=64, n_scales=1)
        offset = (8, 16, 32)
        chunk = _make_chunk_data(
            chunk_index=7, scale_index=0, texture_offset=offset, shape=(16, 16, 16)
        )
        atlas.upload_chunk(chunk)

    region = atlas._regions[(0, 7)]
    assert isinstance(region, TextureRegion)
    assert region.offset == offset
    assert region.shape == (16, 16, 16)
