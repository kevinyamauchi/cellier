"""Tests for ImageMemoryStore."""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.data.image._image_requests import ChunkRequest


def _req(*axis_selections) -> ChunkRequest:
    return ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=axis_selections,
    )


# ── Construction ────────────────────────────────────────────────────────────


def test_coerces_to_float32():
    data = np.ones((4, 4, 4), dtype=np.uint8)
    store = ImageMemoryStore(data=data)
    assert store.data.dtype == np.float32


def test_shape_and_ndim():
    data = np.zeros((10, 20, 30))
    store = ImageMemoryStore(data=data)
    assert store.shape == (10, 20, 30)
    assert store.ndim == 3
    assert store.n_levels == 1
    assert store.level_shapes == [(10, 20, 30)]


# ── Serialisation round-trip ────────────────────────────────────────────────


def test_json_roundtrip():
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    store = ImageMemoryStore(data=data)
    restored = ImageMemoryStore.model_validate_json(store.model_dump_json())
    np.testing.assert_array_equal(store.data, restored.data)


# ── get_data: 2D slice from 3D volume ──────────────────────────────────────


async def test_get_data_2d_slice_yx():
    """Slice z=1 from (Z=3, Y=4, X=5) array; displayed=(Y,X)."""
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    store = ImageMemoryStore(data=data)
    req = _req(1, (0, 4), (0, 5))  # z=1 fixed, full Y and X
    result = await store.get_data(req)
    assert result.shape == (4, 5)
    np.testing.assert_array_equal(result, data[1, :, :])


async def test_get_data_2d_slice_zx():
    """Slice y=2 from (Z=3, Y=4, X=5) array; displayed=(Z,X)."""
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    store = ImageMemoryStore(data=data)
    req = _req((0, 3), 2, (0, 5))  # full Z and X, y=2 fixed
    result = await store.get_data(req)
    assert result.shape == (3, 5)
    np.testing.assert_array_equal(result, data[:, 2, :])


# ── get_data: full 3D volume ────────────────────────────────────────────────


async def test_get_data_3d_full():
    """Request the full (Z, Y, X) volume."""
    data = np.random.rand(5, 6, 7).astype(np.float32)
    store = ImageMemoryStore(data=data)
    req = _req((0, 5), (0, 6), (0, 7))
    result = await store.get_data(req)
    assert result.shape == (5, 6, 7)
    np.testing.assert_array_equal(result, data)


# ── get_data: 5D dataset, 2D scene ─────────────────────────────────────────


async def test_get_data_5d_to_2d():
    """5D (T=2, C=3, Z=10, Y=20, X=30), display Y-X at t=0, c=1, z=5."""
    shape = (2, 3, 10, 20, 30)
    data = np.random.rand(*shape).astype(np.float32)
    store = ImageMemoryStore(data=data)
    req = _req(0, 1, 5, (0, 20), (0, 30))
    result = await store.get_data(req)
    assert result.shape == (20, 30)
    np.testing.assert_array_equal(result, data[0, 1, 5, :, :])


# ── get_data: 5D dataset, 3D scene ─────────────────────────────────────────


async def test_get_data_5d_to_3d():
    """5D (T=2, C=3, Z=10, Y=20, X=30), display Z-Y-X at t=0, c=1."""
    shape = (2, 3, 10, 20, 30)
    data = np.random.rand(*shape).astype(np.float32)
    store = ImageMemoryStore(data=data)
    req = _req(0, 1, (0, 10), (0, 20), (0, 30))
    result = await store.get_data(req)
    assert result.shape == (10, 20, 30)
    np.testing.assert_array_equal(result, data[0, 1, :, :, :])


# ── get_data: out-of-bounds clamping ───────────────────────────────────────


async def test_get_data_clamps_negative_start():
    """Negative start is clamped; result is zero-padded on the left."""
    data = np.ones((10, 10), dtype=np.float32)
    store = ImageMemoryStore(data=data)
    req = _req((-2, 8), (0, 10))  # Y starts at -2 → first 2 rows zero-padded
    result = await store.get_data(req)
    assert result.shape == (10, 10)
    np.testing.assert_array_equal(result[:2, :], 0.0)  # padded rows
    np.testing.assert_array_equal(result[2:, :], 1.0)  # real data
