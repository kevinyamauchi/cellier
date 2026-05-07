"""Tests for LabelMemoryStore."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.v2.data.image._image_requests import ChunkRequest
from cellier.v2.data.label._label_memory_store import LabelMemoryStore


def _req(*axis_selections) -> ChunkRequest:
    return ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=axis_selections,
    )


# ── Construction / dtype validation ────────────────────────────────────────


def test_accepts_int8():
    data = np.ones((4, 4, 4), dtype=np.int8)
    store = LabelMemoryStore(data=data)
    assert store.data.dtype == np.int8


def test_accepts_int16():
    data = np.ones((4, 4, 4), dtype=np.int16)
    store = LabelMemoryStore(data=data)
    assert store.data.dtype == np.int16


def test_accepts_int32():
    data = np.ones((4, 4, 4), dtype=np.int32)
    store = LabelMemoryStore(data=data)
    assert store.data.dtype == np.int32


@pytest.mark.parametrize(
    "dtype", [np.int64, np.uint8, np.uint16, np.uint32, np.float32]
)
def test_rejects_bad_dtypes(dtype):
    data = np.ones((4, 4, 4), dtype=dtype)
    with pytest.raises(ValueError, match="int8, int16, or int32"):
        LabelMemoryStore(data=data)


# ── Properties ──────────────────────────────────────────────────────────────


def test_ndim():
    store = LabelMemoryStore(data=np.zeros((2, 3, 4), dtype=np.int32))
    assert store.ndim == 3


def test_shape():
    store = LabelMemoryStore(data=np.zeros((2, 3, 4), dtype=np.int32))
    assert store.shape == (2, 3, 4)


def test_n_levels():
    store = LabelMemoryStore(data=np.zeros((4, 4, 4), dtype=np.int32))
    assert store.n_levels == 1


def test_level_shapes():
    store = LabelMemoryStore(data=np.zeros((2, 3, 4), dtype=np.int32))
    assert store.level_shapes == [(2, 3, 4)]


# ── get_data ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_data_full_returns_int32():
    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int16)
    store = LabelMemoryStore(data=data)
    req = _req((0, 2), (0, 2), (0, 2))
    result = await store.get_data(req)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, data.astype(np.int32))


@pytest.mark.asyncio
async def test_get_data_slice_drops_axis():
    data = np.zeros((5, 6, 7), dtype=np.int32)
    store = LabelMemoryStore(data=data)
    req = _req(2, (0, 6), (0, 7))
    result = await store.get_data(req)
    assert result.shape == (6, 7)


@pytest.mark.asyncio
async def test_get_data_negative_labels_survive():
    data = np.array([[[0, -5, 3]]], dtype=np.int32)
    store = LabelMemoryStore(data=data)
    req = _req((0, 1), (0, 1), (0, 3))
    result = await store.get_data(req)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, data)


@pytest.mark.asyncio
async def test_get_data_int16_upcasts_to_int32():
    data = np.array([[[100, -200]]], dtype=np.int16)
    store = LabelMemoryStore(data=data)
    req = _req((0, 1), (0, 1), (0, 2))
    result = await store.get_data(req)
    assert result.dtype == np.int32
    assert result[0, 0, 1] == -200
