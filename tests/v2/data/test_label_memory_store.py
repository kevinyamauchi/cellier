"""Tests for LabelMemoryStore."""

from __future__ import annotations

import re
from uuid import uuid4

import numpy as np
import pytest

from cellier.data.image._image_requests import ChunkRequest
from cellier.data.label._label_memory_store import LabelMemoryStore


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


# ── Serialisation round trip ────────────────────────────────────────────────
#
# This store validates its dtype where every sibling coerces, which makes the
# dtype load-bearing on the way back in.  A bare ``tolist()`` payload dropped
# it, and numpy infers int64 from a list of Python ints, so a label store
# could be serialised but never deserialised -- for any dtype it accepts.


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32])
def test_round_trip_preserves_dtype(dtype):
    """Every accepted dtype survives, at its original width."""
    store = LabelMemoryStore(data=np.arange(6, dtype=dtype).reshape(2, 3))
    restored = LabelMemoryStore.model_validate(store.model_dump())
    assert restored.data.dtype == dtype
    np.testing.assert_array_equal(restored.data, store.data)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32])
def test_round_trip_through_json(dtype):
    """JSON is the form that actually crosses a wire."""
    store = LabelMemoryStore(data=np.arange(6, dtype=dtype).reshape(2, 3), name="seg")
    restored = LabelMemoryStore.model_validate_json(store.model_dump_json())
    assert restored.data.dtype == dtype
    assert restored.name == "seg"
    np.testing.assert_array_equal(restored.data, store.data)


def test_serialised_payload_names_its_dtype():
    """The payload is self-describing; nothing has to be inferred from it."""
    store = LabelMemoryStore(data=np.zeros((2, 2), dtype=np.int16))
    payload = store.model_dump()["data"]
    assert payload == {"dtype": "int16", "values": [[0, 0], [0, 0]]}


def test_a_payload_declaring_a_rejected_dtype_is_still_rejected():
    """One dtype rule, applied to a payload's claim as to a live array.

    Otherwise the serialised form would be a way around the validation that
    the in-memory constructor enforces.
    """
    with pytest.raises(ValueError, match="int8, int16, or int32"):
        LabelMemoryStore.model_validate(
            {"data": {"dtype": "int64", "values": [[0, 0]]}}
        )


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"values": [[0, 0]]}, "missing"),
        ({"dtype": "int16"}, "missing"),
        ({"dtype": "banana", "values": [[0, 0]]}, "Unknown dtype"),
    ],
)
def test_malformed_payloads_raise_a_clear_error(payload, match):
    """A malformed payload should not surface as a numpy TypeError."""
    with pytest.raises(ValueError, match=match):
        LabelMemoryStore.model_validate({"data": payload})


def test_an_int64_array_is_still_rejected_not_narrowed():
    """The deliberate rejection is the point, and the fix must not soften it.

    Coercing like the sibling stores would have made the round trip work by
    silently truncating a caller's int64 labels; the message tells them to
    narrow it themselves instead.
    """
    with pytest.raises(ValueError, match=re.escape("Cast your data to np.int32 first")):
        LabelMemoryStore(data=np.zeros((2, 2), dtype=np.int64))


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
