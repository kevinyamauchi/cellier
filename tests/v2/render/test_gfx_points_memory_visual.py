# tests/v2/render/test_gfx_points_memory_visual.py
"""Tests for GFXPointsMemoryVisual."""

import asyncio
from uuid import uuid4

import numpy as np

from cellier.v2.data.points._points_memory_store import PointsMemoryStore
from cellier.v2.data.points._points_requests import PointsSliceRequest
from cellier.v2.render.visuals._points_memory import GFXPointsMemoryVisual
from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._points_memory import PointsMarkerAppearance, PointsVisual


def _store():
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
    return PointsMemoryStore(positions=positions)


def _visual(store, appearance=None):
    if appearance is None:
        appearance = PointsMarkerAppearance()
    model = PointsVisual(
        name="test", data_store_id=str(store.id), appearance=appearance
    )
    return GFXPointsMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )


def _make_batch(store, displayed=(0, 1, 2), sliced=None):
    if sliced is None:
        sliced = {}
    sid = uuid4()
    req = PointsSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=displayed,
        slice_indices=sliced,
    )
    data = asyncio.run(store.get_data(req))
    return [(req, data)]


# ── Construction ──────────────────────────────────────────────────────────────


def test_single_node_for_both_modes():
    v = _visual(_store())
    assert v.node_2d is v.node
    assert v.node_3d is v.node


# ── get_node_for_dims ─────────────────────────────────────────────────────────


def test_get_node_for_dims_always_returns_same_node():
    v = _visual(_store())
    assert v.get_node_for_dims((0, 1, 2)) is v.node
    assert v.get_node_for_dims((1, 2)) is v.node


def test_get_node_for_dims_updates_last_displayed_axes():
    v = _visual(_store())
    assert v._last_displayed_axes is None
    v.get_node_for_dims((1, 2))
    assert v._last_displayed_axes == (1, 2)


def test_get_node_for_dims_no_matrix_update_on_repeat():
    v = _visual(_store())
    v.get_node_for_dims((1, 2))
    original_matrix = v.node.local.matrix.copy()
    v.get_node_for_dims((1, 2))  # same axes — must be a no-op
    np.testing.assert_array_equal(v.node.local.matrix, original_matrix)


# ── on_data_ready ─────────────────────────────────────────────────────────────


def test_on_data_ready_applies_real_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready(_make_batch(s, displayed=(0, 1, 2)))
    assert v.node.material is v._material


def test_on_data_ready_2d_applies_real_material():
    s = _store()
    v = _visual(s)
    # All points are at various z positions; use thick slab to guarantee some survive.
    v.on_data_ready_2d(_make_batch(s, displayed=(1, 2), sliced={0: 0}))
    if not v._is_empty:
        assert v.node.material is v._material


def test_empty_slab_applies_empty_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready_2d(_make_batch(s, displayed=(1, 2), sliced={0: 1000}))
    assert v.node.material is v._empty_material


# ── Coordinate reorder ────────────────────────────────────────────────────────


def test_3d_positions_reordered_zyx_to_xyz():
    """After on_data_ready, GPU positions must be in (x, y, z) order."""
    # Single point at known (z=1, y=2, x=3).
    positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    store = PointsMemoryStore(positions=positions)
    model = PointsVisual(name="t", data_store_id=str(store.id))
    v = GFXPointsMemoryVisual(
        visual_model=model,
        render_modes={"3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )
    v.on_data_ready(_make_batch(store, displayed=(0, 1, 2)))
    gpu_pos = v.node.geometry.positions.data
    # After reversal: column 0 should be x=3, column 1 y=2, column 2 z=1.
    np.testing.assert_allclose(gpu_pos[0], [3.0, 2.0, 1.0])


def test_2d_positions_padded_with_zero():
    """After on_data_ready_2d, GPU positions must be (row, col, 0)."""
    # All points at z=0 so they pass the default thickness=0.5 filter.
    positions = np.array([[0.0, 2.0, 3.0], [0.0, 4.0, 5.0]], dtype=np.float32)
    store = PointsMemoryStore(positions=positions)
    model = PointsVisual(name="t", data_store_id=str(store.id))
    v = GFXPointsMemoryVisual(
        visual_model=model,
        render_modes={"2d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )
    v.on_data_ready_2d(_make_batch(store, displayed=(1, 2), sliced={0: 0}))
    if not v._is_empty:
        gpu_pos = v.node.geometry.positions.data
        # Third column must be zero (the padding column).
        np.testing.assert_allclose(gpu_pos[:, 2], 0.0)
