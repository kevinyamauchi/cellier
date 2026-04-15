# tests/v2/render/test_gfx_lines_memory_visual.py
"""Tests for GFXLinesMemoryVisual."""

import asyncio
from uuid import uuid4

import numpy as np

from cellier.v2.data.lines._lines_memory_store import LinesMemoryStore
from cellier.v2.data.lines._lines_requests import LinesSliceRequest
from cellier.v2.render.visuals._lines_memory import GFXLinesMemoryVisual
from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._lines_memory import LinesMemoryAppearance, LinesVisual


def _store():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],  # segment 0
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 3.0],  # segment 1
        ],
        dtype=np.float32,
    )
    return LinesMemoryStore(positions=positions)


def _visual(store, appearance=None):
    if appearance is None:
        appearance = LinesMemoryAppearance()
    model = LinesVisual(name="test", data_store_id=str(store.id), appearance=appearance)
    return GFXLinesMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )


def _make_batch(store, displayed=(0, 1, 2), sliced=None):
    if sliced is None:
        sliced = {}
    sid = uuid4()
    req = LinesSliceRequest(
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


def test_initial_material_is_empty():
    v = _visual(_store())
    assert v.node.material is v._empty_material


def test_n_levels_is_one():
    v = _visual(_store())
    assert v.n_levels == 1


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
    v.get_node_for_dims((1, 2))
    np.testing.assert_array_equal(v.node.local.matrix, original_matrix)


# ── on_data_ready ─────────────────────────────────────────────────────────────


def test_on_data_ready_applies_real_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready(_make_batch(s, displayed=(0, 1, 2)))
    assert v.node.material is v._material


def test_empty_slab_applies_empty_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready_2d(_make_batch(s, displayed=(1, 2), sliced={0: 1000}))
    assert v.node.material is v._empty_material


# ── Coordinate reorder ────────────────────────────────────────────────────────


def test_3d_positions_reordered_zyx_to_xyz():
    """After on_data_ready, GPU positions must be in (x, y, z) order."""
    # Segment: (z=1, y=2, x=3) → (z=4, y=5, x=6).
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    store = LinesMemoryStore(positions=positions)
    model = LinesVisual(name="t", data_store_id=str(store.id))
    v = GFXLinesMemoryVisual(
        visual_model=model,
        render_modes={"3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )
    v.on_data_ready(_make_batch(store, displayed=(0, 1, 2)))
    gpu_pos = v.node.geometry.positions.data
    # After reversal: first vertex should be (x=3, y=2, z=1).
    np.testing.assert_allclose(gpu_pos[0], [3.0, 2.0, 1.0])
    # Second vertex should be (x=6, y=5, z=4).
    np.testing.assert_allclose(gpu_pos[1], [6.0, 5.0, 4.0])


def test_2d_positions_padded_with_zero():
    """After on_data_ready_2d, GPU positions must be (row, col, 0)."""
    # Both vertices at z=0 so they pass the default thickness=0.5 filter.
    positions = np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]], dtype=np.float32)
    store = LinesMemoryStore(positions=positions)
    model = LinesVisual(name="t", data_store_id=str(store.id))
    v = GFXLinesMemoryVisual(
        visual_model=model,
        render_modes={"2d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )
    v.on_data_ready_2d(_make_batch(store, displayed=(1, 2), sliced={0: 0}))
    if not v._is_empty:
        gpu_pos = v.node.geometry.positions.data
        np.testing.assert_allclose(gpu_pos[:, 2], 0.0)
