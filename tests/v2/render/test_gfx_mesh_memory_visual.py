"""Tests for GFXMeshMemoryVisual."""

import asyncio
from uuid import uuid4

import numpy as np
import pygfx as gfx

from cellier.v2.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.v2.data.mesh._mesh_requests import MeshSliceRequest
from cellier.v2.render.visuals._mesh_memory import GFXMeshMemoryVisual
from cellier.v2.visuals._mesh_memory import (
    MeshFlatAppearance,
    MeshPhongAppearance,
    MeshVisual,
)


def _store():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    idx = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return MeshMemoryStore(positions=pos, indices=idx)


def _visual(store, appearance=None):
    if appearance is None:
        appearance = MeshFlatAppearance()
    model = MeshVisual(name="test", data_store_id=str(store.id), appearance=appearance)
    return GFXMeshMemoryVisual(
        visual_model=model,
        data_store=store,
        render_modes={"2d", "3d"},
    )


# ── Construction ──────────────────────────────────────────────────────────────


def test_single_node_for_both_modes():
    v = _visual(_store())
    assert v.node_2d is v.node
    assert v.node_3d is v.node


def test_node_2d_is_node_3d():
    v = _visual(_store())
    assert v.node_2d is v.node_3d


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
    """Second call with same axes must not call _update_node_matrix."""
    v = _visual(_store())
    v.get_node_for_dims((1, 2))
    original_matrix = v.node.local.matrix.copy()
    v.get_node_for_dims((1, 2))  # same axes — should be a no-op
    np.testing.assert_array_equal(v.node.local.matrix, original_matrix)


# ── on_data_ready / on_data_ready_2d ─────────────────────────────────────────


def _make_batch(store, displayed=(0, 1, 2), sliced=None):
    if sliced is None:
        sliced = {}
    sid = uuid4()
    req = MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=displayed,
        slice_indices=sliced,
    )
    data = asyncio.run(store.get_data(req))
    return [(req, data)]


def test_on_data_ready_applies_3d_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready(_make_batch(s, displayed=(0, 1, 2)))
    assert v.node.material is v._material_3d


def test_on_data_ready_2d_applies_2d_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready_2d(_make_batch(s, displayed=(1, 2), sliced={0: 0}))
    if not v._is_empty:
        assert v.node.material is v._material_2d


def test_empty_slab_applies_empty_material():
    s = _store()
    v = _visual(s)
    v.on_data_ready_2d(_make_batch(s, displayed=(1, 2), sliced={0: 1000}))
    assert v.node.material is v._empty_material


def test_phong_appearance_builds_phong_material():
    v = _visual(_store(), appearance=MeshPhongAppearance())
    assert isinstance(v._material_3d, gfx.MeshPhongMaterial)
    assert isinstance(v._material_2d, gfx.MeshBasicMaterial)


def test_flat_appearance_builds_basic_material():
    v = _visual(_store(), appearance=MeshFlatAppearance())
    assert isinstance(v._material_3d, gfx.MeshBasicMaterial)
    assert isinstance(v._material_2d, gfx.MeshBasicMaterial)
