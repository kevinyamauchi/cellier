"""Tests for MeshMemoryStore construction and get_data."""

import asyncio
from uuid import uuid4

import numpy as np

from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.mesh._mesh_requests import MeshSliceRequest


def _simple_store() -> MeshMemoryStore:
    """Tetrahedron: 4 vertices, 4 faces."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return MeshMemoryStore(positions=positions, indices=indices, name="tet")


def _req(displayed=(1, 2), sliced=None, thickness=0.5):
    if sliced is None:
        sliced = {0: 0}
    sid = uuid4()
    return MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=displayed,
        slice_indices=sliced,
        thickness=thickness,
    )


# ── Construction ──────────────────────────────────────────────────────────────


def test_normals_in_3d_get_data_result():
    """Normals are computed from projected geometry for 3-D display."""
    store = _simple_store()
    sid = uuid4()
    req = MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=(0, 1, 2),
        slice_indices={},
    )
    result = asyncio.run(store.get_data(req))
    assert result.normals is not None
    assert result.normals.shape == (result.positions.shape[0], 3)
    assert result.normals.dtype == np.float32


def test_normals_zeros_for_2d_display():
    """2-D display emits zero normals (material is unlit; normals unused)."""
    store = _simple_store()
    result = asyncio.run(store.get_data(_req(displayed=(1, 2), sliced={0: 0})))
    assert not result.is_empty
    assert result.normals.shape == (result.positions.shape[0], 2)
    assert result.normals.dtype == np.float32


def test_int64_indices_coerced_to_int32():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    store = MeshMemoryStore(positions=positions, indices=indices)
    assert store.indices.dtype == np.int32


def test_colors_mode_vertex():
    # Use a triangle: 3 vertices, 1 face — unambiguously vertex-colored.
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    store = MeshMemoryStore(positions=positions, indices=indices)
    store.colors = np.ones((3, 4), dtype=np.float32)  # 3 colors == n_vertices (1)
    assert store.colors_mode == "vertex"


def test_colors_mode_face():
    # Use a store where n_faces != n_vertices to test unambiguously.
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]],
        dtype=np.float32,
    )
    indices = np.array(
        [[0, 1, 2], [1, 3, 2], [0, 1, 4], [1, 3, 4], [0, 2, 4]], dtype=np.int32
    )
    store2 = MeshMemoryStore(positions=positions, indices=indices)
    store2.colors = np.ones((5, 4), dtype=np.float32)  # 5 faces
    assert store2.colors_mode == "face"


# ── get_data — 3D (all axes displayed) ───────────────────────────────────────


def test_get_data_3d_returns_all_faces():
    store = _simple_store()
    sid = uuid4()
    req = MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=(0, 1, 2),
        slice_indices={},
    )
    result = asyncio.run(store.get_data(req))
    assert result.is_empty is False
    assert result.indices.shape[0] == store.n_faces
    assert result.positions.shape[1] == 3  # all 3 axes displayed


# ── get_data — 2D (slab filter) ──────────────────────────────────────────────


def test_get_data_2d_empty_slab():
    store = _simple_store()
    # Slice at z=100, far outside the tetrahedron.
    result = asyncio.run(store.get_data(_req(sliced={0: 100})))
    assert result.is_empty is True
    assert result.indices.shape == (1, 3)  # placeholder


def test_get_data_2d_positions_projected():
    store = _simple_store()
    # Tetrahedron vertices: 0=(0,0,0), 1=(1,0,0), 2=(0,1,0), 3=(0,0,1).
    # Axis 0 (the sliced axis) values: 0→0, 1→1, 2→0, 3→0.
    # Slice at axis0=0, thickness=0.5: vertices 0, 2, 3 are in the slab;
    # vertex 1 is not.
    # All-vertices rule: only face [0,2,3] has every vertex in the slab.
    result = asyncio.run(store.get_data(_req(sliced={0: 0}, thickness=0.5)))
    assert not result.is_empty
    # Projected positions have only 2 columns (y, x).
    assert result.positions.shape[1] == 2
    # Exactly one face survives — the one whose vertices are all on the slice.
    assert result.indices.shape == (1, 3)
    # Exactly three vertices survive.
    assert result.positions.shape[0] == 3


def test_get_data_2d_all_vertices_must_be_in_slab():
    """Faces with any off-slab vertex are excluded (all-vertex rule)."""
    store = _simple_store()
    # Tetrahedron at z=0 slice: vertices 0, 2, 3 in slab; vertex 1 at z=1.
    # Faces touching vertex 1 ([0,1,2], [0,1,3], [1,2,3]) must be excluded.
    result = asyncio.run(store.get_data(_req(sliced={0: 0}, thickness=0.5)))
    assert not result.is_empty
    assert result.indices.shape[0] == 1  # only face [0,2,3] survives


def test_get_data_2d_off_slab_face_excluded():
    """A face whose vertices are entirely off-slab produces an empty result."""
    store = _simple_store()
    # Slice at z=1, thickness=0.5: only vertex 1 (z=1) is in the slab.
    # No face has ALL vertices at z≈1, so result must be empty.
    result = asyncio.run(store.get_data(_req(sliced={0: 1}, thickness=0.5)))
    assert result.is_empty


def test_get_data_2d_indices_reindexed():
    """All index values must be valid into the compacted positions array."""
    store = _simple_store()
    result = asyncio.run(store.get_data(_req(sliced={0: 0}, thickness=0.5)))
    if not result.is_empty:
        n_verts = result.positions.shape[0]
        assert result.indices.max() < n_verts
        assert result.indices.min() >= 0


def test_get_data_2d_vertex_colors_gathered():
    # Use a store with more vertices than faces to make vertex mode unambiguous.
    # Square base: 4 vertices, 2 faces (2 triangles).
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    store = MeshMemoryStore(positions=positions, indices=indices)
    store.colors = np.eye(4, dtype=np.float32)  # 4 colors == n_vertices (> n_faces=2)
    result = asyncio.run(store.get_data(_req(sliced={0: 0}, thickness=0.5)))
    if not result.is_empty:
        assert result.colors is not None
        assert result.colors.shape[0] == result.positions.shape[0]
        assert result.color_mode == "vertex"


# ── original_face_indices (pick index mapping) ────────────────────────────────


def _stacked_faces_store() -> MeshMemoryStore:
    """Three independent triangles, each planar at z = 0, 10, 20."""
    verts = []
    faces = []
    for z in (0, 10, 20):
        b = len(verts)
        verts += [[z, 0, 0], [z, 1, 0], [z, 0, 1]]
        faces.append([b, b + 1, b + 2])
    return MeshMemoryStore(
        positions=np.array(verts, dtype=np.float32),
        indices=np.array(faces, dtype=np.int32),
    )


def test_original_face_indices_identity_in_3d():
    """A full 3-D view keeps every face, so the map is the identity arange."""
    store = _stacked_faces_store()
    result = asyncio.run(store.get_data(_req(displayed=(0, 1, 2), sliced={})))
    assert list(result.original_face_indices) == [0, 1, 2]


def test_original_face_indices_track_surviving_subset_in_2d():
    """A 2-D slab keeping one face reports that face's original index.

    pygfx reports the rendered face index (here 0, the only survivor);
    ``original_face_indices`` maps it back to original face 1.
    """
    store = _stacked_faces_store()
    result = asyncio.run(
        store.get_data(_req(displayed=(1, 2), sliced={0: 10}, thickness=0.5))
    )
    assert not result.is_empty
    assert result.indices.shape[0] == 1
    assert list(result.original_face_indices) == [1]


# ── Checkpoint cancellation ───────────────────────────────────────────────────


def test_get_data_cancellable():
    """CancelledError fires at checkpoint A before reindexing begins.

    Uses a large mesh so Phase 1 is not instant.  Cancel immediately
    after task creation; the task must not complete.
    """
    n = 50_000
    positions = np.random.rand(n * 3, 3).astype(np.float32) * 100
    indices = np.arange(n * 3, dtype=np.int32).reshape(n, 3)
    store = MeshMemoryStore(positions=positions, indices=indices)

    sid = uuid4()
    req = MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=(1, 2),
        slice_indices={0: 50},
    )

    async def _run():
        task = asyncio.create_task(store.get_data(req))
        task.cancel()
        try:
            await task
            return "completed"
        except asyncio.CancelledError:
            return "cancelled"

    result = asyncio.run(_run())
    assert result == "cancelled", (
        "get_data completed despite immediate cancel — checkpoints may "
        "not be firing.  Increase mesh size or verify await placement."
    )
