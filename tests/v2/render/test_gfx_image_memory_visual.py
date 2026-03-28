"""Tests for GFXImageMemoryVisual."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np

from cellier.v2._state import AxisAlignedSelectionState, DimsState
from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.data.image._image_requests import ChunkRequest
from cellier.v2.visuals._image_memory import ImageMemoryAppearance, ImageVisual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(shape=(10, 20, 30)) -> ImageMemoryStore:
    return ImageMemoryStore(data=np.zeros(shape, dtype=np.float32))


def _make_appearance() -> ImageMemoryAppearance:
    return ImageMemoryAppearance(color_map="viridis", clim=(0.0, 1.0))


def _make_visual_model(store: ImageMemoryStore) -> ImageVisual:
    return ImageVisual(
        name="test",
        data_store_id=str(store.id),
        appearance=_make_appearance(),
    )


def _make_dims_state_2d() -> DimsState:
    """3D store, 2D scene: display (Y, X), slice Z at 5."""
    return DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
            slice_indices={0: 5},
        ),
    )


def _make_dims_state_3d() -> DimsState:
    """3D store, 3D scene: display all axes."""
    return DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(0, 1, 2),
            slice_indices={},
        ),
    )


# ---------------------------------------------------------------------------
# Tests: planning
# ---------------------------------------------------------------------------


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_build_slice_request_2d_returns_one_request(mock_gfx):
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="2d")

    dims = _make_dims_state_2d()
    requests = visual.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
    )

    assert len(requests) == 1
    req = requests[0]
    assert isinstance(req, ChunkRequest)
    assert req.scale_index == 0
    # Axis 0 (z) is sliced at index 5
    assert req.axis_selections[0] == 5
    # Axes 1 and 2 (y, x) are displayed — full extent
    assert req.axis_selections[1] == (0, 20)
    assert req.axis_selections[2] == (0, 30)
    # Both UUIDs are freshly generated
    assert req.chunk_request_id is not None
    assert req.slice_request_id is not None


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_build_slice_request_3d_returns_one_request(mock_gfx):
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="3d")

    dims = _make_dims_state_3d()
    requests = visual.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        thresholds=None,
        dims_state=dims,
    )

    assert len(requests) == 1
    req = requests[0]
    assert req.axis_selections == ((0, 10), (0, 20), (0, 30))


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_build_slice_request_3d_handles_none_dims_state(mock_gfx):
    """With dims_state=None all axes are treated as displayed."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    store = _make_store(shape=(5, 6, 7))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="3d")

    requests = visual.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        thresholds=None,
        dims_state=None,
    )
    assert requests[0].axis_selections == ((0, 5), (0, 6), (0, 7))


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_5d_dims_state_2d_scene(mock_gfx):
    """5D store, 2D scene: display (Y, X), slice T=0, C=1, Z=5."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    store = _make_store(shape=(2, 3, 10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="2d")

    dims = DimsState(
        axis_labels=("t", "c", "z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(3, 4),
            slice_indices={0: 0, 1: 1, 2: 5},
        ),
    )
    requests = visual.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
    )
    req = requests[0]
    assert req.axis_selections == (0, 1, 5, (0, 20), (0, 30))


# ---------------------------------------------------------------------------
# Tests: commit (on_data_ready / on_data_ready_2d)
# ---------------------------------------------------------------------------


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_on_data_ready_2d_transposes_correctly(mock_gfx):
    """Data (H=20, W=30) must be transposed to (W=30, H=20, 1) for pygfx."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="2d")

    data = np.random.rand(20, 30).astype(np.float32)
    req = ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=(5, (0, 20), (0, 30)),
    )

    captured_arrays = []
    mock_gfx.Texture.side_effect = lambda arr, **kw: captured_arrays.append(arr)
    mock_gfx.Geometry.return_value = MagicMock()

    visual.on_data_ready_2d([(req, data)])

    assert len(captured_arrays) == 1
    arr = captured_arrays[0]
    assert arr.shape == (30, 20, 1)
    np.testing.assert_array_equal(arr[:, :, 0], data.T)


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_on_data_ready_3d_transposes_correctly(mock_gfx):
    """Data (D=10, H=20, W=30) → (W=30, H=20, D=10) for pygfx."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="3d")

    data = np.random.rand(10, 20, 30).astype(np.float32)
    req = ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=((0, 10), (0, 20), (0, 30)),
    )

    captured_arrays = []
    mock_gfx.Texture.side_effect = lambda arr, **kw: captured_arrays.append(arr)
    mock_gfx.Geometry.return_value = MagicMock()

    visual.on_data_ready([(req, data)])

    assert len(captured_arrays) == 1
    arr = captured_arrays[0]
    assert arr.shape == (30, 20, 10)
    np.testing.assert_array_equal(arr, data.T)


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_on_data_ready_noop_on_empty_batch(mock_gfx):
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    store = _make_store()
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="3d")

    # Should not raise
    visual.on_data_ready([])
    visual.on_data_ready_2d([])


# ---------------------------------------------------------------------------
# Tests: transform
# ---------------------------------------------------------------------------


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_node_matrix_set_on_construction(mock_gfx):
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.v2.transform import AffineTransform

    store = _make_store()
    model = _make_visual_model(store)
    t = AffineTransform.from_translation((10.0, 20.0, 30.0))
    visual = GFXImageMemoryVisual(model, store, render_mode="3d", transform=t)

    np.testing.assert_array_equal(visual.node_3d.local.matrix, t.matrix)


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_on_transform_changed_updates_node_and_field(mock_gfx):
    from cellier.v2.events._events import TransformChangedEvent
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.v2.transform import AffineTransform

    store = _make_store()
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="2d")

    new_t = AffineTransform.from_scale((3.0, 3.0, 3.0))
    event = TransformChangedEvent(
        source_id=uuid4(),
        scene_id=uuid4(),
        visual_id=model.id,
        transform=new_t,
    )
    visual.on_transform_changed(event)

    assert visual._transform is new_t
    np.testing.assert_array_equal(visual.node_2d.local.matrix, new_t.matrix)


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_identity_transform_is_noop_3d(mock_gfx):
    """Identity transform should produce the same result as no transform."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="3d")

    dims = _make_dims_state_3d()
    requests = visual.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        thresholds=None,
        dims_state=dims,
    )
    assert len(requests) == 1
    assert requests[0].axis_selections == ((0, 10), (0, 20), (0, 30))
