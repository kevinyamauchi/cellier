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


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_scaled_transform_halves_slice_index_2d(mock_gfx):
    """With scale=(2,2,2), world z=10 should map to data z=5."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.v2.transform import AffineTransform

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    t = AffineTransform.from_scale((2.0, 2.0, 2.0))
    visual = GFXImageMemoryVisual(model, store, render_mode="2d", transform=t)

    dims = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
            slice_indices={0: 10},  # world z=10
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
    # world z=10, scale=2 → data z=5
    assert req.axis_selections[0] == 5
    assert req.axis_selections[1] == (0, 20)
    assert req.axis_selections[2] == (0, 30)


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_non_spatial_axis_not_transformed_3d(mock_gfx):
    """4D store, 3D scene: non-spatial axis (t) is NOT transformed."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.v2.transform import AffineTransform

    store = _make_store(shape=(8, 10, 20, 30))
    model = _make_visual_model(store)
    t = AffineTransform.from_scale((2.0, 2.0, 2.0))
    visual = GFXImageMemoryVisual(model, store, render_mode="3d", transform=t)

    dims = DimsState(
        axis_labels=("t", "z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2, 3),
            slice_indices={0: 6},  # t=6, non-spatial
        ),
    )
    requests = visual.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        thresholds=None,
        dims_state=dims,
    )
    req = requests[0]
    # t is non-spatial (outside 3D transform) → stays at 6
    assert req.axis_selections[0] == 6
    assert req.axis_selections[1] == (0, 10)
    assert req.axis_selections[2] == (0, 20)
    assert req.axis_selections[3] == (0, 30)


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_scaled_transform_on_spatial_slice_in_4d(mock_gfx):
    """4D store, 2D scene: spatial z-axis IS transformed by scale."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.v2.transform import AffineTransform

    store = _make_store(shape=(8, 10, 20, 30))
    model = _make_visual_model(store)
    t = AffineTransform.from_scale((2.0, 2.0, 2.0))
    visual = GFXImageMemoryVisual(model, store, render_mode="2d", transform=t)

    dims = DimsState(
        axis_labels=("t", "z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(2, 3),
            slice_indices={0: 3, 1: 8},  # t=3 (non-spatial), z=8 (spatial)
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
    # t=3 non-spatial → stays at 3
    assert req.axis_selections[0] == 3
    # z=8 spatial, scale=2 → data z=4
    assert req.axis_selections[1] == 4
    assert req.axis_selections[2] == (0, 20)
    assert req.axis_selections[3] == (0, 30)


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_identity_transform_preserves_slice_index(mock_gfx):
    """Identity transform should not alter slice indices."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="2d")

    dims = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
            slice_indices={0: 7},
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
    assert requests[0].axis_selections[0] == 7


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_slice_index_clamped_to_store_bounds(mock_gfx):
    """Transformed slice index should be clamped to valid range."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.v2.transform import AffineTransform

    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    # scale=0.5 means world z=5 → data z=10, which is out of bounds (max 9)
    t = AffineTransform.from_scale((0.5, 0.5, 0.5))
    visual = GFXImageMemoryVisual(model, store, render_mode="2d", transform=t)

    dims = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
            slice_indices={0: 5},  # world z=5, data z=10 → clamped to 9
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
    assert requests[0].axis_selections[0] == 9  # clamped


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
def test_node_matrix_set_lazily_on_slice_request(mock_gfx):
    """Node matrix is set on first build_slice_request, not construction."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.v2.transform import AffineTransform

    store = _make_store()
    model = _make_visual_model(store)
    t = AffineTransform.from_scale((4.0, 2.0, 3.0))
    visual = GFXImageMemoryVisual(model, store, render_mode="3d", transform=t)

    # Before any slice request, _last_displayed_axes is None.
    assert visual._last_displayed_axes is None

    # Trigger a slice request with displayed axes (0, 1, 2).
    dims = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(0, 1, 2),
            slice_indices={},
        ),
    )
    visual.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        thresholds=None,
        dims_state=dims,
    )
    assert visual._last_displayed_axes == (0, 1, 2)
    # set_slice((0,1,2)) from a 3D transform returns the full matrix.
    np.testing.assert_array_equal(visual.node_3d.local.matrix, t.matrix)


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_on_transform_changed_updates_node_after_initial_slice(mock_gfx):
    """Transform change updates node matrix if displayed axes are known."""
    from cellier.v2.events._events import TransformChangedEvent
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.v2.transform import AffineTransform

    store = _make_store()
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="2d")

    # First, trigger a slice to establish displayed axes.
    dims = DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
            slice_indices={0: 5},
        ),
    )
    visual.build_slice_request_2d(
        camera_pos_world=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims,
    )

    # Now change the transform.
    new_t = AffineTransform.from_scale((3.0, 5.0, 7.0))
    event = TransformChangedEvent(
        source_id=uuid4(),
        scene_id=uuid4(),
        visual_id=model.id,
        transform=new_t,
    )
    visual.on_transform_changed(event)

    assert visual._transform is new_t
    # 2D node receives set_slice((1, 2)) -> diag(5, 7) -> padded to 4x4.
    expected_2d = np.diag([5.0, 7.0, 1.0, 1.0]).astype(np.float32)
    np.testing.assert_array_equal(visual.node_2d.local.matrix, expected_2d)


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
