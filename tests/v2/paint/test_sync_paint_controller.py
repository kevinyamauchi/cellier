"""Tests for :class:`cellier.v2.paint.SyncPaintController`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from cellier.v2.controller import CellierController
from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.paint._history import PaintStrokeCommand
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.visuals._image_memory import ImageMemoryAppearance

if TYPE_CHECKING:
    from cellier.v2.paint import SyncPaintController


@pytest.fixture
def paint_setup(qtbot):
    """Build a 2D scene/canvas/visual and a SyncPaintController.

    The fixture is ``async`` so pytest-asyncio (mode=auto) installs an
    event loop — ``reslice_scene`` calls ``asyncio.ensure_future`` from
    within ``_write_values``, and a running loop is required.
    """
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("y", "x"))
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=cs,
        name="paint_scene",
        render_modes={"2d"},
    )

    data = np.zeros((32, 32), dtype=np.float32)
    store = ImageMemoryStore(data=data, name="paint_store")
    appearance = ImageMemoryAppearance(color_map="grays", clim=(0.0, 1.0))
    visual = controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=appearance,
        name="paint_image",
    )

    controller.add_canvas(scene_id=scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]

    paint_ctrl = controller.add_paint_controller(
        visual_id=visual.id,
        canvas_id=canvas_id,
        brush_value=1.0,
        brush_radius_voxels=2.0,
    )
    return controller, paint_ctrl, store, canvas_id


def _push_stroke(
    paint_ctrl: SyncPaintController,
    indices: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
) -> None:
    """Append a stroke directly to history (mimics what mouse handlers do)."""
    paint_ctrl._history.push(
        PaintStrokeCommand(
            visual_id=paint_ctrl._visual_id,
            data_store_id=paint_ctrl._data_store_id,
            voxel_indices=indices,
            old_values=old,
            new_values=new,
        )
    )


async def test_brush_paints_voxels(paint_setup):
    _controller, paint_ctrl, store, _canvas_id = paint_setup
    indices = np.array([[5, 5], [5, 6], [6, 5]], dtype=np.int64)
    values = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    paint_ctrl._write_values(indices, values)

    assert store.data[5, 5] == pytest.approx(1.0)
    assert store.data[5, 6] == pytest.approx(1.0)
    assert store.data[6, 5] == pytest.approx(1.0)


async def test_undo_reverts_voxels(paint_setup):
    _controller, paint_ctrl, store, _canvas_id = paint_setup
    indices = np.array([[10, 10], [10, 11]], dtype=np.int64)
    old = paint_ctrl._read_old_values(indices)
    new = np.array([1.0, 1.0], dtype=np.float32)

    paint_ctrl._write_values(indices, new)
    _push_stroke(paint_ctrl, indices, old, new)

    paint_ctrl.undo()

    assert store.data[10, 10] == pytest.approx(0.0)
    assert store.data[10, 11] == pytest.approx(0.0)


async def test_redo_replays_voxels(paint_setup):
    _controller, paint_ctrl, store, _canvas_id = paint_setup
    indices = np.array([[20, 20]], dtype=np.int64)
    old = paint_ctrl._read_old_values(indices)
    new = np.array([1.0], dtype=np.float32)
    paint_ctrl._write_values(indices, new)
    _push_stroke(paint_ctrl, indices, old, new)

    paint_ctrl.undo()
    assert store.data[20, 20] == pytest.approx(0.0)

    paint_ctrl.redo()

    assert store.data[20, 20] == pytest.approx(1.0)


async def test_abort_reverts_all_strokes(paint_setup):
    _controller, paint_ctrl, store, _canvas_id = paint_setup
    for r, c in [(3, 3), (4, 4)]:
        idx = np.array([[r, c]], dtype=np.int64)
        old = paint_ctrl._read_old_values(idx)
        new = np.array([1.0], dtype=np.float32)
        paint_ctrl._write_values(idx, new)
        _push_stroke(paint_ctrl, idx, old, new)
    assert store.data[3, 3] == pytest.approx(1.0)
    assert store.data[4, 4] == pytest.approx(1.0)

    paint_ctrl.abort()

    assert store.data[3, 3] == pytest.approx(0.0)
    assert store.data[4, 4] == pytest.approx(0.0)
    assert paint_ctrl._history.can_undo is False


async def test_commit_clears_history(paint_setup):
    _controller, paint_ctrl, _store, _canvas_id = paint_setup
    idx = np.array([[1, 1]], dtype=np.int64)
    _push_stroke(
        paint_ctrl,
        idx,
        np.array([0.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    )
    assert paint_ctrl._history.can_undo is True

    paint_ctrl.commit()

    assert paint_ctrl._history.can_undo is False


async def test_camera_controller_disabled_on_init(paint_setup):
    controller, _paint_ctrl, _store, canvas_id = paint_setup
    canvas_view = controller._render_manager._canvases[canvas_id]
    assert canvas_view._controller.enabled is False


async def test_camera_controller_restored_on_commit(paint_setup):
    controller, paint_ctrl, _store, canvas_id = paint_setup
    canvas_view = controller._render_manager._canvases[canvas_id]
    assert canvas_view._controller.enabled is False

    paint_ctrl.commit()

    assert canvas_view._controller.enabled is True


async def test_camera_controller_restored_on_abort(paint_setup):
    controller, paint_ctrl, _store, canvas_id = paint_setup
    canvas_view = controller._render_manager._canvases[canvas_id]
    assert canvas_view._controller.enabled is False

    paint_ctrl.abort()

    assert canvas_view._controller.enabled is True


async def test_add_paint_controller_unsupported_store_raises(qtbot):
    """Non-ImageMemoryStore visuals raise TypeError in Phase 2."""
    from cellier.v2.data.points._points_memory_store import PointsMemoryStore
    from cellier.v2.visuals._points_memory import PointsMarkerAppearance

    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("y", "x"))
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=cs,
        name="bad_scene",
        render_modes={"2d"},
    )
    points = PointsMemoryStore(
        positions=np.array([[1.0, 1.0]], dtype=np.float32),
        name="pts",
    )
    visual = controller.add_points(
        data=points,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(
            color=(1.0, 0.0, 0.0, 1.0), size=4.0, size_space="screen"
        ),
        name="points_visual",
    )
    controller.add_canvas(scene_id=scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]

    with pytest.raises(TypeError, match="No PaintController"):
        controller.add_paint_controller(
            visual_id=visual.id,
            canvas_id=canvas_id,
        )
