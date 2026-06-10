"""Defensive gesture-hardening tests for the paint controller.

These cover the capture-broken edge cases: a stroke whose release never
arrives, and stray moves belonging to a different gesture.  The code under
test lives in ``cellier.v2.paint._abstract`` and ``_history``.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.v2.controller import CellierController
from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.events._events import (
    CanvasMouseMove2DEvent,
    CanvasMousePress2DEvent,
    CanvasMouseRelease2DEvent,
    CanvasPickInfo,
)
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.visuals._image_memory import InMemoryImageAppearance


@pytest.fixture
def paint_setup(qtbot):
    """Build a 2D scene/canvas/visual and a SyncPaintController."""
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=cs,
        name="paint_scene",
        render_modes={"2d"},
    )

    data = np.zeros((8, 32, 32), dtype=np.float32)
    store = ImageMemoryStore(data=data, name="paint_store")
    appearance = InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0))
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
    return controller, paint_ctrl, store, scene.id, canvas_id


def _press(canvas_id, scene_id, coord, gesture_id):
    return CanvasMousePress2DEvent(
        source_id=canvas_id,
        scene_id=scene_id,
        world_coordinate=np.asarray(coord, dtype=np.float64),
        pick_info=CanvasPickInfo(hit_visual_id=None),
        gesture_id=gesture_id,
    )


def _move(canvas_id, scene_id, coord, gesture_id):
    return CanvasMouseMove2DEvent(
        source_id=canvas_id,
        scene_id=scene_id,
        world_coordinate=np.asarray(coord, dtype=np.float64),
        pick_info=CanvasPickInfo(hit_visual_id=None),
        gesture_id=gesture_id,
    )


def _release(canvas_id, scene_id, coord, gesture_id):
    return CanvasMouseRelease2DEvent(
        source_id=canvas_id,
        scene_id=scene_id,
        world_coordinate=np.asarray(coord, dtype=np.float64),
        pick_info=CanvasPickInfo(hit_visual_id=None),
        gesture_id=gesture_id,
    )


async def test_dangling_stroke_finalised_on_next_press(paint_setup):
    _controller, paint_ctrl, _store, scene_id, canvas_id = paint_setup
    gesture_a = uuid4()
    gesture_b = uuid4()

    # press -> moves, but NO release (capture broken).
    paint_ctrl._on_mouse_press(_press(canvas_id, scene_id, [0, 10, 10], gesture_a))
    paint_ctrl._on_mouse_move(_move(canvas_id, scene_id, [0, 11, 11], gesture_a))

    # A fresh press should finalise the dangling stroke into history and
    # start a new active stroke.
    paint_ctrl._on_mouse_press(_press(canvas_id, scene_id, [0, 20, 20], gesture_b))

    assert paint_ctrl._history.can_undo
    assert len(paint_ctrl._history._undo_stack) == 1
    assert paint_ctrl._active_stroke is not None
    assert paint_ctrl._active_stroke.gesture_id == gesture_b


async def test_move_from_other_gesture_ignored(paint_setup):
    _controller, paint_ctrl, _store, scene_id, canvas_id = paint_setup
    gesture_a = uuid4()
    gesture_b = uuid4()

    paint_ctrl._on_mouse_press(_press(canvas_id, scene_id, [0, 10, 10], gesture_a))
    chunks_after_press = len(paint_ctrl._active_stroke._voxel_chunks)

    # A move tagged with a different gesture must be ignored.
    paint_ctrl._on_mouse_move(_move(canvas_id, scene_id, [0, 12, 12], gesture_b))

    assert len(paint_ctrl._active_stroke._voxel_chunks) == chunks_after_press


async def test_normal_stroke_produces_one_command(paint_setup):
    _controller, paint_ctrl, _store, scene_id, canvas_id = paint_setup
    gesture = uuid4()

    paint_ctrl._on_mouse_press(_press(canvas_id, scene_id, [0, 10, 10], gesture))
    paint_ctrl._on_mouse_move(_move(canvas_id, scene_id, [0, 11, 11], gesture))
    paint_ctrl._on_mouse_move(_move(canvas_id, scene_id, [0, 12, 12], gesture))
    paint_ctrl._on_mouse_release(_release(canvas_id, scene_id, [0, 12, 12], gesture))

    assert len(paint_ctrl._history._undo_stack) == 1
    assert paint_ctrl._active_stroke is None
