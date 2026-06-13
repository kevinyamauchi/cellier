"""Tests for canvas mouse event context-field wiring and gesture synthesis."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from cellier.controller import CellierController
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.events._events import (
    CanvasMouseMove2DEvent,
    CanvasMousePress2DEvent,
    CanvasMouseRelease2DEvent,
    CanvasPickInfo,
    ImagePickInfo,
    LinesPickInfo,
    MeshPickInfo,
    PointsPickInfo,
    ViewRay,
    _CanvasRawPointerEvent,
)
from cellier.render.render_manager import RenderManager, _ImageDisplayedDataCoord
from cellier.scene.dims import CoordinateSystem
from cellier.visuals import LinesMemoryAppearance, MeshFlatAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance


def _raw_2d(
    canvas_id,
    scene_id,
    action,
    *,
    button=1,
    buttons=(1,),
    modifiers=("Control",),
    gesture_id=None,
) -> _CanvasRawPointerEvent:
    return _CanvasRawPointerEvent(
        canvas_id=canvas_id,
        scene_id=scene_id,
        action=action,
        camera_type="2d",
        position_2d=np.array([1.0, 2.0], dtype=np.float64),
        ray=None,
        hit_visual_id=None,
        button=button,
        modifiers=modifiers,
        buttons=buttons,
        gesture_id=gesture_id,
    )


def _make_2d_controller():
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="2d", coordinate_system=cs, name="main")
    return controller, scene.id


def test_context_fields_propagate_to_public_event():
    controller, scene_id = _make_2d_controller()
    canvas_id = uuid4()
    received: list = []
    controller.on_mouse_press_2d(canvas_id, received.append, owner_id=uuid4())

    gid = uuid4()
    controller._on_raw_pointer_event(
        _raw_2d(
            canvas_id,
            scene_id,
            "press",
            button=1,
            buttons=(1, 2),
            modifiers=("Control", "Shift"),
            gesture_id=gid,
        )
    )

    assert len(received) == 1
    event = received[0]
    assert isinstance(event, CanvasMousePress2DEvent)
    assert event.button == 1
    assert event.buttons == (1, 2)
    assert event.modifiers == ("Control", "Shift")
    assert event.gesture_id == gid
    assert isinstance(event.pick_info, CanvasPickInfo)


def test_gesture_id_threads_through_phases():
    controller, scene_id = _make_2d_controller()
    canvas_id = uuid4()
    received: list = []
    owner = uuid4()
    controller.on_mouse_press_2d(canvas_id, received.append, owner_id=owner)
    controller.on_mouse_move_2d(canvas_id, received.append, owner_id=owner)
    controller.on_mouse_release_2d(canvas_id, received.append, owner_id=owner)

    gid = uuid4()
    for action in ("press", "move", "release"):
        controller._on_raw_pointer_event(
            _raw_2d(canvas_id, scene_id, action, gesture_id=gid)
        )

    assert [type(e) for e in received] == [
        CanvasMousePress2DEvent,
        CanvasMouseMove2DEvent,
        CanvasMouseRelease2DEvent,
    ]
    assert {e.gesture_id for e in received} == {gid}


# ---------------------------------------------------------------------------
# render_manager gesture synthesis
# ---------------------------------------------------------------------------


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)


def _fake_pygfx_event(event_type, *, button=1, buttons=(1,)):
    return SimpleNamespace(
        type=event_type,
        x=10.0,
        y=20.0,
        button=button,
        buttons=buttons,
        modifiers=("Control",),
        pick_info={},
    )


def _fake_canvas_view():
    camera = SimpleNamespace(
        width=100.0,
        height=100.0,
        local=SimpleNamespace(position=(0.0, 0.0, 0.0)),
    )
    canvas = SimpleNamespace(get_logical_size=lambda: (100, 100))
    return SimpleNamespace(_canvas=canvas, _camera=camera, _dim="2d")


def _render_manager_with_canvas():
    rm = RenderManager()
    bus = _RecordingBus()
    rm._event_bus = bus
    canvas_id = uuid4()
    scene_id = uuid4()
    rm._canvas_to_scene[canvas_id] = scene_id
    rm._canvases[canvas_id] = _fake_canvas_view()
    return rm, bus, canvas_id


def test_gesture_synthesis_press_move_release_share_id():
    rm, bus, canvas_id = _render_manager_with_canvas()

    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_down"), canvas_id)
    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_move"), canvas_id)
    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_up"), canvas_id)

    actions = [e.action for e in bus.events]
    assert actions == ["press", "move", "release"]
    gesture_ids = {e.gesture_id for e in bus.events}
    assert len(gesture_ids) == 1
    assert next(iter(gesture_ids)) is not None
    assert bus.events[0].buttons == (1,)


def test_fresh_press_has_new_gesture_id():
    rm, bus, canvas_id = _render_manager_with_canvas()

    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_down"), canvas_id)
    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_up"), canvas_id)
    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_down"), canvas_id)

    assert bus.events[0].gesture_id != bus.events[2].gesture_id


def test_hover_move_without_press_has_none_gesture_id():
    rm, bus, canvas_id = _render_manager_with_canvas()

    # A move with no preceding press (e.g. after release cleared state).
    hover = _fake_pygfx_event("pointer_move", buttons=())
    rm._on_canvas_pointer_event(hover, canvas_id)

    assert bus.events[0].action == "move"
    assert bus.events[0].gesture_id is None


# ---------------------------------------------------------------------------
# element-level pick detail extraction
# ---------------------------------------------------------------------------


def _controller_with_points():
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32
    )
    visual = controller.add_points(
        data=PointsMemoryStore(positions=positions),
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(),
        name="points",
    )
    return controller, scene.id, visual.id


def _controller_with_lines():
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    # 3 edges -> 6 vertices.
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    visual = controller.add_lines(
        data=LinesMemoryStore(positions=positions),
        scene_id=scene.id,
        appearance=LinesMemoryAppearance(),
        name="lines",
    )
    return controller, scene.id, visual.id


def test_extract_points_pick_details():
    controller, scene_id, visual_id = _controller_with_points()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)

    details = rm._extract_pick_details(scene_id, node, {"vertex_index": 2})
    assert details == PointsPickInfo(point_index=2)


def test_extract_points_missing_index_returns_none():
    controller, scene_id, visual_id = _controller_with_points()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)

    assert rm._extract_pick_details(scene_id, node, {}) is None


def test_extract_lines_vertex_to_edge_mapping():
    controller, scene_id, visual_id = _controller_with_lines()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)

    # vertex_index 2k and 2k+1 both map to edge k.
    for vertex_index, edge_index in [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)]:
        details = rm._extract_pick_details(
            scene_id, node, {"vertex_index": vertex_index}
        )
        assert details == LinesPickInfo(edge_index=edge_index)


def _controller_with_mesh():
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    idx = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    visual = controller.add_mesh(
        data=MeshMemoryStore(positions=pos, indices=idx, name="msh"),
        scene_id=scene.id,
        appearance=MeshFlatAppearance(),
        name="mesh",
    )
    return controller, scene.id, visual.id


def test_extract_mesh_pick_details():
    controller, scene_id, visual_id = _controller_with_mesh()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)

    details = rm._extract_pick_details(scene_id, node, {"face_index": 1})
    assert details == MeshPickInfo(face_index=1)


def test_extract_mesh_missing_index_returns_none():
    controller, scene_id, visual_id = _controller_with_mesh()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)

    assert rm._extract_pick_details(scene_id, node, {}) is None


def test_raw_event_details_forwarded_to_public_event():
    controller, scene_id = _make_2d_controller()
    received: list = []
    canvas_id = uuid4()
    visual_id = uuid4()
    controller.on_mouse_press_2d(canvas_id, received.append, owner_id=uuid4())

    raw = _CanvasRawPointerEvent(
        canvas_id=canvas_id,
        scene_id=scene_id,
        action="press",
        camera_type="2d",
        position_2d=np.array([0.0, 0.0], dtype=np.float64),
        ray=None,
        hit_visual_id=visual_id,
        button=1,
        modifiers=(),
        buttons=(1,),
        gesture_id=uuid4(),
        pick_details=PointsPickInfo(point_index=3),
    )
    controller._on_raw_pointer_event(raw)

    assert received[0].pick_info.hit_visual_id == visual_id
    assert received[0].pick_info.details == PointsPickInfo(point_index=3)


def test_raw_event_miss_has_none_details():
    controller, scene_id = _make_2d_controller()
    received: list = []
    canvas_id = uuid4()
    controller.on_mouse_press_2d(canvas_id, received.append, owner_id=uuid4())

    controller._on_raw_pointer_event(_raw_2d(canvas_id, scene_id, "press"))

    assert received[0].pick_info.hit_visual_id is None
    assert received[0].pick_info.details is None


# ---------------------------------------------------------------------------
# Phase 4: readback gate
# ---------------------------------------------------------------------------


def _register_fake_canvas(rm, scene_id):
    bus = _RecordingBus()
    rm._event_bus = bus
    canvas_id = uuid4()
    rm._canvas_to_scene[canvas_id] = scene_id
    rm._canvases[canvas_id] = _fake_canvas_view()
    return bus, canvas_id


def test_gate_off_by_default_details_none_but_hit_visual_set():
    controller, scene_id, visual_id = _controller_with_points()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)
    bus, canvas_id = _register_fake_canvas(rm, scene_id)

    ev = _fake_pygfx_event("pointer_down")
    ev.pick_info = {"world_object": node, "vertex_index": 1}
    rm._on_canvas_pointer_event(ev, canvas_id)

    raw = bus.events[0]
    assert raw.hit_visual_id == visual_id
    assert raw.pick_details is None


def test_gate_on_extracts_details():
    controller, scene_id, visual_id = _controller_with_points()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)
    bus, canvas_id = _register_fake_canvas(rm, scene_id)
    rm.set_pick_details_enabled(canvas_id, True)

    ev = _fake_pygfx_event("pointer_down")
    ev.pick_info = {"world_object": node, "vertex_index": 1}
    rm._on_canvas_pointer_event(ev, canvas_id)

    assert bus.events[0].pick_details == PointsPickInfo(point_index=1)


def test_on_mouse_subscription_enables_gate_and_unsubscribe_disables():
    controller, _scene_id, _ = _controller_with_points()
    rm = controller._render_manager
    canvas_id = uuid4()

    assert rm._pick_details_enabled.get(canvas_id, False) is False

    handle = controller.on_mouse_press_2d(canvas_id, lambda e: None, owner_id=uuid4())
    assert rm._pick_details_enabled.get(canvas_id) is True

    controller.unsubscribe_mouse(handle)
    assert rm._pick_details_enabled.get(canvas_id) is False


def test_gate_stays_enabled_until_last_subscriber_removed():
    controller, _scene_id, _ = _controller_with_points()
    rm = controller._render_manager
    canvas_id = uuid4()
    owner = uuid4()

    h1 = controller.on_mouse_press_2d(canvas_id, lambda e: None, owner_id=owner)
    h2 = controller.on_mouse_move_2d(canvas_id, lambda e: None, owner_id=owner)
    assert rm._pick_details_enabled.get(canvas_id) is True

    controller.unsubscribe_mouse(h1)
    assert rm._pick_details_enabled.get(canvas_id) is True

    controller.unsubscribe_mouse(h2)
    assert rm._pick_details_enabled.get(canvas_id) is False


def test_direct_bus_subscription_does_not_enable_gate():
    controller, _scene_id, _ = _controller_with_points()
    rm = controller._render_manager
    canvas_id = uuid4()

    # Mimic the paint controller: subscribe directly on the bus.
    controller._outgoing_events.subscribe(
        CanvasMouseMove2DEvent,
        lambda e: None,
        entity_id=canvas_id,
        owner_id=uuid4(),
    )
    assert rm._pick_details_enabled.get(canvas_id, False) is False


# ---------------------------------------------------------------------------
# Image/labels pick coordinate is promoted into true (z, y, x) data order
# ---------------------------------------------------------------------------


def test_image_pick_coord_2d_promoted_in_data_axis_order():
    """The displayed-axis values land on their true data axes, not transposed.

    The render layer decodes the displayed coordinate in pygfx ``(x, y)`` order
    (column, row).  For a 2-D view of a (z, y, x) volume with z sliced, the
    promoted ``data_coordinate`` must be ``(z_slice, y=row, x=col)`` — i.e. the
    pygfx order is reversed onto the ascending displayed axes.
    """
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="s")
    scene.dims.selection.slice_indices = {0: 5}
    scene.dims.selection.displayed_axes = (1, 2)

    canvas_id = uuid4()
    received: list = []
    controller.on_mouse_press_2d(canvas_id, received.append, owner_id=uuid4())

    # pygfx-order displayed coord: x (col) = 10.5, y (row) = 3.5.
    raw = _CanvasRawPointerEvent(
        canvas_id=canvas_id,
        scene_id=scene.id,
        action="press",
        camera_type="2d",
        position_2d=np.array([10.5, 3.5], dtype=np.float64),
        ray=None,
        hit_visual_id=uuid4(),
        button=1,
        modifiers=(),
        buttons=(1,),
        gesture_id=None,
        pick_details=_ImageDisplayedDataCoord(displayed_data_coord=(10.5, 3.5)),
    )
    controller._on_raw_pointer_event(raw)

    assert len(received) == 1
    details = received[0].pick_info.details
    assert isinstance(details, ImagePickInfo)
    # axis 0 (z) = slice index, axis 1 (y) = row, axis 2 (x) = column.
    assert tuple(details.data_coordinate) == (5.0, 3.5, 10.5)


def test_image_pick_coord_3d_promoted_in_data_axis_order():
    """3-D pick: pygfx ``(x, y, z)`` is reversed onto displayed axes (z, y, x)."""
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="s")
    scene.dims.selection.displayed_axes = (0, 1, 2)

    canvas_id = uuid4()
    received: list = []
    controller.on_mouse_press_3d(canvas_id, received.append, owner_id=uuid4())

    ray = ViewRay(
        origin=np.zeros(3, dtype=np.float64),
        direction=np.array([0.0, 0.0, 1.0]),
    )
    # pygfx-order displayed coord: x = 20.5, y = 4.5, z = 1.5.
    raw = _CanvasRawPointerEvent(
        canvas_id=canvas_id,
        scene_id=scene.id,
        action="press",
        camera_type="3d",
        position_2d=None,
        ray=ray,
        hit_visual_id=uuid4(),
        button=1,
        modifiers=(),
        buttons=(1,),
        gesture_id=None,
        pick_details=_ImageDisplayedDataCoord(displayed_data_coord=(20.5, 4.5, 1.5)),
    )
    controller._on_raw_pointer_event(raw)

    assert len(received) == 1
    details = received[0].pick_info.details
    assert isinstance(details, ImagePickInfo)
    # (z, y, x) — z and x swap relative to the pygfx (x, y, z) order.
    assert tuple(details.data_coordinate) == (1.5, 4.5, 20.5)
