"""Tests for the typed pick-info payloads and event context fields."""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from cellier.events._events import (
    CanvasMouseMove2DEvent,
    CanvasMouseMove3DEvent,
    CanvasMousePress2DEvent,
    CanvasMousePress3DEvent,
    CanvasMouseRelease2DEvent,
    CanvasMouseRelease3DEvent,
    CanvasPickInfo,
    ImagePickInfo,
    LabelsPickInfo,
    LinesPickInfo,
    MeshPickInfo,
    PointsPickInfo,
    ViewRay,
)


def test_points_pick_info_fields() -> None:
    info = PointsPickInfo(point_index=7)
    assert info.point_index == 7


def test_lines_pick_info_fields() -> None:
    info = LinesPickInfo(edge_index=3)
    assert info.edge_index == 3


def test_image_pick_info_fields() -> None:
    info = ImagePickInfo(data_coordinate=(1.5, 2.5, 3.0))
    assert info.data_coordinate == (1.5, 2.5, 3.0)


def test_mesh_pick_info_fields() -> None:
    info = MeshPickInfo(face_index=9)
    assert info.face_index == 9


def test_labels_pick_info_fields() -> None:
    info = LabelsPickInfo(data_coordinate=(0.0, 1.0, 2.0, 4.0, 5.0))
    assert info.data_coordinate == (0.0, 1.0, 2.0, 4.0, 5.0)


def test_canvas_pick_info_details_defaults_none() -> None:
    info = CanvasPickInfo(hit_visual_id=None)
    assert info.details is None

    vid = uuid4()
    hit = CanvasPickInfo(hit_visual_id=vid, details=PointsPickInfo(point_index=4))
    assert hit.hit_visual_id == vid
    assert hit.details.point_index == 4


def test_2d_events_context_fields_default() -> None:
    pick = CanvasPickInfo(hit_visual_id=None)
    coord = np.zeros(3, dtype=np.float64)
    for cls in (
        CanvasMousePress2DEvent,
        CanvasMouseMove2DEvent,
        CanvasMouseRelease2DEvent,
    ):
        event = cls(
            source_id=uuid4(),
            scene_id=uuid4(),
            world_coordinate=coord,
            pick_info=pick,
        )
        assert event.button == 0
        assert event.buttons == ()
        assert event.modifiers == ()
        assert event.gesture_id is None


def test_3d_events_context_fields_default() -> None:
    pick = CanvasPickInfo(hit_visual_id=None)
    ray = ViewRay(
        origin=np.zeros(3, dtype=np.float64),
        direction=np.array([0.0, 0.0, 1.0]),
    )
    for cls in (
        CanvasMousePress3DEvent,
        CanvasMouseMove3DEvent,
        CanvasMouseRelease3DEvent,
    ):
        event = cls(
            source_id=uuid4(),
            scene_id=uuid4(),
            ray=ray,
            pick_info=pick,
        )
        assert event.button == 0
        assert event.buttons == ()
        assert event.modifiers == ()
        assert event.gesture_id is None
