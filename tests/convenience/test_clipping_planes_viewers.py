"""Clipping planes through the convenience viewers (design 3.3, D20, D30)."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.convenience import OrthoViewer, Viewer
from cellier.data import LabelMemoryStore, PointsMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import ClippingPlane


def _system() -> DataCoordinateSystem:
    return DataCoordinateSystem(
        name="data",
        datastore_id=uuid4(),
        axes=tuple(Axis(name=n, axis_type="space", sampling="discrete") for n in "zyx"),
    )


def _plane(system, x=4.0) -> ClippingPlane:
    return ClippingPlane.from_point_normal(system, (4, 4, x), (0, 0, 1))


def test_store_then_planes_then_visual(qtbot):
    """D30: the plane is built from the store's system before the visual."""
    system = _system()
    store = LabelMemoryStore(
        data=np.ones((8, 9, 10), np.int32), data_coordinate_systems=[system]
    )
    plane = _plane(store.data_coordinate_systems[0])
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    try:
        visual = viewer.add_labels(store, clipping_planes=(plane,))
        assert visual.clipping_planes == (plane,)
        visual.clipping_planes = (plane, _plane(system, 6.0))
        assert len(visual.clipping_planes) == 2
    finally:
        viewer.controller.close()


def test_a_store_with_no_system_gets_planes_after_the_visual(qtbot):
    store = PointsMemoryStore(positions=np.zeros((3, 3), dtype=np.float32))
    assert store.data_coordinate_systems == []
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    try:
        visual = viewer.add_points(store)
        visual.clipping_planes = (_plane(store.data_coordinate_systems[0]),)
        assert len(visual.clipping_planes) == 1
    finally:
        viewer.controller.close()


def test_the_ortho_panels_share_one_tuple(qtbot):
    """D20: one store, one system, so the planes are linked across views."""
    system = _system()
    store = LabelMemoryStore(
        data=np.ones((8, 9, 10), np.int32), data_coordinate_systems=[system]
    )
    plane = _plane(system)
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    try:
        visuals = ortho.add_labels(store, clipping_planes=(plane,))
        assert {v.clipping_planes for v in visuals.values()} == {(plane,)}

        moved = (_plane(system, 6.0),)
        visuals["xz"].clipping_planes = moved
        assert {v.clipping_planes for v in visuals.values()} == {moved}
        visuals["vol"].clipping_planes = ()
        assert {v.clipping_planes for v in visuals.values()} == {()}

        # Refused on one panel: no panel changes.
        with pytest.raises(Exception, match="data coordinate system"):
            visuals["xy"].clipping_planes = (_plane(_system()),)
        assert {v.clipping_planes for v in visuals.values()} == {()}
    finally:
        ortho.controller.close()


def test_planes_survive_save_and_load(qtbot, tmp_path):
    system = _system()
    store = PointsMemoryStore(
        positions=np.arange(12, dtype=np.float32).reshape(4, 3),
        data_coordinate_systems=[system],
    )
    planes = (_plane(system), _plane(system, 6.0).model_copy(update={"enabled": False}))
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    try:
        visual = viewer.add_points(store, clipping_planes=planes)
        visual_id = visual.id
        viewer.to_file(tmp_path / "viewer.json")
    finally:
        viewer.controller.close()

    restored = Viewer.from_file(tmp_path / "viewer.json")
    try:
        controller = restored.controller
        visual = controller.get_visual_model(visual_id)
        assert visual.clipping_planes == planes
        # Still the store's system, and already on the render visual.
        store = controller.get_data_store(system.datastore_id)
        assert visual.clipping_planes[0].plane.coordinate_system == (
            store.data_coordinate_system.id
        )
        scene_manager = controller._render_manager._scenes[restored.scene.id]
        assert scene_manager.get_visual(visual_id).clipping_planes == planes
    finally:
        restored.controller.close()
