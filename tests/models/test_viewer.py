"""Test the viewer model."""

import numpy as np

from cellier.models.data_manager import DataManager
from cellier.models.data_stores import PointsMemoryStore
from cellier.models.scene import (
    AxisAlignedRegionSelector,
    Canvas,
    CoordinateSystem,
    DimsManager,
    OrbitCameraController,
    PerspectiveCamera,
    RangeTuple,
    Scene,
)
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.models.visuals import PointsUniformMaterial, PointsVisual
from cellier.types import CoordinateSpace


def test_viewer(tmp_path):
    """Test serialization/deserialization of the viewer model."""
    # the point data_stores
    coordinates = np.array([[10, 10, 10], [30, 10, 20], [10, 20, 20]], dtype=np.float32)
    points_store = PointsMemoryStore(coordinates=coordinates)

    # make the data_stores manager
    data = DataManager(stores={points_store.id: points_store})

    # make the scene coordinate system
    data_range = (
        RangeTuple(start=0, stop=250, step=1),
        RangeTuple(start=0, stop=250, step=1),
        RangeTuple(start=0, stop=250, step=1),
    )
    coordinate_system_3d = CoordinateSystem(
        name="scene_3d", axis_labels=("z", "y", "x")
    )
    selection = AxisAlignedRegionSelector(
        space_type=CoordinateSpace.WORLD,
        ordered_dims=(0, 1, 2),
        n_displayed_dims=3,
        index_selection=(0, slice(0, 10, 1), slice(None, None, None)),
    )
    dims = DimsManager(
        range=data_range,
        coordinate_system=coordinate_system_3d,
        selection=selection,
    )

    # make the points visual
    points_material = PointsUniformMaterial(
        color=(1, 0, 0, 1),
        size=1,
    )
    points_visual = PointsVisual(
        name="points_visual", data_store_id=points_store.id, material=points_material
    )

    # make the canvas
    camera = PerspectiveCamera(
        width=110, height=110, controller=OrbitCameraController(enabled=True)
    )
    canvas = Canvas(camera=camera)

    # make the scene
    scene = Scene(dims=dims, visuals=[points_visual], canvases={canvas.id: canvas})
    scene_manager = SceneManager(scenes={scene.id: scene})

    viewer_model = ViewerModel(data=data, scenes=scene_manager)

    # serialize
    output_path = tmp_path / "test.json"
    viewer_model.to_json_file(output_path)

    # deserialize
    deserialized_viewer = ViewerModel.from_json_file(output_path)

    assert viewer_model.scenes == deserialized_viewer.scenes
