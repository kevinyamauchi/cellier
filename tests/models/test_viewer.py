"""Test the viewer model."""

import numpy as np

from cellier.models.data_manager import DataManager
from cellier.models.data_stores import PointsMemoryStore
from cellier.models.scene.cameras import PerspectiveCamera
from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import CoordinateSystem, DimsManager
from cellier.models.scene.scene import Scene
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.models.visuals import PointsUniformMaterial, PointsVisual


def test_viewer(tmp_path):
    """Test serialization/deserialization of the viewer model."""
    # the point data_stores
    coordinates = np.array([[10, 10, 10], [30, 10, 20], [10, 20, 20]], dtype=np.float32)
    points_store = PointsMemoryStore(coordinates=coordinates)

    # make the data_stores manager
    data = DataManager(stores={points_store.id: points_store})

    # make the scene coordinate system
    coordinate_system = CoordinateSystem(name="scene_0", axis_labels=["z", "y", "x"])
    dims = DimsManager(
        coordinate_system=coordinate_system, displayed_dimensions=(0, 1, 2)
    )

    # make the mesh visual
    points_material = PointsUniformMaterial(
        color=(1, 0, 0, 1),
        size=1,
    )
    points_visual = PointsVisual(
        name="mesh_visual", data_store_id=points_store.id, material=points_material
    )

    # make the canvas
    camera = PerspectiveCamera(width=110, height=110)
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
