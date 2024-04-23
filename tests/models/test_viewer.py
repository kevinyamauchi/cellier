"""Test the viewer model."""

import numpy as np

from cellier.models.data_stores.mesh import MeshMemoryStore
from cellier.models.data_streams.mesh import MeshSynchronousDataStream
from cellier.models.scene.cameras import PerspectiveCamera
from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import CoordinateSystem, DimsManager
from cellier.models.scene.scene import Scene
from cellier.models.viewer import DataManager, SceneManager, ViewerModel
from cellier.models.visuals.mesh_visual import MeshPhongMaterial, MeshVisual


def test_viewer(tmp_path):
    """Test serialization/deserialization of the viewer model."""
    # the mesh data_stores
    vertices = np.array([[10, 10, 10], [30, 10, 20], [10, 20, 20]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    # make the mesh store
    mesh_store = MeshMemoryStore(vertices=vertices, faces=faces)

    # make the mesh stream
    mesh_stream = MeshSynchronousDataStream(data_store=mesh_store, selectors=[])

    # make the data_stores manager
    data = DataManager(stores=[mesh_store], streams=[mesh_stream])

    # make the scene coordinate system
    coordinate_system = CoordinateSystem(name="scene_0", axis_labels=["z", "y", "x"])
    dims = DimsManager(
        coordinate_system=coordinate_system, displayed_dimensions=("z", "y", "x")
    )

    # make the mesh visual
    mesh_material = MeshPhongMaterial()
    mesh_visual = MeshVisual(
        name="mesh_visual", data_stream=mesh_stream, material=mesh_material
    )

    # make the canvas
    camera = PerspectiveCamera(width=110, height=110)
    canvas = Canvas(camera=camera)

    # make the scene
    scene = Scene(dims=dims, visuals=[mesh_visual], canvases=[canvas])
    scene_manager = SceneManager(scenes=[scene])

    viewer_model = ViewerModel(data=data, scenes=scene_manager)

    # serialize
    output_path = tmp_path / "test.json"
    viewer_model.to_json_file(output_path)

    # deserialize
    deserialized_viewer = ViewerModel.from_json_file(output_path)

    assert viewer_model.scenes == deserialized_viewer.scenes
