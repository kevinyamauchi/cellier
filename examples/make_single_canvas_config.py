"""Script to show how to assemble a viewer model."""

import numpy as np

from cellier.models.data_manager import DataManager
from cellier.models.data_stores.mesh import MeshMemoryStore
from cellier.models.data_streams.mesh import MeshSynchronousDataStream
from cellier.models.nodes.mesh_node import MeshNode, MeshPhongMaterial
from cellier.models.scene.cameras import PerspectiveCamera
from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import CoordinateSystem, DimsManager
from cellier.models.scene.scene import Scene
from cellier.models.viewer import SceneManager, ViewerModel

# the mesh data_stores
vertices = np.array([[10, 10, 10], [10, 10, 20], [10, 20, 20]], dtype=np.float32)
faces = np.array([[0, 1, 2]], dtype=np.float32)

colors = np.array(
    [
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ],
    dtype=np.float32,
)

# make the mesh store
mesh_store = MeshMemoryStore(vertices=vertices, faces=faces)

# make the mesh stream
mesh_stream = MeshSynchronousDataStream(data_store_id=mesh_store.id, selectors=[])

# make the data_stores manager
data = DataManager(
    stores={mesh_store.id: mesh_store}, streams={mesh_stream.id: mesh_stream}
)

# make the scene coordinate system
coordinate_system = CoordinateSystem(name="scene_0", axis_labels=["z", "y", "x"])
dims = DimsManager(coordinate_system=coordinate_system, displayed_dimensions=(0, 1, 2))

# make the mesh visual
mesh_material = MeshPhongMaterial()
mesh_visual = MeshNode(
    name="mesh_visual", data_stream_id=mesh_stream.id, material=mesh_material
)

# make the canvas
camera = PerspectiveCamera()
canvas = Canvas(camera=camera)

# make the scene
scene = Scene(dims=dims, visuals=[mesh_visual], canvases=[canvas])
scene_manager = SceneManager(scenes={scene.id: scene})

# make the viewer model
viewer_model = ViewerModel(data=data, scenes=scene_manager)

print(viewer_model)

viewer_model.to_json_file("single_canvas_config.json")
