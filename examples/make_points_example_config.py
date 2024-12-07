"""Script to to create a points viewer configuration."""

import numpy as np

from cellier.models.data_manager import DataManager
from cellier.models.data_stores.points import PointsMemoryStore
from cellier.models.data_streams.points import PointsSynchronousDataStream
from cellier.models.nodes.points_node import PointsNode, PointsUniformMaterial
from cellier.models.scene.cameras import PerspectiveCamera
from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import CoordinateSystem, DimsManager
from cellier.models.scene.scene import Scene
from cellier.models.viewer import SceneManager, ViewerModel

# make a 4D point cloud
n_points = 500
rng = np.random.default_rng(0)
spatial_coordinates = 50 * rng.uniform(0, 1, (n_points, 3))
temporal_coordinates = rng.choice(np.arange(10), (n_points, 1))
coordinates = np.column_stack((temporal_coordinates, spatial_coordinates))
print(coordinates.shape)


# make the points store
points_store = PointsMemoryStore(coordinates=coordinates)

# make the points stream
points_stream = PointsSynchronousDataStream(data_store_id=points_store.id, selectors=[])

# make the data_stores manager
data = DataManager(
    stores={points_store.id: points_store}, streams={points_stream.id: points_stream}
)

# make the scene coordinate system
coordinate_system_3d = CoordinateSystem(
    name="scene_3d", axis_labels=["t", "z", "y", "x"]
)
dims_3d = DimsManager(
    point=(0, 0, 0, 0),
    margin_negative=(0, 0, 0, 0),
    margin_positive=(0, 0, 0, 0),
    coordinate_system=coordinate_system_3d,
    displayed_dimensions=(1, 2, 3),
)

coordinate_system_2d = CoordinateSystem(
    name="scene_2d", axis_labels=["t", "z", "y", "x"]
)
dims_2d = DimsManager(
    point=(0, 0, 0, 0),
    margin_negative=(0, 0.5, 0, 0),
    margin_positive=(0, 0.5, 0, 0),
    coordinate_system=coordinate_system_2d,
    displayed_dimensions=(2, 3),
)

# make the points visual
points_material_3d = PointsUniformMaterial(
    size=1, color=(1, 1, 1, 1), size_coordinate_space="data"
)
points_visual_3d = PointsNode(
    name="points_node_3d", data_stream_id=points_stream.id, material=points_material_3d
)

points_material_2d = PointsUniformMaterial(
    size=5, color=(1, 1, 1, 1), size_coordinate_space="data"
)
points_visual_2d = PointsNode(
    name="points_node_2d", data_stream_id=points_stream.id, material=points_material_2d
)

# make the canvas
camera_3d = PerspectiveCamera()
canvas_3d = Canvas(camera=camera_3d)

camera_2d = PerspectiveCamera()
canvas_2d = Canvas(camera=camera_2d)

# make the scene
scene_3d = Scene(
    dims=dims_3d, visuals=[points_visual_3d], canvases={canvas_3d.id: canvas_3d}
)
scene_2d = Scene(
    dims=dims_2d, visuals=[points_visual_2d], canvases={canvas_2d.id: canvas_2d}
)

scene_manager = SceneManager(scenes={scene_3d.id: scene_3d, scene_2d.id: scene_2d})

# make the viewer model
viewer_model = ViewerModel(data=data, scenes=scene_manager)

print(viewer_model)

viewer_model.to_json_file("points_example_config.json")
