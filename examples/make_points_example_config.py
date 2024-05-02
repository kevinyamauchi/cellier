"""Script to to create a points viewer configuration."""

import numpy as np

from cellier.models.data_manager import DataManager
from cellier.models.data_stores.points import PointsMemoryStore
from cellier.models.data_streams.points import PointsSynchronousDataStream
from cellier.models.scene.cameras import PerspectiveCamera
from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import CoordinateSystem, DimsManager
from cellier.models.scene.scene import Scene
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.models.visuals.points_visual import PointsUniformMaterial, PointsVisual

# make a 4D point cloud
coordinates = np.array(
    [[0, 10, 10, 10], [0, 10, 10, 20], [5, 10, 20, 20]], dtype=np.float32
)


# make the points store
points_store = PointsMemoryStore(coordinates=coordinates)

# make the points stream
points_stream = PointsSynchronousDataStream(data_store_id=points_store.id, selectors=[])

# make the data_stores manager
data = DataManager(
    stores={points_store.id: points_store}, streams={points_stream.id: points_stream}
)

# make the scene coordinate system
coordinate_system = CoordinateSystem(name="scene_0", axis_labels=["t", "z", "y", "x"])
dims = DimsManager(
    point=(0, 0, 0, 0),
    margin_negative=(0, 0, 0, 0),
    margin_positive=(0, 0, 0, 0),
    coordinate_system=coordinate_system,
    displayed_dimensions=(1, 2, 3),
)

# make the points visual
points_material = PointsUniformMaterial(
    size=1, color=(1, 1, 1, 1), size_coordinate_space="data"
)
points_visual = PointsVisual(
    name="points_visual", data_stream_id=points_stream.id, material=points_material
)

# make the canvas
camera = PerspectiveCamera()
canvas = Canvas(camera=camera)

# make the scene
scene = Scene(dims=dims, visuals=[points_visual], canvases=[canvas])
scene_manager = SceneManager(scenes={scene.id: scene})

# make the viewer model
viewer_model = ViewerModel(data=data, scenes=scene_manager)

print(viewer_model)

viewer_model.to_json_file("points_example_config.json")
