"""Script to make the config file for a single config."""

import numpy as np

from cellier.models.data_manager import DataManager
from cellier.models.data_stores.image import MockLatentImageStore
from cellier.models.data_streams.image import ImageSynchronousDataStream
from cellier.models.nodes.image_node import ImageMIPMaterial, ImageNode
from cellier.models.scene.cameras import PerspectiveCamera
from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import CoordinateSystem, DimsManager
from cellier.models.scene.scene import Scene
from cellier.models.viewer import SceneManager, ViewerModel

# make a 4D point cloud

rng = np.random.default_rng(0)
volume_data = rng.random((3, 10, 10, 10))
print(volume_data.shape)


# make the points store
image_store = MockLatentImageStore(data=volume_data, slice_time=1, name="image store")

# make the points stream
image_stream = ImageSynchronousDataStream(data_store_id=image_store.id, selectors=[])

# make the data_stores manager
data = DataManager(
    stores={image_store.id: image_store}, streams={image_stream.id: image_stream}
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

# make the image visual
image_material_3d = ImageMIPMaterial()
image_visual_3d = ImageNode(
    name="points_node_3d", data_stream_id=image_stream.id, material=image_material_3d
)

# make the canvas
camera_3d = PerspectiveCamera()
canvas_3d = Canvas(camera=camera_3d)

# make the scene
scene_3d = Scene(
    dims=dims_3d, visuals=[image_visual_3d], canvases={canvas_3d.id: canvas_3d}
)

scene_manager = SceneManager(scenes={scene_3d.id: scene_3d})

# make the viewer model
viewer_model = ViewerModel(data=data, scenes=scene_manager)

print(viewer_model)

viewer_model.to_json_file("latent_volume_example_config.json")
