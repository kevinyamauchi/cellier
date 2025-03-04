"""Demo making a simple multiscale viewer using Cellier."""

import numpy as np
from qtpy import QtWidgets

from cellier.models.data_manager import DataManager
from cellier.models.data_stores.image import MultiScaleImageZarrStore
from cellier.models.data_streams.image import MultiscaleImageDataStream
from cellier.models.nodes.image_node import ImageIsoMaterial, MultiscaleImageNode
from cellier.models.scene import (
    Canvas,
    CoordinateSystem,
    DimsManager,
    PerspectiveCamera,
    Scene,
)
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.slicer.slicer import SlicerType
from cellier.viewer_controller import CellierController

n_scales = 5

# make the image store
image_store = MultiScaleImageZarrStore(
    root_path="multiscale_blobs.zarr",
    scale_paths=[f"level_{scale_index}" for scale_index in range(5)],
    scales=[
        (2**scale_index, 2**scale_index, 2**scale_index) for scale_index in range(5)
    ],
    translations=[(0, 0, 0) for scale_index in range(n_scales)],
)

# make the points stream
image_stream = MultiscaleImageDataStream(data_store_id=image_store.id, selectors=[])

# make the data_stores manager
data = DataManager(
    stores={image_store.id: image_store}, streams={image_stream.id: image_stream}
)

# make the scene coordinate system
coordinate_system_3d = CoordinateSystem(name="scene_3d", axis_labels=["z", "y", "x"])
dims_3d = DimsManager(
    point=(0, 0, 0),
    margin_negative=(0, 0, 0),
    margin_positive=(0, 0, 0),
    coordinate_system=coordinate_system_3d,
    displayed_dimensions=(0, 1, 3),
)

# make the image visual
image_material_3d = ImageIsoMaterial()
image_visual_3d = MultiscaleImageNode(
    name="image_node_3d",
    n_scales=n_scales,
    data_stream_id=image_stream.id,
    material=image_material_3d,
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


class Main(QtWidgets.QWidget):
    """Example widget with viewer."""

    def __init__(self, viewer_config_path: str):
        super().__init__(None)
        self.resize(640, 480)

        # make the viewer
        self.viewer = CellierController(
            model=viewer_model, slicer_type=SlicerType.ASYNCHRONOUS, widget_parent=self
        )

        for cam in self.viewer._render_manager.cameras.values():
            cam.set_state(
                {
                    "fov": 0,
                    "x": 512,
                    "y": -1261,
                    "z": 512,
                    "width": 1773.6200269505302,
                    "height": 1773.6200269505302,
                    "rotation": np.array(
                        [0.7071067811865475, 0.0, 0.0, 0.7071067811865476]
                    ),
                    "reference_up": [0, 0, 1],
                    "scale": [1, 1, 1],
                }
            )
            print(cam.get_state())

        for scene in self.viewer._model.scenes.scenes.values():
            scene_id = scene.id
            canvas_id = next(iter(scene.canvases))
            self.viewer._render_manager.animate(scene_id=scene_id, canvas_id=canvas_id)

        print("reslicing")
        self.viewer.reslice_all()

        self._button = QtWidgets.QPushButton("Draw current view", self)
        self._button.clicked.connect(self._on_button_click)

        layout = QtWidgets.QHBoxLayout()
        for canvas in self.viewer._canvas_widgets.values():
            # add the canvas widgets
            canvas.update()
            layout.addWidget(canvas)
        layout.addWidget(self._button)
        self.setLayout(layout)

    def _on_button_click(self):
        self.viewer.reslice_all()


CONFIG_PATH = "simple_volume_example_config.json"

app = QtWidgets.QApplication([])
m = Main(CONFIG_PATH)
m.show()


if __name__ == "__main__":
    app.exec()
