"""Example app showing 2D and 3D labels rendering."""

from qtpy import QtWidgets
from skimage.data import binary_blobs
from skimage.measure import label
from superqt import QLabeledSlider

from cellier.models.data_manager import DataManager
from cellier.models.data_stores import ImageMemoryStore
from cellier.models.scene import (
    Canvas,
    CoordinateSystem,
    DimsManager,
    PerspectiveCamera,
    Scene,
)
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.models.visuals import LabelsMaterial, MultiscaleLabelsVisual
from cellier.viewer_controller import CellierController


class Main(QtWidgets.QWidget):
    """Example widget with viewer."""

    def __init__(self, viewer_model):
        super().__init__(None)
        self.resize(640, 480)

        # make the viewer
        self.viewer = CellierController(model=viewer_model, widget_parent=self)

        # make the slider for the z axis in the 2D canvas
        self.z_slider_widget = QLabeledSlider(parent=self)
        self.z_slider_widget.valueChanged.connect(self._on_z_slider_changed)
        self.z_slider_widget.setValue(125)
        self.z_slider_widget.setRange(0, 249)

        layout = QtWidgets.QHBoxLayout()
        # layout.addWidget(self.z_slider_widget)
        for canvas in self.viewer._canvas_widgets.values():
            # add the canvas widgets
            canvas.update()
            print(canvas)
            layout.addWidget(canvas)
        self.setLayout(layout)

    def _on_z_slider_changed(self, slider_value: int):
        for scene in self.viewer._model.scenes.scenes.values():
            dims_manager = scene.dims
            coordinate_system = dims_manager.coordinate_system
            if coordinate_system.name == "scene_2d":
                dims_point = list(dims_manager.point)
                dims_point[0] = slider_value
                dims_manager.point = dims_point

                self.viewer.reslice_scene(scene_id=scene.id)


# make the data
im = binary_blobs(length=250, volume_fraction=0.1, n_dim=3)
label_image = label(im)

# make the data store
data_store = ImageMemoryStore(data=label_image, name="label_image")

# make the data manager
# make the data_stores manager
data = DataManager(stores={data_store.id: data_store})

# make the 2D scene coordinate system
coordinate_system_2d = CoordinateSystem(name="scene_2d", axis_labels=("z", "y", "x"))
dims_2d = DimsManager(
    point=(125, 0, 0),
    margin_negative=(0, 0, 0),
    margin_positive=(0, 0, 0),
    coordinate_system=coordinate_system_2d,
    displayed_dimensions=(1, 2),
)

# make the 2D labels visual
labels_material = LabelsMaterial(color_map="glasbey:glasbey")
labels_visual_2d = MultiscaleLabelsVisual(
    name="labels_node_2d",
    data_store_id=data_store.id,
    material=labels_material,
    downscale_factors=[1],
)


# make the 3D scene coordinate system
coordinate_system_3d = CoordinateSystem(name="scene_3d", axis_labels=("z", "y", "x"))
dims_3d = DimsManager(
    point=(0, 0, 0),
    margin_negative=(0, 0, 0),
    margin_positive=(0, 0, 0),
    coordinate_system=coordinate_system_3d,
    displayed_dimensions=(0, 1, 2),
)

# make the 3D labels visual
labels_material = LabelsMaterial(color_map="glasbey:glasbey")
labels_visual_3d = MultiscaleLabelsVisual(
    name="labels_node_3d",
    data_store_id=data_store.id,
    material=labels_material,
    downscale_factors=[1],
)

# make the 2D canvas
camera_2d = PerspectiveCamera(fov=0)
canvas_2d = Canvas(camera=camera_2d)

# make the 3D canvas
camera_3d = PerspectiveCamera()
canvas_3d = Canvas(camera=camera_3d)

# make the scenes
scene_2d = Scene(
    dims=dims_2d, visuals=[labels_visual_2d], canvases={canvas_2d.id: canvas_2d}
)
scene_3d = Scene(
    dims=dims_3d, visuals=[labels_visual_3d], canvases={canvas_3d.id: canvas_3d}
)

scene_manager = SceneManager(
    scenes={
        scene_2d.id: scene_2d,
        scene_3d.id: scene_3d,
    }
)

# make the viewer model
model = ViewerModel(data=data, scenes=scene_manager)


app = QtWidgets.QApplication([])
m = Main(model)
m.show()

if __name__ == "__main__":
    app.exec()
