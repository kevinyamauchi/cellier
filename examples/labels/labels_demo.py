"""Example app showing 2D and 3D labels rendering."""

from qtpy import QtWidgets
from skimage.data import binary_blobs
from skimage.measure import label

from cellier.app.interactivity import LabelsPaintingManager
from cellier.app.qt import QtCanvasWidget
from cellier.models.data_manager import DataManager
from cellier.models.data_stores import ImageMemoryStore
from cellier.models.scene import (
    Canvas,
    CoordinateSystem,
    DimsManager,
    PerspectiveCamera,
    RangeTuple,
    Scene,
)
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.models.visuals import LabelsMaterial, MultiscaleLabelsVisual
from cellier.viewer_controller import CellierController


class Main(QtWidgets.QWidget):
    """Example widget with viewer."""

    def __init__(
        self,
        viewer_model,
        labels_visual_model,
        labels_data_store,
    ):
        super().__init__(None)
        self.resize(640, 480)

        # make the viewer
        self.viewer = CellierController(model=viewer_model, widget_parent=self)

        # make the painting manager
        self._labels_painting_manager = LabelsPaintingManager(
            model=labels_visual_model,
            data_store=labels_data_store,
        )

        layout = QtWidgets.QHBoxLayout()
        # layout.addWidget(self.z_slider_widget)
        for canvas_id, canvas in self.viewer._canvas_widgets.items():
            # add the canvas widgets
            canvas.update()

            scenes = self.viewer._model.scenes.scenes

            for scene in scenes.values():
                # get the dims model the canvas
                if canvas_id in scene.canvases:
                    dims_model = scene.dims

            canvas_widget = QtCanvasWidget(
                dims_id=dims_model.id, canvas_widget=canvas, parent=self
            )

            # create the sliders
            dims_ranges = dims_model.range
            sliders_data = {
                axis_label: range(int(start), int(stop), int(step))
                for axis_label, (start, stop, step) in zip(
                    dims_model.coordinate_system.axis_labels,
                    dims_ranges,
                )
            }
            canvas_widget._dims_sliders.create_sliders(sliders_data)

            self.viewer.events.scene.add_dims_with_controls(
                dims_model=dims_model,
                dims_controls=canvas_widget._dims_sliders,
            )

            # redraw the scene when the dims model updates
            dims_model.events.point.connect(self._on_dims_update)

            dims_model.point = (10, 0, 0)

            layout.addWidget(canvas_widget)
            self.setLayout(layout)

            self._labels_visual_id = labels_visual_model.id
            # self.viewer.add_visual_callback(
            #     visual_id=self._labels_visual_id, callback=self._on_mouse_click
            # )

            # register the canvas
            for scene in self.viewer._model.scenes.scenes.values():
                for visual in scene.visuals:
                    if visual.id == self._labels_visual_id:
                        canvas_model = next(iter(scene.canvases.values()))
                        canvas_id = canvas_model.id
            self.viewer.events.mouse.register_canvas(canvas_id)
            self.viewer.events.mouse.subscribe_to_canvas(
                canvas_id, self._labels_painting_manager._on_mouse_press
            )
            self.viewer.events.mouse.subscribe_to_canvas(
                canvas_id, self._on_mouse_click
            )

        # renderer = list(self.viewer._render_manager.renderers.values())[0]
        # renderer.add_event_handler(self._on_mouse_other, "pointer_down")

    def _on_dims_update(self, event):
        for scene in self.viewer._model.scenes.scenes.values():
            dims_manager = scene.dims
            coordinate_system = dims_manager.coordinate_system
            if coordinate_system.name == "scene_2d":
                self.viewer.reslice_scene(scene_id=scene.id)


# make the data
im = binary_blobs(length=250, volume_fraction=0.1, n_dim=3)
label_image = label(im)

# make the data store
data_store = ImageMemoryStore(data=label_image, name="label_image")

# make the data manager
data = DataManager(stores={data_store.id: data_store})

# the range of the data in the scene
data_range = (
    RangeTuple(start=0, stop=250, step=1),
    RangeTuple(start=0, stop=250, step=1),
    RangeTuple(start=0, stop=250, step=1),
)

# make the 2D scene coordinate system
coordinate_system_2d = CoordinateSystem(name="scene_2d", axis_labels=("z", "y", "x"))
dims_2d = DimsManager(
    point=(125, 0, 0),
    margin_negative=(0, 0, 0),
    margin_positive=(0, 0, 0),
    range=data_range,
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
    range=data_range,
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
m = Main(
    model,
    labels_visual_model=labels_visual_2d,
    labels_data_store=data_store,
)
m.show()

if __name__ == "__main__":
    app.exec()
