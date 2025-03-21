"""Example app showing 2D and 3D labels rendering."""

from qtpy import QtWidgets
from skimage.data import binary_blobs
from skimage.measure import label

from cellier.app.interactivity import LabelsPaintingManager
from cellier.app.qt import QtCanvasWidget, QtQuadview
from cellier.convenience import get_canvas_with_visual_id, get_dims_with_canvas_id
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
from cellier.types import DataStoreId
from cellier.viewer_controller import CellierController


class Main(QtWidgets.QWidget):
    """Example widget with viewer."""

    def __init__(
        self,
        viewer_model,
        labels_xy,
        labels_xz,
        labels_zy,
        labels_data_store,
    ):
        super().__init__(None)
        # self.resize(640, 480)

        # make the viewer
        self.viewer = CellierController(model=viewer_model, widget_parent=self)

        # make the painting managers
        self._painting_manager_xy = LabelsPaintingManager(
            model=labels_xy,
            data_store=labels_data_store,
        )
        self._painting_manager_xz = LabelsPaintingManager(
            model=labels_xz,
            data_store=labels_data_store,
        )
        self._painting_manager_zy = LabelsPaintingManager(
            model=labels_zy,
            data_store=labels_data_store,
        )

        # make the xz canvas widget
        canvas_xz = get_canvas_with_visual_id(
            viewer_model=viewer_model,
            visual_id=labels_xz.id,
        )
        dims_xz = get_dims_with_canvas_id(
            viewer_model=viewer_model, canvas_id=canvas_xz.id
        )
        canvas_widget_xz = QtCanvasWidget.from_models(
            dims_model=dims_xz,
            render_canvas_widget=self.viewer._canvas_widgets[canvas_xz.id],
        )

        # make the xy canvas widget
        canvas_xy = get_canvas_with_visual_id(
            viewer_model=viewer_model,
            visual_id=labels_xy.id,
        )
        dims_xy = get_dims_with_canvas_id(
            viewer_model=viewer_model, canvas_id=canvas_xy.id
        )
        canvas_widget_xy = QtCanvasWidget.from_models(
            dims_model=dims_xy,
            render_canvas_widget=self.viewer._canvas_widgets[canvas_xy.id],
        )

        # make the zy canvas widget
        canvas_zy = get_canvas_with_visual_id(
            viewer_model=viewer_model,
            visual_id=labels_zy.id,
        )
        dims_zy = get_dims_with_canvas_id(
            viewer_model=viewer_model, canvas_id=canvas_zy.id
        )
        canvas_widget_zy = QtCanvasWidget.from_models(
            dims_model=dims_zy,
            render_canvas_widget=self.viewer._canvas_widgets[canvas_zy.id],
        )

        # make the main widget
        self._ortho_view_widget = QtQuadview(
            widget_0=canvas_widget_xz,
            widget_1=canvas_widget_xy,
            widget_2=canvas_widget_zy,
        )

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._ortho_view_widget)
        self.setLayout(layout)
        # layout.addWidget(self.z_slider_widget)
        # for canvas_id, canvas in self.viewer._canvas_widgets.items():
        #     # add the canvas widgets
        #     canvas.update()
        #
        #     scenes = self.viewer._model.scenes.scenes
        #
        #     for scene in scenes.values():
        #         # get the dims model the canvas
        #         if canvas_id in scene.canvases:
        #             dims_model = scene.dims
        #
        #     canvas_widget = QtCanvasWidget(
        #         dims_id=dims_model.id, canvas_widget=canvas, parent=self
        #     )
        #
        #     # create the sliders
        #     dims_ranges = dims_model.range
        #     sliders_data = {
        #         axis_label: range(int(start), int(stop), int(step))
        #         for axis_label, (start, stop, step) in zip(
        #             dims_model.coordinate_system.axis_labels,
        #             dims_ranges,
        #         )
        #     }
        #     canvas_widget._dims_sliders.create_sliders(sliders_data)
        #
        #     self.viewer.events.scene.add_dims_with_controls(
        #         dims_model=dims_model,
        #         dims_controls=canvas_widget._dims_sliders,
        #     )
        #
        #     # redraw the scene when the dims model updates
        #     dims_model.events.point.connect(self._on_dims_update)
        #
        #     dims_model.point = (10, 0, 0)
        #
        #     layout.addWidget(canvas_widget)
        #     self.setLayout(layout)

        # register the canvas
        # for scene in self.viewer._model.scenes.scenes.values():
        #     for visual in scene.visuals:
        #         if visual.id == self._labels_visual_id:
        #             canvas_model = next(iter(scene.canvases.values()))
        #             canvas_id = canvas_model.id
        # self.viewer.events.mouse.register_canvas(canvas_id)
        # self.viewer.events.mouse.subscribe_to_canvas(
        #     canvas_id, self._labels_painting_manager._on_mouse_press
        # )

        # renderer = list(self.viewer._render_manager.renderers.values())[0]
        # renderer.add_event_handler(self._on_mouse_other, "pointer_down")

    def _on_dims_update(self, event):
        for scene in self.viewer._model.scenes.scenes.values():
            dims_manager = scene.dims
            coordinate_system = dims_manager.coordinate_system
            if coordinate_system.name == "scene_2d":
                self.viewer.reslice_scene(scene_id=scene.id)


def make_2d_view(
    coordinate_system_name: str,
    data_store_id: DataStoreId,
    data_range,
    displayed_dimensions: tuple[int, ...],
):
    """Make a 2D view of a label image."""
    # make the 2D scene coordinate system
    coordinate_system = CoordinateSystem(
        name=coordinate_system_name, axis_labels=("z", "y", "x")
    )
    dims = DimsManager(
        point=(125, 0, 0),
        margin_negative=(0, 0, 0),
        margin_positive=(0, 0, 0),
        range=data_range,
        coordinate_system=coordinate_system,
        displayed_dimensions=displayed_dimensions,
    )

    # make the 2D labels visual
    labels_material = LabelsMaterial(color_map="glasbey:glasbey")
    labels_visual_model = MultiscaleLabelsVisual(
        name="labels_node_2d",
        data_store_id=data_store_id,
        material=labels_material,
        downscale_factors=[1],
    )

    return labels_visual_model, dims


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

labels_xy, dims_xy = make_2d_view(
    coordinate_system_name="xy",
    data_store_id=data_store.id,
    data_range=data_range,
    displayed_dimensions=(1, 2),
)

labels_xz, dims_xz = make_2d_view(
    coordinate_system_name="xz",
    data_store_id=data_store.id,
    data_range=data_range,
    displayed_dimensions=(1, 2),
)

labels_zy, dims_zy = make_2d_view(
    coordinate_system_name="zy",
    data_store_id=data_store.id,
    data_range=data_range,
    displayed_dimensions=(1, 2),
)

# make the cameras
camera_xy = PerspectiveCamera(fov=0)
camera_xz = PerspectiveCamera(fov=0)
camera_zy = PerspectiveCamera(fov=0)

# make the canvases
canvas_xy = Canvas(camera=camera_xy)
canvas_xz = Canvas(camera=camera_xz)
canvas_zy = Canvas(camera=camera_zy)

# make the scenes
scene_xy = Scene(dims=dims_xy, visuals=[labels_xy], canvases={canvas_xy.id: canvas_xy})
scene_xz = Scene(dims=dims_xz, visuals=[labels_xz], canvases={canvas_xz.id: canvas_xz})
scene_zy = Scene(dims=dims_zy, visuals=[labels_zy], canvases={canvas_zy.id: canvas_zy})

scene_manager = SceneManager(
    scenes={
        scene_xy.id: scene_xy,
        scene_xz.id: scene_xz,
        scene_zy.id: scene_zy,
    }
)

# make the viewer model
model = ViewerModel(data=data, scenes=scene_manager)


app = QtWidgets.QApplication([])
m = Main(
    model,
    labels_xy=labels_xy,
    labels_xz=labels_xz,
    labels_zy=labels_zy,
    labels_data_store=data_store,
)
m.show()

if __name__ == "__main__":
    app.exec()
