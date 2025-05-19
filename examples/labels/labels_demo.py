"""Example app showing 2D and 3D labels rendering."""

from qtpy.QtWidgets import QApplication, QHBoxLayout, QPushButton, QWidget
from skimage.data import binary_blobs
from skimage.measure import label

from cellier.app.interactivity import LabelsPaintingManager, LabelsPaintingMode
from cellier.app.qt import QtCanvasWidget, QtQuadview
from cellier.convenience import (
    get_canvas_with_visual_id,
    get_dims_with_canvas_id,
    get_scene_with_dims_id,
)
from cellier.models.data_manager import DataManager
from cellier.models.data_stores import ImageMemoryStore
from cellier.models.scene import (
    AxisAlignedRegionSelector,
    Canvas,
    CoordinateSystem,
    DimsManager,
    DimsState,
    PanZoomCameraController,
    PerspectiveCamera,
    RangeTuple,
    Scene,
)
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.models.visuals import LabelsMaterial, MultiscaleLabelsVisual
from cellier.types import CoordinateSpace, DataStoreId
from cellier.viewer_controller import CellierController


class Main(QWidget):
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
        self.resize(800, 600)

        # make the viewer
        self.viewer = CellierController(model=viewer_model, widget_parent=self)

        # set up the canvases
        self._painting_manager_xy, canvas_widget_xy = self._setup_canvas(
            labels_model=labels_xy,
            labels_data_store=labels_data_store,
        )
        self._painting_manager_xz, canvas_widget_xz = self._setup_canvas(
            labels_model=labels_xz,
            labels_data_store=labels_data_store,
        )
        self._painting_manager_zy, canvas_widget_zy = self._setup_canvas(
            labels_model=labels_zy,
            labels_data_store=labels_data_store,
        )

        # make a button
        self._painting_button = QPushButton("Paint", parent=self)
        self._painting_button.setCheckable(True)
        self._painting_button.setChecked(False)
        self._painting_button.clicked.connect(self._on_painting_button)

        # connect the data update event to the refresh callback
        labels_data_store.events.data.connect(self._on_data_update)

        # make the main widget
        self._ortho_view_widget = QtQuadview(
            widget_0=canvas_widget_xz,
            widget_1=canvas_widget_xy,
            widget_2=canvas_widget_zy,
        )

        layout = QHBoxLayout()
        layout.addWidget(self._ortho_view_widget)
        layout.addWidget(self._painting_button)
        self.setLayout(layout)

    def _setup_canvas(
        self, labels_model: MultiscaleLabelsVisual, labels_data_store: ImageMemoryStore
    ) -> tuple[LabelsPaintingManager, QtCanvasWidget]:
        """Set up the canvas for the labels visual."""
        # make the canvas widget
        canvas_model = get_canvas_with_visual_id(
            viewer_model=self.viewer._model,
            visual_id=labels_model.id,
        )
        canvas_id = canvas_model.id
        dims_model = get_dims_with_canvas_id(
            viewer_model=self.viewer._model, canvas_id=canvas_id
        )
        canvas_widget = QtCanvasWidget.from_models(
            dims_model=dims_model,
            render_canvas_widget=self.viewer._canvas_widgets[canvas_id],
        )

        # connect the dims model to the canvas widget
        self.viewer.events.scene.add_dims_with_controls(
            dims_model=dims_model,
            dims_controls=canvas_widget._dims_sliders,
        )

        # connect the redraw to the dims model
        self.viewer.events.scene.subscribe_to_dims(
            dims_id=dims_model.id, callback=self._on_dims_update
        )

        # make the painting manager
        painting_manager = LabelsPaintingManager(
            labels_model=labels_model,
            camera_model=canvas_model.camera,
            data_store=labels_data_store,
            mode=LabelsPaintingMode.NONE,
        )

        # connect the painting event
        self.viewer.events.mouse.register_canvas(canvas_id)
        self.viewer.events.mouse.subscribe_to_canvas(
            canvas_id, painting_manager._on_mouse_press
        )

        return painting_manager, canvas_widget

    def _on_dims_update(self, new_dims_state: DimsState):
        scene_model = get_scene_with_dims_id(
            viewer_model=self.viewer._model,
            dims_id=new_dims_state.id,
        )
        self.viewer.reslice_scene(scene_id=scene_model.id)

    def _on_data_update(self):
        self.viewer.reslice_all()

    def _on_painting_button(self):
        if self._painting_button.isChecked():
            # set the painting mode
            self._painting_manager_xy.mode = LabelsPaintingMode.PAINT
            self._painting_manager_xz.mode = LabelsPaintingMode.PAINT
            self._painting_manager_zy.mode = LabelsPaintingMode.PAINT
        else:
            # set the painting mode
            self._painting_manager_xy.mode = LabelsPaintingMode.NONE
            self._painting_manager_xz.mode = LabelsPaintingMode.NONE
            self._painting_manager_zy.mode = LabelsPaintingMode.NONE


def make_2d_view(
    coordinate_system_name: str,
    data_store_id: DataStoreId,
    data_range,
    ordered_dims: tuple[int, int, int],
):
    """Make a 2D view of a label image."""
    # make the 2D scene coordinate system
    coordinate_system = CoordinateSystem(
        name=coordinate_system_name, axis_labels=("z", "y", "x")
    )
    dims = DimsManager(
        range=data_range,
        coordinate_system=coordinate_system,
        selection=AxisAlignedRegionSelector(
            space_type=CoordinateSpace.WORLD,
            ordered_dims=ordered_dims,
            n_displayed_dims=2,
            index_selection=(125, slice(None, None, None), slice(None, None, None)),
        ),
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
    ordered_dims=(0, 1, 2),
)

labels_xz, dims_xz = make_2d_view(
    coordinate_system_name="xz",
    data_store_id=data_store.id,
    data_range=data_range,
    ordered_dims=(0, 1, 2),
)

labels_zy, dims_zy = make_2d_view(
    coordinate_system_name="zy",
    data_store_id=data_store.id,
    data_range=data_range,
    ordered_dims=(0, 1, 2),
)

# make the cameras
controller_xy = PanZoomCameraController(enabled=False)
camera_xy = PerspectiveCamera(fov=0, controller=controller_xy)

controller_xz = PanZoomCameraController(enabled=True)
camera_xz = PerspectiveCamera(fov=0, controller=controller_xz)

controller_zy = PanZoomCameraController(enabled=False)
camera_zy = PerspectiveCamera(fov=0, controller=controller_zy)

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


app = QApplication([])
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
