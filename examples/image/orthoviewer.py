"""Example app showing 2D and 3D image rendering in an orthoviewer.

All views are sliced from a single 3D image data store.
"""

from qtpy.QtWidgets import QApplication, QHBoxLayout, QWidget
from skimage.data import binary_blobs

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
from cellier.models.visuals import ImageAppearance, MultiscaleImageVisual
from cellier.types import CoordinateSpace, DataStoreId
from cellier.viewer_controller import CellierController


class Main(QWidget):
    """Example widget with viewer."""

    def __init__(
        self,
        viewer_model,
        image_xy,
        image_xz,
        image_zy,
        labels_data_store,
    ):
        super().__init__(None)
        self.resize(800, 600)

        # make the viewer
        self.viewer = CellierController(model=viewer_model, widget_parent=self)

        # set up the canvases
        canvas_widget_xy = self._setup_canvas(
            image_model=image_xy,
        )
        canvas_widget_xz = self._setup_canvas(
            image_model=image_xz,
        )
        canvas_widget_zy = self._setup_canvas(
            image_model=image_zy,
        )

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
        self.setLayout(layout)

        # slice everything
        self.viewer.reslice_all()

    def _setup_canvas(self, image_model: MultiscaleImageVisual) -> QtCanvasWidget:
        """Set up the canvas for the labels visual."""
        # make the canvas widget
        canvas_model = get_canvas_with_visual_id(
            viewer_model=self.viewer._model,
            visual_id=image_model.id,
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

        return canvas_widget

    def _on_dims_update(self, new_dims_state: DimsState):
        scene_model = get_scene_with_dims_id(
            viewer_model=self.viewer._model,
            dims_id=new_dims_state.id,
        )
        self.viewer.reslice_scene(scene_id=scene_model.id)

    def _on_data_update(self):
        self.viewer.reslice_all()


def make_2d_view(
    coordinate_system_name: str,
    index_selection: tuple[int | slice, int | slice, int | slice],
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
            index_selection=index_selection,
        ),
    )

    # make the 2D labels visual
    image_material = ImageAppearance(color_map="viridis")
    image_visual_model = MultiscaleImageVisual(
        name="image_node_2d",
        data_store_id=data_store_id,
        appearance=image_material,
        downscale_factors=[1],
    )

    return image_visual_model, dims


# make the data
im = binary_blobs(length=250, volume_fraction=0.1, n_dim=3).astype(float)

# make the data store
data_store = ImageMemoryStore(data=im, name="volume_data")

# make the data manager
data = DataManager(stores={data_store.id: data_store})

# the range of the data in the scene
data_range = (
    RangeTuple(start=0, stop=250, step=1),
    RangeTuple(start=0, stop=250, step=1),
    RangeTuple(start=0, stop=250, step=1),
)

image_xy, dims_xy = make_2d_view(
    coordinate_system_name="xy",
    index_selection=(125, slice(None, None, None), slice(None, None, None)),
    data_store_id=data_store.id,
    data_range=data_range,
    ordered_dims=(0, 1, 2),
)

image_zx, dims_xz = make_2d_view(
    coordinate_system_name="xz",
    data_store_id=data_store.id,
    index_selection=(slice(None, None, None), 125, slice(None, None, None)),
    data_range=data_range,
    ordered_dims=(1, 0, 2),
)

image_zy, dims_zy = make_2d_view(
    coordinate_system_name="zy",
    index_selection=(slice(None, None, None), slice(None, None, None), 125),
    data_store_id=data_store.id,
    data_range=data_range,
    ordered_dims=(2, 0, 1),
)

# make the cameras
controller_xy = PanZoomCameraController(enabled=True)
camera_xy = PerspectiveCamera(fov=0, controller=controller_xy)

controller_xz = PanZoomCameraController(enabled=True)
camera_xz = PerspectiveCamera(fov=0, controller=controller_xz)

controller_zy = PanZoomCameraController(enabled=True)
camera_zy = PerspectiveCamera(fov=0, controller=controller_zy)

# make the canvases
canvas_xy = Canvas(camera=camera_xy)
canvas_xz = Canvas(camera=camera_xz)
canvas_zy = Canvas(camera=camera_zy)

# make the scenes
scene_xy = Scene(dims=dims_xy, visuals=[image_xy], canvases={canvas_xy.id: canvas_xy})
scene_xz = Scene(dims=dims_xz, visuals=[image_zx], canvases={canvas_xz.id: canvas_xz})
scene_zy = Scene(dims=dims_zy, visuals=[image_zy], canvases={canvas_zy.id: canvas_zy})

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
    image_xy=image_xy,
    image_xz=image_zx,
    image_zy=image_zy,
    labels_data_store=data_store,
)
m.show()

if __name__ == "__main__":
    app.exec()
