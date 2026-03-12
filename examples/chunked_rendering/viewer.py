"""Chunked 3D image viewer — new Cellier API example.

This script is the direct replacement for
``examples/tiled_rendering/zarr_example_viewer.py``
using the current chunked rendering API.

Run ``make_zarr_sample.py`` first to generate ``chunked_blobs.zarr``, then:

    python viewer.py

Controls
--------
- Orbit  : left-click drag
- Pan    : middle-click drag (or shift + left-click drag)
- Zoom   : scroll wheel
- Reslice: click the "Reslice current view" button to load chunks
            visible from the current camera position.
"""

import pathlib
import sys

import cmap
from qtpy import QtWidgets

from cellier.models.data_manager import DataManager
from cellier.models.data_stores import ChunkedImageStore
from cellier.models.scene import (
    AxisAlignedRegionSelector,
    CoordinateSystem,
    DimsManager,
    RangeTuple,
)
from cellier.models.scene.cameras import PerspectiveCamera
from cellier.models.scene.canvas import Canvas
from cellier.models.scene.scene import Scene
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.models.visuals.chunked_image import ChunkedImageVisual
from cellier.models.visuals.image import ImageAppearance
from cellier.types import CoordinateSpace
from cellier.utils.chunked_image._data_classes import TextureConfiguration
from cellier.utils.chunked_image._multiscale_image_model import MultiscaleImageModel
from cellier.viewer_controller import CellierController

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STORE_PATH = str(pathlib.Path(__file__).parent / "chunked_blobs.zarr")
SHAPE: tuple[int, int, int] = (256, 256, 256)
CHUNK: tuple[int, int, int] = (32, 32, 32)

# Texture atlas edge length in voxels.
# 128 voxels → 4 chunks per axis → up to 64 chunks visible at once.
TEXTURE_WIDTH = 128

# ---------------------------------------------------------------------------
# Build the multiscale descriptor (single scale for this example)
# ---------------------------------------------------------------------------
multiscale_model = MultiscaleImageModel.from_shape_and_scales(
    shape=SHAPE,
    chunk_shapes=[CHUNK],
    downscale_factors=[1.0],
)

# ---------------------------------------------------------------------------
# Data store  (backed by chunked_blobs.zarr)
# ---------------------------------------------------------------------------
store = ChunkedImageStore(
    multiscale_model=multiscale_model,
    store_path=STORE_PATH,
    texture_config=TextureConfiguration(texture_width=TEXTURE_WIDTH),
    name="blobs",
)

# ---------------------------------------------------------------------------
# Visual model
# ---------------------------------------------------------------------------
visual = ChunkedImageVisual(
    name="blobs volume",
    data_store_id=store.id,
    appearance=ImageAppearance(color_map=cmap.Colormap("grays")),
)

# ---------------------------------------------------------------------------
# Data manager  (no streams needed for ChunkedImageVisual)
# ---------------------------------------------------------------------------
data = DataManager(stores={store.id: store})

# ---------------------------------------------------------------------------
# Scene coordinate system and dims
# ---------------------------------------------------------------------------
coord_sys = CoordinateSystem(name="scene_3d", axis_labels=("z", "y", "x"))

data_range = (
    RangeTuple(start=0, stop=SHAPE[0], step=1),
    RangeTuple(start=0, stop=SHAPE[1], step=1),
    RangeTuple(start=0, stop=SHAPE[2], step=1),
)
selection = AxisAlignedRegionSelector(
    space_type=CoordinateSpace.WORLD,
    ordered_dims=(0, 1, 2),
    n_displayed_dims=3,
    index_selection=(slice(None), slice(None), slice(None)),
)
dims = DimsManager(range=data_range, coordinate_system=coord_sys, selection=selection)

# ---------------------------------------------------------------------------
# Camera / canvas / scene / viewer model
# ---------------------------------------------------------------------------
camera = PerspectiveCamera()
canvas = Canvas(camera=camera)
scene = Scene(dims=dims, visuals=[visual], canvases={canvas.id: canvas})

viewer_model = ViewerModel(
    data=data,
    scenes=SceneManager(scenes={scene.id: scene}),
)


# ---------------------------------------------------------------------------
# Qt widget
# ---------------------------------------------------------------------------
class ChunkedBlobViewer(QtWidgets.QWidget):
    """Main window for the chunked 3D image viewer example."""

    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("Cellier — Chunked 3D Image Viewer")
        self.resize(900, 700)

        # Create the controller — this wires up the ChunkManager, async slicer,
        # and all render/event connections automatically.
        self.viewer = CellierController(model=viewer_model, widget_parent=self)

        # Point the camera at the centre of the volume from outside its bounds.
        self.viewer.look_at_visual(
            visual_id=visual.id,
            view_direction=(1.0, 1.0, 1.0),
            up_direction=(0.0, 0.0, 1.0),
        )

        # Populate the first frame using the initial camera position.
        self.viewer.reslice_all()

        # "Reslice" button — after orbiting, click to load chunks for the new view.
        self._button = QtWidgets.QPushButton("Reslice current view")
        self._button.setToolTip(
            "Load the chunks visible from the current camera position."
        )
        self._button.clicked.connect(self.viewer.reslice_all)

        # Layout: canvas on top, button along the bottom.
        canvas_row = QtWidgets.QHBoxLayout()
        for canvas_widget in self.viewer._canvas_widgets.values():
            canvas_row.addWidget(canvas_widget)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(canvas_row)
        layout.addWidget(self._button)
        self.setLayout(layout)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ChunkedBlobViewer()
    window.show()
    sys.exit(app.exec())
