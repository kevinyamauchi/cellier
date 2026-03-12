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

Debug overlays (updated on every reslice)
------------------------------------------
- Green  : camera frustum.
- Yellow : axis-aligned bounding box of the current texture atlas placement.
"""

import pathlib
import sys

import cmap
import numpy as np
from qtpy import QtWidgets

from cellier.models.data_manager import DataManager
from cellier.models.data_stores import ChunkedImageStore
from cellier.models.data_stores.lines import LinesMemoryStore
from cellier.models.scene import (
    AxisAlignedRegionSelector,
    CoordinateSystem,
    DimsManager,
    OrbitCameraController,
    RangeTuple,
)
from cellier.models.scene.cameras import PerspectiveCamera
from cellier.models.scene.canvas import Canvas
from cellier.models.scene.scene import Scene
from cellier.models.viewer import SceneManager, ViewerModel
from cellier.models.visuals.chunked_image import ChunkedImageVisual
from cellier.models.visuals.image import ImageAppearance
from cellier.models.visuals.lines import LinesUniformAppearance, LinesVisual
from cellier.types import CoordinateSpace
from cellier.utils.chunked_image._data_classes import TextureConfiguration
from cellier.utils.chunked_image._multiscale_image_model import MultiscaleImageModel
from cellier.utils.geometry import frustum_edges_from_corners
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
# Debug overlay — frustum wireframe (12 edges x 2 endpoints = 24 rows)
# ---------------------------------------------------------------------------
frustum_store = LinesMemoryStore(
    coordinates=np.zeros((24, 3), dtype=np.float32),
    name="frustum_debug",
)
frustum_visual = LinesVisual(
    name="frustum_debug",
    data_store_id=frustum_store.id,
    appearance=LinesUniformAppearance(
        size=2,
        color=(0.0, 1.0, 0.0, 1.0),
        size_coordinate_space="screen",
    ),
)

# ---------------------------------------------------------------------------
# Debug overlay — texture bounding box (12 edges x 2 endpoints = 24 rows)
# ---------------------------------------------------------------------------
texture_bbox_store = LinesMemoryStore(
    coordinates=np.zeros((24, 3), dtype=np.float32),
    name="texture_bbox_debug",
)
texture_bbox_visual = LinesVisual(
    name="texture_bbox_debug",
    data_store_id=texture_bbox_store.id,
    appearance=LinesUniformAppearance(
        size=2,
        color=(1.0, 1.0, 0.0, 1.0),
        size_coordinate_space="screen",
    ),
)

# ---------------------------------------------------------------------------
# Data manager  (no streams needed for any of these stores)
# ---------------------------------------------------------------------------
data = DataManager(
    stores={
        store.id: store,
        frustum_store.id: frustum_store,
        texture_bbox_store.id: texture_bbox_store,
    }
)

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
controller_3d = OrbitCameraController(enabled=True)
camera = PerspectiveCamera(controller=controller_3d)
canvas = Canvas(camera=camera)
scene = Scene(
    dims=dims,
    visuals=[visual, frustum_visual, texture_bbox_visual],
    canvases={canvas.id: canvas},
)

viewer_model = ViewerModel(
    data=data,
    scenes=SceneManager(scenes={scene.id: scene}),
)


# ---------------------------------------------------------------------------
# Debug geometry helpers
# ---------------------------------------------------------------------------


def _aabb_to_line_pairs(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    """Return the 12 edges of an AABB as ``(24, 3)`` start/end pairs.

    Parameters
    ----------
    bbox_min, bbox_max : np.ndarray
        Minimum and maximum corners of the AABB, each ``(3,)`` in
        ``(z, y, x)`` world coordinates.

    Returns
    -------
    np.ndarray
        Shape ``(24, 3)`` — 12 edges x 2 endpoints, ready to be assigned to
        a ``LinesMemoryStore.coordinates``.
    """
    z0, y0, x0 = bbox_min
    z1, y1, x1 = bbox_max

    # 8 corners enumerated in binary order (z-bit, y-bit, x-bit).
    corners = np.array(
        [
            [z0, y0, x0],
            [z0, y0, x1],
            [z0, y1, x0],
            [z0, y1, x1],
            [z1, y0, x0],
            [z1, y0, x1],
            [z1, y1, x0],
            [z1, y1, x1],
        ],
        dtype=np.float32,
    )

    # 12 edges: 4 x-parallel, 4 y-parallel, 4 z-parallel.
    edge_indices = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    edges = np.array(
        [[corners[a], corners[b]] for a, b in edge_indices], dtype=np.float32
    )
    return edges.reshape(24, 3)


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

        # Populate the first frame using the initial camera position.
        self._reslice_and_update_debug()

        # "Reslice" button — after orbiting, click to load chunks for the new view.
        self._button = QtWidgets.QPushButton("Reslice current view")
        self._button.setToolTip(
            "Load the chunks visible from the current camera position."
        )
        self._button.clicked.connect(self._reslice_and_update_debug)

        # Layout: canvas on top, button along the bottom.
        canvas_row = QtWidgets.QHBoxLayout()
        for canvas_widget in self.viewer._canvas_widgets.values():
            canvas_row.addWidget(canvas_widget)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(canvas_row)
        layout.addWidget(self._button)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _reslice_and_update_debug(self) -> None:
        """Reslice the volume then refresh all debug overlays."""
        self._update_frustum_lines()
        self.viewer.reslice_all()
        self._update_texture_bbox()

    def _update_frustum_lines(self) -> None:
        """Refresh the green frustum wireframe."""
        corners = camera.frustum  # (2, 4, 3) updated by the render manager
        edges = frustum_edges_from_corners(corners)  # (12, 2, 3)
        frustum_store.coordinates = edges.reshape(24, 3)
        self.viewer.reslice_visual(scene.id, frustum_visual.id, canvas.id)

    def _update_texture_bbox(self) -> None:
        """Refresh the yellow texture-atlas bounding-box wireframe.

        The texture placement is read from the GFX render node's current
        volume transform (set by the most recent ``set_slice`` call).
        """
        gfx_node = self.viewer._render_manager._visuals[visual.id]

        # The atlas and volume node are created lazily on the first reslice.
        if gfx_node._atlas is None or gfx_node._volume_node is None:
            return

        tw = float(gfx_node._atlas.texture_width)
        # The volume's local matrix is the texture-to-world affine transform.
        # AffineTransform.from_translation(positioning_corner) puts the
        # (z, y, x) corner in matrix column 3 (rows 0-2).
        mat = gfx_node._volume_node.local.matrix  # (4, 4)
        tex_min = np.array(mat[:3, 3], dtype=np.float32)  # (z, y, x) min corner
        tex_max = tex_min + tw

        texture_bbox_store.coordinates = _aabb_to_line_pairs(tex_min, tex_max)
        self.viewer.reslice_visual(scene.id, texture_bbox_visual.id, canvas.id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ChunkedBlobViewer()
    window.show()
    sys.exit(app.exec())
