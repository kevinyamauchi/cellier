"""Multiscale chunked 3D image viewer — Cellier example.

Demonstrates automatic scale-level selection: the viewer picks the coarsest
resolution level whose visible chunks still fit in the texture atlas, so
zooming out switches to a coarser level and zooming in restores the finest.

Run ``make_zarr_sample.py`` first to generate the three zarr stores, then:

    python viewer.py

Controls
--------
- Orbit  : left-click drag
- Pan    : middle-click drag (or shift + left-click drag)
- Zoom   : scroll wheel
- Reslice: click the "Reslice current view" button

Debug overlays
--------------
- White  : full data extent (scale 1 AABB in world space) — always visible.
- Green  : camera frustum (updated on every reslice).
- Red    : texture atlas bounding box for scale 0 (finest).
- Cyan   : texture atlas bounding box for scale 1 (medium).
- Yellow : texture atlas bounding box for scale 2 (coarsest).

Only the texture bounding box for the currently active scale is shown.
The active scale index is also printed to stdout on every reslice.
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
SHAPE: tuple[int, int, int] = (1024, 1024, 1024)
CHUNK: tuple[int, int, int] = (64, 64, 64)

# 256/64 = 4 chunks per axis → up to 64 chunks visible at once in a texture.
TEXTURE_WIDTH = 256

DOWNSCALE_FACTORS = [1.0, 2.0, 4.0]

STORE_PATHS = [
    str(pathlib.Path(__file__).parent / "multiscale_blobs_s0.zarr"),
    str(pathlib.Path(__file__).parent / "multiscale_blobs_s1.zarr"),
    str(pathlib.Path(__file__).parent / "multiscale_blobs_s2.zarr"),
]

# One colour per scale level (red → finest, cyan → medium, yellow → coarsest).
BBOX_COLORS = [
    (1.0, 0.2, 0.2, 1.0),  # scale 0 — red
    (0.2, 1.0, 1.0, 1.0),  # scale 1 — cyan
    (1.0, 1.0, 0.2, 1.0),  # scale 2 — yellow
]


# ---------------------------------------------------------------------------
# Geometry helper (defined early so module-level code can use it)
# ---------------------------------------------------------------------------


def _aabb_to_line_pairs(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    """Return the 12 edges of an AABB as (24, 3) start/end pairs.

    Parameters
    ----------
    bbox_min, bbox_max : np.ndarray
        Minimum and maximum corners of the AABB, each (3,) in (z, y, x).

    Returns
    -------
    np.ndarray
        Shape (24, 3) — 12 edges x 2 endpoints.
    """
    z0, y0, x0 = bbox_min
    z1, y1, x1 = bbox_max

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
# Build the multiscale descriptor
# ---------------------------------------------------------------------------
multiscale_model = MultiscaleImageModel.from_shape_and_scales(
    shape=SHAPE,
    chunk_shapes=[CHUNK, CHUNK, CHUNK],
    downscale_factors=DOWNSCALE_FACTORS,
)

# ---------------------------------------------------------------------------
# Pre-compute the full data AABB from scale 1 in world (scale_0) coordinates.
# All scale levels cover the same world space; scale 1 is used as the
# reference since it has a reasonable number of chunks to aggregate.
# ---------------------------------------------------------------------------
_s1_corners = multiscale_model.scales[1].chunk_corners_scale_0.reshape(-1, 3)
DATA_BBOX_MIN: np.ndarray = _s1_corners.min(axis=0)
DATA_BBOX_MAX: np.ndarray = _s1_corners.max(axis=0)

# ---------------------------------------------------------------------------
# Data store — backed by three zarr arrays, one per scale level
# ---------------------------------------------------------------------------
store = ChunkedImageStore(
    multiscale_model=multiscale_model,
    store_paths=STORE_PATHS,
    texture_config=TextureConfiguration(texture_width=TEXTURE_WIDTH),
    name="multiscale_blobs",
)

# ---------------------------------------------------------------------------
# Visual model
# ---------------------------------------------------------------------------
visual = ChunkedImageVisual(
    name="multiscale blobs volume",
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
# Debug overlay — full data extent (permanent white box, scale 1 AABB)
# ---------------------------------------------------------------------------
data_extent_store = LinesMemoryStore(
    coordinates=_aabb_to_line_pairs(DATA_BBOX_MIN, DATA_BBOX_MAX),
    name="data_extent_debug",
)
data_extent_visual = LinesVisual(
    name="data_extent_debug",
    data_store_id=data_extent_store.id,
    appearance=LinesUniformAppearance(
        size=2,
        color=(0.0, 0.0, 0.0, 1.0),
        size_coordinate_space="screen",
    ),
)

# ---------------------------------------------------------------------------
# Debug overlays — one bounding-box wireframe per scale level
# ---------------------------------------------------------------------------
bbox_stores: list[LinesMemoryStore] = []
bbox_visuals: list[LinesVisual] = []

for i, color in enumerate(BBOX_COLORS):
    bs = LinesMemoryStore(
        coordinates=np.zeros((24, 3), dtype=np.float32),
        name=f"bbox_scale{i}_debug",
    )
    bv = LinesVisual(
        name=f"bbox_scale{i}_debug",
        data_store_id=bs.id,
        appearance=LinesUniformAppearance(
            size=2,
            color=color,
            size_coordinate_space="screen",
        ),
    )
    bbox_stores.append(bs)
    bbox_visuals.append(bv)

# ---------------------------------------------------------------------------
# Data manager
# ---------------------------------------------------------------------------
all_stores: dict = {
    store.id: store,
    frustum_store.id: frustum_store,
    data_extent_store.id: data_extent_store,
}
for bs in bbox_stores:
    all_stores[bs.id] = bs

data = DataManager(stores=all_stores)

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
    visuals=[visual, frustum_visual, data_extent_visual, *bbox_visuals],
    canvases={canvas.id: canvas},
)

viewer_model = ViewerModel(
    data=data,
    scenes=SceneManager(scenes={scene.id: scene}),
)


# ---------------------------------------------------------------------------
# Qt widget
# ---------------------------------------------------------------------------
class MultiscaleBlobViewer(QtWidgets.QWidget):
    """Main window for the multiscale chunked 3D image viewer example."""

    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("Cellier — Multiscale Chunked 3D Image Viewer")
        self.resize(900, 700)

        self.viewer = CellierController(model=viewer_model, widget_parent=self)

        # Populate the first frame.
        self._reslice_and_update_debug()

        self._button = QtWidgets.QPushButton("Reslice current view")
        self._button.setToolTip(
            "Load chunks for the current camera position and select the "
            "appropriate scale level automatically."
        )
        self._button.clicked.connect(self._reslice_and_update_debug)

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
        self._update_texture_bboxes()

    def _update_frustum_lines(self) -> None:
        """Refresh the green frustum wireframe."""
        corners = camera.frustum  # (2, 4, 3) updated by the render manager
        edges = frustum_edges_from_corners(corners)  # (12, 2, 3)
        frustum_store.coordinates = edges.reshape(24, 3)
        self.viewer.reslice_visual(scene.id, frustum_visual.id, canvas.id)

    def _update_texture_bboxes(self) -> None:
        """Refresh the per-scale bounding-box overlays.

        Reads ``_active_scale`` and ``_volume_nodes`` from the GFX render node
        and shows only the bbox for the active scale.
        """
        gfx_node = self.viewer._render_manager._visuals[visual.id]

        # Atlas and volume nodes are created lazily on the first reslice.
        if gfx_node._atlas is None or not gfx_node._volume_nodes:
            return

        active = gfx_node._active_scale
        scale_labels = {0: "finest (s0)", 1: "medium (s1)", 2: "coarsest (s2)"}
        print(f"Active scale: {active} — {scale_labels.get(active, str(active))}")
        tw = float(gfx_node._atlas.texture_width)

        for i, (bbox_store, bbox_visual) in enumerate(zip(bbox_stores, bbox_visuals)):
            if i == active:
                # Read the world-space extent from the active volume's full
                # affine matrix.  The texture_to_world transform encodes both
                # translation (mat[:3,3]) AND scale (diagonal of mat[:3,:3]),
                # so the far corner must be computed via matrix multiplication
                # rather than simply adding tw.
                mat = gfx_node._volume_nodes[active].local.matrix  # (4, 4)
                tex_min = np.array(mat[:3, 3], dtype=np.float32)  # (z, y, x)
                far = np.array([tw, tw, tw, 1.0], dtype=np.float64)
                tex_max = (mat @ far)[:3].astype(np.float32)
                bbox_store.coordinates = _aabb_to_line_pairs(tex_min, tex_max)
                self.viewer.reslice_visual(scene.id, bbox_visual.id, canvas.id)

            # Toggle render-node visibility directly.
            rn = self.viewer._render_manager._visuals[bbox_visual.id]
            rn.node.visible = i == active


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MultiscaleBlobViewer()
    window.show()
    sys.exit(app.exec())
