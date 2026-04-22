"""Transform Validation Viewer — dataset generator + 4-panel orthoviewer.

Running with --make-files writes the dataset to disk; running without it
opens the viewer.  If all transforms are correct the image cubes, points,
lines, and mesh will be spatially aligned across every panel.

Usage
-----
    # Generate dataset:
    uv run scripts/v2/transform_validation_viewer.py --output-dir /tmp --make-files

    # Open viewer:
    uv run scripts/v2/transform_validation_viewer.py --output-dir /tmp
"""

from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path

import numpy as np
import PySide6.QtAsyncio as QtAsyncio
import tensorstore as ts
from cmap import Colormap
from PySide6 import QtCore, QtWidgets

from cellier.v2.controller import CellierController
from cellier.v2.data.image import OMEZarrImageDataStore
from cellier.v2.data.lines._lines_memory_store import LinesMemoryStore
from cellier.v2.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.v2.data.points._points_memory_store import PointsMemoryStore
from cellier.v2.gui._scene import QtCanvasWidget, QtDimsSliders
from cellier.v2.scene.cameras import OrbitCameraController, PerspectiveCamera
from cellier.v2.scene.canvas import Canvas
from cellier.v2.scene.dims import AxisAlignedSelection, CoordinateSystem, DimsManager
from cellier.v2.scene.scene import Scene
from cellier.v2.transform import AffineTransform
from cellier.v2.viewer_model import DataManager, ViewerModel
from cellier.v2.visuals._image import (
    ImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageVisual,
)
from cellier.v2.visuals._lines_memory import LinesMemoryAppearance, LinesVisual
from cellier.v2.visuals._mesh_memory import MeshFlatAppearance, MeshVisual
from cellier.v2.visuals._points_memory import PointsMarkerAppearance, PointsVisual

# ---------------------------------------------------------------------------
# Module-level geometry constants
# ---------------------------------------------------------------------------

# World bounding box: [0, 300]^3 in (z, y, x) world units
WORLD_SHAPE = np.array([300, 300, 300])
CUBE_HALF = 25  # half-side of each octant cube in world units
DATA_TO_WORLD = np.array([3.0, 2.0, 2.0])  # physical voxel size at level 0 (z, y, x)

# 8 octant indices (iz, iy, ix) ∈ {0,1}^3, matching itertools.product order
OCTANT_INDICES = list(itertools.product([0, 1], repeat=3))

# Octant centres in world space: (iz, iy, ix) × 150 + 75
OCTANT_CENTRES = np.array(
    [np.array(idx) * 150 + 75 for idx in OCTANT_INDICES],
    dtype=np.float32,
)  # shape (8, 3)

# One RGBA color per octant
OCTANT_COLORS = {
    (0, 0, 0): (0.9, 0.2, 0.2, 1.0),  # red
    (0, 0, 1): (0.2, 0.7, 0.2, 1.0),  # green
    (0, 1, 0): (0.2, 0.4, 0.9, 1.0),  # blue
    (0, 1, 1): (0.9, 0.8, 0.1, 1.0),  # yellow
    (1, 0, 0): (0.8, 0.4, 0.1, 1.0),  # orange
    (1, 0, 1): (0.6, 0.1, 0.8, 1.0),  # purple
    (1, 1, 0): (0.1, 0.8, 0.8, 1.0),  # cyan
    (1, 1, 1): (0.9, 0.9, 0.9, 1.0),  # white
}

# Discrete LUT for scalar values 0..8 with CLIM=(0, 8): value n maps to n/8.
OCTANT_CMAP = Colormap(
    [
        (0.0, (0.0, 0.0, 0.0, 1.0)),
        *[
            (linear_idx / 8.0, OCTANT_COLORS[idx_tuple])
            for linear_idx, idx_tuple in enumerate(OCTANT_INDICES, start=1)
        ],
    ]
)

LEVEL_SCALES = [
    [3.0, 2.0, 2.0],  # level 0
    [3.0, 4.0, 4.0],  # level 1
    [3.0, 8.0, 8.0],  # level 2
]

AABB_EDGES = [
    # x-axis edges (ix flips)
    ((0, 0, 0), (0, 0, 1)),
    ((0, 1, 0), (0, 1, 1)),
    ((1, 0, 0), (1, 0, 1)),
    ((1, 1, 0), (1, 1, 1)),
    # y-axis edges (iy flips)
    ((0, 0, 0), (0, 1, 0)),
    ((0, 0, 1), (0, 1, 1)),
    ((1, 0, 0), (1, 1, 0)),
    ((1, 0, 1), (1, 1, 1)),
    # z-axis edges (iz flips)
    ((0, 0, 0), (1, 0, 0)),
    ((0, 0, 1), (1, 0, 1)),
    ((0, 1, 0), (1, 1, 0)),
    ((0, 1, 1), (1, 1, 1)),
]

FACE_QUADS = [
    (0, 1, 3, 2),  # z=0
    (4, 6, 7, 5),  # z=1
    (0, 4, 5, 1),  # y=0
    (2, 3, 7, 6),  # y=1
    (0, 2, 6, 4),  # x=0
    (1, 5, 7, 3),  # x=1
]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(output_dir: Path) -> None:
    """Generate the transform validation dataset under output_dir/transform_validation/."""
    base = output_dir / "transform_validation"
    base.mkdir(parents=True, exist_ok=True)

    # ── Step 2a: build level-0 image array ───────────────────────────────────
    # Shape = WORLD_SHAPE / DATA_TO_WORLD = (300/3, 300/2, 300/2) = (100, 150, 150)
    data_l0 = np.zeros((100, 150, 150), dtype=np.uint8)

    for linear_idx, idx_tuple in enumerate(OCTANT_INDICES):
        centre = OCTANT_CENTRES[linear_idx]
        lo_world = centre - CUBE_HALF
        hi_world = centre + CUBE_HALF
        lo_data = np.floor(lo_world / DATA_TO_WORLD).astype(int)
        hi_data = np.ceil(hi_world / DATA_TO_WORLD).astype(int)
        data_l0[
            lo_data[0] : hi_data[0],
            lo_data[1] : hi_data[1],
            lo_data[2] : hi_data[2],
        ] = linear_idx + 1  # scalar values 1–8

    # ── Step 2b: downsample to levels 1 and 2 ────────────────────────────────
    data_l1 = data_l0[:, ::2, ::2]  # shape (100, 75, 75)
    data_l2 = data_l0[:, ::4, ::4]  # shape (100, 38, 38)

    # ── Step 2c: write OME-Zarr v0.5 via tensorstore ─────────────────────────
    store_root = base / "image.ome.zarr"
    store_root.mkdir(parents=True, exist_ok=True)

    print("[generate_dataset] Writing image arrays...")
    for level, arr in enumerate([data_l0, data_l1, data_l2]):
        level_path = store_root / str(level)
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(level_path)},
            "metadata": {
                "shape": list(arr.shape),
                "data_type": "uint8",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [64, 64, 64]},
                },
                "chunk_key_encoding": {"name": "default"},
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            },
            "create": True,
            "delete_existing": True,
        }
        ts_store = ts.open(spec).result()
        ts_store[...].write(arr).result()
        print(f"  Level {level}: shape={arr.shape}  scale={LEVEL_SCALES[level]}")

    # ── Step 2c-ii: write root zarr.json (group + OME metadata) ──────────────
    ome_meta = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": str(level),
                        "coordinateTransformations": [
                            {"type": "scale", "scale": LEVEL_SCALES[level]}
                        ],
                    }
                    for level in range(3)
                ],
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                ],
            }
        ],
    }

    root_zarr_json = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"ome": ome_meta},
    }
    (store_root / "zarr.json").write_text(json.dumps(root_zarr_json, indent=2))
    print(f"  Written zarr.json to {store_root}")

    # ── Step 2d: points layer ─────────────────────────────────────────────────
    points = OCTANT_CENTRES.copy()  # (8, 3) world-space (z, y, x)
    points_colors = np.array(
        [OCTANT_COLORS[idx] for idx in OCTANT_INDICES], dtype=np.float32
    )  # (8, 4)

    np.save(base / "points.npy", points)
    np.save(base / "points_colors.npy", points_colors)
    print(f"  Written points: {points.shape}")

    # ── Step 2e: lines layer ──────────────────────────────────────────────────
    def corner_world(idx_tuple):
        return np.array(idx_tuple, dtype=np.float32) * 300.0

    lines_vertices = []
    lines_colors = []
    for a_idx, b_idx in AABB_EDGES:
        for corner_idx in (a_idx, b_idx):
            lines_vertices.append(corner_world(corner_idx))
            lines_colors.append(OCTANT_COLORS[corner_idx])

    lines_vertices = np.array(lines_vertices, dtype=np.float32)  # (24, 3)
    lines_colors = np.array(lines_colors, dtype=np.float32)  # (24, 4)

    np.save(base / "lines.npy", lines_vertices)
    np.save(base / "lines_colors.npy", lines_colors)
    print(f"  Written lines: {lines_vertices.shape}")

    # ── Step 2f: mesh layer ───────────────────────────────────────────────────
    mesh_vertices = np.array(
        [np.array(idx, dtype=np.float32) * 300.0 for idx in OCTANT_INDICES],
        dtype=np.float32,
    )  # (8, 3)

    mesh_colors = np.array(
        [OCTANT_COLORS[idx] for idx in OCTANT_INDICES], dtype=np.float32
    )  # (8, 4)

    mesh_faces = np.array(
        [
            (a, b, c)
            for (a, b, c, d) in FACE_QUADS
            for (a, b, c) in [(a, b, c), (a, c, d)]
        ],
        dtype=np.int32,
    )  # (12, 3)

    np.save(base / "mesh_vertices.npy", mesh_vertices)
    np.save(base / "mesh_faces.npy", mesh_faces)
    np.save(base / "mesh_colors.npy", mesh_colors)
    print(f"  Written mesh: vertices={mesh_vertices.shape}  faces={mesh_faces.shape}")

    # ── Step 2g: verification checks ─────────────────────────────────────────
    assert data_l0.shape == (100, 150, 150)
    assert data_l1.shape == (100, 75, 75)
    assert data_l2.shape == (100, 38, 38)
    assert list(np.unique(data_l0)) == list(range(9))  # 0–8
    assert (data_l0 > 0).sum() == 90_000  # 8 × 18×25×25
    assert points.shape == (8, 3)
    assert lines_vertices.shape == (24, 3)
    assert mesh_vertices.shape == (8, 3)
    assert mesh_faces.shape == (12, 3)
    print("[generate_dataset] All verification checks passed.")


# ---------------------------------------------------------------------------
# Viewer class
# ---------------------------------------------------------------------------

SLIDER_STYLE_XY = "QSlider::handle { background: #4488ff; }"
SLIDER_STYLE_XZ = "QSlider::handle { background: #ff8844; }"
SLIDER_STYLE_YZ = "QSlider::handle { background: #44cc44; }"


class TransformValidationViewer:
    """4-panel orthoviewer window for transform validation.

    Parameters
    ----------
    controller : CellierController
        The active controller.
    canvas_widgets : dict
        Mapping of panel key → QtCanvasWidget.
    image_visuals : dict
        Mapping of panel key → image visual model.
    points_visuals : dict
        Mapping of panel key → points visual model.
    lines_visuals : dict
        Mapping of panel key → lines visual model.
    mesh_visuals : dict
        Mapping of panel key → mesh visual model.
    clim_range : tuple[float, float]
        (min, max) for the contrast-limits slider.
    """

    def __init__(
        self,
        controller,
        *,
        canvas_widgets: dict,
        image_visuals: dict,
        points_visuals: dict,
        lines_visuals: dict,
        mesh_visuals: dict,
        clim_range: tuple[float, float],
    ) -> None:
        from superqt import QLabeledDoubleRangeSlider

        self._controller = controller
        self._canvas_widgets = canvas_widgets

        # ── Main window ───────────────────────────────────────────────────────
        self._window = QtWidgets.QMainWindow()
        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        # ── 2×2 canvas grid ───────────────────────────────────────────────────
        grid_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_widget)
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)

        panels = [
            (0, 0, "xy", "XY  (slice Z)"),
            (0, 1, "xz", "XZ  (slice Y)"),
            (1, 0, "yz", "YZ  (slice X)"),
            (1, 1, "vol", "3D Volume"),
        ]
        for row, col, key, label in panels:
            cell = QtWidgets.QWidget()
            cell_layout = QtWidgets.QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(0)

            lbl = QtWidgets.QLabel(label)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 11px; padding: 2px;")
            cell_layout.addWidget(lbl)
            cell_layout.addWidget(canvas_widgets[key].widget, stretch=1)
            grid.addWidget(cell, row, col)

        root_layout.addWidget(grid_widget, stretch=1)

        # ── Side panel ────────────────────────────────────────────────────────
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(280)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        root_layout.addWidget(panel)

        # ── "Layers" group box ────────────────────────────────────────────────
        layers_group = QtWidgets.QGroupBox("Layers")
        layers_layout = QtWidgets.QVBoxLayout(layers_group)

        def _make_toggle(visuals_dict):
            def _on_changed(state):
                visible = bool(state)
                for v in visuals_dict.values():
                    v.appearance.visible = visible

            return _on_changed

        for label, visuals_dict in [
            ("Image", image_visuals),
            ("Points", points_visuals),
            ("Lines", lines_visuals),
            ("Mesh", mesh_visuals),
        ]:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            cb.stateChanged.connect(_make_toggle(visuals_dict))
            layers_layout.addWidget(cb)

        panel_layout.addWidget(layers_group)

        # ── "Image" group box with contrast-limits slider ─────────────────────
        image_group = QtWidgets.QGroupBox("Image")
        image_layout = QtWidgets.QVBoxLayout(image_group)

        clim_slider = QLabeledDoubleRangeSlider(QtCore.Qt.Orientation.Horizontal)
        clim_slider.setRange(*clim_range)
        clim_slider.setValue(clim_range)
        clim_slider.setDecimals(1)
        self._clim_slider = clim_slider

        def _on_clim_changed(value):
            for v in image_visuals.values():
                controller.update_appearance_field(v.id, "clim", tuple(value))

        clim_slider.valueChanged.connect(_on_clim_changed)
        image_layout.addWidget(clim_slider)
        panel_layout.addWidget(image_group)

        # Push all controls to the top
        panel_layout.addStretch(1)

    @property
    def window(self):
        return self._window

    def close_widgets(self) -> None:
        for w in self._canvas_widgets.values():
            w.close()
        self._clim_slider.close()


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------


def _build_viewer_model(
    image_store: OMEZarrImageDataStore,
    points_store: PointsMemoryStore,
    lines_store: LinesMemoryStore,
    mesh_store: MeshMemoryStore,
    initial_slice_position: float = 0.5,
) -> tuple[ViewerModel, dict[str, MultiscaleImageVisual]]:
    """Construct the ViewerModel for the transform validation viewer.

    Creates all scenes, visuals, and canvases as model objects and
    assembles them into a ViewerModel. No render layer is touched.

    Parameters
    ----------
    image_store : OMEZarrImageDataStore
        Multiscale image data store, already opened.
    points_store : PointsMemoryStore
        In-memory points data store.
    lines_store : LinesMemoryStore
        In-memory lines data store.
    mesh_store : MeshMemoryStore
        In-memory mesh data store.
    initial_slice_position : float
        Starting slice position for each 2D panel, normalised to [0, 1]
        relative to the world-space length of the sliced axis.
        ``0.0`` is the near edge, ``1.0`` is the far edge, ``0.5`` (default)
        is the midpoint. Converted to integer world units internally using
        the per-axis lengths from ``WORLD_SHAPE``.

    Returns
    -------
    viewer_model : ViewerModel
        Fully assembled model, ready for CellierController.from_model.
    image_visuals : dict[str, MultiscaleImageVisual]
        Mapping of panel key to image visual model, for post-construction
        access (e.g. camera-fit callback).
    """
    # Convert normalised position to integer world-space coordinates per axis.
    # WORLD_SHAPE = (300, 300, 300), so 0.5 → 150 on every axis.
    z_slice_index = int(initial_slice_position * WORLD_SHAPE[0])
    y_slice_index = int(initial_slice_position * WORLD_SHAPE[1])
    x_slice_index = int(initial_slice_position * WORLD_SHAPE[2])

    coordinate_system = CoordinateSystem(name="world", axis_labels=["z", "y", "x"])
    voxel_to_world_transform = AffineTransform.from_scale((3.0, 2.0, 2.0))
    depth_range = (1.0, 5000.0)
    clim = (0, 8)

    # ── Appearance shared across 2D image visuals ─────────────────────────
    common_2d_image_appearance = ImageAppearance(
        color_map=OCTANT_CMAP,
        clim=clim,
        lod_bias=1.0,
        force_level=None,
        frustum_cull=True,
        iso_threshold=0.5,
        render_mode="mip",
    )
    image_render_config_2d = MultiscaleImageRenderConfig(
        block_size=32,
        gpu_budget_bytes=256 * 1024**2,
        gpu_budget_bytes_2d=64 * 1024**2,
        use_brick_shader=True,
    )

    vol_image_appearance = ImageAppearance(
        color_map=OCTANT_CMAP,
        clim=clim,
        lod_bias=1.0,
        force_level=image_store.n_levels - 1,
        frustum_cull=False,
        iso_threshold=0.05,
        render_mode="iso",
    )
    image_render_config_vol = MultiscaleImageRenderConfig(
        block_size=32,
        gpu_budget_bytes=512 * 1024**2,
        gpu_budget_bytes_2d=64 * 1024**2,
        use_brick_shader=True,
    )

    # ── Points appearance ─────────────────────────────────────────────────
    points_appearance = PointsMarkerAppearance(
        color_mode="vertex",
        size=10.0,
        size_space="screen",
    )

    # ── Lines appearance ──────────────────────────────────────────────────
    lines_appearance = LinesMemoryAppearance(
        color_mode="vertex",
        thickness=2.0,
        thickness_space="screen",
    )

    # ── Mesh appearance ───────────────────────────────────────────────────
    mesh_appearance = MeshFlatAppearance(
        color_mode="vertex",
        side="front",
        opacity=0.8,
    )

    # ── Camera model (shared template — each canvas gets its own instance) ─
    def _make_perspective_camera() -> PerspectiveCamera:
        return PerspectiveCamera(
            fov=70.0,
            near_clipping_plane=depth_range[0],
            far_clipping_plane=depth_range[1],
            controller=OrbitCameraController(enabled=True),
        )

    # ── Helper: build a visual model for one panel ────────────────────────
    def _make_image_visual(
        name: str,
        appearance: ImageAppearance,
        render_config: MultiscaleImageRenderConfig,
    ) -> MultiscaleImageVisual:
        return MultiscaleImageVisual(
            name=name,
            data_store_id=str(image_store.id),
            level_transforms=image_store.level_transforms,
            appearance=appearance,
            render_config=render_config,
            transform=voxel_to_world_transform,
        )

    def _make_points_visual(name: str) -> PointsVisual:
        return PointsVisual(
            name=name,
            data_store_id=str(points_store.id),
            appearance=points_appearance,
        )

    def _make_lines_visual(name: str) -> LinesVisual:
        return LinesVisual(
            name=name,
            data_store_id=str(lines_store.id),
            appearance=lines_appearance,
        )

    def _make_mesh_visual(name: str) -> MeshVisual:
        return MeshVisual(
            name=name,
            data_store_id=str(mesh_store.id),
            appearance=mesh_appearance,
        )

    # ── XY scene (slice Z) ────────────────────────────────────────────────
    xy_image_visual = _make_image_visual(
        "xy_image", common_2d_image_appearance, image_render_config_2d
    )
    xy_scene = Scene(
        name="xy",
        dims=DimsManager(
            coordinate_system=coordinate_system,
            selection=AxisAlignedSelection(
                displayed_axes=(1, 2),
                slice_indices={0: z_slice_index},
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=[
            xy_image_visual,
            _make_points_visual("xy_points"),
            _make_lines_visual("xy_lines"),
            _make_mesh_visual("xy_mesh"),
        ],
        canvases={},
    )
    xy_canvas = Canvas(cameras={"2d": _make_perspective_camera()})
    xy_scene.canvases[xy_canvas.id] = xy_canvas

    # ── XZ scene (slice Y) ────────────────────────────────────────────────
    xz_image_visual = _make_image_visual(
        "xz_image", common_2d_image_appearance, image_render_config_2d
    )
    xz_scene = Scene(
        name="xz",
        dims=DimsManager(
            coordinate_system=coordinate_system,
            selection=AxisAlignedSelection(
                displayed_axes=(0, 2),
                slice_indices={1: y_slice_index},
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=[
            xz_image_visual,
            _make_points_visual("xz_points"),
            _make_lines_visual("xz_lines"),
            _make_mesh_visual("xz_mesh"),
        ],
        canvases={},
    )
    xz_canvas = Canvas(cameras={"2d": _make_perspective_camera()})
    xz_scene.canvases[xz_canvas.id] = xz_canvas

    # ── YZ scene (slice X) ────────────────────────────────────────────────
    yz_image_visual = _make_image_visual(
        "yz_image", common_2d_image_appearance, image_render_config_2d
    )
    yz_scene = Scene(
        name="yz",
        dims=DimsManager(
            coordinate_system=coordinate_system,
            selection=AxisAlignedSelection(
                displayed_axes=(0, 1),
                slice_indices={2: x_slice_index},
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=[
            yz_image_visual,
            _make_points_visual("yz_points"),
            _make_lines_visual("yz_lines"),
            _make_mesh_visual("yz_mesh"),
        ],
        canvases={},
    )
    yz_canvas = Canvas(cameras={"2d": _make_perspective_camera()})
    yz_scene.canvases[yz_canvas.id] = yz_canvas

    # ── Volume scene (3D) ─────────────────────────────────────────────────
    vol_image_visual = _make_image_visual(
        "vol_image", vol_image_appearance, image_render_config_vol
    )
    vol_scene = Scene(
        name="vol",
        dims=DimsManager(
            coordinate_system=coordinate_system,
            selection=AxisAlignedSelection(
                displayed_axes=(0, 1, 2),
                slice_indices={},
            ),
        ),
        render_modes={"3d"},
        lighting="default",
        visuals=[
            vol_image_visual,
            _make_points_visual("vol_points"),
            _make_lines_visual("vol_lines"),
            _make_mesh_visual("vol_mesh"),
        ],
        canvases={},
    )
    vol_canvas = Canvas(cameras={"3d": _make_perspective_camera()})
    vol_scene.canvases[vol_canvas.id] = vol_canvas

    # ── Assemble ViewerModel ──────────────────────────────────────────────
    viewer_model = ViewerModel(
        data=DataManager(
            stores={
                image_store.id: image_store,
                points_store.id: points_store,
                lines_store.id: lines_store,
                mesh_store.id: mesh_store,
            }
        ),
        scenes={
            xy_scene.id: xy_scene,
            xz_scene.id: xz_scene,
            yz_scene.id: yz_scene,
            vol_scene.id: vol_scene,
        },
    )

    image_visuals = {
        "xy": xy_image_visual,
        "xz": xz_image_visual,
        "yz": yz_image_visual,
        "vol": vol_image_visual,
    }

    return viewer_model, image_visuals


async def async_main(dataset_dir: Path, image_store: OMEZarrImageDataStore) -> None:
    """Build and run the transform validation viewer."""
    # ── Load npy files ────────────────────────────────────────────────────────
    points_positions = np.load(dataset_dir / "points.npy")
    points_colors = np.load(dataset_dir / "points_colors.npy")
    lines_positions = np.load(dataset_dir / "lines.npy")
    lines_colors = np.load(dataset_dir / "lines_colors.npy")
    mesh_vertices = np.load(dataset_dir / "mesh_vertices.npy")
    mesh_faces = np.load(dataset_dir / "mesh_faces.npy")
    mesh_colors = np.load(dataset_dir / "mesh_colors.npy")

    # ── Build data stores ─────────────────────────────────────────────────────
    points_store = PointsMemoryStore(
        positions=points_positions,
        colors=points_colors,
        name="points",
    )
    lines_store = LinesMemoryStore(
        positions=lines_positions,
        colors=lines_colors,
        name="lines",
    )
    mesh_store = MeshMemoryStore(
        positions=mesh_vertices,
        indices=mesh_faces,
        colors=mesh_colors,
        name="mesh",
    )

    # ── Build ViewerModel (no render layer) ───────────────────────────────────
    viewer_model, image_visuals = _build_viewer_model(
        image_store=image_store,
        points_store=points_store,
        lines_store=lines_store,
        mesh_store=mesh_store,
    )

    # ── Construct controller from model ───────────────────────────────────────
    controller = CellierController.from_model(viewer_model)

    # ── Post-construction: wire vol camera-fit on first data delivery ─────────
    vol_scene = controller.get_scene_by_name("vol")
    vol_image_visual = image_visuals["vol"]
    vol_scene_manager = controller._render_manager._scenes[vol_scene.id]
    vol_gfx_visual = vol_scene_manager.get_visual(vol_image_visual.id)
    vol_camera_fitted = [False]
    original_on_data_ready = vol_gfx_visual.on_data_ready

    def _on_vol_data_ready(batch):
        original_on_data_ready(batch)
        if not vol_camera_fitted[0]:
            vol_camera_fitted[0] = True
            controller.fit_camera(vol_scene.id)

    vol_gfx_visual.on_data_ready = _on_vol_data_ready

    # ── Build canvas widgets ──────────────────────────────────────────────────
    axis_ranges = {0: (0, 300), 1: (0, 300), 2: (0, 300)}

    def _get_canvas_view(scene_id):
        return controller.get_canvas_view(controller.get_canvas_ids(scene_id)[0])

    def _make_2d_canvas_widget(scene, slider_style):
        canvas_view = _get_canvas_view(scene.id)
        axis_labels = dict(enumerate(scene.dims.coordinate_system.axis_labels))
        dims_sliders = QtDimsSliders(
            controller=controller,
            scene_id=scene.id,
            axis_ranges=axis_ranges,
            axis_labels=axis_labels,
            initial_slice_indices=dict(scene.dims.selection.slice_indices),
            initial_displayed_axes=scene.dims.selection.displayed_axes,
        )
        dims_sliders.widget.setStyleSheet(slider_style)
        return QtCanvasWidget(canvas_view=canvas_view, dims_sliders=dims_sliders)

    xy_scene = controller.get_scene_by_name("xy")
    xz_scene = controller.get_scene_by_name("xz")
    yz_scene = controller.get_scene_by_name("yz")

    canvas_widgets = {
        "xy": _make_2d_canvas_widget(xy_scene, SLIDER_STYLE_XY),
        "xz": _make_2d_canvas_widget(xz_scene, SLIDER_STYLE_XZ),
        "yz": _make_2d_canvas_widget(yz_scene, SLIDER_STYLE_YZ),
        "vol": QtCanvasWidget.from_scene_and_canvas(
            controller,
            vol_scene,
            _get_canvas_view(vol_scene.id),
            axis_ranges=axis_ranges,
        ),
    }

    # ── Collect visual models for the viewer UI ───────────────────────────────
    def _visuals_by_name_prefix(prefix: str) -> dict:
        result = {}
        for scene_key, scene in [
            ("xy", xy_scene),
            ("xz", xz_scene),
            ("yz", yz_scene),
            ("vol", vol_scene),
        ]:
            for visual_model in scene.visuals:
                if visual_model.name == f"{scene_key}_{prefix}":
                    result[scene_key] = visual_model
        return result

    # ── Build and show the viewer window ──────────────────────────────────────
    viewer = TransformValidationViewer(
        controller,
        canvas_widgets=canvas_widgets,
        image_visuals=image_visuals,
        points_visuals=_visuals_by_name_prefix("points"),
        lines_visuals=_visuals_by_name_prefix("lines"),
        mesh_visuals=_visuals_by_name_prefix("mesh"),
        clim_range=(0.0, 8.0),
    )
    viewer.window.setWindowTitle("Transform Validation Viewer")
    viewer.window.resize(1400, 900)
    viewer.window.show()

    for scene in [xy_scene, xz_scene, yz_scene, vol_scene]:
        controller.fit_camera(scene.id)
        controller.reslice_scene(scene.id)

    app = QtWidgets.QApplication.instance()
    app.aboutToQuit.connect(viewer.close_widgets)
    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Transform validation dataset generator and viewer."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory under which transform_validation/ will be written.",
    )
    parser.add_argument(
        "--make-files",
        action="store_true",
        help="Generate dataset files and exit without opening the viewer.",
    )
    args = parser.parse_args()

    dataset_dir = args.output_dir / "transform_validation"

    if args.make_files:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        generate_dataset(args.output_dir)
        print(f"[main] Dataset written to {dataset_dir}")
        return

    # Viewer path — validate files exist before starting Qt
    for fname in [
        "points.npy",
        "points_colors.npy",
        "lines.npy",
        "lines_colors.npy",
        "mesh_vertices.npy",
        "mesh_faces.npy",
        "mesh_colors.npy",
        "image.ome.zarr",
    ]:
        p = dataset_dir / fname
        if not p.exists():
            print(
                f"[main] Required file missing: {p}\n"
                "Run with --make-files first to generate the dataset."
            )
            raise SystemExit(1)

    # Open OME-Zarr store before starting the event loop (TensorStore handles
    # must be opened synchronously).
    zarr_uri = (dataset_dir / "image.ome.zarr").resolve().as_uri()
    image_store = OMEZarrImageDataStore.from_path(zarr_uri)

    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    QtAsyncio.run(async_main(dataset_dir, image_store), handle_sigint=True)


if __name__ == "__main__":
    main()
