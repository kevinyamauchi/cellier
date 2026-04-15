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
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._image import ImageAppearance
from cellier.v2.visuals._lines_memory import LinesMemoryAppearance
from cellier.v2.visuals._mesh_memory import MeshFlatAppearance
from cellier.v2.visuals._points_memory import PointsMarkerAppearance

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


async def async_main(dataset_dir: Path, image_store: OMEZarrImageDataStore) -> None:
    """Build and run the transform validation viewer."""
    # ── Load npy files ────────────────────────────────────────────────────────
    points_pos = np.load(dataset_dir / "points.npy")
    points_col = np.load(dataset_dir / "points_colors.npy")
    lines_pos = np.load(dataset_dir / "lines.npy")
    lines_col = np.load(dataset_dir / "lines_colors.npy")
    mesh_verts = np.load(dataset_dir / "mesh_vertices.npy")
    mesh_faces_np = np.load(dataset_dir / "mesh_faces.npy")
    mesh_col = np.load(dataset_dir / "mesh_colors.npy")

    # ── Controller and coordinate system ─────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=["z", "y", "x"])

    # ── Geometry constants ────────────────────────────────────────────────────
    Z_SLICE = 150
    Y_SLICE = 150
    X_SLICE = 150

    axis_ranges = {0: (0, 300), 1: (0, 300), 2: (0, 300)}
    depth_range = (1.0, 5000.0)

    # ── Four scenes ───────────────────────────────────────────────────────────
    xy_scene = controller.add_scene(
        dim="2d", coordinate_system=cs, name="xy", render_modes={"2d"}
    )
    xy_scene.dims.selection.displayed_axes = (1, 2)
    xy_scene.dims.selection.slice_indices = {0: Z_SLICE}

    xz_scene = controller.add_scene(
        dim="2d", coordinate_system=cs, name="xz", render_modes={"2d"}
    )
    xz_scene.dims.selection.displayed_axes = (0, 2)
    xz_scene.dims.selection.slice_indices = {1: Y_SLICE}

    yz_scene = controller.add_scene(
        dim="2d", coordinate_system=cs, name="yz", render_modes={"2d"}
    )
    yz_scene.dims.selection.displayed_axes = (0, 1)
    yz_scene.dims.selection.slice_indices = {2: X_SLICE}

    vol_scene = controller.add_scene(
        dim="3d",
        coordinate_system=cs,
        name="vol",
        render_modes={"3d"},
        lighting="default",
    )

    scenes = {"xy": xy_scene, "xz": xz_scene, "yz": yz_scene, "vol": vol_scene}

    # ── Image visuals ─────────────────────────────────────────────────────────
    # pygfx convention: voxel_spacing is (x, y, z)
    voxel_to_world = AffineTransform.from_scale((3.0, 2.0, 2.0))
    voxel_spacing_xyz = (2.0, 2.0, 3.0)
    CLIM = (0, 8)

    common_2d_img_appearance = ImageAppearance(
        color_map=OCTANT_CMAP,
        clim=CLIM,
        lod_bias=1.0,
        force_level=None,
        frustum_cull=True,
        iso_threshold=0.5,
        render_mode="mip",
    )

    common_img_kwargs = {
        "data": image_store,
        "block_size": 32,
        "gpu_budget_bytes": 256 * 1024**2,
        "gpu_budget_bytes_2d": 64 * 1024**2,
        "threshold": 0.5,
        "use_brick_shader": True,
        "voxel_spacing": voxel_spacing_xyz,
    }

    image_visuals = {}
    for key, scene in [("xy", xy_scene), ("xz", xz_scene), ("yz", yz_scene)]:
        v = controller.add_image(
            scene_id=scene.id,
            appearance=common_2d_img_appearance,
            name=f"{key}_image",
            **common_img_kwargs,
        )
        v.transform = voxel_to_world
        image_visuals[key] = v

    vol_img_appearance = ImageAppearance(
        color_map=OCTANT_CMAP,
        clim=CLIM,
        lod_bias=1.0,
        force_level=image_store.n_levels - 1,  # coarsest level for 3D
        frustum_cull=False,
        iso_threshold=0.05,
        render_mode="iso",
    )
    v = controller.add_image(
        scene_id=vol_scene.id,
        appearance=vol_img_appearance,
        name="vol_image",
        data=image_store,
        block_size=32,
        gpu_budget_bytes=512 * 1024**2,
        gpu_budget_bytes_2d=64 * 1024**2,
        threshold=0.5,
        use_brick_shader=True,
        voxel_spacing=voxel_spacing_xyz,
    )
    v.transform = voxel_to_world
    image_visuals["vol"] = v

    # ── Fit vol camera on first data delivery ─────────────────────────────────
    _vol_scene_mgr = controller._render_manager._scenes[vol_scene.id]
    _gfx_vol_vis = _vol_scene_mgr.get_visual(v.id)
    _vol_fitted = [False]
    _orig_on_data_ready = _gfx_vol_vis.on_data_ready

    def _on_vol_data_ready(batch):
        _orig_on_data_ready(batch)
        if not _vol_fitted[0]:
            _vol_fitted[0] = True
            controller.fit_camera(vol_scene.id)

    _gfx_vol_vis.on_data_ready = _on_vol_data_ready

    # ── Points visuals ────────────────────────────────────────────────────────
    points_store = PointsMemoryStore(
        positions=points_pos,
        colors=points_col,
        name="points",
    )

    points_appearance = PointsMarkerAppearance(
        color_mode="vertex",
        size=10.0,
        size_space="screen",
    )

    points_visuals = {}
    for key, scene in scenes.items():
        v = controller.add_points(
            data=points_store,
            scene_id=scene.id,
            appearance=points_appearance,
            name=f"{key}_points",
        )
        points_visuals[key] = v

    # ── Lines visuals ─────────────────────────────────────────────────────────
    lines_store = LinesMemoryStore(
        positions=lines_pos,
        colors=lines_col,
        name="lines",
    )

    lines_appearance = LinesMemoryAppearance(
        color_mode="vertex",
        thickness=2.0,
        thickness_space="screen",
    )

    lines_visuals = {}
    for key, scene in scenes.items():
        v = controller.add_lines(
            data=lines_store,
            scene_id=scene.id,
            appearance=lines_appearance,
            name=f"{key}_lines",
        )
        lines_visuals[key] = v

    # ── Mesh visuals ──────────────────────────────────────────────────────────
    mesh_store = MeshMemoryStore(
        positions=mesh_verts,
        indices=mesh_faces_np,
        colors=mesh_col,
        name="mesh",
    )

    mesh_appearance = MeshFlatAppearance(
        color_mode="vertex",
        side="front",
        opacity=0.8,
    )

    mesh_visuals = {}
    for key, scene in scenes.items():
        v = controller.add_mesh(
            data=mesh_store,
            scene_id=scene.id,
            appearance=mesh_appearance,
            name=f"{key}_mesh",
        )
        mesh_visuals[key] = v

    # ── Canvases ──────────────────────────────────────────────────────────────
    controller.add_canvas(xy_scene.id, depth_range=depth_range)
    controller.add_canvas(xz_scene.id, depth_range=depth_range)
    controller.add_canvas(yz_scene.id, depth_range=depth_range)
    controller.add_canvas(vol_scene.id, depth_range=depth_range)

    def _canvas_view(scene_id):
        return controller._render_manager._find_canvas_for_scene(scene_id)

    def _make_2d_canvas_widget(scene, slider_style):
        canvas_view = _canvas_view(scene.id)
        axis_labels = dict(enumerate(scene.dims.coordinate_system.axis_labels))
        selection = scene.dims.selection
        dims_sliders = QtDimsSliders(
            controller=controller,
            scene_id=scene.id,
            axis_ranges=axis_ranges,
            axis_labels=axis_labels,
            initial_slice_indices=dict(getattr(selection, "slice_indices", {})),
            initial_displayed_axes=getattr(selection, "displayed_axes", ()),
        )
        dims_sliders.widget.setStyleSheet(slider_style)
        return QtCanvasWidget(canvas_view=canvas_view, dims_sliders=dims_sliders)

    canvas_widgets = {
        "xy": _make_2d_canvas_widget(xy_scene, SLIDER_STYLE_XY),
        "xz": _make_2d_canvas_widget(xz_scene, SLIDER_STYLE_XZ),
        "yz": _make_2d_canvas_widget(yz_scene, SLIDER_STYLE_YZ),
        "vol": QtCanvasWidget.from_scene_and_canvas(
            controller,
            vol_scene,
            _canvas_view(vol_scene.id),
            axis_ranges=axis_ranges,
        ),
    }

    # ── Build and show viewer window ──────────────────────────────────────────
    viewer = TransformValidationViewer(
        controller,
        canvas_widgets=canvas_widgets,
        image_visuals=image_visuals,
        points_visuals=points_visuals,
        lines_visuals=lines_visuals,
        mesh_visuals=mesh_visuals,
        clim_range=(0.0, 8.0),
    )
    viewer.window.setWindowTitle("Transform Validation Viewer")
    viewer.window.resize(1400, 900)
    viewer.window.show()

    for scene in scenes.values():
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
