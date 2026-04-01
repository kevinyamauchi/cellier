"""OME-Zarr 2D/3D viewer with automatic camera-settle redraw.

Demonstrates ``OMEZarrImageDataStore`` with a synthetic OME-Zarr v0.5 file.
Includes an iso-threshold spinbox for 3D mode.

Generate the OME-Zarr store once::

    uv run scripts/v2/integration_2d_3d/example_ome_zarr.py --make-files

Then launch the viewer::

    uv run scripts/v2/integration_2d_3d/example_ome_zarr.py [--zarr-path PATH]

Controls:
    Mouse              — pan/zoom (2D) or orbit (3D)
    Toggle 2D/3D       — switch rendering mode
    Auto-redraw cb     — redraw on camera settle
    Z-slice spinbox    — change z-slice in 2D mode
    Iso threshold sb   — adjust iso threshold in 3D mode (3D mode only)
    Contrast min/max   — adjust contrast limits (both 2D and 3D)
    Colormap btn       — cycle viridis / gray
    Force-level radios — lock LOD level
    LOD bias spinbox   — scale LOD thresholds
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from typing import ClassVar

import numpy as np
import pygfx as gfx
import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellier.v2.controller import CellierController
from cellier.v2.data.image import OMEZarrImageDataStore
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._image import ImageAppearance

# ---------------------------------------------------------------------------
# Debug-logging CLI helper
# ---------------------------------------------------------------------------


def _setup_debug_logging(spec: str) -> None:
    """Parse ``--debug-log`` spec and configure cellier loggers."""
    import logging as _logging

    from cellier.v2.logging import _CATEGORY_MAP, enable_debug_logging

    _LEVEL_NAMES = {
        "debug": _logging.DEBUG,
        "info": _logging.INFO,
        "warning": _logging.WARNING,
    }

    overrides: dict[str, int] = {}
    all_cats = tuple(_CATEGORY_MAP.keys())
    default_level = _logging.DEBUG

    if spec in ("", "all"):
        pass
    elif spec.startswith("all:"):
        level_str = spec.split(":", 1)[1].strip().lower()
        default_level = _LEVEL_NAMES.get(level_str, _logging.DEBUG)
    else:
        for token in spec.split(","):
            token = token.strip()
            if ":" in token:
                cat, level_str = token.split(":", 1)
                overrides[cat.strip()] = _LEVEL_NAMES.get(
                    level_str.strip().lower(), _logging.DEBUG
                )
            else:
                overrides[token] = _logging.DEBUG

    if overrides:
        cats = tuple(overrides.keys())
    else:
        cats = all_cats

    enable_debug_logging(categories=cats)
    for cat in cats:
        target_level = overrides.get(cat, default_level)
        logger = _CATEGORY_MAP.get(cat)
        if logger is not None:
            logger.setLevel(target_level)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZARR_PATH = pathlib.Path(__file__).parent / "ome_zarr_blobs.ome.zarr"
BLOCK_SIZE = 32
GPU_BUDGET_2D = 64 * 1024**2
GPU_BUDGET_3D = 512 * 1024**2
LOD_BIAS = 1.0
ISO_THRESHOLD = 0.2

AABB_COLOR = "#ff00ff"

# ---------------------------------------------------------------------------
# OME-Zarr generation
# ---------------------------------------------------------------------------


def _make_ome_zarr(path: pathlib.Path) -> None:
    """Write a valid 3D OME-Zarr v0.5 store with binary blobs."""
    import zarr

    try:
        from skimage.data import binary_blobs
    except ImportError:
        print("scikit-image required for --make-files.  Install with:")
        print("    uv pip install scikit-image")
        sys.exit(1)

    path.mkdir(parents=True, exist_ok=True)

    # Generate synthetic volume.
    vol = binary_blobs(length=128, n_dim=3, blob_size_fraction=0.05, rng=42)
    vol = vol.astype(np.float32)

    # Build 3 pyramid levels: (64,128,128), (32,64,64), (16,32,32)
    # with anisotropic z (z voxels are 2x larger than y/x at level 0).
    shapes = [(64, 128, 128), (32, 64, 64), (16, 32, 32)]
    datasets = []
    scale_configs = [
        {"scale": [2.0, 1.0, 1.0], "translation": [0.5, 0.5, 0.5]},
        {"scale": [4.0, 2.0, 2.0], "translation": [1.5, 1.0, 1.0]},
        {"scale": [8.0, 4.0, 4.0], "translation": [3.5, 3.0, 3.0]},
    ]

    for i, (shape, cfg) in enumerate(zip(shapes, scale_configs)):
        ds_path = str(i)
        datasets.append(
            {
                "path": ds_path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": cfg["scale"]},
                    {"type": "translation", "translation": cfg["translation"]},
                ],
            }
        )

        # Downsample volume for this level.
        if i == 0:
            level_data = vol[: shape[0], : shape[1], : shape[2]]
        else:
            # Simple 2x downsample via slicing.
            factor = 2**i
            level_data = vol[::factor, ::factor, ::factor]
            level_data = level_data[: shape[0], : shape[1], : shape[2]]

        # Write zarr v3 array.
        store = zarr.storage.LocalStore(str(path / ds_path))
        arr = zarr.create(
            store=store,
            shape=shape,
            dtype="float32",
            chunks=(min(32, shape[0]), min(32, shape[1]), min(32, shape[2])),
            zarr_format=3,
        )
        arr[...] = level_data.astype(np.float32)
        print(f"  Level {i}: shape={shape}")

    # Write root zarr.json with OME metadata.
    root_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": "blobs",
                        "axes": [
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": datasets,
                        "version": "0.5",
                    }
                ],
            }
        },
    }
    (path / "zarr.json").write_text(json.dumps(root_meta, indent=2))
    print(f"Wrote OME-Zarr v0.5 store to {path}")


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _make_box_wireframe(
    box_min: np.ndarray, box_max: np.ndarray, color: str
) -> gfx.Line:
    """Build an AABB wireframe as disconnected edge pairs."""
    x0, y0, z0 = box_min
    x1, y1, z1 = box_max
    positions = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y1, z0],
            [x0, y0, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x1, y1, z1],
            [x0, y1, z1],
            [x0, y1, z1],
            [x0, y0, z1],
            [x0, y0, z0],
            [x0, y0, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z0],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    return gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineSegmentMaterial(color=color, thickness=1.0),
    )


def _make_separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


def _dtype_max(dtype: np.dtype) -> float:
    """Return the maximum representable value for a numpy dtype."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return float(np.finfo(dtype).max)


class OMEZarrApp(QMainWindow):
    """Single-window viewer toggling between 2D and 3D cellier scenes."""

    _COLORMAPS: ClassVar[list[str]] = ["viridis", "gray"]

    def __init__(self, data_store: OMEZarrImageDataStore) -> None:
        super().__init__()
        self._force_level: int | None = None
        self._colormap_index: int = 0
        self._active_mode: str = "2d"
        self._dtype_max = _dtype_max(data_store.dtype)

        self._controller = CellierController(widget_parent=self, slicer_batch_size=128)

        # Build coordinate system from store axes.
        space_axes = [ax for ax in data_store.axes if ax.type == "space"]
        cs = CoordinateSystem(
            name="world",
            axis_labels=tuple(ax.name for ax in data_store.axes),
        )

        z_depth = data_store.level_shapes[0][
            next(ax.array_dim for ax in data_store.axes if ax.name == "z")
        ]
        self._z_max = z_depth - 1

        # Determine displayed axes from the store's spatial axes.
        spatial_dims = tuple(ax.array_dim for ax in space_axes)
        non_spatial_slice = {
            ax.array_dim: 0 for ax in data_store.axes if ax.type != "space"
        }

        # Build a voxel-to-world transform from the level-0 physical scale.
        # The level_transforms on the store are voxel ratios (level-k → level-0);
        # the physical voxel size at level 0 comes from the OME scale metadata.
        # We extract it by reading the diagonal of the store's internal data:
        # for this dataset, the OME level-0 scale is [2.0, 1.0, 1.0] (z, y, x).
        t0 = data_store.level_transforms[0]  # identity in voxel-ratio space
        ndim = t0.ndim
        # Re-read the OME metadata to get the raw level-0 physical scale.
        import yaozarrs

        group = yaozarrs.open_group(data_store.zarr_path)
        ome_image = group.ome_metadata()
        ms = ome_image.multiscales[data_store.multiscale_index]
        level_0_scale = tuple(ms.datasets[0].scale_transform.scale)
        voxel_to_world = AffineTransform.from_scale_and_translation(scale=level_0_scale)

        # ── 2D scene ──────────────────────────────────────────────────
        self._scene_2d = self._controller.add_scene(
            dim="2d", coordinate_system=cs, name="scene_2d"
        )
        # Override displayed axes to the last two spatial dims.
        self._scene_2d.dims.selection.displayed_axes = spatial_dims[-2:]
        # Slice the z axis at the midpoint, plus any non-spatial axes.
        z_dim = next(ax.array_dim for ax in space_axes if ax.name == "z")
        slice_indices_2d = dict(non_spatial_slice)
        slice_indices_2d[z_dim] = z_depth // 2
        self._scene_2d.dims.selection.slice_indices = slice_indices_2d

        self._visual_2d = self._controller.add_image(
            data=data_store,
            scene_id=self._scene_2d.id,
            appearance=ImageAppearance(
                color_map="viridis",
                clim=(0.0, self._dtype_max),
                lod_bias=LOD_BIAS,
                force_level=None,
                frustum_cull=True,
                iso_threshold=ISO_THRESHOLD,
            ),
            name="image_2d",
            block_size=BLOCK_SIZE,
            gpu_budget_bytes=GPU_BUDGET_2D,
        )
        self._visual_2d.transform = voxel_to_world
        self._canvas_widget_2d = self._controller.add_canvas(self._scene_2d.id)

        # ── 3D scene ──────────────────────────────────────────────────
        self._scene_3d = self._controller.add_scene(
            dim="3d", coordinate_system=cs, name="scene_3d"
        )
        self._scene_3d.dims.selection.displayed_axes = spatial_dims
        self._scene_3d.dims.selection.slice_indices = dict(non_spatial_slice)

        self._visual_3d = self._controller.add_image(
            data=data_store,
            scene_id=self._scene_3d.id,
            appearance=ImageAppearance(
                color_map="viridis",
                clim=(0.0, self._dtype_max),
                lod_bias=LOD_BIAS,
                force_level=None,
                frustum_cull=True,
                iso_threshold=ISO_THRESHOLD,
            ),
            name="volume_3d",
            block_size=BLOCK_SIZE,
            gpu_budget_bytes=GPU_BUDGET_3D,
        )
        self._visual_3d.transform = voxel_to_world

        # Compute world extents for the depth range and wireframe.
        vox_shape = data_store.level_shapes[0]
        spatial_shape = tuple(vox_shape[d] for d in spatial_dims)
        spatial_scale = tuple(level_0_scale[d] for d in spatial_dims)
        world_extents = tuple(s * sc for s, sc in zip(spatial_shape, spatial_scale))
        max_extent = max(world_extents)

        print(f"\n scale: {level_0_scale}")
        print(f" world extents: {world_extents}")
        print(f" max extent: {max_extent}\n")

        # Set depth range large enough to see the full volume from any angle.
        depth_range_3d = (0.1, max_extent * 5.0)
        self._canvas_widget_3d = self._controller.add_canvas(
            self._scene_3d.id, depth_range=depth_range_3d
        )

        # Add AABB wireframe to the 3D scene in world coordinates.
        # pygfx uses (x, y, z) order → reverse from data order (z, y, x).
        wx, wy, wz = world_extents[2], world_extents[1], world_extents[0]
        pad = 1.0
        gfx_scene_3d = self._controller._render_manager.get_scene(self._scene_3d.id)
        gfx_scene_3d.add(
            _make_box_wireframe(
                np.array([-0.5 - pad, -0.5 - pad, -0.5 - pad]),
                np.array([wx - 0.5 + pad, wy - 0.5 + pad, wz - 0.5 + pad]),
                AABB_COLOR,
            )
        )

        # Re-fit the 3D camera now that the wireframe is in the scene.
        self._controller.look_at_visual(
            self._visual_3d.id,
            view_direction=(-1, -1, -1),
            up=(0, 0, 1),
        )

        self._setup_ui()

        # Trigger an initial reslice so tiles load before the first
        # camera-settle event.
        self._controller.reslice_scene(self._scene_2d.id)

    # ── Render-layer visual accessors ─────────────────────────────────

    def _get_gfx_visual_2d(self):
        return self._controller._render_manager._scenes[self._scene_2d.id]._visuals[
            self._visual_2d.id
        ]

    def _get_gfx_visual_3d(self):
        return self._controller._render_manager._scenes[self._scene_3d.id]._visuals[
            self._visual_3d.id
        ]

    # ── UI setup ──────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        panel = QWidget()
        panel.setFixedWidth(230)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(8, 8, 8, 8)

        # ── Shared controls ───────────────────────────────────────────
        self._toggle_btn = QPushButton("Toggle 2D / 3D")
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        pl.addWidget(self._toggle_btn)

        self._auto_redraw_cb = QCheckBox("Auto-redraw on camera move")
        self._auto_redraw_cb.setChecked(True)
        self._auto_redraw_cb.toggled.connect(self._on_auto_redraw_toggled)
        pl.addWidget(self._auto_redraw_cb)

        pl.addWidget(QLabel("Settle threshold (ms):"))
        self._settle_sb = QDoubleSpinBox()
        self._settle_sb.setRange(50.0, 5000.0)
        self._settle_sb.setSingleStep(50.0)
        self._settle_sb.setDecimals(0)
        self._settle_sb.setValue(300.0)
        self._settle_sb.valueChanged.connect(self._on_settle_threshold_changed)
        pl.addWidget(self._settle_sb)

        self._colormap_btn = QPushButton(
            f"Colormap: {self._COLORMAPS[self._colormap_index]}"
        )
        self._colormap_btn.clicked.connect(self._on_toggle_colormap)
        pl.addWidget(self._colormap_btn)

        self._mode_label = QLabel("Mode: 2D")
        pl.addWidget(self._mode_label)

        pl.addWidget(_make_separator())

        # ── 2D-only controls ─────────────────────────────────────────
        self._widget_2d: list[QWidget] = []

        lbl_z = QLabel("Z-slice:")
        pl.addWidget(lbl_z)
        self._widget_2d.append(lbl_z)

        self._z_slice_sb = QSpinBox()
        self._z_slice_sb.setRange(0, self._z_max)
        z_dim = (
            next(ax.array_dim for ax in self._visual_2d.appearance._store_axes)
            if hasattr(self._visual_2d.appearance, "_store_axes")
            else 0
        )
        self._z_slice_sb.setValue(self._z_max // 2)
        self._z_slice_sb.valueChanged.connect(self._on_z_slice_changed)
        pl.addWidget(self._z_slice_sb)
        self._widget_2d.append(self._z_slice_sb)

        self._viewport_cull_cb = QCheckBox("Viewport cull")
        self._viewport_cull_cb.setChecked(True)
        pl.addWidget(self._viewport_cull_cb)
        self._widget_2d.append(self._viewport_cull_cb)

        _sep_2d = _make_separator()
        pl.addWidget(_sep_2d)
        self._widget_2d.append(_sep_2d)

        # ── 3D-only controls ─────────────────────────────────────────
        self._widget_3d: list[QWidget] = []

        self._frustum_cull_cb = QCheckBox("Frustum cull")
        self._frustum_cull_cb.setChecked(True)
        pl.addWidget(self._frustum_cull_cb)
        self._widget_3d.append(self._frustum_cull_cb)

        pl.addWidget(_make_separator())

        # ── Iso threshold (3D mode) ──────────────────────────────────
        lbl_thresh = QLabel("Iso threshold:")
        pl.addWidget(lbl_thresh)
        self._widget_3d.append(lbl_thresh)

        self._threshold_sb = QDoubleSpinBox()
        self._threshold_sb.setRange(0.0, self._dtype_max)
        self._threshold_sb.setSingleStep(0.01)
        self._threshold_sb.setDecimals(3)
        self._threshold_sb.setValue(self._visual_3d.appearance.iso_threshold)
        self._threshold_sb.valueChanged.connect(self._on_iso_threshold_changed)
        pl.addWidget(self._threshold_sb)
        self._widget_3d.append(self._threshold_sb)

        _sep_3d = _make_separator()
        pl.addWidget(_sep_3d)
        self._widget_3d.append(_sep_3d)

        # ── Contrast limits ──────────────────────────────────────────
        pl.addWidget(QLabel("Contrast min:"))
        self._clim_min_sb = QDoubleSpinBox()
        self._clim_min_sb.setRange(0.0, self._dtype_max)
        self._clim_min_sb.setSingleStep(0.1)
        self._clim_min_sb.setDecimals(3)
        self._clim_min_sb.setValue(0.0)
        self._clim_min_sb.valueChanged.connect(self._on_clim_changed)
        pl.addWidget(self._clim_min_sb)

        pl.addWidget(QLabel("Contrast max:"))
        self._clim_max_sb = QDoubleSpinBox()
        self._clim_max_sb.setRange(0.0, self._dtype_max)
        self._clim_max_sb.setSingleStep(0.1)
        self._clim_max_sb.setDecimals(3)
        self._clim_max_sb.setValue(self._dtype_max)
        self._clim_max_sb.valueChanged.connect(self._on_clim_changed)
        pl.addWidget(self._clim_max_sb)

        pl.addWidget(_make_separator())

        # ── Shared LOD controls ──────────────────────────────────────
        pl.addWidget(QLabel("Force level:"))
        self._level_group = QButtonGroup(self)
        for label, value in [("Auto", None), ("1", 1), ("2", 2), ("3", 3)]:
            rb = QRadioButton(label)
            if value is None:
                rb.setChecked(True)
            self._level_group.addButton(rb)
            rb.setProperty("force_level", value)
            pl.addWidget(rb)
        self._level_group.buttonClicked.connect(self._on_level_radio_clicked)

        pl.addWidget(QLabel("LOD bias:"))
        self._lod_bias_sb = QDoubleSpinBox()
        self._lod_bias_sb.setRange(0.1, 10.0)
        self._lod_bias_sb.setSingleStep(0.1)
        self._lod_bias_sb.setDecimals(2)
        self._lod_bias_sb.setValue(LOD_BIAS)
        pl.addWidget(self._lod_bias_sb)

        pl.addStretch()

        self._status_label = QLabel("Auto-redraw active — move camera to update")
        self._status_label.setWordWrap(True)
        pl.addWidget(self._status_label)

        # Start with 3D widgets hidden (2D is active).
        for w in self._widget_3d:
            w.setVisible(False)

        # Canvas container.
        self._canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self._canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self._canvas_widget_2d)
        canvas_layout.addWidget(self._canvas_widget_3d)
        self._canvas_widget_3d.setVisible(False)

        root.addWidget(panel)
        root.addWidget(self._canvas_container, stretch=1)

    # ── Toggle button ─────────────────────────────────────────────────

    def _on_toggle_clicked(self) -> None:
        coordinator = self._controller._render_manager._slice_coordinator
        if self._active_mode == "2d":
            coordinator.cancel_scene(self._scene_2d.id)
            self._active_mode = "3d"
            self._canvas_widget_2d.setVisible(False)
            self._canvas_widget_3d.setVisible(True)
            self._mode_label.setText("Mode: 3D")
            for w in self._widget_2d:
                w.setVisible(False)
            for w in self._widget_3d:
                w.setVisible(True)
        else:
            coordinator.cancel_scene(self._scene_3d.id)
            self._active_mode = "2d"
            self._canvas_widget_3d.setVisible(False)
            self._canvas_widget_2d.setVisible(True)
            self._mode_label.setText("Mode: 2D")
            for w in self._widget_3d:
                w.setVisible(False)
            for w in self._widget_2d:
                w.setVisible(True)

        self._threshold_sb.setEnabled(self._active_mode == "3d")
        self._status_label.setText("Auto-redraw active — move camera to update")

    # ── UI callbacks ──────────────────────────────────────────────────

    def _on_auto_redraw_toggled(self, checked: bool) -> None:
        self._controller.camera_reslice_enabled = checked
        self._settle_sb.setEnabled(checked)

    def _on_settle_threshold_changed(self, value_ms: float) -> None:
        self._controller._camera_settle_threshold_s = value_ms / 1000.0

    def _on_toggle_colormap(self) -> None:
        self._colormap_index = (self._colormap_index + 1) % len(self._COLORMAPS)
        new_cmap = self._COLORMAPS[self._colormap_index]
        self._visual_2d.appearance.color_map = new_cmap
        self._visual_3d.appearance.color_map = new_cmap
        self._colormap_btn.setText(f"Colormap: {new_cmap}")
        print(f"[colormap] switched to '{new_cmap}'")

    def _on_level_radio_clicked(self, button: QRadioButton) -> None:
        self._force_level = button.property("force_level")

    def _on_z_slice_changed(self, value: int) -> None:
        # z is always axis dim 0 for this 3D spatial store.
        self._scene_2d.dims.selection.slice_indices = {0: value}
        gfx_visual = self._get_gfx_visual_2d()
        gfx_visual._block_cache_2d.tile_manager.clear()
        gfx_visual._lut_manager_2d.rebuild(gfx_visual._block_cache_2d.tile_manager)
        self._controller.reslice_scene(self._scene_2d.id)

    def _on_iso_threshold_changed(self, value: float) -> None:
        self._visual_3d.appearance.iso_threshold = value

    def _on_clim_changed(self) -> None:
        clim = (self._clim_min_sb.value(), self._clim_max_sb.value())
        self._visual_2d.appearance.clim = clim
        self._visual_3d.appearance.clim = clim


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------


async def async_main(data_store: OMEZarrImageDataStore) -> None:
    app = QApplication.instance()
    window = OMEZarrApp(data_store)
    window.resize(1280, 800)
    window.setWindowTitle("OME-Zarr 2D / 3D — auto camera-settle redraw")
    window.show()

    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OME-Zarr 2D/3D viewer with automatic camera-settle redraw"
    )
    parser.add_argument(
        "--zarr-path",
        default=str(ZARR_PATH),
        help="Path or URI to the OME-Zarr store (file://, s3://, gs://, https://).",
    )
    parser.add_argument(
        "--anonymous",
        action="store_true",
        help="Use anonymous credentials for S3/GCS public buckets.",
    )
    parser.add_argument(
        "--series-index",
        type=int,
        default=0,
        help="For Bf2Raw containers, which image series to open (default: 0).",
    )
    parser.add_argument(
        "--make-files",
        action="store_true",
        help="Generate the synthetic OME-Zarr store and exit.",
    )
    parser.add_argument(
        "--debug-log",
        metavar="SPEC",
        default=None,
        const="all",
        nargs="?",
        help=(
            "Enable debug logging. See example_combined_camera_redraw.py "
            "for full spec documentation."
        ),
    )
    args = parser.parse_args()

    if args.make_files:
        print("Generating OME-Zarr v0.5 store ...")
        _make_ome_zarr(pathlib.Path(args.zarr_path))
        print("Done.")
        sys.exit(0)

    zarr_uri = args.zarr_path
    # Detect whether the path is a URI or a local filesystem path.
    has_scheme = "://" in zarr_uri
    if not has_scheme:
        # Local path — check existence and convert to file:// URI.
        local_path = pathlib.Path(zarr_uri)
        if not local_path.exists():
            print(f"Error: zarr store not found at '{local_path}'")
            print("Generate it first with:")
            print(
                "    uv run scripts/v2/integration_2d_3d/example_ome_zarr.py"
                " --make-files"
            )
            sys.exit(1)
        zarr_uri = f"file://{local_path.resolve()}"

    if args.debug_log is not None:
        _setup_debug_logging(args.debug_log)

    print("Opening OME-Zarr store via OMEZarrImageDataStore.from_path() ...")
    data_store = OMEZarrImageDataStore.from_path(
        zarr_uri,
        anonymous=args.anonymous,
        series_index=args.series_index,
    )
    print(f"  {data_store.n_levels} levels opened.")
    for i, shape in enumerate(data_store.level_shapes):
        print(f"  Level {i}: shape={shape}")
    print(f"  Axes: {data_store.axis_names}")
    print(f"  Units: {data_store.axis_units}")
    print()
    print(
        "Move the camera to trigger automatic redraw after the settle threshold.\n"
        "Adjust the iso threshold spinbox in the sidebar for 3D mode.\n"
    )

    _app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(data_store), handle_sigint=True)


if __name__ == "__main__":
    main()
