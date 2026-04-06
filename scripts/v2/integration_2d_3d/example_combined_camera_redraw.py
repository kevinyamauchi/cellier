"""Combined 2D/3D viewer with automatic camera-settle redraw.

This is a variant of example_combined.py that replaces the manual "Update"
button with automatic redraw triggered when the camera has been still for a
configurable threshold duration (default 300 ms).

A spinbox in the sidebar lets the user adjust the settle threshold at runtime.
"""

from __future__ import annotations

import argparse
import asyncio
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
from cellier.v2.data.image import MultiscaleZarrDataStore
from cellier.v2.render._config import RenderManagerConfig, SlicingConfig
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.visuals._image import ImageAppearance

# ---------------------------------------------------------------------------
# Debug-logging CLI helper (power-user per-category:level syntax)
# ---------------------------------------------------------------------------


def _setup_debug_logging(spec: str) -> None:
    """Parse ``--debug-log`` spec and configure cellier loggers.

    Supports several forms::

        "all"                           -> all categories at DEBUG
        "all:info"                      -> all categories at INFO
        "perf,cache"                    -> perf+cache at DEBUG
        "perf:info,cache:debug"         -> per-category levels

    A bare category name (no colon) defaults to DEBUG.
    """
    import logging as _logging

    from cellier.v2.logging import _CATEGORY_MAP, enable_debug_logging

    _LEVEL_NAMES = {
        "debug": _logging.DEBUG,
        "info": _logging.INFO,
        "warning": _logging.WARNING,
    }

    # Parse spec into {category: level_int} pairs.
    overrides: dict[str, int] = {}
    all_cats = tuple(_CATEGORY_MAP.keys())
    default_level = _logging.DEBUG

    if spec in ("", "all"):
        # All categories at DEBUG — no overrides needed.
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

    # Determine which categories to enable.
    if overrides:
        cats = tuple(overrides.keys())
    else:
        cats = all_cats

    # Step 1: enable handler + set all requested categories to DEBUG.
    enable_debug_logging(categories=cats)

    # Step 2: apply per-category level overrides (or global default).
    for cat in cats:
        target_level = overrides.get(cat, default_level)
        logger = _CATEGORY_MAP.get(cat)
        if logger is not None:
            logger.setLevel(target_level)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32
GPU_BUDGET_2D = 64 * 1024**2  # 64 MiB for 2D tile cache
GPU_BUDGET_3D = 512 * 1024**2  # 512 MiB for 3D brick cache
LOD_BIAS = 1.0

ZARR_PATH = pathlib.Path(__file__).parent / "multiscale_blobs.zarr"
ZARR_SCALE_NAMES = ["s0", "s1", "s2"]

FRUSTUM_COLOR = "#00cc44"
AABB_COLOR = "#ff00ff"

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


def _make_frustum_wireframe(corners: np.ndarray, color: str = "#00cc44") -> gfx.Line:
    """Build a wireframe from 8 frustum corners (shape 2x4x3)."""
    edge_indices = [
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((0, 3), (0, 0)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 2), (1, 3)),
        ((1, 3), (1, 0)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((0, 2), (1, 2)),
        ((0, 3), (1, 3)),
    ]
    positions = np.array(
        [[corners[a], corners[b]] for (a, b) in edge_indices],
        dtype=np.float32,
    ).reshape(-1, 3)
    return gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineSegmentMaterial(color=color, thickness=1.5),
    )


def _make_separator() -> QFrame:
    """Horizontal separator line for the side panel."""
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class CombinedApp(QMainWindow):
    """Single-window viewer toggling between 2D and 3D cellier scenes."""

    _COLORMAPS: ClassVar[list[str]] = ["viridis", "gray"]

    def __init__(self, data_store: MultiscaleZarrDataStore) -> None:
        super().__init__()
        self._force_level: int | None = None
        self._colormap_index: int = 0
        self._active_mode: str = "2d"
        self._frustum_line: gfx.Line | None = None

        self._controller = CellierController(
            widget_parent=self,
            render_config=RenderManagerConfig(slicing=SlicingConfig(batch_size=128)),
        )
        cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))

        z_depth = data_store.level_shapes[0][0]
        self._z_max = z_depth - 1

        # ── 2D scene ──────────────────────────────────────────────────
        self._scene_2d = self._controller.add_scene(
            dim="2d", coordinate_system=cs, name="scene_2d"
        )
        self._scene_2d.dims.selection.slice_indices = {0: z_depth // 2}

        self._visual_2d = self._controller.add_image(
            data=data_store,
            scene_id=self._scene_2d.id,
            appearance=ImageAppearance(
                color_map="viridis",
                clim=(0.0, 1.0),
                lod_bias=LOD_BIAS,
                force_level=None,
                frustum_cull=True,
            ),
            name="image_2d",
            block_size=BLOCK_SIZE,
            gpu_budget_bytes=GPU_BUDGET_2D,
        )
        self._canvas_widget_2d = self._controller.add_canvas(self._scene_2d.id)

        # ── 3D scene ──────────────────────────────────────────────────
        self._scene_3d = self._controller.add_scene(
            dim="3d", coordinate_system=cs, name="scene_3d"
        )

        self._visual_3d = self._controller.add_image(
            data=data_store,
            scene_id=self._scene_3d.id,
            appearance=ImageAppearance(
                color_map="viridis",
                clim=(0.0, 1.0),
                lod_bias=LOD_BIAS,
                force_level=None,
                frustum_cull=True,
            ),
            name="volume_3d",
            block_size=BLOCK_SIZE,
            gpu_budget_bytes=GPU_BUDGET_3D,
            threshold=0.2,
        )
        self._canvas_widget_3d = self._controller.add_canvas(self._scene_3d.id)

        # Add AABB wireframe to the 3D scene.
        gfx_visual_3d = self._get_gfx_visual_3d()
        d, h, w = gfx_visual_3d._volume_geometry.base_layout.volume_shape
        pad = 1.0
        gfx_scene_3d = self._controller._render_manager.get_scene(self._scene_3d.id)
        gfx_scene_3d.add(
            _make_box_wireframe(
                np.array([-0.5 - pad, -0.5 - pad, -0.5 - pad]),
                np.array([w - 0.5 + pad, h - 0.5 + pad, d - 0.5 + pad]),
                AABB_COLOR,
            )
        )

        self._setup_ui()

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

        # Auto-redraw toggle
        self._auto_redraw_cb = QCheckBox("Auto-redraw on camera move")
        self._auto_redraw_cb.setChecked(True)
        self._auto_redraw_cb.toggled.connect(self._on_auto_redraw_toggled)
        pl.addWidget(self._auto_redraw_cb)

        # Settle threshold — only meaningful when auto-redraw is on
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
        self._z_slice_sb.setValue(self._scene_2d.dims.selection.slice_indices[0])
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

        self._show_frustum_cb = QCheckBox("Show frustum")
        self._show_frustum_cb.setChecked(False)
        self._show_frustum_cb.toggled.connect(self._on_show_frustum_toggled)
        pl.addWidget(self._show_frustum_cb)
        self._widget_3d.append(self._show_frustum_cb)

        lbl_far = QLabel("Far plane:")
        pl.addWidget(lbl_far)
        self._widget_3d.append(lbl_far)

        self._far_plane_sb = QDoubleSpinBox()
        self._far_plane_sb.setRange(100.0, 200_000.0)
        self._far_plane_sb.setSingleStep(500.0)
        self._far_plane_sb.setDecimals(0)
        self._far_plane_sb.setValue(8000.0)
        self._far_plane_sb.valueChanged.connect(self._on_far_plane_changed)
        pl.addWidget(self._far_plane_sb)
        self._widget_3d.append(self._far_plane_sb)

        _sep_3d = _make_separator()
        pl.addWidget(_sep_3d)
        self._widget_3d.append(_sep_3d)

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

        # Canvas container — holds both canvas widgets, only one visible.
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
        # Cancel in-flight slicing requests for the scene we are leaving.
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

        self._status_label.setText("Auto-redraw active — move camera to update")

    # ── UI callbacks ──────────────────────────────────────────────────

    def _on_auto_redraw_toggled(self, checked: bool) -> None:
        """Enable or disable camera-settle reslicing from the checkbox."""
        self._controller.camera_reslice_enabled = checked
        self._settle_sb.setEnabled(checked)

    def _on_settle_threshold_changed(self, value_ms: float) -> None:
        """Update the controller's settle threshold from the spinbox."""
        self._controller.camera_settle_threshold_s = value_ms / 1000.0

    def _on_toggle_colormap(self) -> None:
        self._colormap_index = (self._colormap_index + 1) % len(self._COLORMAPS)
        new_cmap = self._COLORMAPS[self._colormap_index]
        # Update both visuals.
        self._visual_2d.appearance.color_map = new_cmap
        self._visual_3d.appearance.color_map = new_cmap
        self._colormap_btn.setText(f"Colormap: {new_cmap}")
        print(f"[colormap] switched to '{new_cmap}'")

    def _on_level_radio_clicked(self, button: QRadioButton) -> None:
        self._force_level = button.property("force_level")

    def _on_z_slice_changed(self, value: int) -> None:
        self._scene_2d.dims.selection.slice_indices = {0: value}
        # All cached tiles are from the old z-plane — clear and re-fetch.
        gfx_visual = self._get_gfx_visual_2d()
        gfx_visual._block_cache_2d.tile_manager.clear()
        gfx_visual._lut_manager_2d.rebuild(gfx_visual._block_cache_2d.tile_manager)
        # Trigger immediate reslice (not camera-driven, so bypass settle).
        self._controller.reslice_scene(self._scene_2d.id)

    def _on_show_frustum_toggled(self, checked: bool) -> None:
        if self._frustum_line is not None:
            self._frustum_line.visible = checked

    def _on_far_plane_changed(self, value: float) -> None:
        self._controller.set_camera_depth_range(
            scene_id=self._scene_3d.id,
            depth_range=(1.0, value),
        )

    def _rebuild_frustum_wireframe(self, corners: np.ndarray) -> None:
        gfx_scene = self._controller._render_manager.get_scene(self._scene_3d.id)
        if self._frustum_line is not None:
            gfx_scene.remove(self._frustum_line)
        self._frustum_line = _make_frustum_wireframe(corners, color=FRUSTUM_COLOR)
        self._frustum_line.visible = self._show_frustum_cb.isChecked()
        gfx_scene.add(self._frustum_line)


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------


async def async_main(data_store: MultiscaleZarrDataStore) -> None:
    """Create the main window and run the Qt event loop."""
    app = QApplication.instance()
    window = CombinedApp(data_store)
    window.resize(1280, 800)
    window.setWindowTitle("Combined 2D / 3D — auto camera-settle redraw")
    window.show()

    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args, build the data store, and launch the viewer."""
    parser = argparse.ArgumentParser(
        description="Combined 2D/3D viewer with automatic camera-settle redraw"
    )
    parser.add_argument(
        "--zarr-path",
        type=pathlib.Path,
        default=ZARR_PATH,
        help="Path to the multiscale zarr store.",
    )
    parser.add_argument(
        "--debug-log",
        metavar="SPEC",
        default=None,
        const="all",
        nargs="?",
        help=(
            "Enable debug logging. Comma-separated list of categories "
            "(perf, gpu, cache, slicer). "
            "Omit value or use 'all' for all categories at DEBUG level. "
            "Power-user: append :LEVEL to set per-category levels. "
            "Examples:\n"
            "  --debug-log                    (all at DEBUG)\n"
            "  --debug-log perf,cache         (perf+cache at DEBUG)\n"
            "  --debug-log all:info           (all at INFO — summaries only)\n"
            "  --debug-log perf:info,cache:debug,slicer:info"
        ),
    )
    args = parser.parse_args()

    if not args.zarr_path.exists():
        print(f"Error: zarr store not found at '{args.zarr_path}'")
        print("Run example.py with --make-files first:")
        print("    uv run example.py --make-files")
        sys.exit(1)

    if args.debug_log is not None:
        _setup_debug_logging(args.debug_log)

    print("Opening tensorstore stores via MultiscaleZarrDataStore ...")
    data_store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(args.zarr_path),
        scale_names=ZARR_SCALE_NAMES,
        level_scales=[(1, 1, 1), (2, 2, 2), (4, 4, 4)],
        level_translations=[(0, 0, 0), (0.5, 0.5, 0.5), (1.5, 1.5, 1.5)],
    )
    print(f"  {data_store.n_levels} levels opened.")
    for i, shape in enumerate(data_store.level_shapes):
        print(f"  s{i}: shape={shape}")
    print()
    print(
        "Move the camera to trigger automatic redraw after the settle threshold.\n"
        "Adjust the settle threshold with the spinbox in the sidebar.\n"
    )

    _app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(data_store), handle_sigint=True)


if __name__ == "__main__":
    main()
