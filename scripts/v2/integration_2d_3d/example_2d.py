"""Async 2D multiscale image viewer using CellierController.

Uses the full cellier v2 pipeline: CellierController -> RenderManager ->
SliceCoordinator -> SceneManager -> GFXMultiscaleImageVisual.build_slice_request_2d()
-> AsyncSlicer -> data_store.get_data() -> on_data_ready_2d().

No BlockState2D is needed.

Usage
-----
Generate the multiscale zarr store (once):

    uv run example.py --make-files

Then launch the 2D viewer:

    uv run example_2d.py [--zarr-path PATH] [--z-slice N]
"""

import argparse
import asyncio
import pathlib
import sys
import time
from typing import ClassVar

import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
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
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.visuals import ImageAppearance

GPU_BUDGET = 64 * 1024**2  # 64 MB for 2D
LOD_BIAS = 1.0
COMMIT_BATCH_SIZE = 16

ZARR_PATH = pathlib.Path(__file__).parent / "multiscale_blobs.zarr"
ZARR_SCALE_NAMES = ["s0", "s1", "s2"]


class MainWindow(QMainWindow):
    """PySide6 main window using CellierController for 2D image viewing."""

    _COLORMAPS: ClassVar[list[str]] = ["viridis", "gray"]

    def __init__(
        self,
        data_store: MultiscaleZarrDataStore,
        z_slice: int | None = None,
    ) -> None:
        super().__init__()
        self._update_count = 0
        self._force_level: int | None = None
        self._colormap_index: int = 0

        # Build the cellier scene with 2D dimensionality.
        self._controller = CellierController(widget_parent=self, slicer_batch_size=32)
        cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
        self._scene = self._controller.add_scene(
            dim="2d", coordinate_system=cs, name="main"
        )

        # Determine the initial z-slice.
        z_depth = data_store.level_shapes[0][0]
        if z_slice is None:
            z_slice = z_depth // 2

        # Set initial slice index on the scene's dims.
        self._scene.dims.slice_indices = (z_slice,)

        # Add the image through the standard controller API.
        appearance = ImageAppearance(
            color_map="viridis",
            clim=(0.0, 1.0),
            lod_bias=LOD_BIAS,
            force_level=None,
            frustum_cull=True,
        )
        self._visual = self._controller.add_image(
            data=data_store,
            scene_id=self._scene.id,
            appearance=appearance,
            name="image",
            block_size=32,
            gpu_budget_bytes=GPU_BUDGET,
        )

        # Add a canvas (auto-detects 2D -> orthographic + pan/zoom).
        canvas_widget = self._controller.add_canvas(self._scene.id)

        self._z_max = z_depth - 1
        self._setup_ui(canvas_widget)

    def _setup_ui(self, canvas_widget: QWidget) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        panel = QWidget()
        panel.setFixedWidth(220)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)

        # Update button.
        self._update_btn = QPushButton("Update")
        self._update_btn.clicked.connect(
            lambda: asyncio.ensure_future(self._on_update_clicked())
        )
        panel_layout.addWidget(self._update_btn)

        # Colormap toggle.
        self._colormap_btn = QPushButton(
            f"Colormap: {self._COLORMAPS[self._colormap_index]}"
        )
        self._colormap_btn.clicked.connect(self._on_toggle_colormap)
        panel_layout.addWidget(self._colormap_btn)

        # Viewport cull checkbox.
        self._viewport_cull_cb = QCheckBox("Viewport cull")
        self._viewport_cull_cb.setChecked(True)
        panel_layout.addWidget(self._viewport_cull_cb)

        # Force level radio buttons.
        panel_layout.addWidget(QLabel("Force level:"))
        self._level_group = QButtonGroup(self)
        for label, value in [("Auto", None), ("1", 1), ("2", 2), ("3", 3)]:
            rb = QRadioButton(label)
            if value is None:
                rb.setChecked(True)
            self._level_group.addButton(rb)
            rb.setProperty("force_level", value)
            panel_layout.addWidget(rb)
        self._level_group.buttonClicked.connect(self._on_level_radio_clicked)

        # LOD bias spinbox.
        panel_layout.addWidget(QLabel("LOD bias:"))
        self._lod_bias_sb = QDoubleSpinBox()
        self._lod_bias_sb.setRange(0.1, 10.0)
        self._lod_bias_sb.setSingleStep(0.1)
        self._lod_bias_sb.setDecimals(2)
        self._lod_bias_sb.setValue(LOD_BIAS)
        panel_layout.addWidget(self._lod_bias_sb)

        # Z slice spinbox.
        panel_layout.addWidget(QLabel("Z slice:"))
        self._z_slice_sb = QSpinBox()
        self._z_slice_sb.setRange(0, self._z_max)
        self._z_slice_sb.setValue(self._scene.dims.slice_indices[0])
        self._z_slice_sb.valueChanged.connect(self._on_z_slice_changed)
        panel_layout.addWidget(self._z_slice_sb)

        panel_layout.addStretch()

        # Status label.
        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        panel_layout.addWidget(self._status_label)

        root.addWidget(panel)
        root.addWidget(canvas_widget, stretch=1)

    # ── UI callbacks ───────────────────────────────────────────────────

    def _on_toggle_colormap(self) -> None:
        self._colormap_index = (self._colormap_index + 1) % len(self._COLORMAPS)
        new_cmap = self._COLORMAPS[self._colormap_index]
        self._visual.appearance.color_map = new_cmap
        self._colormap_btn.setText(f"Colormap: {new_cmap}")
        print(f"[colormap] switched to '{new_cmap}'")

    def _on_level_radio_clicked(self, button: QRadioButton) -> None:
        self._force_level = button.property("force_level")

    def _on_z_slice_changed(self, value: int) -> None:
        self._scene.dims.slice_indices = (value,)
        # All cached tiles are from the old z-plane — clear and re-fetch.
        gfx_visual = self._get_gfx_visual()
        gfx_visual._block_cache_2d.tile_manager.clear()
        gfx_visual._lut_manager_2d.rebuild(gfx_visual._block_cache_2d.tile_manager)
        self._reslice_task = asyncio.ensure_future(self._on_update_clicked())

    def _get_gfx_visual(self):
        """Access the render-layer visual (for stats / cache access)."""
        return self._controller._render_manager._scenes[self._scene.id]._visuals[
            self._visual.id
        ]

    # ── Update button handler ──────────────────────────────────────────

    async def _on_update_clicked(self) -> None:
        """Main update flow: write UI values into model, then reslice."""
        self._update_count += 1
        t_total = time.perf_counter()

        # Write current UI values into the visual appearance model.
        self._visual.appearance.lod_bias = self._lod_bias_sb.value()
        self._visual.appearance.force_level = self._force_level
        self._visual.appearance.frustum_cull = self._viewport_cull_cb.isChecked()

        # Trigger the full cellier reslicing pipeline.
        self._controller.reslice_scene(self._scene.id)

        t_ms = (time.perf_counter() - t_total) * 1000

        # Read debug stats from the render layer.
        gfx_visual = self._get_gfx_visual()
        stats = gfx_visual._last_plan_stats

        total = stats.get("fills", 0)
        if total == 0:
            self._status_label.setText(
                f"Ready  (all {stats.get('total_required', 0)} tiles cached)"
            )
        else:
            self._status_label.setText(f"Loading: 0 / {total} tiles")

        print(
            f"[update #{self._update_count}]  planning={t_ms:.1f}ms  "
            f"fills={stats.get('fills', 0)}  hits={stats.get('hits', 0)}  "
            f"culled={stats.get('n_culled', 0)}  "
            f"dropped={stats.get('n_dropped', 0)}"
        )


async def async_main(data_store: MultiscaleZarrDataStore, z_slice: int | None) -> None:
    """Create the main window and run the Qt event loop."""
    app = QApplication.instance()
    window = MainWindow(data_store, z_slice=z_slice)
    window.resize(1200, 800)
    window.setWindowTitle("Async 2D multiscale image viewer — CellierController")
    window.show()

    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


def main() -> None:
    """Parse CLI args, build the data store, and launch the viewer."""
    parser = argparse.ArgumentParser(
        description="Async 2D multiscale image viewer (CellierController)"
    )
    parser.add_argument(
        "--zarr-path",
        type=pathlib.Path,
        default=ZARR_PATH,
        help="Path to the multiscale zarr store.",
    )
    parser.add_argument(
        "--z-slice",
        type=int,
        default=None,
        help="Z slice index (default: mid-slice).",
    )
    args = parser.parse_args()

    if not args.zarr_path.exists():
        print(f"Error: zarr store not found at '{args.zarr_path}'")
        print("Run example.py with --make-files first:")
        print("    uv run example.py --make-files")
        sys.exit(1)

    print("Opening tensorstore stores via MultiscaleZarrDataStore ...")
    data_store = MultiscaleZarrDataStore(
        zarr_path=str(args.zarr_path),
        scale_names=ZARR_SCALE_NAMES,
    )
    print(f"  {data_store.n_levels} levels opened.")
    for i, shape in enumerate(data_store.level_shapes):
        print(f"  s{i}: shape={shape}")
    print()
    print(
        "Press 'Update' to run the pipeline.  "
        "Pan/zoom with the mouse between presses.\n"
    )

    _app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(data_store, args.z_slice), handle_sigint=True)


if __name__ == "__main__":
    main()
