"""Async 2D multiscale image viewer via CellierController.

Same visual behaviour as ``example_2d.py`` (in chunked_rendering_2d_3d/)
but construction and update wiring use ``CellierController`` with the
new 2D rendering path instead of direct BlockState2D access.

Usage
-----
Generate the multiscale zarr store (once, if not already present):

    uv run example.py --make-files

Then launch the 2D controller-based viewer:

    uv run example_cellier_2d.py [--zarr-path PATH] [--z-slice 42]

Controls
--------
Mouse               — pan / zoom at any time
Update btn          — freeze camera, plan, submit async data load
Colormap btn        — toggle between viridis and gray (live, no Update needed)
Viewport cull       — enable / disable tile culling outside viewport
Auto / 1 / 2 / 3   — select force-level
LOD bias spinbox    — multiplicative bias on zoom-based LOD selection
Z slice spinbox     — select which Z slice to view
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import pathlib
import sys
import time
from typing import ClassVar

import numpy as np
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
from cellier.v2.visuals._image import ImageAppearance

BLOCK_SIZE = 32
GPU_BUDGET = 64 * 1024**2  # 64 MB for 2D
LOD_BIAS = 1.0

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
        self._controller = CellierController(widget_parent=self)
        cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
        self._scene = self._controller.add_scene(
            dim="2d", coordinate_system=cs, name="main"
        )

        # Determine the initial z-slice.
        z_depth = data_store.level_shapes[0][0]
        if z_slice is None:
            z_slice = z_depth // 2
        self._z_slice = z_slice

        # Set initial slice index on the scene's dims.
        self._scene.dims.selection.slice_indices = {0: self._z_slice}

        self._visual = self._controller.add_image(
            data=data_store,
            scene_id=self._scene.id,
            appearance=ImageAppearance(
                color_map="viridis",
                clim=(0.0, 1.0),
                lod_bias=LOD_BIAS,
                force_level=None,
                frustum_cull=False,
            ),
            name="image",
            block_size=BLOCK_SIZE,
            gpu_budget_bytes=GPU_BUDGET,
        )

        canvas_widget = self._controller.add_canvas(self._scene.id, available_dims="2d")

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
        self._z_slice_sb.setValue(self._z_slice)
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

    def _on_level_radio_clicked(self, button: QRadioButton) -> None:
        self._force_level = button.property("force_level")

    def _on_toggle_colormap(self) -> None:
        self._colormap_index = (self._colormap_index + 1) % len(self._COLORMAPS)
        new_cmap = self._COLORMAPS[self._colormap_index]
        self._visual.appearance.color_map = new_cmap
        self._colormap_btn.setText(f"Colormap: {new_cmap}")
        print(f"[colormap] switched to '{new_cmap}'")

    def _on_z_slice_changed(self, value: int) -> None:
        self._z_slice = value
        # Update the scene's dims — this is what SceneManager reads
        # when building slice requests.
        self._scene.dims.selection.slice_indices = {0: value}
        # Clear the 2D tile cache so all tiles are re-fetched at the
        # new Z slice.
        gfx_visual = self._get_gfx_visual()
        if gfx_visual._image_state is not None:
            gfx_visual._image_state.tile_manager.clear()

    def _get_gfx_visual(self):
        """Access the render-layer visual (for stats / cache access)."""
        return self._controller._render_manager._scenes[self._scene.id]._visuals[
            self._visual.id
        ]

    # ── Update button handler ──────────────────────────────────────────

    async def _on_update_clicked(self) -> None:
        self._update_count += 1
        t_total = time.perf_counter()

        # Write current UI values into the appearance model.
        self._visual.appearance.lod_bias = self._lod_bias_sb.value()
        self._visual.appearance.force_level = self._force_level
        # frustum_cull is not used for 2D; viewport culling is handled
        # by the use_culling parameter in build_slice_request_2d.
        # We pass it here for completeness.
        self._visual.appearance.frustum_cull = self._viewport_cull_cb.isChecked()

        # Trigger the reslice pipeline: planning (sync) + data load (async).
        self._controller.reslice_scene(self._scene.id)

        # Collect stats from the render layer.
        gfx_visual = self._get_gfx_visual()
        stats = gfx_visual._last_plan_stats
        t_planning_ms = (time.perf_counter() - t_total) * 1000

        # Get camera info for debug display.
        canvas_view = next(iter(self._controller._render_manager._canvases.values()))
        cam = canvas_view._camera
        cam_pos = np.array(cam.local.position, dtype=np.float64)
        logical_w, logical_h = canvas_view._canvas.get_logical_size()

        self._print_update_summary(cam_pos, logical_w, logical_h, stats, t_planning_ms)

        total_fills = stats.get("fills", 0)
        if total_fills == 0:
            self._status_label.setText(
                f"Ready  (all {stats.get('total_required', 0)} tiles cached)"
            )
        else:
            self._status_label.setText(f"Loading: 0 / {total_fills} tiles")

    # ── Debug print ────────────────────────────────────────────────────

    def _print_update_summary(
        self,
        cam_pos: np.ndarray,
        viewport_w: float,
        viewport_h: float,
        stats: dict,
        t_planning_ms: float,
    ) -> None:
        W = 70
        sep_heavy = "=" * W
        sep_light = "-" * W
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(sep_heavy)
        print(f"[update #{self._update_count}]  {now}")
        print(sep_light)

        # Camera.
        print("camera")
        print(f"  position:       [{cam_pos[0]:.1f}, {cam_pos[1]:.1f}]")
        print(f"  z_slice:        {self._z_slice}")
        print(f"  viewport:       {viewport_w:.0f} x {viewport_h:.0f}")

        # LOD assignment.
        lc = stats.get("level_counts", {})
        force = self._force_level
        lod_bias = self._lod_bias_sb.value()
        print()
        print(f"lod assignment  (force_level={force}  lod_bias={lod_bias:.2f})")
        for level in sorted(k for k in lc if k > 0):
            print(f"  level {level}:  {lc[level]} tiles")
        print(f"  lod_select:     {stats.get('lod_select_ms', 0):.2f} ms")
        print(f"  distance_sort:  {stats.get('distance_sort_ms', 0):.2f} ms")

        # Viewport cull.
        culling_on = self._viewport_cull_cb.isChecked()
        print()
        print(f"viewport cull  [{'ENABLED' if culling_on else 'DISABLED'}]")
        if culling_on:
            print(f"  total tiles:     {stats.get('total_required', 0)}")
            visible = stats.get("total_required", 0) - stats.get("n_culled", 0)
            print(f"  visible tiles:   {visible}")
            print(f"  n_culled:        {stats.get('n_culled', 0)}")
            print(f"  cull_time:       {stats.get('cull_ms', 0):.2f} ms")
        else:
            print(f"  total tiles:     {stats.get('total_required', 0)}  (no culling)")

        # Cache budget.
        n_needed = stats.get("n_needed", 0)
        n_budget = stats.get("n_budget", 0)
        n_dropped = stats.get("n_dropped", 0)
        print()
        print("cache budget")
        drop_flag = (
            f"  *** {n_dropped} DROPPED (nearest-first) ***" if n_dropped else ""
        )
        print(f"  needed / budget:  {n_needed} / {n_budget}{drop_flag}")

        # Cache stats.
        gfx_visual = self._get_gfx_visual()
        tilemap_size = (
            len(gfx_visual._image_state.tile_manager.tilemap)
            if gfx_visual._image_state is not None
            else 0
        )
        print()
        print("cache  (planning phase)")
        print(f"  hits:           {stats.get('hits', 0)}")
        print(f"  fills queued:   {stats.get('fills', 0)}")
        print(f"  tilemap:        {tilemap_size}")
        print(f"  stage:          {stats.get('stage_ms', 0):.2f} ms")
        print(f"  plan_total:     {stats.get('plan_total_ms', 0):.2f} ms")
        print("  (commit is async -- see status label for progress)")

        print()
        print("timings (outer)")
        print(f"  planning_total:    {t_planning_ms:.2f} ms")
        print(sep_heavy)


async def async_main(data_store: MultiscaleZarrDataStore, z_slice: int | None) -> None:
    """Create the main window and run the Qt event loop."""
    app = QApplication.instance()
    window = MainWindow(data_store, z_slice=z_slice)
    window.resize(1200, 800)
    window.setWindowTitle("Async 2D multiscale image viewer -- CellierController")
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
