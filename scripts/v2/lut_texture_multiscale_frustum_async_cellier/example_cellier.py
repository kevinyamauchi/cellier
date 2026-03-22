"""Async brick loading via CellierController.

Same visual behaviour as ``example_v2.py`` but construction and update wiring
use ``CellierController`` instead of direct RenderManager access.

Usage
-----
Generate the multiscale zarr store (once):

    uv run example.py --make-files

Then launch the controller-based viewer:

    uv run example_cellier.py [--zarr-path PATH]

Controls
--------
Mouse                  — orbit / zoom at any time
Update btn             — freeze camera, plan, submit async commit task
Frustum cull checkbox  — enable / disable brick culling
Show frustum checkbox  — toggle wireframe visibility (no pipeline re-run)
Auto / 1 / 2 / 3 radio — select force-level
LOD bias spinbox       — scale LOD thresholds
Far plane spinbox      — camera far clip distance
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import pathlib
import sys
import time

import numpy as np
import pygfx as gfx
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
    QVBoxLayout,
    QWidget,
)

from cellier.v2.controller import CellierController
from cellier.v2.data.image import MultiscaleZarrDataStore
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.visuals._image import ImageAppearance

# ---------------------------------------------------------------------------
# Constants (identical to example_v2.py)
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32
GPU_BUDGET = 1024 * 1024**2  # 1 GiB
LOD_BIAS = 1.0

ZARR_PATH = pathlib.Path(__file__).parent / "multiscale_blobs.zarr"
ZARR_SCALE_NAMES = ["s0", "s1", "s2"]

FRUSTUM_COLOR = "#00cc44"
AABB_COLOR = "#ff00ff"


# ---------------------------------------------------------------------------
# Scene helpers (same as example_v2.py)
# ---------------------------------------------------------------------------


def _make_box_wireframe(
    box_min: np.ndarray, box_max: np.ndarray, color: str
) -> gfx.Line:
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


def make_frustum_wireframe(corners: np.ndarray, color: str = "#00cc44") -> gfx.Line:
    """Build a wireframe gfx.Line from 8 frustum corner points (shape 2x4x3)."""
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


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """PySide6 main window using CellierController."""

    def __init__(self, data_store: MultiscaleZarrDataStore) -> None:
        super().__init__()
        self._frustum_line: gfx.Line | None = None
        self._update_count = 0
        self._force_level: int | None = None

        self._controller = CellierController(widget_parent=self)
        cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
        self._scene = self._controller.add_scene(
            dim="3d", coordinate_system=cs, name="main"
        )
        self._visual = self._controller.add_image(
            data=data_store,
            scene_id=self._scene.id,
            appearance=ImageAppearance(
                color_map="viridis",
                clim=(0.0, 1.0),
                lod_bias=LOD_BIAS,
                force_level=None,
                frustum_cull=True,
            ),
            name="volume",
            block_size=BLOCK_SIZE,
            gpu_budget_bytes=GPU_BUDGET,
            threshold=0.2,
        )
        canvas_widget = self._controller.add_canvas(self._scene.id)

        # Add AABB wireframe directly to the pygfx scene (acceptable in a script).
        gfx_visual = self._controller._render_manager._scenes[self._scene.id]._visuals[
            self._visual.id
        ]
        d, h, w = gfx_visual._volume_geometry.base_layout.volume_shape
        pad = 1.0
        aabb_min = np.array([-0.5 - pad, -0.5 - pad, -0.5 - pad])
        aabb_max = np.array([w - 0.5 + pad, h - 0.5 + pad, d - 0.5 + pad])
        gfx_scene = self._controller._render_manager.get_scene(self._scene.id)
        gfx_scene.add(_make_box_wireframe(aabb_min, aabb_max, AABB_COLOR))

        self._setup_ui(canvas_widget)

    # ── UI ─────────────────────────────────────────────────────────────

    def _setup_ui(self, canvas_widget: QWidget) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        panel = QWidget()
        panel.setFixedWidth(220)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)

        self._update_btn = QPushButton("Update")
        self._update_btn.clicked.connect(
            lambda: asyncio.ensure_future(self._on_update_clicked())
        )
        panel_layout.addWidget(self._update_btn)

        self._frustum_cull_cb = QCheckBox("Frustum cull")
        self._frustum_cull_cb.setChecked(True)
        panel_layout.addWidget(self._frustum_cull_cb)

        self._show_frustum_cb = QCheckBox("Show frustum")
        self._show_frustum_cb.setChecked(True)
        self._show_frustum_cb.toggled.connect(self._on_show_frustum_toggled)
        panel_layout.addWidget(self._show_frustum_cb)

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

        panel_layout.addWidget(QLabel("LOD bias:"))
        self._lod_bias_sb = QDoubleSpinBox()
        self._lod_bias_sb.setRange(0.1, 10.0)
        self._lod_bias_sb.setSingleStep(0.1)
        self._lod_bias_sb.setDecimals(2)
        self._lod_bias_sb.setValue(LOD_BIAS)
        panel_layout.addWidget(self._lod_bias_sb)

        panel_layout.addWidget(QLabel("Far plane:"))
        self._far_plane_sb = QDoubleSpinBox()
        self._far_plane_sb.setRange(100.0, 200_000.0)
        self._far_plane_sb.setSingleStep(500.0)
        self._far_plane_sb.setDecimals(0)
        self._far_plane_sb.setValue(8000.0)
        self._far_plane_sb.valueChanged.connect(self._on_far_plane_changed)
        panel_layout.addWidget(self._far_plane_sb)

        panel_layout.addStretch()

        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        panel_layout.addWidget(self._status_label)

        root.addWidget(panel)
        root.addWidget(canvas_widget, stretch=1)

    # ── UI callbacks ───────────────────────────────────────────────────

    def _on_show_frustum_toggled(self, checked: bool) -> None:
        if self._frustum_line is not None:
            self._frustum_line.visible = checked

    def _on_level_radio_clicked(self, button: QRadioButton) -> None:
        self._force_level = button.property("force_level")

    def _on_far_plane_changed(self, value: float) -> None:
        self._controller.set_camera_depth_range(
            scene_id=self._scene.id,
            depth_range=(1.0, value),
        )

    # ── Update button handler ───────────────────────────────────────────

    async def _on_update_clicked(self) -> None:
        self._update_count += 1
        t_total = time.perf_counter()

        # Write current UI values into the visual appearance model.
        # CellierController reads these at reslice time.
        self._visual.appearance.lod_bias = self._lod_bias_sb.value()
        self._visual.appearance.force_level = self._force_level
        self._visual.appearance.frustum_cull = self._frustum_cull_cb.isChecked()

        self._controller.reslice_scene(self._scene.id)

        # Debug stats from the render layer (planning is synchronous).
        gfx_visual = self._controller._render_manager._scenes[self._scene.id]._visuals[
            self._visual.id
        ]
        stats = gfx_visual._last_plan_stats
        t_planning_ms = (time.perf_counter() - t_total) * 1000

        # Frustum wireframe snapshot.
        canvas_view = next(iter(self._controller._render_manager._canvases.values()))
        cam = canvas_view._camera
        corners = np.asarray(cam.frustum, dtype=np.float64).copy()
        self._rebuild_frustum_wireframe(corners)

        from cellier.v2.render._frustum import frustum_planes_from_corners

        planes = frustum_planes_from_corners(corners)
        cam_pos = np.array(cam.world.position, dtype=np.float64)
        view_dir = self._get_view_direction(cam)
        n_levels = gfx_visual._volume_geometry.n_levels
        fov_y_rad = float(np.radians(cam.fov))
        _, screen_h = canvas_view._canvas.get_logical_size()
        focal_half_height = (screen_h / 2) / np.tan(fov_y_rad / 2)
        thresholds = [
            (2 ** (k - 1)) * focal_half_height * self._visual.appearance.lod_bias
            for k in range(1, n_levels)
        ]

        self._print_update_summary(
            cam_pos,
            view_dir,
            corners,
            planes,
            thresholds,
            stats,
            t_planning_ms,
        )

        total = stats.get("fills", 0)
        if total == 0:
            self._status_label.setText(
                f"Ready  (all {stats.get('total_required', 0)} bricks cached)"
            )
        else:
            self._status_label.setText(f"Loading: 0 / {total} bricks")

    def _rebuild_frustum_wireframe(self, corners: np.ndarray) -> None:
        gfx_scene = self._controller._render_manager.get_scene(self._scene.id)
        if self._frustum_line is not None:
            gfx_scene.remove(self._frustum_line)
        self._frustum_line = make_frustum_wireframe(corners, color=FRUSTUM_COLOR)
        gfx_scene.add(self._frustum_line)
        self._frustum_line.visible = self._show_frustum_cb.isChecked()

    @staticmethod
    def _get_view_direction(camera: gfx.PerspectiveCamera) -> np.ndarray:
        mat = np.asarray(camera.world.matrix, dtype=np.float64)
        forward = -mat[:3, 2]
        norm = np.linalg.norm(forward)
        return forward / norm if norm > 1e-12 else np.array([0.0, 0.0, -1.0])

    # ── Debug print ────────────────────────────────────────────────────

    def _print_update_summary(
        self,
        cam_pos: np.ndarray,
        view_dir: np.ndarray,
        corners: np.ndarray,
        planes: np.ndarray,
        thresholds: list[float],
        stats: dict,
        t_planning_ms: float,
    ) -> None:
        W = 70
        sep_heavy = "═" * W
        sep_light = "─" * W
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(sep_heavy)
        print(f"[update #{self._update_count}]  {now}")
        print(sep_light)

        print("camera")
        px, py, pz = cam_pos
        print(f"  position:   [{px:>8.2f}, {py:>8.2f}, {pz:>8.2f}]")
        dx, dy, dz = view_dir
        print(f"  view_dir:   [{dx:>8.3f}, {dy:>8.3f}, {dz:>8.3f}]")

        plane_names = ["near  ", "far   ", "left  ", "right ", "top   ", "bottom"]
        print()
        print("frustum planes (6 x 4)  [a, b, c, d]")
        for name, p in zip(plane_names, planes):
            print(f"  {name}: [{p[0]:>8.4f}, {p[1]:>8.4f}, {p[2]:>8.4f}, {p[3]:>8.4f}]")

        lc = stats.get("level_counts", {})
        force = self._force_level
        print()
        print(
            f"lod assignment  (force_level={force}  "
            f"lod_bias={self._visual.appearance.lod_bias:.2f})"
        )
        if force is None and thresholds:
            for k, t in enumerate(thresholds):
                print(f"  threshold L{k + 1}→L{k + 2}:  {t:.1f} world units")
        for level in sorted(k for k in lc if k > 0):
            print(f"  level {level}:  {lc[level]} bricks")
        print(f"  lod_select:     {stats.get('lod_select_ms', 0):.2f} ms")
        print(f"  distance_sort:  {stats.get('distance_sort_ms', 0):.2f} ms")

        culling_on = self._frustum_cull_cb.isChecked()
        print()
        print(f"frustum cull  [{'ENABLED' if culling_on else 'DISABLED'}]")
        if culling_on:
            ct = stats.get("cull_timings", {})
            print(f"  total bricks:    {stats.get('total_required', 0)}")
            n_visible = stats.get("total_required", 0) - stats.get("n_culled", 0)
            print(f"  visible bricks:  {n_visible}")
            print(f"  culled bricks:   {stats.get('n_culled', 0)}")
            print(f"  build_corners:   {ct.get('build_corners_ms', 0):.2f} ms")
            print(f"  einsum:          {ct.get('einsum_ms', 0):.2f} ms")
            print(f"  mask:            {ct.get('mask_ms', 0):.2f} ms")
            print(f"  total_cull:      {stats.get('frustum_cull_ms', 0):.2f} ms")
        else:
            print(f"  total bricks:    {stats.get('total_required', 0)}  (no culling)")

        n_needed = stats.get("n_needed", 0)
        n_budget = stats.get("n_budget", 0)
        n_dropped = stats.get("n_dropped", 0)
        print()
        print("cache budget")
        drop_flag = (
            f"  *** {n_dropped} DROPPED (nearest-first) ***" if n_dropped else ""
        )
        print(f"  chunks needed / cache size:  {n_needed} / {n_budget}{drop_flag}")

        gfx_visual = self._controller._render_manager._scenes[self._scene.id]._visuals[
            self._visual.id
        ]
        tilemap_size = len(gfx_visual._block_cache.tile_manager.tilemap)
        print()
        print("cache  (planning phase)")
        print(f"  hits:           {stats.get('hits', 0)}")
        print(f"  fills queued:   {stats.get('fills', 0)}")
        print(f"  tilemap:        {tilemap_size}")
        print(f"  stage:          {stats.get('stage_ms', 0):.2f} ms")
        print(f"  plan_total:     {stats.get('plan_total_ms', 0):.2f} ms")
        print("  (commit is async — see status label for progress)")

        print()
        print("timings (outer)")
        print(f"  planning_total:    {t_planning_ms:.2f} ms")
        print(sep_heavy)


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------


async def async_main(data_store: MultiscaleZarrDataStore) -> None:
    """Create the main window and run the Qt event loop."""
    app = QApplication.instance()
    window = MainWindow(data_store)
    window.resize(1200, 800)
    window.setWindowTitle("Async multiscale volume renderer — CellierController")
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
        description="Async multiscale volume renderer (CellierController)"
    )
    parser.add_argument(
        "--zarr-path",
        type=pathlib.Path,
        default=ZARR_PATH,
        help="Path to the multiscale zarr store.",
    )
    args = parser.parse_args()

    if not args.zarr_path.exists():
        print(f"Error: zarr store not found at '{args.zarr_path}'")
        print("Run example.py with --make-files first:")
        print("    uv run example.py --make-files")
        sys.exit(1)

    print("Opening tensorstore stores via MultiscaleZarrDataStore …")
    data_store = MultiscaleZarrDataStore(
        zarr_path=str(args.zarr_path),
        scale_names=ZARR_SCALE_NAMES,
    )
    print(f"  {data_store.n_levels} levels opened.")
    for i, shape in enumerate(data_store.level_shapes):
        print(f"  s{i}: shape={shape}")
    print()
    print(
        "Press 'Update' to run the pipeline.  Orbit with the mouse between presses.\n"
    )

    _app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(data_store), handle_sigint=True)


if __name__ == "__main__":
    main()
