"""Phase 3 — Async brick loading with PySide6.QtAsyncio + tensorstore.

Camera movement is free at all times; pressing **Update** freezes the
current camera state, runs the synchronous planning phase immediately,
then submits an async commit task that loads bricks one-by-one.  Each
arriving brick is visible in the next rendered frame.

Pressing **Update** while a load is in progress cancels the previous
task and starts a fresh one from the current camera position.

Usage
-----
Generate the multiscale zarr store (once):

    uv run example.py --make-files

Then launch the viewer:

    uv run example.py [--zarr-path PATH]

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
from rendercanvas.qt import QRenderWidget
import PySide6.QtAsyncio as QtAsyncio

from image_block.volume import BlockVolumeState, make_block_volume, _open_ts_stores
from image_block.volume.frustum import (
    frustum_planes_from_corners,
    get_camera_position_world,
    get_frustum_corners_world,
    get_view_direction_world,
    make_frustum_wireframe,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32
GPU_BUDGET = 1024 * 1024**2  # 1 GiB — fits all L3 (512) + all L2 (4096) bricks for a 1024³ volume

# Bricks committed per Qt yield in the async load loop.
# Higher → fewer render interruptions → faster total load, less visual feedback.
# Lower  → more frequent screen updates, slower total load.
COMMIT_BATCH_SIZE = 8

LOD_BIAS = 1.0

ZARR_PATH = pathlib.Path(__file__).parent / "multiscale_blobs.zarr"
ZARR_SCALE_NAMES = ["s0", "s1", "s2"]

FRUSTUM_COLOR = "#00cc44"
AABB_COLOR = "#ffffff"


# ---------------------------------------------------------------------------
# Zarr helpers
# ---------------------------------------------------------------------------


def _make_box_wireframe(box_min: np.ndarray, box_max: np.ndarray, color: str) -> gfx.Line:
    """Build a wireframe AABB as a gfx.Line."""
    x0, y0, z0 = box_min
    x1, y1, z1 = box_max
    positions = np.array(
        [
            [x0, y0, z0], [x1, y0, z0],
            [x1, y0, z0], [x1, y1, z0],
            [x1, y1, z0], [x0, y1, z0],
            [x0, y1, z0], [x0, y0, z0],
            [x0, y0, z1], [x1, y0, z1],
            [x1, y0, z1], [x1, y1, z1],
            [x1, y1, z1], [x0, y1, z1],
            [x0, y1, z1], [x0, y0, z1],
            [x0, y0, z0], [x0, y0, z1],
            [x1, y0, z0], [x1, y0, z1],
            [x1, y1, z0], [x1, y1, z1],
            [x0, y1, z0], [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    geometry = gfx.Geometry(positions=positions)
    material = gfx.LineSegmentMaterial(color=color, thickness=1.0)
    return gfx.Line(geometry, material)


def make_multiscale_zarr(zarr_path: pathlib.Path) -> None:
    """Generate a 3-level multiscale zarr store using tensorstore (zarr v3).

    Uses tensorstore's write path so no zarr Python package is required.
    Produces zarr v3 stores readable by tensorstore's ``"zarr3"`` driver.
    """
    import tensorstore as ts
    from skimage.data import binary_blobs
    from scipy.ndimage import zoom

    print(f"Generating multiscale zarr (v3) at '{zarr_path}' …")
    print("  generating 1024³ base volume — this may take a minute …")

    # Level 0: full resolution 1024³ synthetic blob volume.
    # binary_blobs only supports cubic volumes up to the length parameter,
    # so we tile four 512³ blocks to reach 1024³.
    half = 512
    quadrant = binary_blobs(
        length=half, blob_size_fraction=0.05, n_dim=3, volume_fraction=0.2, rng=42
    ).astype(np.float32)
    base = np.tile(quadrant, (2, 2, 2))   # (1024, 1024, 1024)

    chunk = (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)

    for name, factor in [("s0", 1), ("s1", 2), ("s2", 4)]:
        if factor > 1:
            data = zoom(base, 1.0 / factor, order=1).astype(np.float32)
        else:
            data = base

        level_path = zarr_path / name
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(level_path)},
            "metadata": {
                "shape": list(data.shape),
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": list(chunk)},
                },
                "chunk_key_encoding": {"name": "default"},
                "data_type": "float32",
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                ],
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        store[...].write(data).result()
        print(f"  {name}: shape={data.shape}  chunks={chunk}")

    print("Done.\n")


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------


class BlockVolumeApp(QMainWindow):
    """PySide6 main window for the Phase 3 async volume viewer."""

    def __init__(self, state: BlockVolumeState, vol: gfx.Volume) -> None:
        super().__init__()
        self._state = state
        self._vol = vol
        self._frustum_line: gfx.Line | None = None
        self._update_count = 0
        self._force_level: int | None = None

        # Async task tracking.
        self._load_task: asyncio.Task | None = None
        self._load_task_start_time: float = 0.0
        self._prev_arrived: int = 0   # bricks committed by the previous task

        self._setup_scene()
        self._setup_ui()

    # ── Scene ──────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        self._canvas = QRenderWidget(parent=self, update_mode="continuous")
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        self._scene.add(self._vol)

        d, h, w = self._state.base_layout.volume_shape
        pad = 1.0
        aabb_min = np.array([-0.5 - pad, -0.5 - pad, -0.5 - pad])
        aabb_max = np.array([w - 0.5 + pad, h - 0.5 + pad, d - 0.5 + pad])
        self._scene.add(_make_box_wireframe(aabb_min, aabb_max, AABB_COLOR))

        self._camera = gfx.PerspectiveCamera(70, 16 / 9, depth_range=(1, 8000))
        self._camera.show_object(self._scene, view_dir=(-1, -1, -1), up=(0, 0, 1))
        self._controller = gfx.OrbitController(
            camera=self._camera, register_events=self._renderer
        )

        self._canvas.request_draw(self._draw_frame)

    # ── UI ─────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        panel = QWidget()
        panel.setFixedWidth(220)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)

        # Update button — wired to async handler via ensure_future.
        self._update_btn = QPushButton("Update")
        self._update_btn.clicked.connect(
            lambda: asyncio.ensure_future(self._on_update_clicked())
        )
        panel_layout.addWidget(self._update_btn)

        # Frustum cull checkbox
        self._frustum_cull_cb = QCheckBox("Frustum cull")
        self._frustum_cull_cb.setChecked(True)
        panel_layout.addWidget(self._frustum_cull_cb)

        # Show frustum checkbox
        self._show_frustum_cb = QCheckBox("Show frustum")
        self._show_frustum_cb.setChecked(True)
        self._show_frustum_cb.toggled.connect(self._on_show_frustum_toggled)
        panel_layout.addWidget(self._show_frustum_cb)

        # Force-level radio buttons
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

        # LOD bias spinbox
        panel_layout.addWidget(QLabel("LOD bias:"))
        self._lod_bias_sb = QDoubleSpinBox()
        self._lod_bias_sb.setRange(0.1, 10.0)
        self._lod_bias_sb.setSingleStep(0.1)
        self._lod_bias_sb.setDecimals(2)
        self._lod_bias_sb.setValue(LOD_BIAS)
        self._lod_bias_sb.setToolTip(
            "Scales LOD distance thresholds.\n"
            "> 1: coarser LOD sooner (fewer bricks)\n"
            "< 1: finer LOD longer (more bricks)"
        )
        panel_layout.addWidget(self._lod_bias_sb)

        # Far plane spinbox
        panel_layout.addWidget(QLabel("Far plane:"))
        self._far_plane_sb = QDoubleSpinBox()
        self._far_plane_sb.setRange(100.0, 200_000.0)
        self._far_plane_sb.setSingleStep(500.0)
        self._far_plane_sb.setDecimals(0)
        self._far_plane_sb.setValue(8000.0)
        self._far_plane_sb.setToolTip(
            "Camera far clip plane in world units.\n"
            "Reducing this tightens the depth range and\n"
            "can improve raycaster performance."
        )
        self._far_plane_sb.valueChanged.connect(self._on_far_plane_changed)
        panel_layout.addWidget(self._far_plane_sb)

        panel_layout.addStretch()

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        panel_layout.addWidget(self._status_label)

        root.addWidget(panel)
        root.addWidget(self._canvas, stretch=1)

    # ── Draw frame (vsync, no pipeline run) ────────────────────────────

    def _draw_frame(self) -> None:
        self._renderer.render(self._scene, self._camera)

    # ── UI callbacks ───────────────────────────────────────────────────

    def _on_show_frustum_toggled(self, checked: bool) -> None:
        if self._frustum_line is not None:
            self._frustum_line.visible = checked

    def _on_level_radio_clicked(self, button: QRadioButton) -> None:
        self._force_level = button.property("force_level")

    def _on_far_plane_changed(self, value: float) -> None:
        near, _ = self._camera.depth_range
        self._camera.depth_range = (near, value)

    # ── Update button — async handler ──────────────────────────────────

    async def _on_update_clicked(self) -> None:
        self._update_count += 1
        t_total = time.perf_counter()

        # ── Cancel any in-flight commit task ──────────────────────────
        prev_was_cancelled = False
        task_age_ms = 0.0
        if self._load_task is not None and not self._load_task.done():
            task_age_ms = (time.perf_counter() - self._load_task_start_time) * 1000
            self._load_task.cancel()
            prev_was_cancelled = True

        # ── Snapshot camera and compute thresholds synchronously ──────
        cam_pos   = get_camera_position_world(self._camera)
        view_dir  = get_view_direction_world(self._camera)
        corners   = get_frustum_corners_world(self._camera)

        fov_y_rad = np.radians(self._camera.fov)
        _, screen_h = self._canvas.get_logical_size()
        lod_bias = self._lod_bias_sb.value()  # captured at submission time
        focal_half_height = (screen_h / 2) / np.tan(fov_y_rad / 2)
        thresholds = [
            (2 ** (k - 1)) * focal_half_height / lod_bias
            for k in range(1, self._state.n_levels)
        ]

        planes = frustum_planes_from_corners(corners)
        frustum_planes = planes if self._frustum_cull_cb.isChecked() else None

        # ── Synchronous planning phase ────────────────────────────────
        fill_plan, stats = self._state.plan_update(
            camera_pos=cam_pos,
            thresholds=thresholds,
            force_level=self._force_level,
            frustum_planes=frustum_planes,
        )

        # ── Rebuild frustum wireframe ─────────────────────────────────
        self._rebuild_frustum_wireframe(corners)

        # ── Print planning summary (immediately, like Phase 2) ────────
        t_planning_ms = (time.perf_counter() - t_total) * 1000
        self._print_update_summary(
            cam_pos, view_dir, corners, planes, thresholds, stats,
            t_planning_ms, prev_was_cancelled, task_age_ms,
        )

        # ── Submit the commit loop as a cancellable asyncio task ──────
        self._load_task_start_time = time.perf_counter()
        self._load_task = asyncio.ensure_future(
            self._state.commit_bricks_async(
                fill_plan,
                status_callback=self._status_label.setText,
                batch_size=COMMIT_BATCH_SIZE,
            )
        )

    def _rebuild_frustum_wireframe(self, corners: np.ndarray) -> None:
        if self._frustum_line is not None:
            self._scene.remove(self._frustum_line)
        self._frustum_line = make_frustum_wireframe(corners, color=FRUSTUM_COLOR)
        self._scene.add(self._frustum_line)
        self._frustum_line.visible = self._show_frustum_cb.isChecked()

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
        prev_was_cancelled: bool,
        task_age_ms: float,
    ) -> None:
        W = 70
        sep_heavy = "═" * W
        sep_light = "─" * W
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(sep_heavy)
        print(f"[update #{self._update_count}]  {now}")
        print(sep_light)

        # ── Async task status (Phase 3 new fields) ────────────────────
        print("async task (previous)")
        if prev_was_cancelled:
            print(f"  status:          cancelled")
            print(f"  task_age:        {task_age_ms:.1f} ms")
        else:
            print(f"  status:          none / complete")

        # Camera
        print()
        print("camera")
        print(f"  position:   [{cam_pos[0]:>8.2f}, {cam_pos[1]:>8.2f}, {cam_pos[2]:>8.2f}]")
        print(f"  view_dir:   [{view_dir[0]:>8.3f}, {view_dir[1]:>8.3f}, {view_dir[2]:>8.3f}]")
        print("  near corners (lb, rb, rt, lt):")
        for c in corners[0]:
            print(f"    [{c[0]:>8.1f}, {c[1]:>8.1f}, {c[2]:>8.1f}]")
        print("  far corners:")
        for c in corners[1]:
            print(f"    [{c[0]:>8.1f}, {c[1]:>8.1f}, {c[2]:>8.1f}]")

        # Frustum planes
        plane_names = ["near  ", "far   ", "left  ", "right ", "top   ", "bottom"]
        print()
        print("frustum planes (6 × 4)  [a, b, c, d]")
        for name, p in zip(plane_names, planes):
            print(f"  {name}: [{p[0]:>8.4f}, {p[1]:>8.4f}, {p[2]:>8.4f}, {p[3]:>8.4f}]")

        # LOD assignment
        lc = stats.get("level_counts", {})
        force = self._force_level
        print()
        print(f"lod assignment  (force_level={force}  lod_bias={self._lod_bias_sb.value():.2f})")
        if force is None and thresholds:
            for k, t in enumerate(thresholds):
                print(f"  threshold L{k+1}→L{k+2}:  {t:.1f} world units")
        for level in sorted(k for k in lc if k > 0):
            print(f"  level {level}:  {lc[level]} bricks")
        print(f"  lod_select:     {stats['lod_select_ms']:.2f} ms")
        print(f"  distance_sort:  {stats['distance_sort_ms']:.2f} ms")

        # Frustum cull
        culling_on = self._frustum_cull_cb.isChecked()
        print()
        print(f"frustum cull  [{'ENABLED' if culling_on else 'DISABLED'}]")
        if culling_on:
            ct = stats.get("cull_timings", {})
            print(f"  total bricks:    {stats['total_required']}")
            print(f"  visible bricks:  {stats['total_required'] - stats['n_culled']}")
            print(f"  culled bricks:   {stats['n_culled']}")
            print(f"  build_corners:   {ct.get('build_corners_ms', 0):.2f} ms")
            print(f"  einsum:          {ct.get('einsum_ms', 0):.2f} ms")
            print(f"  mask:            {ct.get('mask_ms', 0):.2f} ms")
            print(f"  total_cull:      {stats['frustum_cull_ms']:.2f} ms")
        else:
            print(f"  total bricks:    {stats['total_required']}  (no culling)")

        # Cache budget
        n_needed  = stats.get("n_needed", 0)
        n_budget  = stats.get("n_budget", 0)
        n_dropped = stats.get("n_dropped", 0)
        print()
        print("cache budget")
        drop_flag = f"  *** {n_dropped} DROPPED (nearest-first) ***" if n_dropped else ""
        print(f"  chunks needed / cache size:  {n_needed} / {n_budget}{drop_flag}")

        # Cache / async
        print()
        print("cache  (planning phase)")
        print(f"  hits:           {stats['hits']}")
        print(f"  fills queued:   {stats['fills']}")
        print(f"  tilemap:        {len(self._state.tile_manager.tilemap)}")
        print(f"  stage:          {stats['stage_ms']:.2f} ms")
        print(f"  plan_total:     {stats['plan_total_ms']:.2f} ms")
        print(f"  (commit is async — see status label for progress)")

        # Outer timings
        print()
        print("timings (outer)")
        print(f"  planning_total:    {t_planning_ms:.2f} ms")
        print(sep_heavy)


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------


async def async_main(state: BlockVolumeState, vol: gfx.Volume) -> None:
    """Top-level coroutine driven by PySide6.QtAsyncio."""
    app = QApplication.instance()
    window = BlockVolumeApp(state=state, vol=vol)
    window.resize(1200, 800)
    window.setWindowTitle("LUT Volume Renderer — Phase 3 (async brick loading)")
    window.show()

    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 async multi-LOD volume viewer")
    parser.add_argument(
        "--make-files",
        action="store_true",
        help="Generate the multiscale zarr store and exit.",
    )
    parser.add_argument(
        "--zarr-path",
        type=pathlib.Path,
        default=ZARR_PATH,
        help="Path to the multiscale zarr store.",
    )
    # Argparse flags must not reach Qt — parse before QApplication is created.
    args = parser.parse_args()

    if args.make_files:
        make_multiscale_zarr(args.zarr_path)
        sys.exit(0)

    if not args.zarr_path.exists():
        print(f"Error: zarr store not found at '{args.zarr_path}'")
        print("Run with --make-files first:")
        print("    uv run example.py --make-files")
        sys.exit(1)

    # Open tensorstore stores synchronously before the event loop starts.
    print("Opening tensorstore stores …")
    ts_stores = _open_ts_stores(args.zarr_path, ZARR_SCALE_NAMES)
    print(f"  {len(ts_stores)} levels opened.")
    for i, store in enumerate(ts_stores):
        print(f"  s{i}: shape={tuple(int(d) for d in store.domain.shape)}")
    print()

    vol, state = make_block_volume(
        ts_stores=ts_stores,
        block_size=BLOCK_SIZE,
        gpu_budget_bytes=GPU_BUDGET,
        clim=(0.0, 1.0),
        threshold=0.2,
    )

    print(
        f"Cache: {state.cache_info.grid_side}³ grid = "
        f"{state.cache_info.n_slots} slots "
        f"({state.cache_info.cache_shape[0]}³ voxels, "
        f"overlap={state.cache_info.overlap}, "
        f"padded_brick={state.cache_info.padded_block_size}³)"
    )
    print(f"LUT:   {state.base_layout.grid_dims} grid")
    print(f"Levels: {state.n_levels}\n")
    print("Press 'Update' to run the pipeline.  Orbit with the mouse between presses.\n")

    # Pass only sys.argv[0] so Qt never sees argparse flags.
    app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(state, vol), handle_sigint=True)


if __name__ == "__main__":
    main()
