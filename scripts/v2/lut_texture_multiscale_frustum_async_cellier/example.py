"""Async brick loading with GFXMultiscaleImageVisual + AsyncSlicer + MultiscaleZarrDataStore.

I/O responsibilities are clearly separated:

* ``MultiscaleZarrDataStore`` owns the tensorstore handles and the
  padded-read / boundary-clamping logic.
* ``AsyncSlicer`` drives the async batch loop and fires per-batch callbacks.
* ``GFXMultiscaleImageVisual.build_slice_request()`` returns ``chunk_requests`` and
  populates an internal ``slot_map``; the app submits to the slicer and
  passes ``visual.on_data_ready`` as the callback.
* ``GFXMultiscaleImageVisual.on_data_ready()`` commits each arriving batch directly
  into the cache texture and rebuilds the LUT.

Camera movement is free at all times; pressing **Update** freezes the
current camera state, runs the synchronous planning phase immediately,
then submits an async commit task.  Each arriving batch is visible in
the next rendered frame.

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
from uuid import UUID

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

from cellier.v2.render._frustum import frustum_planes_from_corners


# ---------------------------------------------------------------------------
# App-layer camera / scene helpers (pygfx-specific, not part of the package)
# ---------------------------------------------------------------------------

def get_frustum_corners_world(camera: gfx.PerspectiveCamera) -> np.ndarray:
    """Return world-space frustum corners, shape (2, 4, 3)."""
    return np.asarray(camera.frustum, dtype=np.float64)


def get_camera_position_world(camera: gfx.PerspectiveCamera) -> np.ndarray:
    """Return camera world position as (3,)."""
    return np.array(camera.world.position, dtype=np.float64)


def get_view_direction_world(camera: gfx.PerspectiveCamera) -> np.ndarray:
    """Return normalised view direction in world space as (3,)."""
    mat = np.asarray(camera.world.matrix, dtype=np.float64)
    forward = -mat[:3, 2]
    norm = np.linalg.norm(forward)
    return forward / norm if norm > 1e-12 else np.array([0.0, 0.0, -1.0])


def make_frustum_wireframe(corners: np.ndarray, color: str = "#00cc44") -> gfx.Line:
    """Build a frustum wireframe as a gfx.Line with segment material."""
    edge_indices = [
        ((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (0, 3)), ((0, 3), (0, 0)),
        ((1, 0), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (1, 3)), ((1, 3), (1, 0)),
        ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((0, 3), (1, 3)),
    ]
    positions = np.array(
        [[corners[a], corners[b]] for (a, b) in edge_indices],
        dtype=np.float32,
    ).reshape(-1, 3)
    return gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineSegmentMaterial(color=color, thickness=1.5),
    )

# Cellier package imports
from cellier.v2.data.image import MultiscaleZarrDataStore
from cellier.v2.slicer import AsyncSlicer
from cellier.v2.visuals import MultiscaleImageVisual, ImageAppearance
from cellier.v2.render.visuals import GFXMultiscaleImageVisual

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32
GPU_BUDGET = 1024 * 1024**2   # 1 GiB
COMMIT_BATCH_SIZE = 8
LOD_BIAS = 1.0

ZARR_PATH = pathlib.Path(__file__).parent / "multiscale_blobs.zarr"
ZARR_SCALE_NAMES = ["s0", "s1", "s2"]
# Downscale factors matching the scale names above.
ZARR_DOWNSCALE_FACTORS = [1, 2, 4]

FRUSTUM_COLOR = "#00cc44"
AABB_COLOR = "#ff00ff"


# ---------------------------------------------------------------------------
# Scene helpers
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


# ---------------------------------------------------------------------------
# Zarr generation helper
# ---------------------------------------------------------------------------

def make_multiscale_zarr(zarr_path: pathlib.Path) -> None:
    """Generate a 3-level multiscale zarr store using tensorstore (zarr v3)."""
    import tensorstore as ts
    from skimage.data import binary_blobs
    from scipy.ndimage import zoom

    print(f"Generating multiscale zarr (v3) at '{zarr_path}' …")
    print("  generating 1024³ base volume — this may take a minute …")

    half = 512
    quadrant = binary_blobs(
        length=half, blob_size_fraction=0.05, n_dim=3, volume_fraction=0.2, rng=42
    ).astype(np.float32)
    base = np.tile(quadrant, (2, 2, 2))   # (1024, 1024, 1024)

    chunk = (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)

    for name, factor in [("s0", 1), ("s1", 2), ("s2", 4)]:
        data = zoom(base, 1.0 / factor, order=1).astype(np.float32) if factor > 1 else base
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
    """PySide6 main window for the async volume viewer."""

    def __init__(
        self,
        visual: GFXMultiscaleImageVisual,
        data_store: MultiscaleZarrDataStore,
    ) -> None:
        super().__init__()
        self._visual = visual
        self._data_store = data_store
        self._frustum_line: gfx.Line | None = None
        self._update_count = 0
        self._force_level: int | None = None

        self._slicer = AsyncSlicer(batch_size=COMMIT_BATCH_SIZE)

        self._current_slice_id: UUID | None = None
        self._load_task_start_time: float = 0.0

        self._setup_scene()
        self._setup_ui()

    # ── Scene ──────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        self._canvas = QRenderWidget(parent=self, update_mode="continuous")
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        self._scene.add(self._visual.node_3d)

        d, h, w = self._visual._volume_geometry.base_layout.volume_shape
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
        root.addWidget(self._canvas, stretch=1)

    # ── Draw frame ─────────────────────────────────────────────────────

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

    # ── Update button handler ───────────────────────────────────────────

    async def _on_update_clicked(self) -> None:
        self._update_count += 1
        t_total = time.perf_counter()

        # ── Cancel any in-flight commit task ──────────────────────────
        prev_was_cancelled = False
        task_age_ms = 0.0
        if self._current_slice_id is not None:
            task_age_ms = (time.perf_counter() - self._load_task_start_time) * 1000
            prev_was_cancelled = self._slicer.cancel(self._current_slice_id)
            self._current_slice_id = None
            if prev_was_cancelled:
                self._visual.cancel_pending()
                self._status_label.setText("Cancelled — new update starting …")

        # ── Snapshot camera and compute thresholds synchronously ──────
        cam_pos  = get_camera_position_world(self._camera)
        view_dir = get_view_direction_world(self._camera)
        corners  = get_frustum_corners_world(self._camera)

        fov_y_rad = np.radians(self._camera.fov)
        _, screen_h = self._canvas.get_logical_size()
        lod_bias = self._lod_bias_sb.value()
        focal_half_height = (screen_h / 2) / np.tan(fov_y_rad / 2)
        n_levels = self._visual._volume_geometry.n_levels
        thresholds = [
            (2 ** (k - 1)) * focal_half_height / lod_bias
            for k in range(1, n_levels)
        ]

        planes = frustum_planes_from_corners(corners)
        frustum_planes = planes if self._frustum_cull_cb.isChecked() else None

        # ── Build slice request (synchronous planning) ─────────────────
        chunk_requests = self._visual.build_slice_request(
            camera_pos=cam_pos,
            frustum_planes=frustum_planes,
            thresholds=thresholds,
            force_level=self._force_level,
        )
        stats = self._visual._last_plan_stats

        # ── Rebuild frustum wireframe ─────────────────────────────────
        self._rebuild_frustum_wireframe(corners)

        # ── Print planning summary ────────────────────────────────────
        t_planning_ms = (time.perf_counter() - t_total) * 1000
        self._print_update_summary(
            cam_pos, view_dir, corners, planes, thresholds, stats,
            t_planning_ms, prev_was_cancelled, task_age_ms,
        )

        if not chunk_requests:
            self._status_label.setText(
                f"Ready  (all {stats['total_required']} bricks cached)"
            )
            return

        # ── Submit to the AsyncSlicer ─────────────────────────────────
        total = len(chunk_requests)
        arrived_counter = [0]
        self._status_label.setText(f"Loading: 0 / {total} bricks")

        def _on_batch_done(batch):
            self._visual.on_data_ready(batch)
            arrived_counter[0] += len(batch)
            n = arrived_counter[0]
            if n >= total:
                self._status_label.setText(f"Ready  ({n} bricks loaded)")
            else:
                self._status_label.setText(f"Loading: {n} / {total} bricks")

        self._load_task_start_time = time.perf_counter()
        self._current_slice_id = self._slicer.submit(
            chunk_requests,
            fetch_fn=self._data_store.get_data,
            callback=_on_batch_done,
            consumer_id="BlockVolumeApp",
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

        print("async task (previous)")
        if prev_was_cancelled:
            print(f"  status:          cancelled")
            print(f"  task_age:        {task_age_ms:.1f} ms")
        else:
            print(f"  status:          none / complete")

        print()
        print("camera")
        print(f"  position:   [{cam_pos[0]:>8.2f}, {cam_pos[1]:>8.2f}, {cam_pos[2]:>8.2f}]")
        print(f"  view_dir:   [{view_dir[0]:>8.3f}, {view_dir[1]:>8.3f}, {view_dir[2]:>8.3f}]")

        plane_names = ["near  ", "far   ", "left  ", "right ", "top   ", "bottom"]
        print()
        print("frustum planes (6 × 4)  [a, b, c, d]")
        for name, p in zip(plane_names, planes):
            print(f"  {name}: [{p[0]:>8.4f}, {p[1]:>8.4f}, {p[2]:>8.4f}, {p[3]:>8.4f}]")

        lc = stats.get("level_counts", {})
        force = self._force_level
        print()
        print(
            f"lod assignment  (force_level={force}  "
            f"lod_bias={self._lod_bias_sb.value():.2f})"
        )
        if force is None and thresholds:
            for k, t in enumerate(thresholds):
                print(f"  threshold L{k+1}→L{k+2}:  {t:.1f} world units")
        for level in sorted(k for k in lc if k > 0):
            print(f"  level {level}:  {lc[level]} bricks")
        print(f"  lod_select:     {stats['lod_select_ms']:.2f} ms")
        print(f"  distance_sort:  {stats['distance_sort_ms']:.2f} ms")

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

        n_needed  = stats.get("n_needed", 0)
        n_budget  = stats.get("n_budget", 0)
        n_dropped = stats.get("n_dropped", 0)
        print()
        print("cache budget")
        drop_flag = f"  *** {n_dropped} DROPPED (nearest-first) ***" if n_dropped else ""
        print(f"  chunks needed / cache size:  {n_needed} / {n_budget}{drop_flag}")

        tilemap_size = len(self._visual._block_cache.tile_manager.tilemap)
        print()
        print("cache  (planning phase)")
        print(f"  hits:           {stats['hits']}")
        print(f"  fills queued:   {stats['fills']}")
        print(f"  tilemap:        {tilemap_size}")
        print(f"  stage:          {stats['stage_ms']:.2f} ms")
        print(f"  plan_total:     {stats['plan_total_ms']:.2f} ms")
        print(f"  (commit is async — see status label for progress)")

        print()
        print("timings (outer)")
        print(f"  planning_total:    {t_planning_ms:.2f} ms")
        print(sep_heavy)


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------

async def async_main(
    visual: GFXMultiscaleImageVisual,
    data_store: MultiscaleZarrDataStore,
) -> None:
    app = QApplication.instance()
    window = BlockVolumeApp(visual=visual, data_store=data_store)
    window.resize(1200, 800)
    window.setWindowTitle("Async multiscale volume renderer — GFXMultiscaleImageVisual")
    window.show()

    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Async multiscale volume renderer")
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
    args = parser.parse_args()

    if args.make_files:
        make_multiscale_zarr(args.zarr_path)
        sys.exit(0)

    if not args.zarr_path.exists():
        print(f"Error: zarr store not found at '{args.zarr_path}'")
        print("Run with --make-files first:")
        print("    uv run example.py --make-files")
        sys.exit(1)

    # ── Open data store ────────────────────────────────────────────────
    print("Opening tensorstore stores via MultiscaleZarrDataStore …")
    data_store = MultiscaleZarrDataStore(
        zarr_path=str(args.zarr_path),
        scale_names=ZARR_SCALE_NAMES,
    )
    print(f"  {data_store.n_levels} levels opened.")
    for i, shape in enumerate(data_store.level_shapes):
        print(f"  s{i}: shape={shape}")
    print()

    # ── Build visual model ─────────────────────────────────────────────
    model = MultiscaleImageVisual(
        name="volume",
        data_store_id="zarr-store-0",
        downscale_factors=ZARR_DOWNSCALE_FACTORS,
        appearance=ImageAppearance(
            color_map="viridis",
            clim=(0.0, 1.0),
        ),
    )

    # ── Build GFXMultiscaleImageVisual ─────────────────────────────────
    level_shapes = list(data_store.level_shapes)
    visual = GFXMultiscaleImageVisual.from_cellier_model(
        model=model,
        level_shapes=level_shapes,
        render_modes={"3d"},
        block_size=BLOCK_SIZE,
        gpu_budget_bytes=GPU_BUDGET,
        threshold=0.2,
    )

    geo = visual._volume_geometry
    cache_info = visual._block_cache.info
    print(
        f"Cache: {cache_info.grid_side}³ grid = "
        f"{cache_info.n_slots} slots "
        f"({cache_info.cache_shape[0]}³ voxels, "
        f"overlap={cache_info.overlap}, "
        f"padded_brick={cache_info.padded_block_size}³)"
    )
    print(f"LUT:   {geo.base_layout.grid_dims} grid")
    print(f"Levels: {geo.n_levels}  downscale_factors={geo.downscale_factors}\n")
    print("Press 'Update' to run the pipeline.  Orbit with the mouse between presses.\n")

    app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(visual, data_store), handle_sigint=True)


if __name__ == "__main__":
    main()
