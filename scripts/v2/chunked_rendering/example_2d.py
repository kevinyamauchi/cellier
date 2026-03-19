"""Async multiscale 2D image viewer with LUT indirection.

Usage
-----
    uv run example_2d.py --make-files       # generate multiscale zarr, then exit
    uv run example_2d.py                    # launch viewer
    uv run example_2d.py --zarr-path /path  # custom zarr store
    uv run example_2d.py --z-slice 42       # override Z slice
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import math
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
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from rendercanvas.pyside6 import QRenderWidget

from image_block import BlockState2D, make_block_image, open_ts_stores

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZARR_PATH = pathlib.Path(__file__).parent / "multiscale_blobs.zarr"
SCALE_NAMES = ["s0", "s1", "s2"]
BLOCK_SIZE = 32
GPU_BUDGET = 64 * 1024**2  # 64 MB for 2D
COMMIT_BATCH_SIZE = 16
AABB_COLOR = (0.8, 0.8, 0.8, 1.0)
VIEWPORT_COLOR = (0.3, 0.9, 0.3, 1.0)


# ---------------------------------------------------------------------------
# Dataset generation (--make-files)
# ---------------------------------------------------------------------------


def make_zarr_sample(path: pathlib.Path) -> None:
    """Generate a multiscale 3D zarr v3 store with binary blobs.

    Uses tensorstore's write path so no zarr Python package is required.
    Produces zarr v3 stores readable by tensorstore's ``"zarr3"`` driver.

    Produces s0 (256^3), s1 (128^3), s2 (64^3) with chunks of 32^3.
    The same store can be reused by the 3D example; the 2D viewer
    slices at z_mid for each level.
    """
    import tensorstore as ts
    from skimage.data import binary_blobs
    from skimage.measure import block_reduce

    print(f"Generating multiscale zarr (v3) at '{path}' ...")

    base = binary_blobs(
        length=1024,
        n_dim=3,
        blob_size_fraction=0.08,
        volume_fraction=0.25,
        rng=42,
    ).astype(np.float32)

    chunk = (BLOCK_SIZE,) * 3

    for name, factor in [("s0", 1), ("s1", 2), ("s2", 4)]:
        if factor > 1:
            data = block_reduce(base, (factor, factor, factor), np.max).astype(
                np.float32
            )
        else:
            data = base

        level_path = path / name
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
                    {
                        "name": "bytes",
                        "configuration": {"endian": "little"},
                    },
                ],
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        store[...].write(data).result()
        print(
            f"  {name}: shape={data.shape}  "
            f"range=[{data.min():.2f}, {data.max():.2f}]  "
            f"chunks={chunk}"
        )

    print("Done.\n")


# ---------------------------------------------------------------------------
# Wireframe helpers
# ---------------------------------------------------------------------------


def make_aabb_wireframe(w: float, h: float, color: tuple) -> gfx.Line:
    """Create a wireframe rectangle for the full image extent.

    LineSegmentMaterial draws disconnected pairs: (0,1), (2,3), etc.
    So we provide 8 vertices for 4 edges.
    """
    positions = np.array(
        [
            [0, 0, 0],
            [w, 0, 0],  # bottom edge
            [w, 0, 0],
            [w, h, 0],  # right edge
            [w, h, 0],
            [0, h, 0],  # top edge
            [0, h, 0],
            [0, 0, 0],  # left edge
        ],
        dtype=np.float32,
    )
    colors = np.array([color] * 8, dtype=np.float32)
    geo = gfx.Geometry(positions=positions, colors=colors)
    mat = gfx.LineSegmentMaterial(color_mode="vertex", thickness=2)
    return gfx.Line(geo, mat)


def make_viewport_wireframe(
    view_min: np.ndarray, view_max: np.ndarray, color: tuple
) -> gfx.Line:
    """Create a wireframe rectangle for the viewport AABB.

    Same segment-pair layout as the AABB wireframe.
    """
    x0, y0 = view_min
    x1, y1 = view_max
    positions = np.array(
        [
            [x0, y0, 0.1],
            [x1, y0, 0.1],  # bottom
            [x1, y0, 0.1],
            [x1, y1, 0.1],  # right
            [x1, y1, 0.1],
            [x0, y1, 0.1],  # top
            [x0, y1, 0.1],
            [x0, y0, 0.1],  # left
        ],
        dtype=np.float32,
    )
    colors = np.array([color] * 8, dtype=np.float32)
    geo = gfx.Geometry(positions=positions, colors=colors)
    mat = gfx.LineSegmentMaterial(color_mode="vertex", thickness=2)
    return gfx.Line(geo, mat)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------


class ImageBlockApp(QWidget):
    """Main application window for the 2D tiled image viewer."""

    def __init__(
        self,
        state: BlockState2D,
        image: gfx.Image,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._image = image
        self._update_count = 0
        self._load_task: asyncio.Task | None = None
        self._load_task_start_time: float = 0.0

        # LOAD controls.
        self._force_level: int | None = None

        # ---- Build scene ----
        self._scene = gfx.Scene()
        self._scene.add(gfx.Background.from_color("#1a1a2e"))
        self._scene.add(image)

        # AABB wireframe.
        bs = state.block_size
        gh, gw = state.base_layout.grid_dims
        img_w = gw * bs
        img_h = gh * bs
        self._aabb_line = make_aabb_wireframe(img_w, img_h, AABB_COLOR)
        self._scene.add(self._aabb_line)

        # Viewport wireframe (rebuilt on each update).
        self._viewport_line: gfx.Line | None = None

        # ---- Camera ----
        self._camera = gfx.OrthographicCamera()
        self._camera.show_rect(0, img_w, 0, img_h)

        # ---- Build UI ----
        main_layout = QHBoxLayout(self)

        # Left panel.
        panel = QVBoxLayout()
        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(220)

        # Update button.
        self._update_btn = QPushButton("Update")
        self._update_btn.clicked.connect(self._on_update_clicked_sync)
        panel.addWidget(self._update_btn)

        # Viewport cull checkbox.
        self._viewport_cull_cb = QCheckBox("Viewport cull")
        self._viewport_cull_cb.setChecked(True)
        panel.addWidget(self._viewport_cull_cb)

        # Show viewport checkbox.
        self._show_viewport_cb = QCheckBox("Show viewport")
        self._show_viewport_cb.setChecked(True)
        self._show_viewport_cb.toggled.connect(self._on_show_viewport_toggled)
        panel.addWidget(self._show_viewport_cb)

        # Force level radio buttons.
        panel.addWidget(QLabel("Force level:"))
        self._level_group = QButtonGroup(self)
        for i, label in enumerate(["Auto", "1", "2", "3"]):
            rb = QRadioButton(label)
            if i == 0:
                rb.setChecked(True)
            self._level_group.addButton(rb, i)
            panel.addWidget(rb)
        self._level_group.idClicked.connect(self._on_level_changed)

        # LOAD bias spinbox (multiplicative: 1.0 = neutral, matching 3D viewer).
        panel.addWidget(QLabel("LOAD bias:"))
        self._load_bias_sb = QDoubleSpinBox()
        self._load_bias_sb.setRange(0.1, 10.0)
        self._load_bias_sb.setSingleStep(0.1)
        self._load_bias_sb.setValue(1.0)
        panel.addWidget(self._load_bias_sb)

        # Z slice spinbox.
        z_max = int(state.ts_stores[0].domain.shape[0]) - 1
        panel.addWidget(QLabel("Z slice:"))
        self._z_slice_sb = QSpinBox()
        self._z_slice_sb.setRange(0, z_max)
        self._z_slice_sb.setValue(state.z_slice)
        self._z_slice_sb.valueChanged.connect(self._on_z_slice_changed)
        panel.addWidget(self._z_slice_sb)

        # Stretch spacer.
        panel.addStretch()

        # Status label.
        self._status_label = QLabel("Ready")
        panel.addWidget(self._status_label)

        main_layout.addWidget(panel_widget)

        # Right: render canvas.
        self._canvas = QRenderWidget(update_mode="continuous")
        self._renderer = gfx.renderers.WgpuRenderer(self._canvas)
        self._controller = gfx.PanZoomController(
            self._camera, register_events=self._renderer
        )
        main_layout.addWidget(self._canvas, stretch=1)

        # Register the draw callback with rendercanvas.
        # update_mode="continuous" drives _draw_frame every vsync via
        # an internal Qt timer; texture updates appear on the next frame
        # without any explicit canvas.update() call.
        self._canvas.request_draw(self._draw_frame)

    def _draw_frame(self):
        """Render callback (called every vsync by rendercanvas)."""
        self._renderer.render(self._scene, self._camera)

    # ---- UI callbacks ----

    def _on_level_changed(self, idx: int) -> None:
        if idx == 0:
            self._force_level = None
        else:
            self._force_level = idx

    def _on_show_viewport_toggled(self, checked: bool) -> None:
        if self._viewport_line is not None:
            self._viewport_line.visible = checked

    def _on_z_slice_changed(self, value: int) -> None:
        self._state.z_slice = value
        # Recompute per-level z slices.
        for i in range(self._state.n_levels):
            scale = 2**i
            self._state._z_slices[i] = value // scale
        # Clear cache so all tiles are re-fetched.
        self._state.tile_manager.clear()

    def _on_update_clicked_sync(self) -> None:
        """Wrapper to launch the async update from a sync Qt signal."""
        asyncio.ensure_future(self._on_update_clicked())

    async def _on_update_clicked(self) -> None:
        """Main update flow: cancel previous, plan, submit new task."""
        t_total = time.perf_counter()
        self._update_count += 1

        # 1. Cancel previous task if running.
        prev_was_cancelled = False
        task_age_ms = 0.0
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
            task_age_ms = (time.perf_counter() - self._load_task_start_time) * 1000
            prev_was_cancelled = True
            # Give the cancelled task a chance to clean up.
            try:
                await self._load_task
            except asyncio.CancelledError:
                pass

        # 2. Snapshot camera_info.
        camera_info = self._snapshot_camera()

        # 3. Plan update.
        fill_plan, stats = self._state.plan_update(
            camera_info=camera_info,
            load_bias=self._load_bias_sb.value(),
            force_level=self._force_level,
            use_culling=self._viewport_cull_cb.isChecked(),
        )

        # 4. Rebuild viewport wireframe.
        self._rebuild_viewport_wireframe(camera_info)

        # 5. Print debug summary.
        t_planning_ms = (time.perf_counter() - t_total) * 1000
        self._print_update_summary(
            camera_info,
            stats,
            t_planning_ms,
            prev_was_cancelled,
            task_age_ms,
        )

        # 6. Submit the commit loop.
        self._load_task_start_time = time.perf_counter()
        self._load_task = asyncio.ensure_future(
            self._state.commit_tiles_async(
                fill_plan,
                status_callback=self._status_label.setText,
                batch_size=COMMIT_BATCH_SIZE,
            )
        )

    def _snapshot_camera(self) -> dict:
        """Capture current camera state as a dict.

        pygfx's OrthographicCamera has both ``width`` and ``height``
        which define the *minimum* visible extent.  The actual visible
        extent is expanded in one dimension to match the canvas aspect
        ratio.  This method computes the true visible AABB.

        Uses ``get_logical_size()`` because the camera projection
        operates in logical (CSS) pixels, not physical/retina pixels.
        """
        cam = self._camera

        # Logical size -- matches the coordinate space the camera uses.
        logical_size = self._canvas.get_logical_size()
        vw = logical_size[0] if logical_size[0] > 0 else 800
        vh = logical_size[1] if logical_size[1] > 0 else 600
        canvas_aspect = vw / vh

        # Camera's requested extents.
        cam_w = cam.width if cam.width > 0 else 1.0
        cam_h = cam.height if cam.height > 0 else 1.0
        cam_aspect = cam_w / cam_h

        # Actual visible world extent: expand one dimension to fill canvas.
        if canvas_aspect >= cam_aspect:
            # Canvas is relatively wider -> width expands
            world_height = cam_h
            world_width = cam_h * canvas_aspect
        else:
            # Canvas is relatively taller -> height expands
            world_width = cam_w
            world_height = cam_w / canvas_aspect

        # Camera position (centre of view).
        pos = cam.local.position
        cx = float(pos[0])
        cy = float(pos[1])

        # Viewport AABB in world space.
        half_w = world_width / 2.0
        half_h = world_height / 2.0
        view_min = np.array([cx - half_w, cy - half_h], dtype=np.float64)
        view_max = np.array([cx + half_w, cy + half_h], dtype=np.float64)

        return {
            "viewport_width": vw,
            "viewport_height": vh,
            "world_width": world_width,
            "world_height": world_height,
            "position": np.array([cx, cy, float(pos[2])]),
            "view_center": (cx, cy),
            "view_min": view_min,
            "view_max": view_max,
        }

    def _rebuild_viewport_wireframe(self, camera_info: dict) -> None:
        if self._viewport_line is not None:
            self._scene.remove(self._viewport_line)
        self._viewport_line = make_viewport_wireframe(
            camera_info["view_min"],
            camera_info["view_max"],
            VIEWPORT_COLOR,
        )
        self._scene.add(self._viewport_line)
        self._viewport_line.visible = self._show_viewport_cb.isChecked()

    # ---- Debug print ----

    def _print_update_summary(
        self,
        camera_info: dict,
        stats: dict,
        t_planning_ms: float,
        prev_was_cancelled: bool,
        task_age_ms: float,
    ) -> None:
        W = 70
        sep_heavy = "=" * W
        sep_light = "-" * W
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(sep_heavy)
        print(f"[update #{self._update_count}]  {now}")
        print(sep_light)

        # Async task status.
        print("async task (previous)")
        if prev_was_cancelled:
            print("  status:          cancelled")
            print(f"  task_age:        {task_age_ms:.1f} ms")
        else:
            print("  status:          none / complete")

        # Camera.
        pos = camera_info["position"]
        cam = self._camera
        print()
        print("camera")
        print(f"  position:        [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
        print(f"  cam.width:       {cam.width:.1f}  cam.height: {cam.height:.1f}")
        print(
            f"  logical_size:    {camera_info['viewport_width']:.0f} x "
            f"{camera_info['viewport_height']:.0f}"
        )
        print(f"  visible_width:   {camera_info['world_width']:.1f}")
        print(f"  visible_height:  {camera_info['world_height']:.1f}")
        ppwu = camera_info["viewport_width"] / camera_info["world_width"]
        print(f"  pixels/world:    {ppwu:.3f}")
        print(
            f"  view_min:        [{camera_info['view_min'][0]:.1f}, "
            f"{camera_info['view_min'][1]:.1f}]"
        )
        print(
            f"  view_max:        [{camera_info['view_max'][0]:.1f}, "
            f"{camera_info['view_max'][1]:.1f}]"
        )

        # LOAD assignment.
        lc = stats.get("level_counts", {})
        force = self._force_level
        load_bias = self._load_bias_sb.value()
        ww = max(camera_info["world_width"], 1.0)
        vw = camera_info["viewport_width"]
        screen_px_size = ww / vw if vw > 0 else 1.0
        biased = screen_px_size * max(load_bias, 1e-6)
        ideal = 1.0 + math.log2(max(biased, 1e-6))
        shift = math.log2(max(load_bias, 1e-6))
        print()
        print(
            f"load assignment  (force_level={force}  "
            f"load_bias={load_bias:.2f}  shift={shift:+.2f} levels)"
        )
        print(f"  screen_pixel_size: {screen_px_size:.4f} wu/px")
        print(f"  biased:            {biased:.4f} wu/px")
        print(
            f"  ideal_level:       {ideal:.2f}  "
            f"-> rounded: {max(1, min(self._state.n_levels, round(ideal)))}"
        )
        if force is None:
            for k in range(1, self._state.n_levels + 1):
                data_px_size = 2 ** (k - 1)
                flag = " <-- selected" if k in lc else ""
                print(
                    f"  L{k}: data_pixel={data_px_size} wu/px  "
                    f"(switches at screen_pixel={data_px_size:.1f}){flag}"
                )
        for level in sorted(k for k in lc if k > 0):
            print(f"  level {level}:  {lc[level]} tiles")
        print(f"  load_select:     {stats['load_select_ms']:.2f} ms")
        print(f"  distance_sort:  {stats['distance_sort_ms']:.2f} ms")

        # Viewport cull.
        culling_on = self._viewport_cull_cb.isChecked()
        print()
        print(f"viewport cull  [{'ENABLED' if culling_on else 'DISABLED'}]")
        if culling_on:
            print(f"  total tiles:     {stats['total_required']}")
            visible = stats["total_required"] - stats["n_culled"]
            print(f"  visible tiles:   {visible}")
            print(f"  n_culled:        {stats['n_culled']}")
            print(f"  cull_time:       {stats.get('cull_ms', 0):.2f} ms")
        else:
            print(f"  total tiles:     {stats['total_required']}  (no culling)")

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

        # Cache.
        print()
        print("cache  (planning phase)")
        print(f"  hits:           {stats['hits']}")
        print(f"  fills queued:   {stats['fills']}")
        print(f"  tilemap:        {len(self._state.tile_manager.tilemap)}")
        print(f"  stage:          {stats['stage_ms']:.2f} ms")
        print(f"  plan_total:     {stats['plan_total_ms']:.2f} ms")
        print("  (commit is async -- see status label for progress)")

        # Outer timings.
        print()
        print("timings (outer)")
        print(f"  planning_total:    {t_planning_ms:.2f} ms")
        print(sep_heavy)


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------


async def async_main(state: BlockState2D, image: gfx.Image) -> None:
    """Top-level coroutine driven by PySide6.QtAsyncio."""
    app = QApplication.instance()
    window = ImageBlockApp(state=state, image=image)
    window.resize(1200, 800)
    window.setWindowTitle("Chunked 2D Image Viewer -- LUT Indirection (async)")
    window.show()

    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Async multiscale 2D image viewer with LUT indirection"
    )
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
    parser.add_argument(
        "--z-slice",
        type=int,
        default=None,
        help="Z slice index (default: mid-slice).",
    )
    args = parser.parse_args()

    if args.make_files:
        make_zarr_sample(args.zarr_path)
        return

    # Open stores synchronously before starting the event loop.
    print("Opening tensorstore handles ...")
    ts_stores = open_ts_stores(args.zarr_path, SCALE_NAMES)

    # Create the pygfx image + state.
    print("Building block image ...")
    image, state = make_block_image(
        ts_stores,
        block_size=BLOCK_SIZE,
        gpu_budget_bytes=GPU_BUDGET,
        z_slice=args.z_slice,
    )

    # Launch Qt + asyncio.
    print("Launching viewer ...")
    app = QApplication.instance() or QApplication(sys.argv)

    import PySide6.QtAsyncio as QtAsyncio

    QtAsyncio.run(async_main(state, image), handle_sigint=True)


if __name__ == "__main__":
    main()
