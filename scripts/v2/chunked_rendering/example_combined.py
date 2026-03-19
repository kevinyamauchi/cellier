"""Combined 2D / 3D async multiscale chunked image viewer.

Single QRenderWidget canvas that toggles between a 2D tiled image
(XY slice, PanZoom) and a 3D LUT brick-cache volume (OrbitController).
Both render pipelines are driven manually by pressing **Update**.
Camera state is preserved independently for each mode across toggles.

Usage
-----
Generate the multiscale zarr store once (shared with example_2d / example_3d):

    uv run example_combined.py --make-files

Then launch the viewer:

    uv run example_combined.py [--zarr-path PATH]

Controls
--------
Mouse                  — pan/zoom (2D) or orbit (3D) at any time
Update btn             — trigger async render pipeline for active mode
Toggle 2D/3D btn       — switch between 2D and 3D (camera state preserved)
Z-slice spinbox (2D)   — change XY slice; takes effect on next Update
Frustum cull (3D)      — enable/disable brick frustum culling
Show frustum (3D)      — toggle frustum wireframe visibility
Force-level radios (3D) — override automatic LOAD selection
LOAD bias spinbox (3D)  — scale LOAD distance thresholds
Far plane spinbox (3D) — camera far clipping distance
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
from rendercanvas.qt import QRenderWidget

from image_block import (
    BlockState2D,
    BlockState3D,
    make_block_image,
    open_ts_stores,
)
from image_block.volume import make_block_volume
from image_block.volume.frustum import (
    frustum_planes_from_corners,
    get_camera_position_world,
    get_frustum_corners_world,
    get_view_direction_world,
    make_frustum_wireframe,
)

# ---------------------------------------------------------------------------
# Constants — edit these to tune memory usage and behaviour
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32

# GPU memory budgets — one cache texture per mode, independent.
GPU_BUDGET_2D = 256 * 1024**2  # 256 MiB  (2D tile cache)
GPU_BUDGET_3D = 1024 * 1024**2  # 512 MiB  (3D brick cache)

COMMIT_BATCH_SIZE = 8  # bricks/tiles committed per Qt yield in async loop

LOAD_BIAS_DEFAULT = 1.0

ZARR_PATH = pathlib.Path(__file__).parent / "multiscale_blobs.zarr"
ZARR_SCALE_NAMES = ["s0", "s1", "s2"]

FRUSTUM_COLOR = "#00cc44"
AABB_COLOR = "#ffffff"

# ---------------------------------------------------------------------------
# Zarr generation helper (copy from existing examples)
# ---------------------------------------------------------------------------


def make_multiscale_zarr(zarr_path: pathlib.Path) -> None:
    """Generate a 3-level multiscale zarr v3 store via tensorstore."""
    import tensorstore as ts
    from scipy.ndimage import zoom as scipy_zoom
    from skimage.data import binary_blobs

    print(f"Generating multiscale zarr at '{zarr_path}' …")

    base_length = 256  # 256³ is fast to generate for a demo
    print(f"  generating {base_length}³ base volume …")
    vol = binary_blobs(
        length=base_length,
        n_dim=3,
        blob_size_fraction=0.2,
        volume_fraction=0.2,
    ).astype(np.float32)

    for name, factor in zip(ZARR_SCALE_NAMES, [1, 2, 4], strict=False):
        if factor == 1:
            data = vol
        else:
            data = scipy_zoom(vol, 1.0 / factor, order=1).astype(np.float32)

        level_path = zarr_path / name
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(level_path)},
            "metadata": {
                "shape": list(data.shape),
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [BLOCK_SIZE] * 3},
                },
                "data_type": "float32",
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        store[...].write(data).result()
        print(f"  {name}: shape={data.shape}  ({data.nbytes // 2**20} MiB)")
        del data

    print(f"Done → {zarr_path}")


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _make_box_wireframe(
    box_min: np.ndarray, box_max: np.ndarray, color: str
) -> gfx.Line:
    """Build an AABB wireframe as disconnected edge pairs (LineSegmentMaterial)."""
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
    geometry = gfx.Geometry(positions=positions)
    material = gfx.LineSegmentMaterial(color=color, thickness=1.0)
    return gfx.Line(geometry, material)


def _make_rect_wireframe(
    x0: float, y0: float, x1: float, y1: float, color: str, z: float = 0.0
) -> gfx.Line:
    """Build a 2D rectangle wireframe as disconnected edge pairs."""
    positions = np.array(
        [
            [x0, y0, z],
            [x1, y0, z],
            [x1, y0, z],
            [x1, y1, z],
            [x1, y1, z],
            [x0, y1, z],
            [x0, y1, z],
            [x0, y0, z],
        ],
        dtype=np.float32,
    )
    geometry = gfx.Geometry(positions=positions)
    material = gfx.LineSegmentMaterial(color=color, thickness=1.0)
    return gfx.Line(geometry, material)


def _make_separator() -> QFrame:
    """Horizontal separator line for the side panel."""
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


# ---------------------------------------------------------------------------
# Camera-info extraction helpers
# ---------------------------------------------------------------------------


def _get_camera_info_2d(
    camera: gfx.PerspectiveCamera,
    canvas: QRenderWidget,
) -> dict:
    """Build the camera_info dict required by BlockState2D.plan_update().

    For a pygfx PerspectiveCamera with fov=0 (orthographic) driven by
    PanZoomController:
      - camera.width / camera.zoom  → visible world width
      - camera.height / camera.zoom → visible world height
      - camera.local.position.xy    → view centre in world space

    Returns
    -------
    dict with keys:
        viewport_width, viewport_height,
        world_width, world_height,
        view_center (x, y),
        view_min    ndarray (x, y),
        view_max    ndarray (x, y).
    """
    canvas_w, canvas_h = canvas.get_logical_size()
    zoom = max(camera.zoom, 1e-9)
    world_w = camera.width / zoom
    world_h = camera.height / zoom

    cx = float(camera.local.position[0])
    cy = float(camera.local.position[1])

    return {
        "viewport_width": canvas_w,
        "viewport_height": canvas_h,
        "world_width": world_w,
        "world_height": world_h,
        "view_center": (cx, cy),
        "view_min": np.array([cx - world_w / 2.0, cy - world_h / 2.0]),
        "view_max": np.array([cx + world_w / 2.0, cy + world_h / 2.0]),
    }


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class CombinedApp(QMainWindow):
    """Single-canvas viewer toggling between 2D and 3D chunked rendering.

    Parameters
    ----------
    state_2d : BlockState2D
        Pre-built 2D tile-cache state.
    image : gfx.Image
        pygfx Image world-object for the 2D scene.
    state_3d : BlockState3D
        Pre-built 3D brick-cache state.
    vol : gfx.Volume
        pygfx Volume world-object for the 3D scene.
    """

    def __init__(
        self,
        state_2d: BlockState2D,
        image: gfx.Image,
        state_3d: BlockState3D,
        vol: gfx.Volume,
    ) -> None:
        super().__init__()

        self._state_2d = state_2d
        self._image = image
        self._state_3d = state_3d
        self._vol = vol

        # Active mode: "2d" | "3d"
        self._active_mode: str = "2d"

        # Per-mode async task tracking
        self._load_task_2d: asyncio.Task | None = None
        self._load_task_3d: asyncio.Task | None = None

        # 3D-specific state
        self._frustum_line: gfx.Line | None = None
        self._force_level: int | None = None
        self._update_count = 0

        self._setup_scenes()
        self._setup_ui()

    # ── Scene setup ────────────────────────────────────────────────────

    def _setup_scenes(self) -> None:
        # Shared canvas / renderer (update_mode="continuous" drives GPU uploads)
        self._canvas = QRenderWidget(parent=self, update_mode="continuous")
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._canvas.request_draw(self._draw_frame)

        # ── 2D scene ──────────────────────────────────────────────────
        self._scene_2d = gfx.Scene()
        self._scene_2d.add(self._image)

        h2d, w2d = self._state_2d.base_layout.volume_shape
        self._scene_2d.add(
            _make_rect_wireframe(0, 0, float(w2d), float(h2d), AABB_COLOR, z=0.01)
        )

        # Orthographic camera (fov=0)
        self._camera_2d = gfx.PerspectiveCamera(fov=0, depth_range=(-1000, 1000))
        self._camera_2d.show_object(self._scene_2d, view_dir=(0, 0, -1), up=(0, 1, 0))

        # Both controllers always registered; .enabled gates whether they
        # respond to events (pygfx Controller.enabled property).
        self._controller_2d = gfx.PanZoomController(
            camera=self._camera_2d,
            register_events=self._renderer,
        )
        self._controller_2d.enabled = True  # 2D starts active

        # ── 3D scene ──────────────────────────────────────────────────
        self._scene_3d = gfx.Scene()
        self._scene_3d.add(self._vol)

        d3d, h3d, w3d = self._state_3d.base_layout.volume_shape
        pad = 1.0
        self._scene_3d.add(
            _make_box_wireframe(
                np.array([-0.5 - pad, -0.5 - pad, -0.5 - pad]),
                np.array([w3d - 0.5 + pad, h3d - 0.5 + pad, d3d - 0.5 + pad]),
                AABB_COLOR,
            )
        )

        self._camera_3d = gfx.PerspectiveCamera(70, 16 / 9, depth_range=(1, 8000))
        self._camera_3d.show_object(self._scene_3d, view_dir=(-1, -1, -1), up=(0, 0, 1))

        self._controller_3d = gfx.OrbitController(
            camera=self._camera_3d,
            register_events=self._renderer,
        )
        self._controller_3d.enabled = False  # 3D starts inactive

    # ── UI setup ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        panel = QWidget()
        panel.setFixedWidth(230)
        self._panel_layout = QVBoxLayout(panel)
        self._panel_layout.setContentsMargins(8, 8, 8, 8)
        pl = self._panel_layout

        # ── Section A: shared controls (always visible) ────────────────
        self._update_btn = QPushButton("Update")
        self._update_btn.clicked.connect(
            lambda: asyncio.ensure_future(self._on_update_clicked())
        )
        pl.addWidget(self._update_btn)

        self._toggle_btn = QPushButton("Toggle 2D / 3D")
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        pl.addWidget(self._toggle_btn)

        self._mode_label = QLabel("Mode: 2D")
        pl.addWidget(self._mode_label)

        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        pl.addWidget(self._status_label)

        pl.addWidget(_make_separator())

        # ── Section B: 2D-only controls ───────────────────────────────
        self._widget_2d: list[QWidget] = []

        lbl_z = QLabel("Z-slice  (→ next Update)")
        pl.addWidget(lbl_z)
        self._widget_2d.append(lbl_z)

        z_depth = int(self._state_2d.ts_stores[0].domain.shape[0])
        self._z_spinbox = QSpinBox()
        self._z_spinbox.setRange(0, max(0, z_depth - 1))
        self._z_spinbox.setValue(self._state_2d.z_slice)
        self._z_spinbox.valueChanged.connect(self._on_z_slice_changed)
        pl.addWidget(self._z_spinbox)
        self._widget_2d.append(self._z_spinbox)

        _sep_2d = _make_separator()
        pl.addWidget(_sep_2d)
        self._widget_2d.append(_sep_2d)

        # ── Section C: 3D-only controls ───────────────────────────────
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

        lbl_level = QLabel("Force LOAD level:")
        pl.addWidget(lbl_level)
        self._widget_3d.append(lbl_level)

        self._level_btn_group = QButtonGroup(self)
        for label, value in [("Auto", None), ("L1", 1), ("L2", 2), ("L3", 3)]:
            rb = QRadioButton(label)
            rb.setProperty("force_level", value)
            if value is None:
                rb.setChecked(True)
            self._level_btn_group.addButton(rb)
            pl.addWidget(rb)
            self._widget_3d.append(rb)
        self._level_btn_group.buttonClicked.connect(self._on_level_radio_clicked)

        lbl_load = QLabel("LOAD bias:")
        pl.addWidget(lbl_load)
        self._widget_3d.append(lbl_load)

        self._load_bias_sb = QDoubleSpinBox()
        self._load_bias_sb.setRange(0.1, 10.0)
        self._load_bias_sb.setSingleStep(0.1)
        self._load_bias_sb.setValue(LOAD_BIAS_DEFAULT)
        pl.addWidget(self._load_bias_sb)
        self._widget_3d.append(self._load_bias_sb)

        lbl_far = QLabel("Far plane:")
        pl.addWidget(lbl_far)
        self._widget_3d.append(lbl_far)

        self._far_plane_sb = QDoubleSpinBox()
        self._far_plane_sb.setRange(100.0, 100_000.0)
        self._far_plane_sb.setSingleStep(100.0)
        _, far = self._camera_3d.depth_range
        self._far_plane_sb.setValue(far)
        self._far_plane_sb.valueChanged.connect(self._on_far_plane_changed)
        pl.addWidget(self._far_plane_sb)
        self._widget_3d.append(self._far_plane_sb)

        pl.addStretch()

        # Start with 3D widgets hidden (2D is active)
        for w in self._widget_3d:
            w.setVisible(False)

        root.addWidget(panel)
        root.addWidget(self._canvas, stretch=1)

    # ── Draw frame ─────────────────────────────────────────────────────

    def _draw_frame(self) -> None:
        if self._active_mode == "2d":
            self._renderer.render(self._scene_2d, self._camera_2d)
        else:
            self._renderer.render(self._scene_3d, self._camera_3d)

    # ── Toggle button ──────────────────────────────────────────────────

    def _on_toggle_clicked(self) -> None:
        # Cancel any in-flight task for the mode we are leaving
        if self._active_mode == "2d":
            if self._load_task_2d and not self._load_task_2d.done():
                self._load_task_2d.cancel()
            self._active_mode = "3d"
            self._controller_2d.enabled = False
            self._controller_3d.enabled = True
            self._mode_label.setText("Mode: 3D")
            for w in self._widget_2d:
                w.setVisible(False)
            for w in self._widget_3d:
                w.setVisible(True)
        else:
            if self._load_task_3d and not self._load_task_3d.done():
                self._load_task_3d.cancel()
            self._active_mode = "2d"
            self._controller_3d.enabled = False
            self._controller_2d.enabled = True
            self._mode_label.setText("Mode: 2D")
            for w in self._widget_3d:
                w.setVisible(False)
            for w in self._widget_2d:
                w.setVisible(True)

        self._status_label.setText("Ready")

    # ── Z-slice spinbox ────────────────────────────────────────────────

    def _on_z_slice_changed(self, value: int) -> None:
        """Update per-level z-slice indices.  Takes effect on next Update."""
        for i in range(self._state_2d.n_levels):
            scale = 2**i
            self._state_2d._z_slices[i] = value // scale
        self._state_2d.z_slice = value

    # ── 3D-only UI callbacks ───────────────────────────────────────────

    def _on_show_frustum_toggled(self, checked: bool) -> None:
        if self._frustum_line is not None:
            self._frustum_line.visible = checked

    def _on_level_radio_clicked(self, button: QRadioButton) -> None:
        self._force_level = button.property("force_level")

    def _on_far_plane_changed(self, value: float) -> None:
        near, _ = self._camera_3d.depth_range
        self._camera_3d.depth_range = (near, value)

    # ── Frustum wireframe rebuild (3D) ─────────────────────────────────

    def _rebuild_frustum_wireframe(self, corners: np.ndarray) -> None:
        show = self._show_frustum_cb.isChecked()
        if self._frustum_line is not None:
            self._scene_3d.remove(self._frustum_line)
        self._frustum_line = make_frustum_wireframe(corners, color=FRUSTUM_COLOR)
        self._frustum_line.visible = show
        self._scene_3d.add(self._frustum_line)

    # ── Update button — async dispatcher ──────────────────────────────

    async def _on_update_clicked(self) -> None:
        if self._active_mode == "2d":
            await self._update_2d()
        else:
            await self._update_3d()

    # ── 2D update pipeline ─────────────────────────────────────────────

    async def _update_2d(self) -> None:
        self._update_count += 1
        t_total = time.perf_counter()

        # Cancel any in-flight 2D commit task
        if self._load_task_2d and not self._load_task_2d.done():
            self._load_task_2d.cancel()

        # Snapshot camera
        camera_info = _get_camera_info_2d(self._camera_2d, self._canvas)

        load_bias = LOAD_BIAS_DEFAULT  # extend with a spinbox if desired

        # Synchronous planning phase
        t_plan = time.perf_counter()
        fill_plan, stats = self._state_2d.plan_update(
            camera_info=camera_info,
            load_bias=load_bias,
            force_level=None,
            use_culling=True,
        )
        t_plan_ms = (time.perf_counter() - t_plan) * 1000

        # Print debug summary
        sep = "=" * 60
        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print()
        print(sep)
        print(f"UPDATE 2D  #{self._update_count}  {now}")
        print(sep)
        print(f"  Z-slice:        {self._state_2d.z_slice}")
        print(f"  world_width:    {camera_info['world_width']:.1f}")
        print(
            f"  view_center:    ({camera_info['view_center'][0]:.1f}, "
            f"{camera_info['view_center'][1]:.1f})"
        )
        print(f"  level_counts:   {stats.get('level_counts', {})}")
        print(f"  total_tiles:    {stats.get('total_required', '?')}")
        print(f"  culled:         {stats.get('n_culled', 0)}")
        drops = stats.get("n_dropped", 0)
        if drops:
            print(f"  *** DROPPED:    {drops} (budget exceeded) ***")
        print(f"  hits:           {stats.get('hits', '?')}")
        print(f"  fills queued:   {stats.get('fills', '?')}")
        print(f"  plan_total:     {t_plan_ms:.2f} ms")
        print(sep)

        if not fill_plan:
            self._status_label.setText(
                f"2D: all {stats.get('total_required',0)} tiles cached"
            )
            return

        self._status_label.setText(f"2D: loading 0 / {len(fill_plan)} tiles …")

        def _status_cb(text: str) -> None:
            self._status_label.setText(f"2D: {text}")

        self._load_task_2d = asyncio.ensure_future(
            self._state_2d.commit_tiles_async(
                fill_plan,
                status_callback=_status_cb,
                batch_size=COMMIT_BATCH_SIZE,
            )
        )

    # ── 3D update pipeline ─────────────────────────────────────────────

    async def _update_3d(self) -> None:
        self._update_count += 1
        t_total = time.perf_counter()

        # Cancel any in-flight 3D commit task
        if self._load_task_3d and not self._load_task_3d.done():
            self._load_task_3d.cancel()

        # Snapshot camera
        cam_pos = get_camera_position_world(self._camera_3d)
        view_dir = get_view_direction_world(self._camera_3d)
        corners = get_frustum_corners_world(self._camera_3d)

        fov_y_rad = np.radians(self._camera_3d.fov)
        _, screen_h = self._canvas.get_logical_size()
        load_bias = self._load_bias_sb.value()
        focal_hh = (screen_h / 2.0) / np.tan(fov_y_rad / 2.0)
        thresholds = [
            (2 ** (k - 1)) * focal_hh / load_bias
            for k in range(1, self._state_3d.n_levels)
        ]

        # Frustum planes
        t_planes = time.perf_counter()
        planes = frustum_planes_from_corners(corners)
        t_planes_ms = (time.perf_counter() - t_planes) * 1000

        frustum_planes = planes if self._frustum_cull_cb.isChecked() else None

        # Synchronous planning phase
        t_plan = time.perf_counter()
        fill_plan, stats = self._state_3d.plan_update(
            camera_pos=cam_pos,
            thresholds=thresholds,
            force_level=self._force_level,
            frustum_planes=frustum_planes,
        )
        t_plan_ms = (time.perf_counter() - t_plan) * 1000

        # Rebuild frustum wireframe
        self._rebuild_frustum_wireframe(corners)

        # Print debug summary
        sep = "=" * 60
        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        culling_on = self._frustum_cull_cb.isChecked()
        print()
        print(sep)
        print(f"UPDATE 3D  #{self._update_count}  {now}")
        print(sep)
        print(
            f"  cam_pos:        ({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f})"
        )
        print(f"  thresholds:     {[f'{t:.0f}' for t in thresholds]}")
        print(f"  load_bias:       {load_bias:.2f}")
        print(f"  frustum cull:   {'ON' if culling_on else 'OFF'}")
        if culling_on:
            print(f"  total_bricks:   {stats.get('total_required','?')}")
            print(
                f"  visible:        {stats.get('total_required',0) - stats.get('n_culled',0)}"
            )
            print(f"  culled:         {stats.get('n_culled',0)}")
            print(f"  planes_ms:      {t_planes_ms:.2f} ms")
        else:
            print(f"  total_bricks:   {stats.get('total_required','?')}")
        drops = stats.get("n_dropped", 0)
        if drops:
            print(f"  *** DROPPED:    {drops} (budget exceeded) ***")
        print(f"  hits:           {stats.get('hits','?')}")
        print(f"  fills queued:   {stats.get('fills','?')}")
        print(f"  plan_total:     {t_plan_ms:.2f} ms")
        print(sep)

        if not fill_plan:
            self._status_label.setText(
                f"3D: all {stats.get('total_required',0)} bricks cached"
            )
            return

        self._status_label.setText(f"3D: loading 0 / {len(fill_plan)} bricks …")

        def _status_cb(text: str) -> None:
            self._status_label.setText(f"3D: {text}")

        self._load_task_3d = asyncio.ensure_future(
            self._state_3d.commit_bricks_async(
                fill_plan,
                status_callback=_status_cb,
                batch_size=COMMIT_BATCH_SIZE,
            )
        )


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------


async def async_main(
    state_2d: BlockState2D,
    image: gfx.Image,
    state_3d: BlockState3D,
    vol: gfx.Volume,
) -> None:
    """Top-level coroutine driven by PySide6.QtAsyncio."""
    app = QApplication.instance()
    window = CombinedApp(
        state_2d=state_2d,
        image=image,
        state_3d=state_3d,
        vol=vol,
    )
    window.resize(1280, 800)
    window.setWindowTitle("Combined 2D / 3D chunked viewer")
    window.show()

    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined 2D/3D async multiscale chunked image viewer"
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
    # Parse before QApplication is created so Qt never sees argparse flags.
    args = parser.parse_args()

    if args.make_files:
        make_multiscale_zarr(args.zarr_path)
        sys.exit(0)

    if not args.zarr_path.exists():
        print(f"Error: zarr store not found at '{args.zarr_path}'")
        print("Run with --make-files first:")
        print("    uv run example_combined.py --make-files")
        sys.exit(1)

    # Open tensorstore stores synchronously before the event loop starts.
    # Both 2D and 3D states share the same read-only handles.
    print("Opening tensorstore stores …")
    ts_stores = open_ts_stores(args.zarr_path, ZARR_SCALE_NAMES)
    print(f"  {len(ts_stores)} levels opened.\n")

    # Build 2D renderer
    print("Building 2D tile-cache image …")
    image, state_2d = make_block_image(
        ts_stores=ts_stores,
        gpu_budget_bytes=GPU_BUDGET_2D,
        block_size=BLOCK_SIZE,
        clim=(0.0, 1.0),
    )
    print(f"  2D cache: {state_2d.cache_info.n_slots} slots\n")

    # Build 3D renderer (same ts_stores — shared read-only handles)
    print("Building 3D brick-cache volume …")
    vol, state_3d = make_block_volume(
        ts_stores=ts_stores,
        gpu_budget_bytes=GPU_BUDGET_3D,
        block_size=BLOCK_SIZE,
        clim=(0.0, 1.0),
        threshold=0.2,
    )
    print(
        f"  3D cache: {state_3d.cache_info.grid_side}³ grid = "
        f"{state_3d.cache_info.n_slots} slots\n"
    )

    print("Press 'Update' to render.  Toggle between 2D and 3D with the button.\n")

    # Pass only sys.argv[0] so Qt never sees argparse flags.
    app = QApplication([sys.argv[0]])
    QtAsyncio.run(
        async_main(state_2d, image, state_3d, vol),
        handle_sigint=True,
    )


if __name__ == "__main__":
    main()
