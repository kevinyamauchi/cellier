"""Phase 2 — PySide6 GUI + frustum culling.

A ``QMainWindow`` wrapping the Phase 1 multi-LOD volume renderer.
Camera movement is free at all times; pressing **Update** freezes the
current camera state, recomputes the frustum, optionally culls
off-screen bricks, runs the full pipeline, and refreshes the frustum
wireframe.  Each button press prints a structured debug block to stdout.

Usage
-----
Generate the multiscale zarr store (once):

    uv run example.py --make-files

Then launch the viewer:

    uv run example.py [--zarr-path PATH]

Controls
--------
Mouse       — orbit / zoom (between Update presses)
Update btn  — freeze camera, run pipeline, rebuild wireframe
Frustum cull checkbox  — enable / disable brick culling on next Update
Show frustum checkbox  — toggle wireframe visibility (no pipeline re-run)
Auto / 1 / 2 / 3 radio — select force-level on next Update
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys
import time

import numpy as np
import pygfx as gfx
import zarr
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

from block_volume import BlockVolumeState, make_block_volume
from block_volume.frustum import (
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
GPU_BUDGET = 512 * 1024**2  # 512 MiB — fits all L3 bricks (512) and ~82% of L2 (4,096) for a 1024³ volume

# LOD threshold bias.  Scales all screen-space thresholds uniformly.
# 1.0 = switch LOD when a voxel subtends exactly 1 screen pixel.
# > 1.0 = switch to coarser LOD sooner (less detail, fewer bricks).
# < 1.0 = stay at finer LOD longer (more detail, more bricks).
LOD_BIAS = 1.0

ZARR_PATH = pathlib.Path(__file__).parent / "multiscale_blobs.zarr"
ZARR_SCALE_NAMES = ["s0", "s1", "s2"]

FRUSTUM_COLOR = "#00cc44"
AABB_COLOR = "#ffffff"


# ---------------------------------------------------------------------------
# Zarr helpers (copied from Phase 1 example)
# ---------------------------------------------------------------------------


def _make_box_wireframe(box_min: np.ndarray, box_max: np.ndarray, color: str) -> gfx.Line:
    """Build a wireframe AABB as a gfx.Line."""
    import itertools
    corners = np.array(
        list(itertools.product(
            [box_min[0], box_max[0]],
            [box_min[1], box_max[1]],
            [box_min[2], box_max[2]],
        )),
        dtype=np.float32,
    )
    edge_pairs = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # z edges
        (0, 2), (1, 3), (4, 6), (5, 7),  # y edges
        (0, 4), (1, 5), (2, 6), (3, 7),  # x edges
    ]
    positions = np.array(
        [[corners[a], corners[b]] for a, b in edge_pairs],
        dtype=np.float32,
    ).reshape(-1, 3)
    return gfx.Line(gfx.Geometry(positions=positions), gfx.LineSegmentMaterial(color=color, thickness=1.5))


def make_multiscale_zarr(zarr_path: pathlib.Path) -> None:
    """Generate a 1024³ multiscale zarr store for testing.

    Produces three scales written directly to zarr (never fully
    materialised as float32 at once):

        s0  1024³  ~4 GiB uncompressed  (level 1 — finest)
        s1   512³  ~0.5 GiB             (level 2)
        s2   256³  ~0.06 GiB            (level 3 — coarsest)

    Peak RAM during generation is roughly 1 GiB (bool base array) plus
    the current scale being resized.  Expect the full write to take
    30–90 s depending on disk speed.
    """
    try:
        from skimage.data import binary_blobs
        from skimage.transform import resize
    except ImportError:
        raise SystemExit("scikit-image required.  Install with: pip install scikit-image")

    base_length = 1024
    print(f"Generating {base_length}³ binary blobs …  (this takes ~30 s)")
    # binary_blobs returns bool (1 GiB) — convert to float32 only per-scale
    # to keep peak RAM below 2 GiB.
    base_bool = binary_blobs(
        length=base_length, n_dim=3, blob_size_fraction=0.2, volume_fraction=0.2
    )

    for name, factor in zip(ZARR_SCALE_NAMES, [1, 2, 4]):
        s = base_length // factor
        if factor == 1:
            data = base_bool.astype(np.float32)
        else:
            data = resize(base_bool.astype(np.float32), (s, s, s), order=1).astype(np.float32)
        store = zarr.open(
            str(zarr_path / name),
            mode="w",
            shape=data.shape,
            chunks=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
            dtype="f4",
        )
        store[:] = data
        print(f"  {name}: {data.shape}  ({data.nbytes // 2**20} MiB)")
        del data  # release before allocating next scale
    print(f"Done → {zarr_path}")


def load_multiscale_zarr(zarr_path: pathlib.Path) -> list[np.ndarray]:
    """Load all scales from the zarr store into CPU arrays."""
    levels = []
    for name in ZARR_SCALE_NAMES:
        arr = zarr.open(str(zarr_path / name), mode="r")
        levels.append(np.asarray(arr[:], dtype=np.float32))
        print(f"  {name}: shape={levels[-1].shape}  "
              f"range=[{levels[-1].min():.3f}, {levels[-1].max():.3f}]")
    return levels


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class BlockVolumeApp(QMainWindow):
    """PySide6 viewer for the multi-LOD brick-cache volume renderer.

    Parameters
    ----------
    state : BlockVolumeState
        Pre-built volume state from ``make_block_volume``.
    vol : gfx.Volume
        The pygfx Volume object to add to the scene.
    """

    def __init__(self, state: BlockVolumeState, vol: gfx.Volume) -> None:
        super().__init__()
        self._state = state
        self._vol = vol
        self._frustum_line: gfx.Line | None = None
        self._update_count = 0
        self._force_level: int | None = None

        self._setup_scene()
        self._setup_ui()

    # ── Scene ──────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        # update_mode="continuous" → rendercanvas drives render() every vsync
        # via an internal Qt timer.  No manual canvas.update() needed.
        self._canvas = QRenderWidget(parent=self, update_mode="continuous")
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        self._scene.add(self._vol)

        # AABB wireframe
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

        # ── Left panel ──────────────────────────────────────────────
        panel = QWidget()
        panel.setFixedWidth(220)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)

        # Update button
        self._update_btn = QPushButton("Update")
        self._update_btn.clicked.connect(self._on_update_clicked)
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

    # ── Update button — the full pipeline ──────────────────────────────

    def _on_update_clicked(self) -> None:
        self._update_count += 1
        t_total = time.perf_counter()

        # 1. Snapshot camera state
        t0 = time.perf_counter()
        cam_pos  = get_camera_position_world(self._camera)
        view_dir = get_view_direction_world(self._camera)
        corners  = get_frustum_corners_world(self._camera)
        t_camera_ms = (time.perf_counter() - t0) * 1000

        # 2. Compute screen-space LOD thresholds.
        #    focal_half_height = (H/2) / tan(fov/2) — the distance at which
        #    a 1-world-unit object subtends 1 pixel vertically.
        #    threshold_k = 2^(k-1) * focal_half_height / LOD_BIAS
        #    so level k+1 is chosen when a level-k voxel subtends < 1 px.
        fov_y_rad = np.radians(self._camera.fov)
        _, screen_h = self._canvas.get_logical_size()
        lod_bias = self._lod_bias_sb.value()
        focal_half_height = (screen_h / 2) / np.tan(fov_y_rad / 2)
        thresholds = [
            (2 ** (k - 1)) * focal_half_height / lod_bias
            for k in range(1, self._state.n_levels)
        ]

        # 3. Compute frustum planes
        t0 = time.perf_counter()
        planes = frustum_planes_from_corners(corners)
        t_planes_ms = (time.perf_counter() - t0) * 1000

        # 4. BlockVolumeState.update()
        frustum_planes = planes if self._frustum_cull_cb.isChecked() else None
        t0 = time.perf_counter()
        stats = self._state.update(
            camera_pos=cam_pos,
            thresholds=thresholds,
            force_level=self._force_level,
            frustum_planes=frustum_planes,
        )
        t_update_ms = (time.perf_counter() - t0) * 1000

        # 5. Rebuild frustum wireframe
        t0 = time.perf_counter()
        self._rebuild_frustum_wireframe(corners)
        t_wireframe_ms = (time.perf_counter() - t0) * 1000

        t_total_ms = (time.perf_counter() - t_total) * 1000

        # 6. Print debug summary
        self._print_update_summary(
            cam_pos, view_dir, corners, planes, thresholds, stats,
            t_camera_ms, t_planes_ms, t_update_ms, t_wireframe_ms, t_total_ms,
        )

        # 7. Update status label
        culling_flag = " [cull ON]" if frustum_planes is not None else ""
        dropped = stats.get("n_dropped", 0)
        self._status_label.setText(
            f"update #{self._update_count}{culling_flag}\n"
            f"hits={stats['hits']}  fills={stats['fills']}\n"
            f"culled={stats['n_culled']}  dropped={dropped}\n"
            f"{t_total_ms:.1f} ms"
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
        t_camera_ms: float,
        t_planes_ms: float,
        t_update_ms: float,
        t_wireframe_ms: float,
        t_total_ms: float,
    ) -> None:
        W = 70
        sep_heavy = "═" * W
        sep_light = "─" * W
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(sep_heavy)
        print(f"[update #{self._update_count}]  {now}")
        print(sep_light)

        # Camera
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

        # Cache
        print()
        print("cache")
        print(f"  hits:        {stats['hits']}")
        print(f"  fills:       {stats['fills']}")
        print(f"  tilemap:     {len(self._state.tile_manager.tilemap)}")
        print(f"  stage:       {stats['stage_ms']:.2f} ms")
        print(f"  commit:      {stats['commit_ms']:.2f} ms")
        print(f"  lut_rebuild: {stats['lut_rebuild_ms']:.2f} ms")

        # Outer timings
        print()
        print("timings (outer)")
        print(f"  camera_snapshot:   {t_camera_ms:.2f} ms")
        print(f"  frustum_planes:    {t_planes_ms:.2f} ms")
        print(f"  state_update:      {t_update_ms:.2f} ms")
        print(f"  wireframe_rebuild: {t_wireframe_ms:.2f} ms")
        print(f"  {'─' * 25}")
        print(f"  total:             {t_total_ms:.2f} ms")
        print(sep_heavy)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 multi-LOD volume viewer")
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

    print("Loading multiscale data …")
    levels = load_multiscale_zarr(args.zarr_path)
    print(f"  {len(levels)} levels loaded.\n")

    vol, state = make_block_volume(
        levels,
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

    app = QApplication.instance() or QApplication(sys.argv)
    window = BlockVolumeApp(state=state, vol=vol)
    window.resize(1200, 800)
    window.setWindowTitle("LUT Volume Renderer — Phase 2 (PySide6 + frustum culling)")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
