"""3D brick-shader viewer for real OME-Zarr files.

Opens a 3D Qt viewer for any OME-Zarr multiscale volume using the
Kiln-style brick cache shader.  Physical scale and level transforms are
read automatically from the OME metadata.

Usage
-----
Launch the viewer::

    uv run scripts/v2/debug_multiscale_volume_brick_ome.py \\
        --zarr-file-path /path/to/my.ome.zarr

Remote stores are also supported::

    uv run scripts/v2/debug_multiscale_volume_brick_ome.py \\
        --zarr-file-path s3://my-bucket/my.ome.zarr

Controls
--------
- **Render mode** dropdown — switch between ISO (isosurface) and MIP
  (Maximum Intensity Projection).  Colormap auto-switches to viridis
  for MIP and grays for ISO.
- **Clim max** spinner — adjust the upper contrast limit interactively.
- **LOD bias** spinner — scale the screen-space LOD thresholds.
- **Pipeline mode** radios — toggle between full LOD + frustum culling
  and a flat "load all bricks at the coarsest level" mode.
- **Reslice now** button — manually trigger a brick re-request.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Qt viewer
# ---------------------------------------------------------------------------


class OmeBrickViewer:
    """Main window for the OME-Zarr brick validation viewer."""

    def __init__(
        self,
        controller,
        scene_id,
        visual_model,
        n_levels: int,
        depth_range: tuple[float, float] = (1.0, 8000.0),
    ):
        from PySide6 import QtCore, QtWidgets

        self._controller = controller
        self._scene_id = scene_id
        self._visual_model = visual_model
        self._n_levels = n_levels

        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("MultiscaleVolumeBrick — OME-Zarr viewer")
        self._window.resize(1200, 750)

        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        canvas_widget = self._controller.add_canvas(
            self._scene_id, depth_range=depth_range
        )
        root_layout.addWidget(canvas_widget, stretch=1)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(270)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        root_layout.addWidget(panel)

        # ── Reslice button ────────────────────────────────────────────
        self._reslice_btn = QtWidgets.QPushButton("Reslice now")
        self._reslice_btn.clicked.connect(self._on_reslice_clicked)
        panel_layout.addWidget(self._reslice_btn)

        # ── Pipeline mode ─────────────────────────────────────────────
        mode_group = QtWidgets.QGroupBox("Pipeline mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self._lod_radio = QtWidgets.QRadioButton("LOD + frustum culling")
        coarsest = n_levels - 1
        self._flat_radio = QtWidgets.QRadioButton(f"Level {coarsest} all bricks (flat)")
        self._lod_radio.setChecked(True)
        self._lod_radio.toggled.connect(self._on_pipeline_mode_toggle)
        mode_layout.addWidget(self._lod_radio)
        mode_layout.addWidget(self._flat_radio)
        panel_layout.addWidget(mode_group)

        # ── Render mode ───────────────────────────────────────────────
        render_group = QtWidgets.QGroupBox("Render mode")
        render_layout = QtWidgets.QHBoxLayout(render_group)
        self._render_mode_combo = QtWidgets.QComboBox()
        self._render_mode_combo.addItems(["ISO", "MIP"])
        self._render_mode_combo.currentTextChanged.connect(self._on_render_mode_changed)
        render_layout.addWidget(self._render_mode_combo)
        panel_layout.addWidget(render_group)

        # ── Contrast limits ───────────────────────────────────────────
        clim_group = QtWidgets.QGroupBox("Contrast limits")
        clim_layout = QtWidgets.QHBoxLayout(clim_group)
        clim_layout.addWidget(QtWidgets.QLabel("clim max"))
        self._clim_max_spin = QtWidgets.QDoubleSpinBox()
        self._clim_max_spin.setRange(0.01, 65000)
        self._clim_max_spin.setSingleStep(0.05)
        self._clim_max_spin.setDecimals(3)
        self._clim_max_spin.setValue(1.0)
        self._clim_max_spin.valueChanged.connect(self._on_clim_max_changed)
        clim_layout.addWidget(self._clim_max_spin)
        panel_layout.addWidget(clim_group)

        # ── LOD bias ──────────────────────────────────────────────────
        bias_group = QtWidgets.QGroupBox("LOD bias")
        bias_layout = QtWidgets.QHBoxLayout(bias_group)
        bias_layout.addWidget(QtWidgets.QLabel("lod_bias"))
        self._lod_bias_spin = QtWidgets.QDoubleSpinBox()
        self._lod_bias_spin.setRange(0.1, 10.0)
        self._lod_bias_spin.setSingleStep(0.1)
        self._lod_bias_spin.setDecimals(2)
        self._lod_bias_spin.setValue(1.0)
        self._lod_bias_spin.valueChanged.connect(self._on_lod_bias_changed)
        bias_layout.addWidget(self._lod_bias_spin)
        panel_layout.addWidget(bias_group)

        # ── ISO threshold ─────────────────────────────────────────
        thresh_group = QtWidgets.QGroupBox("ISO threshold")
        thresh_layout = QtWidgets.QHBoxLayout(thresh_group)
        thresh_layout.addWidget(QtWidgets.QLabel("threshold"))
        self._threshold_spin = QtWidgets.QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 65000.0)
        self._threshold_spin.setSingleStep(0.01)
        self._threshold_spin.setDecimals(3)
        self._threshold_spin.setValue(self._visual_model.appearance.iso_threshold)
        self._threshold_spin.valueChanged.connect(self._on_threshold_changed)
        thresh_layout.addWidget(self._threshold_spin)
        panel_layout.addWidget(thresh_group)

        self._status_label = QtWidgets.QLabel("Mode: LOD + frustum culling")
        self._status_label.setWordWrap(True)
        panel_layout.addWidget(self._status_label)
        panel_layout.addStretch()

    @property
    def window(self):
        return self._window

    def _on_reslice_clicked(self):
        print("[DEBUG] Manual reslice triggered")
        self._controller.reslice_scene(self._scene_id)

    def _on_pipeline_mode_toggle(self, lod_active: bool):
        if lod_active:
            self._visual_model.appearance.force_level = None
            self._visual_model.appearance.frustum_cull = True
            self._status_label.setText("Mode: LOD + frustum culling")
        else:
            coarsest = self._n_levels - 1
            self._visual_model.appearance.force_level = coarsest
            self._visual_model.appearance.frustum_cull = False
            self._status_label.setText(f"Mode: Level {coarsest} all bricks (flat)")
        print(
            f"[DEBUG] Pipeline mode changed: "
            f"force_level={self._visual_model.appearance.force_level}, "
            f"frustum_cull={self._visual_model.appearance.frustum_cull}"
        )
        self._controller.reslice_scene(self._scene_id)

    def _on_render_mode_changed(self, text: str):
        mode = text.lower()  # "ISO" -> "iso", "MIP" -> "mip"
        self._visual_model.appearance.render_mode = mode
        print(f"[DEBUG] Render mode changed to {mode}")

        # Switch colormap: viridis for MIP, grays for ISO.
        if mode == "mip":
            self._visual_model.appearance.color_map = "viridis"
        else:
            self._visual_model.appearance.color_map = "grays"

    def _on_threshold_changed(self, value: float):
        self._visual_model.appearance.iso_threshold = value
        print(f"[DEBUG] ISO threshold changed to {value}")

    def _on_clim_max_changed(self, value: float):
        self._visual_model.appearance.clim = (0.0, value)
        print(f"[DEBUG] clim changed to (0.0, {value})")

    def _on_lod_bias_changed(self, value: float):
        self._visual_model.appearance.lod_bias = value
        print(f"[DEBUG] LOD bias changed to {value}")
        self._controller.reslice_scene(self._scene_id)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _make_box_wireframe(box_min, box_max, color):
    import pygfx as gfx

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
        gfx.LineSegmentMaterial(color=color, thickness=2.0),
    )


def _dtype_clim_max(dtype: np.dtype) -> float:
    """Return a sensible initial clim upper bound for the given dtype."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    # Float data: assume normalised [0, 1] unless proven otherwise.
    return 1.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def async_main(zarr_uri: str):
    from PySide6 import QtWidgets

    from cellier.v2.controller import CellierController
    from cellier.v2.data.image import OMEZarrImageDataStore
    from cellier.v2.scene.dims import CoordinateSystem
    from cellier.v2.transform import AffineTransform
    from cellier.v2.visuals._image import ImageAppearance

    # ── Open the OME-Zarr store ───────────────────────────────────────
    print(f"Opening OME-Zarr store: {zarr_uri}")
    data_store = OMEZarrImageDataStore.from_path(zarr_uri)
    print(f"  {data_store.n_levels} levels found.")
    for i, shape in enumerate(data_store.level_shapes):
        print(f"  Level {i}: shape={shape}")
    print(f"  Axes:  {data_store.axis_names}")
    print(f"  Units: {data_store.axis_units}")

    # ── Extract level-0 physical scale from OME metadata (ZYX order) ─
    import yaozarrs

    group = yaozarrs.open_group(data_store.zarr_path)
    ome_image = group.ome_metadata()
    ms = ome_image.multiscales[data_store.multiscale_index]
    level_0_scale_zyx = np.array(ms.datasets[0].scale_transform.scale, dtype=np.float64)

    print(f"\n  Level-0 physical scale (ZYX): {level_0_scale_zyx}")

    # compute_normalized_size() expects spacing in shader order (x=W, y=H, z=D),
    # which is the reverse of data ZYX.  Note: visual_model.transform is set in
    # data ZYX order and _pygfx_matrix() handles that reversal separately — but
    # voxel_spacing is consumed before any matrix conversion, so we must reverse
    # it explicitly here.
    voxel_spacing_xyz = level_0_scale_zyx[::-1].copy()

    # ── Compute world extents for AABB and depth range ────────────────
    vox_shape_zyx = np.array(data_store.level_shapes[0], dtype=np.float64)
    world_extents_zyx = vox_shape_zyx * level_0_scale_zyx
    max_extent = float(world_extents_zyx.max())
    # Depth range: near = max(1.0, max_extent * 0.0001) keeps the near plane
    # tight to the camera without wasting depth buffer bits on the sub-voxel
    # near zone.  Far = max_extent * 10 comfortably encloses the scene from
    # any orbit angle.  Both values scale with the physical dataset size so
    # the script works correctly for datasets of any scale.
    depth_range = (max(1.0, max_extent * 0.0001), max_extent * 10.0)

    print(f"  World extents (ZYX): {world_extents_zyx}")
    print(f"  Max extent:          {max_extent:.3f}")
    print(
        f"  Depth range:         near={depth_range[0]:.2f}  far={depth_range[1]:.0f}\n"
    )

    # ── Controller and scene ──────────────────────────────────────────
    controller = CellierController(
        widget_parent=None,
        camera_reslice_enabled=False,
    )
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    # ── Voxel-to-world transform from level-0 physical scale ──────────
    # AffineTransform uses ZYX (data) axis order.
    voxel_to_world = AffineTransform.from_scale_and_translation(
        scale=tuple(level_0_scale_zyx)
    )

    # ── Initial appearance ────────────────────────────────────────────
    initial_clim_max = _dtype_clim_max(data_store.dtype)
    appearance = ImageAppearance(
        color_map="grays",
        clim=(0.0, initial_clim_max),
        lod_bias=1.0,
        force_level=None,
        frustum_cull=True,
        iso_threshold=0.2,
        render_mode="iso",
    )

    _DEBUG_MODE = "none"  # "ray_dir", "lod_color", or "normal_rgb"

    visual_model = controller.add_image(
        data=data_store,
        scene_id=scene.id,
        appearance=appearance,
        name="volume",
        block_size=32,
        gpu_budget_bytes=2048 * 1024**2,
        threshold=0.2,
        use_brick_shader=True,
        voxel_spacing=voxel_spacing_xyz,
    )
    visual_model.transform = voxel_to_world

    # ── Set debug mode on the material ───────────────────────────────
    vis = controller._render_manager._scenes[scene.id].get_visual(visual_model.id)
    vis.material_3d.debug_mode = _DEBUG_MODE

    # ── Pink AABB wireframe in world space ────────────────────────────
    # world_extents_zyx are physical sizes; reorder to pygfx (x=W, y=H, z=D).
    # The volume spans [0, wx] x [0, wy] x [0, wz] in world space.
    # A fixed half-voxel border (-0.5, extent-0.5) is sufficient — no need
    # to multiply by spacing since these are already world-space coordinates.
    wz, wy, wx = world_extents_zyx
    pad = 1.0
    gfx_scene = controller._render_manager.get_scene(scene.id)
    gfx_scene.add(
        _make_box_wireframe(
            np.array([-0.5 - pad, -0.5 - pad, -0.5 - pad]),
            np.array([wx - 0.5 + pad, wy - 0.5 + pad, wz - 0.5 + pad]),
            "#ff00ff",
        )
    )

    # ── Print diagnostics ─────────────────────────────────────────────
    print(f"[DEBUG] debug_mode     = {_DEBUG_MODE}")
    print(f"[DEBUG] norm_size      = {vis._norm_size}")
    print(f"[DEBUG] dataset_size   = {vis._dataset_size}")
    print(f"[DEBUG] node_3d matrix (before reslice) =\n{vis.node_3d.local.matrix}")

    # ── Show window ───────────────────────────────────────────────────
    viewer = OmeBrickViewer(
        controller,
        scene.id,
        visual_model,
        n_levels=data_store.n_levels,
        depth_range=depth_range,
    )
    # Set clim max spinner to match dtype-derived initial value.
    viewer._clim_max_spin.setValue(initial_clim_max)
    viewer.window.show()

    # Initial reslice to kick off data loading.
    controller.reslice_scene(scene.id)

    # Wait for async data load, then print diagnostics and refit camera.
    await asyncio.sleep(2.0)
    print("[DEBUG] After initial reslice:")
    print(f"  cache n_resident = {vis._block_cache_3d.n_resident}")
    print(f"  LUT max w        = {vis._lut_manager_3d.lut_data[:,:,:,3].max()}")
    print(f"  node_3d matrix   =\n{vis.node_3d.local.matrix}")

    # Refit camera to the full scene.
    canvas_view = next(iter(controller._render_manager._canvases.values()))
    canvas_view.show_object(gfx_scene)
    print("[DEBUG] Camera refit done. Use 'Reslice now' to reload bricks.")

    app = QtWidgets.QApplication.instance()
    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-file-path",
        required=True,
        metavar="PATH",
        help=(
            "Path or URI to the OME-Zarr store "
            "(local path, file://, s3://, gs://, https://)."
        ),
    )
    args = parser.parse_args()

    zarr_input = args.zarr_file_path

    # Resolve local paths to an absolute file:// URI.
    if "://" not in zarr_input:
        local = Path(zarr_input)
        if not local.exists():
            print(
                f"Error: OME-Zarr store not found at '{local}'",
                file=sys.stderr,
            )
            sys.exit(1)
        zarr_uri = f"file://{local.resolve()}"
    else:
        zarr_uri = zarr_input

    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(zarr_uri), handle_sigint=True)
