"""3D brick-shader viewer for real OME-Zarr files with 2D/3D toggle.

Opens a Qt viewer for any OME-Zarr multiscale volume using the
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
- **Toggle 2D / 3D** button — switch between a 2D Z-slice view and the
  full 3D volume renderer.
- **Reslice now** button — manually trigger a brick re-request.
- **Render mode** dropdown *(3D only)* — switch between ISO and MIP.
  Colormap auto-switches to viridis for MIP and grays for ISO.
- **Pipeline mode** radios *(3D only)* — toggle between full LOD +
  frustum culling and a flat "load all bricks at the coarsest level" mode.
- **ISO threshold** spinner *(3D only)* — adjust the isosurface threshold.
- **Z slice** spinner *(2D only)* — select the displayed Z plane.
- **Contrast limits** range slider — adjust the lower and upper contrast limits (both views).
- **LOD bias** spinner — scale the screen-space LOD thresholds (both views).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np


class OmeBrickViewer:
    """Main window for the OME-Zarr brick validation viewer."""

    def __init__(
        self,
        controller,
        scene,
        visual_model,
        canvas_widget: QtCanvasWidget,
        zarr_uri: str,
        n_levels: int,
        z_depth: int,
        clim_range: tuple[float, float],
        slider_decimals: int = 2,
    ):
        from PySide6 import QtCore, QtWidgets

        from cellier.v2.gui._dataset_info import QtOmeZarrMetadataWidget
        from cellier.v2.gui.visuals._aabb import QtAABBWidget
        from cellier.v2.gui.visuals._colormap import QtColormapComboBox
        from cellier.v2.gui.visuals._contrast_limits import QtClimRangeSlider
        from cellier.v2.gui.visuals._image import QtVolumeRenderControls

        self._controller = controller
        self._scene = scene
        self._visual_model = visual_model
        self._canvas_widget = canvas_widget
        self._dims_sliders = canvas_widget.dims_sliders
        self._n_levels = n_levels
        self._z_max = z_depth - 1
        self._active_mode = "3d"

        self._clim_slider = QtClimRangeSlider(
            controller,
            visual_model.id,
            clim_range=clim_range,
            initial_clim=visual_model.appearance.clim,
            decimals=slider_decimals,
        )
        self._colormap_combo = QtColormapComboBox(
            controller,
            visual_model.id,
            initial_colormap=visual_model.appearance.color_map,
        )
        self._aabb_widget = QtAABBWidget(
            controller,
            visual_model.id,
            initial_enabled=visual_model.aabb.enabled,
            initial_line_width=visual_model.aabb.line_width,
            initial_color=visual_model.aabb.color,
        )
        self._render_controls = QtVolumeRenderControls(
            controller,
            visual_model.id,
            dtype_max=clim_range[1],
            initial_render_mode=visual_model.appearance.render_mode,
            initial_threshold=visual_model.appearance.iso_threshold,
            decimals=slider_decimals,
        )
        self._metadata_widget = QtOmeZarrMetadataWidget.from_path(zarr_uri)

        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("MultiscaleVolumeBrick — OME-Zarr viewer")
        self._window.resize(1200, 750)

        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        # ── Canvas + dims sliders ─────────────────────────────────────
        root_layout.addWidget(canvas_widget.widget, stretch=1)

        # ── Side panel ────────────────────────────────────────────────
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        root_layout.addWidget(panel)

        # Track 3D-only widgets for show/hide on toggle.
        self._widget_3d: list = []

        # ── Dataset metadata ──────────────────────────────────────────
        panel_layout.addWidget(self._metadata_widget.widget)

        # ── Shared: toggle + reslice + mode label ─────────────────────
        self._toggle_btn = QtWidgets.QPushButton("Toggle 2D / 3D")
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        panel_layout.addWidget(self._toggle_btn)

        self._reslice_btn = QtWidgets.QPushButton("Reslice now")
        self._reslice_btn.clicked.connect(self._on_reslice_clicked)
        panel_layout.addWidget(self._reslice_btn)

        self._mode_label = QtWidgets.QLabel("Mode: 3D")
        panel_layout.addWidget(self._mode_label)

        # ── 3D-only: pipeline mode ────────────────────────────────────
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
        self._widget_3d.append(mode_group)

        # ── 3D-only: render mode + ISO threshold ─────────────────────
        render_group = QtWidgets.QGroupBox("Render")
        render_layout = QtWidgets.QVBoxLayout(render_group)
        render_layout.addWidget(self._render_controls.widget)
        panel_layout.addWidget(render_group)
        self._widget_3d.append(render_group)

        # ── Shared: colormap ──────────────────────────────────────────
        cmap_group = QtWidgets.QGroupBox("Colormap")
        cmap_layout = QtWidgets.QVBoxLayout(cmap_group)
        cmap_layout.addWidget(self._colormap_combo.widget)
        panel_layout.addWidget(cmap_group)

        # ── Shared: contrast limits ───────────────────────────────────
        clim_group = QtWidgets.QGroupBox("Contrast limits")
        clim_layout = QtWidgets.QVBoxLayout(clim_group)
        clim_layout.addWidget(self._clim_slider.widget)
        panel_layout.addWidget(clim_group)

        # ── Shared: LOD bias ──────────────────────────────────────────
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

        # ── Shared: bounding box ──────────────────────────────────────
        aabb_group = QtWidgets.QGroupBox("Bounding box")
        aabb_layout = QtWidgets.QVBoxLayout(aabb_group)
        aabb_layout.addWidget(self._aabb_widget.widget)
        panel_layout.addWidget(aabb_group)

        self._status_label = QtWidgets.QLabel("Mode: LOD + frustum culling")
        self._status_label.setWordWrap(True)
        panel_layout.addWidget(self._status_label)
        panel_layout.addStretch()

    @property
    def window(self):
        return self._window

    # ── Toggle ────────────────────────────────────────────────────────

    def _on_toggle_clicked(self) -> None:
        coordinator = self._controller._render_manager._slice_coordinator
        if self._active_mode == "3d":
            coordinator.cancel_scene(self._scene.id)
            self._active_mode = "2d"
            self._mode_label.setText("Mode: 2D")
            # Set slice_indices before changing displayed_axes so the dims
            # state is consistent when the geometry-rebuild event fires.
            # Read the last Z value the slider stored (retained even while hidden).
            current_z = self._dims_sliders.current_index().get(0, self._z_max // 2)
            self._scene.dims.selection.slice_indices = {0: current_z}
            self._scene.dims.selection.displayed_axes = (1, 2)
            for w in self._widget_3d:
                w.setVisible(False)
        else:
            coordinator.cancel_scene(self._scene.id)
            self._active_mode = "3d"
            self._mode_label.setText("Mode: 3D")
            # Clear slice_indices before changing displayed_axes.
            self._scene.dims.selection.slice_indices = {}
            self._scene.dims.selection.displayed_axes = (0, 1, 2)
            for w in self._widget_3d:
                w.setVisible(True)
        self._controller.reslice_scene(self._scene.id)

    # ── Shared callbacks ──────────────────────────────────────────────

    def _on_reslice_clicked(self) -> None:
        print("[DEBUG] Manual reslice triggered")
        self._controller.reslice_scene(self._scene.id)

    def _on_lod_bias_changed(self, value: float) -> None:
        self._visual_model.appearance.lod_bias = value
        print(f"[DEBUG] LOD bias changed to {value}")
        self._controller.reslice_scene(self._scene.id)

    # ── 3D-only callbacks ─────────────────────────────────────────────

    def _on_pipeline_mode_toggle(self, lod_active: bool) -> None:
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
        self._controller.reslice_scene(self._scene.id)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _dtype_clim_max(dtype: np.dtype) -> float:
    """Return a sensible initial clim upper bound for the given dtype."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def _dtype_decimals(dtype: np.dtype) -> int:
    """Return slider label decimal places appropriate for the given dtype."""
    if np.issubdtype(dtype, np.integer):
        return 0
    return 2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def async_main(zarr_uri: str):
    from PySide6 import QtWidgets

    from cellier.v2.controller import CellierController
    from cellier.v2.data.image import OMEZarrImageDataStore
    from cellier.v2.gui._scene import QtCanvasWidget
    from cellier.v2.render._config import (
        CameraConfig,
        RenderManagerConfig,
        SlicingConfig,
    )
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

    # ── Compute world extents for AABB and depth range ────────────────
    vox_shape_zyx = np.array(data_store.level_shapes[0], dtype=np.float64)
    world_extents_zyx = vox_shape_zyx * level_0_scale_zyx
    max_extent = float(world_extents_zyx.max())
    depth_range = (max(1.0, max_extent * 0.0001), max_extent * 10.0)

    print(f"  World extents (ZYX): {world_extents_zyx}")
    print(f"  Max extent:          {max_extent:.3f}")
    print(
        f"  Depth range:         near={depth_range[0]:.2f}  far={depth_range[1]:.0f}\n"
    )

    # ── Controller ────────────────────────────────────────────────────
    controller = CellierController(
        widget_parent=None,
        render_config=RenderManagerConfig(
            slicing=SlicingConfig(batch_size=32, render_every=4),
            camera=CameraConfig(reslice_enabled=False),
        ),
    )
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))

    # ── Voxel-to-world transform ──────────────────────────────────────
    voxel_to_world = AffineTransform.from_scale_and_translation(
        scale=tuple(level_0_scale_zyx)
    )

    # ── Initial appearance ────────────────────────────────────────────
    initial_clim_max = _dtype_clim_max(data_store.dtype)
    slider_decimals = _dtype_decimals(data_store.dtype)
    _DEBUG_MODE = "none"

    # ── Single scene supporting both 2D and 3D ────────────────────────
    z_depth = data_store.level_shapes[0][0]
    scene = controller.add_scene(
        dim="3d",
        coordinate_system=cs,
        name="main",
        render_modes={"2d", "3d"},
    )
    # Pre-set the Z slice position for when we toggle to 2D.
    scene.dims.selection.slice_indices = {0: z_depth // 2}

    visual_model = controller.add_image(
        data=data_store,
        scene_id=scene.id,
        appearance=ImageAppearance(
            color_map="grays",
            clim=(0.0, initial_clim_max),
            lod_bias=1.0,
            force_level=None,
            frustum_cull=True,
            iso_threshold=0.2,
            render_mode="iso",
        ),
        name="volume",
        block_size=32,
        gpu_budget_bytes=4096 * 1024**2,
        gpu_budget_bytes_2d=64 * 1024**2,
        threshold=0.2,
        use_brick_shader=True,
        transform=voxel_to_world,
    )

    # ── Set debug mode on the 3D material ────────────────────────────
    vis = controller._render_manager._scenes[scene.id].get_visual(visual_model.id)
    vis.material_3d.debug_mode = _DEBUG_MODE

    # ── AABB colour and default visibility ───────────────────────────────
    visual_model.aabb.color = "#ff00ff"
    visual_model.aabb.enabled = True

    # ── Print diagnostics ─────────────────────────────────────────────
    print(f"[DEBUG] debug_mode     = {_DEBUG_MODE}")
    print(f"[DEBUG] norm_size      = {vis._norm_size}")
    print(f"[DEBUG] dataset_size   = {vis._dataset_size}")
    print(f"[DEBUG] node_3d matrix =\n{vis.node_3d.local.matrix}")

    # ── Canvas + dims sliders ─────────────────────────────────────────
    controller.add_canvas(scene.id, depth_range=depth_range)
    canvas_view = controller._render_manager._find_canvas_for_scene(scene.id)

    # Axis ranges derived from the finest level's voxel shape (ZYX order).
    level0_shape = data_store.level_shapes[0]
    axis_ranges = {i: (0, level0_shape[i] - 1) for i in range(len(level0_shape))}

    canvas_widget = QtCanvasWidget.from_scene_and_canvas(
        controller, scene, canvas_view, axis_ranges=axis_ranges
    )

    # ── Show window ───────────────────────────────────────────────────
    viewer = OmeBrickViewer(
        controller,
        scene=scene,
        visual_model=visual_model,
        canvas_widget=canvas_widget,
        zarr_uri=zarr_uri,
        n_levels=data_store.n_levels,
        z_depth=z_depth,
        clim_range=(0.0, initial_clim_max),
        slider_decimals=slider_decimals,
    )
    viewer.window.show()

    # Fit camera from metadata-derived bounding box, then kick off first reslice.
    controller.fit_camera(scene.id)
    controller.reslice_scene(scene.id)
    print("[DEBUG] Camera fitted. Use 'Reslice now' to reload bricks.")

    app = QtWidgets.QApplication.instance()
    app.aboutToQuit.connect(canvas_widget.close)
    app.aboutToQuit.connect(viewer._clim_slider.close)
    app.aboutToQuit.connect(viewer._colormap_combo.close)
    app.aboutToQuit.connect(viewer._aabb_widget.close)
    app.aboutToQuit.connect(viewer._render_controls.close)
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

    import logging

    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    from cellier.v2.logging import enable_debug_logging

    enable_debug_logging(categories=("perf"), level=logging.WARN)

    app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(zarr_uri), handle_sigint=True)
