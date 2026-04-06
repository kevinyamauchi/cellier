"""Interactive validation viewer for MultiscaleVolumeBrick.

Creates a synthetic anisotropic OME-Zarr on first run, then opens a 3D
Qt viewer with controls for toggling between the full LOD pipeline and
a flat level-1 load for artifact isolation.

Usage
-----
Generate the dataset (once):

    uv run scripts/v2/debug_multiscale_volume_brick_blobs.py --make-files

Launch the viewer:

    uv run scripts/v2/debug_multiscale_volume_brick_blobs.py

What to look for
----------------
- Blobs should appear **spherical** (they are physical spheres stored as
  z-compressed ellipsoids due to 2x z-spacing).
- LOD vs flat toggle should preserve blob positions and roundness.
- `lod_color` debug mode shows LOD distribution.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import zarr

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

DATASET_PATH = "cellier_brick_blobs_small.ome.zarr"

SHAPE_ZYX = (500, 2000, 2000)  # voxel counts at level 0
SPACING_ZYX = (2.0, 1.0, 1.0)  # physical voxel size (um)
N_LEVELS = 3
N_BLOBS = 10000
BLOB_RADIUS_UM = 2.0
CHUNK_ZYX = (64, 64, 64)


def make_blob_volume(
    shape_zyx: tuple,
    spacing_zyx: tuple,
    n_blobs: int,
    radius_um: float,
    seed: int = 0,
) -> np.ndarray:
    """Return a uint8 binary volume with spherical blobs.

    Blobs are specified in physical units; their voxel footprints are
    ellipsoidal for anisotropic spacing. A correct shader renders them
    as spheres.
    """
    z, y, x = shape_zyx
    sz, sy, sx = spacing_zyx
    rng = np.random.default_rng(seed)
    volume = np.zeros(shape_zyx, dtype=np.uint8)

    rz = int(np.ceil(radius_um / sz))
    ry = int(np.ceil(radius_um / sy))
    rx = int(np.ceil(radius_um / sx))

    for _ in range(n_blobs):
        cz = rng.integers(rz, z - rz)
        cy = rng.integers(ry, y - ry)
        cx = rng.integers(rx, x - rx)

        lz = np.arange(-rz, rz + 1)
        ly = np.arange(-ry, ry + 1)
        lx = np.arange(-rx, rx + 1)
        ZZ, YY, XX = np.meshgrid(lz, ly, lx, indexing="ij")
        # Physical-distance ellipsoid mask.
        mask = ((ZZ * sz) ** 2 + (YY * sy) ** 2 + (XX * sx) ** 2) <= radius_um**2

        z0, z1 = cz - rz, cz + rz + 1
        y0, y1 = cy - ry, cy + ry + 1
        x0, x1 = cx - rx, cx + rx + 1

        mz0, mz1 = max(0, -z0), mask.shape[0] - max(0, z1 - z)
        my0, my1 = max(0, -y0), mask.shape[1] - max(0, y1 - y)
        mx0, mx1 = max(0, -x0), mask.shape[2] - max(0, x1 - x)

        z0, z1 = max(0, z0), min(z, z1)
        y0, y1 = max(0, y0), min(y, y1)
        x0, x1 = max(0, x0), min(x, x1)

        volume[z0:z1, y0:y1, x0:x1] |= mask[mz0:mz1, my0:my1, mx0:mx1].astype(np.uint8)

    return volume


def create_dataset(path: Path) -> None:
    """Create a 4-level OME-Zarr with anisotropic XY-only downsampling."""
    if path.exists():
        print(f"Dataset already exists at {path}")
        return

    print(f"Creating synthetic dataset at {path} ...")
    root = zarr.open_group(str(path), mode="w")

    data_l0 = make_blob_volume(SHAPE_ZYX, SPACING_ZYX, N_BLOBS, BLOB_RADIUS_UM)
    datasets_meta = []

    for level in range(N_LEVELS):
        factor = 2**level
        if level == 0:
            data = data_l0
        else:
            data = data_l0[:, ::factor, ::factor]

        arr = root.create_array(
            f"s{level}",
            shape=data.shape,
            chunks=CHUNK_ZYX,
            dtype=np.uint8,
        )
        arr[:] = data.astype(np.uint8)

        sz, sy, sx = SPACING_ZYX
        scale_zyx = [sz, sy * factor, sx * factor]

        datasets_meta.append(
            {
                "path": f"s{level}",
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale_zyx},
                ],
            }
        )

    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "blobs",
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets_meta,
        }
    ]
    print("Dataset created.")


# ---------------------------------------------------------------------------
# Qt viewer
# ---------------------------------------------------------------------------


class BlobViewer:
    """Main window for blob validation viewer."""

    def __init__(self, controller, scene_id, visual_model):
        from PySide6 import QtCore, QtWidgets

        self._controller = controller
        self._scene_id = scene_id
        self._visual_model = visual_model

        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("MultiscaleVolumeBrick - blob validation")
        self._window.resize(1100, 700)

        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        canvas_widget = self._controller.add_canvas(self._scene_id)
        root_layout.addWidget(canvas_widget, stretch=1)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(260)
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
        self._flat_radio = QtWidgets.QRadioButton("Level 3 all bricks (flat)")
        self._lod_radio.setChecked(True)  # matches initial force_level=1
        self._lod_radio.toggled.connect(self._on_mode_toggle)
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
        self._clim_max_spin.setRange(0.01, 10.0)
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

        self._status_label = QtWidgets.QLabel("Mode: Level 1 all bricks (flat)")
        self._status_label.setWordWrap(True)
        panel_layout.addWidget(self._status_label)
        panel_layout.addStretch()

    @property
    def window(self):
        return self._window

    def _on_reslice_clicked(self):
        print("[DEBUG] Manual reslice triggered")
        self._controller.reslice_scene(self._scene_id)

    def _on_mode_toggle(self, lod_active: bool):
        if lod_active:
            self._visual_model.appearance.force_level = None
            self._visual_model.appearance.frustum_cull = True
            self._status_label.setText("Mode: LOD + frustum culling")
        else:
            self._visual_model.appearance.force_level = 3
            self._visual_model.appearance.frustum_cull = False
            self._status_label.setText("Mode: Level 3 all bricks (flat)")
        print(
            f"[DEBUG] Mode changed: force_level={self._visual_model.appearance.force_level}, "
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

    def _on_clim_max_changed(self, value: float):
        self._visual_model.appearance.clim = (0.0, value)
        print(f"[DEBUG] clim changed to (0.0, {value})")

    def _on_lod_bias_changed(self, value: float):
        self._visual_model.appearance.lod_bias = value
        print(f"[DEBUG] LOD bias changed to {value}")
        self._controller.reslice_scene(self._scene_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def async_main():
    from PySide6 import QtWidgets

    from cellier.v2.controller import CellierController
    from cellier.v2.data.image import MultiscaleZarrDataStore
    from cellier.v2.scene.dims import CoordinateSystem
    from cellier.v2.visuals._image import ImageAppearance

    controller = CellierController(
        widget_parent=None,
        camera_reslice_enabled=False,
    )
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")

    # Per-level transforms: level-k voxel → level-0 voxel.
    # Only XY (axes 1, 2 in ZYX order) are downsampled; Z is unchanged.
    level_scales = []
    level_translations = []
    for level in range(N_LEVELS):
        factor = 2**level
        # ZYX order: z unchanged, y and x scaled by factor.
        scale = (1.0, float(factor), float(factor))
        translation = (0.0, (factor - 1) / 2.0, (factor - 1) / 2.0)
        level_scales.append(scale)
        level_translations.append(translation)

    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(DATASET_PATH),
        scale_names=[f"s{i}" for i in range(N_LEVELS)],
        level_scales=level_scales,
        level_translations=level_translations,
    )
    # store = OMEZarrImageDataStore.from_path(
    #     zarr_path=str(DATASET_PATH),
    #     name="ome dataset"
    # )

    appearance = ImageAppearance(
        color_map="grays",
        clim=(0.0, 1.0),
        lod_bias=1.0,
        force_level=None,
        frustum_cull=True,
        iso_threshold=0.2,
    )
    # Activate ray_dir debug mode on the material to verify the box is visible.
    # This renders ray direction as RGB, skipping the isosurface test.
    _DEBUG_MODE = "none"  # set to "ray_dir", "lod_color", or "normal_rgb" to debug

    # Voxel spacing in shader order (x=sx, y=sy, z=sz).
    voxel_spacing = np.array(
        [SPACING_ZYX[2], SPACING_ZYX[1], SPACING_ZYX[0]], dtype=np.float64
    )

    visual_model = controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=appearance,
        name="blobs",
        block_size=32,
        gpu_budget_bytes=2048 * 1024**2,
        threshold=0.2,
        use_brick_shader=True,
        voxel_spacing=voxel_spacing,
    )

    # Set voxel-to-world transform (data-axis order: z, y, x).
    # This scales voxel coords by spacing so the volume appears as a cube.
    from cellier.v2.transform import AffineTransform

    visual_model.transform = AffineTransform.from_scale(SPACING_ZYX)

    # ── Diagnostic access to render-layer visual ────────────────────────
    vis = controller._render_manager._scenes[scene.id].get_visual(visual_model.id)
    vis.material_3d.debug_mode = _DEBUG_MODE

    # ── Add pink AABB wireframe in data space ─────────────────────────
    # The brick shader's world_transform maps normalised→data→world.
    # With identity data_to_world the box spans [0, W] x [0, H] x [0, D].
    gfx_scene = controller._render_manager.get_scene(scene.id)
    d, h, w = vis._volume_geometry.base_layout.volume_shape
    pad = 1.0

    def _make_box_wireframe(box_min, box_max, color):
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
        import pygfx as gfx

        return gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineSegmentMaterial(color=color, thickness=2.0),
        )

    # AABB in world (physical) space: voxel extent * spacing.
    # pygfx order: x=W, y=H, z=D; spacing order matches SPACING_ZYX reversed.
    sx, sy, sz = SPACING_ZYX[2], SPACING_ZYX[1], SPACING_ZYX[0]
    aabb = _make_box_wireframe(
        np.array([(-0.5 - pad) * sx, (-0.5 - pad) * sy, (-0.5 - pad) * sz]),
        np.array([(w - 0.5 + pad) * sx, (h - 0.5 + pad) * sy, (d - 0.5 + pad) * sz]),
        "#ff00ff",
    )
    gfx_scene.add(aabb)

    # ── Print state before reslice ────────────────────────────────────
    print(f"[DEBUG] debug_mode = {_DEBUG_MODE}")
    print(f"[DEBUG] norm_size = {vis._norm_size}")
    print(f"[DEBUG] dataset_size = {vis._dataset_size}")
    print(f"[DEBUG] node_3d matrix (before reslice) =\n{vis.node_3d.local.matrix}")

    # ── Show window ──────────────────────────────────────────────────
    viewer = BlobViewer(controller, scene.id, visual_model)
    viewer.window.show()

    # Initial reslice to kick off data loading.
    controller.reslice_scene(scene.id)

    # Wait for async data load, then print diagnostics and refit camera.
    await asyncio.sleep(2.0)
    print("[DEBUG] After initial reslice:")
    print(f"  cache n_resident = {vis._block_cache_3d.n_resident}")
    print(f"  LUT max w = {vis._lut_manager_3d.lut_data[:,:,:,3].max()}")
    print(f"  node_3d matrix =\n{vis.node_3d.local.matrix}")

    # Re-fit camera now that the world transform is applied.
    canvas_view = next(iter(controller._render_manager._canvases.values()))
    canvas_view.show_object(gfx_scene)
    print("[DEBUG] Camera refit done. Use 'Reslice now' button to reslice.")

    app = QtWidgets.QApplication.instance()
    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--make-files",
        action="store_true",
        help="Generate the synthetic OME-Zarr dataset and exit.",
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        default=str(DATASET_PATH),
        help=f"Path to the OME-Zarr dataset (default: {DATASET_PATH}).",
    )
    args = parser.parse_args()
    DATASET_PATH = Path(args.zarr_path)

    if args.make_files:
        create_dataset(DATASET_PATH)
        sys.exit(0)

    # if not DATASET_PATH.exists():
    #     print(
    #         f"Dataset not found at {DATASET_PATH}\n"
    #         "Run with --make-files first to generate it.",
    #         file=sys.stderr,
    #     )
    #     sys.exit(1)

    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(), handle_sigint=True)
