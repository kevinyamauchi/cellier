#!/usr/bin/env python
"""Demo: aligning two volumes with different voxel spacings using transforms.

Two synthetic volumes represent the same physical region but were acquired
at different resolutions:

  - Volume A ("high-res"): 1.0 um/voxel, shape (64, 64, 64)
  - Volume B ("low-res"):  anisotropic voxel spacing, shape (16, 32, 32)
    z-spacing = 4.0 um, y/x-spacing = 2.0 um

Without a corrective transform the low-res volume appears squashed and
half-sized.  Toggle the "Align transforms" checkbox to apply an
anisotropic scale=(4, 2, 2) transform (axis0=z, axis1=y, axis2=x) to
Volume B so the two volumes overlap in world space.

Run with:
    uv run python scripts/v2/transform_alignment_demo.py
"""

from __future__ import annotations

import sys

import numpy as np
import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellier.v2.controller import CellierController
from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._image_memory import ImageMemoryAppearance

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _make_sphere(shape: tuple[int, ...], radius_frac: float = 0.35) -> np.ndarray:
    """Solid sphere centred in a volume, intensity falls off from centre."""
    coords = np.mgrid[tuple(slice(0, s) for s in shape)]
    centre = np.array([(s - 1) / 2.0 for s in shape])
    dist = np.sqrt(sum((c - cx) ** 2 for c, cx in zip(coords, centre)))
    radius = min(shape) * radius_frac
    vol = np.clip(1.0 - dist / radius, 0.0, 1.0).astype(np.float32)
    return vol


def _make_cube(shape: tuple[int, ...], size_frac: float = 0.4) -> np.ndarray:
    """Solid cube centred in a volume."""
    vol = np.zeros(shape, dtype=np.float32)
    half = [int(s * size_frac / 2) for s in shape]
    centre = [s // 2 for s in shape]
    slices = tuple(slice(c - h, c + h) for c, h in zip(centre, half))
    vol[slices] = 1.0
    return vol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Create data ──────────────────────────────────────────────────
    # Volume A: high-res (1 um/voxel) — a sphere
    vol_a = _make_sphere((64, 64, 64))
    store_a = ImageMemoryStore(data=vol_a, name="high_res_sphere")

    # Volume B: low-res, anisotropic — a cube, same physical extent
    # z-spacing 4 um (64/16=4), y/x-spacing 2 um (64/32=2)
    vol_b = _make_cube((16, 32, 32))
    store_b = ImageMemoryStore(data=vol_b, name="low_res_cube")

    # ── 2. Controller + scenes ──────────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))

    scene_2d = controller.add_scene(dim="2d", coordinate_system=cs, name="slice_view")
    scene_3d = controller.add_scene(dim="3d", coordinate_system=cs, name="volume_view")

    # ── 3. Add visuals ──────────────────────────────────────────────────
    appear_a = ImageMemoryAppearance(color_map="viridis", clim=(0.0, 1.0))
    appear_b = ImageMemoryAppearance(color_map="magma", clim=(0.0, 1.0))

    # 2D scene
    vis_a_2d = controller.add_image(
        data=store_a, scene_id=scene_2d.id, appearance=appear_a, name="A_2d"
    )
    vis_b_2d = controller.add_image(
        data=store_b, scene_id=scene_2d.id, appearance=appear_b, name="B_2d"
    )

    # 3D scene
    vis_a_3d = controller.add_image(
        data=store_a, scene_id=scene_3d.id, appearance=appear_a, name="A_3d"
    )
    vis_b_3d = controller.add_image(
        data=store_b, scene_id=scene_3d.id, appearance=appear_b, name="B_3d"
    )

    # ── 3b. Tweak pygfx materials for overlay visibility ───────────────
    # Volume A is drawn on top; make it semi-transparent so B is visible.
    sm_2d = controller._render_manager._scenes[scene_2d.id]
    gfx_a_2d = sm_2d.get_visual(vis_a_2d.id)
    gfx_a_2d.node_2d.material.opacity = 0.5

    # Nudge B's 2D node forward in z to avoid z-fighting.
    gfx_b_2d = sm_2d.get_visual(vis_b_2d.id)
    gfx_b_2d.node_2d.local.z = 1.0

    # ── 4. Build UI ─────────────────────────────────────────────────────
    root = QWidget()
    outer = QVBoxLayout(root)
    controller.set_widget_parent(root)

    # -- Canvas row
    canvas_row = QWidget()
    canvas_layout = QHBoxLayout(canvas_row)
    canvas_layout.setContentsMargins(0, 0, 0, 0)

    widget_2d = controller.add_canvas(scene_id=scene_2d.id)
    widget_3d = controller.add_canvas(scene_id=scene_3d.id)
    canvas_layout.addWidget(widget_2d)
    canvas_layout.addWidget(widget_3d)
    outer.addWidget(canvas_row)

    # -- Controls row
    controls = QWidget()
    ctrl_layout = QHBoxLayout(controls)
    ctrl_layout.setContentsMargins(4, 4, 4, 4)

    # Z-slice spinner (for 2D scene).
    # The range covers Volume A's full depth.  Volume B's planning
    # method applies the inverse transform and clamps to valid bounds.
    z_max = vol_a.shape[0] - 1
    ctrl_layout.addWidget(QLabel("Z slice (world):"))
    z_spin = QSpinBox()
    z_spin.setRange(0, z_max)
    z_spin.setValue(z_max // 2)
    ctrl_layout.addWidget(z_spin)

    # Align checkbox
    align_cb = QCheckBox("Align transforms (scale B by 4,2,2)")
    align_cb.setChecked(False)
    ctrl_layout.addWidget(align_cb)

    # Show/hide Volume A
    show_a_cb = QCheckBox("Show Volume A")
    show_a_cb.setChecked(True)
    ctrl_layout.addWidget(show_a_cb)

    ctrl_layout.addStretch()
    outer.addWidget(controls)

    # -- Status label
    status = QLabel(
        "Viridis = Volume A (64³, 1 µm/voxel)    "
        "Magma = Volume B (16×32×32, z=4 µm, y/x=2 µm)"
    )
    outer.addWidget(status)

    # ── 5. Wire callbacks ───────────────────────────────────────────────
    IDENTITY = AffineTransform.identity()
    ALIGN_SCALE = AffineTransform.from_scale((4.0, 2.0, 2.0))

    def _on_z_changed(z: int) -> None:
        scene_2d.dims.selection.slice_indices = {0: z}
        controller.reslice_scene(scene_2d.id)

    z_spin.valueChanged.connect(_on_z_changed)

    def _on_align_toggled(checked: bool) -> None:
        t = ALIGN_SCALE if checked else IDENTITY
        vis_b_2d.transform = t
        vis_b_3d.transform = t
        controller.reslice_scene(scene_2d.id)
        controller.reslice_scene(scene_3d.id)
        if checked:
            status.setText("Transforms ALIGNED — both volumes overlap in world space")
        else:
            status.setText("Transforms OFF — Volume B appears half-sized")

    align_cb.toggled.connect(_on_align_toggled)

    def _on_show_a_toggled(checked: bool) -> None:
        vis_a_2d.appearance.visible = checked
        vis_a_3d.appearance.visible = checked

    show_a_cb.toggled.connect(_on_show_a_toggled)

    # ── 6. Initial reslice with camera fit ──────────────────────────────
    visuals_loaded: set[str] = set()

    def _wrap_callback(original, visual_id, label, view_dir, up):
        def _wrapped(batch):
            original(batch)
            if label not in visuals_loaded:
                visuals_loaded.add(label)
                controller.look_at_visual(visual_id, view_direction=view_dir, up=up)

        return _wrapped

    # Patch render-layer visuals for camera fit on first data delivery.
    gfx_a_2d.on_data_ready_2d = _wrap_callback(
        gfx_a_2d.on_data_ready_2d,
        vis_a_2d.id,
        "2d",
        view_dir=(0, 0, -1),
        up=(0, 1, 0),
    )

    sm_3d = controller._render_manager._scenes[scene_3d.id]
    gfx_a_3d = sm_3d.get_visual(vis_a_3d.id)

    gfx_a_3d.on_data_ready = _wrap_callback(
        gfx_a_3d.on_data_ready,
        vis_a_3d.id,
        "3d",
        view_dir=(-1, -1, -1),
        up=(0, 0, 1),
    )

    # Set initial z-slice to the middle of Volume A's range.
    scene_2d.dims.selection.slice_indices = {0: z_max // 2}

    root.resize(1400, 700)
    root.setWindowTitle("Transform alignment demo — in-memory volumes")
    root.show()

    QTimer.singleShot(0, controller.reslice_all)
    QtAsyncio.run()


if __name__ == "__main__":
    main()
