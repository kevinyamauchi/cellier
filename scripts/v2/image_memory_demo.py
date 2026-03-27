#!/usr/bin/env python
"""Demo: in-memory 3D image viewed as 2D slice and 3D volume.

Run with:
    uv run python scripts/v2/image_memory_demo.py
"""

from __future__ import annotations

import sys

import numpy as np
import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage.data import binary_blobs

from cellier.v2.controller import CellierController
from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.visuals._image_memory import ImageMemoryAppearance


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Create data ──────────────────────────────────────────────────
    volume = binary_blobs(length=128, volume_fraction=0.1, n_dim=3).astype(np.float32)
    store = ImageMemoryStore(data=volume, name="demo_volume")

    # ── 2. Set up the controller ────────────────────────────────────────
    controller = CellierController()

    # Coordinate system: 3 spatial axes
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))

    # 2D scene: displayed_axes=(1,2), slice_indices={0:0}
    scene_2d = controller.add_scene(dim="2d", coordinate_system=cs, name="slice")
    # 3D scene: displayed_axes=(0,1,2), slice_indices={}
    scene_3d = controller.add_scene(dim="3d", coordinate_system=cs, name="volume")

    appearance = ImageMemoryAppearance(
        color_map="viridis",
        clim=(0.0, 1.0),
    )

    # Add the same store to both scenes (store is registered only once)
    visual_2d = controller.add_image(
        data=store,
        scene_id=scene_2d.id,
        appearance=appearance,
        name="slice_view",
    )
    visual_3d = controller.add_image(
        data=store,
        scene_id=scene_3d.id,
        appearance=appearance,
        name="volume_view",
    )

    # ── 3. Create canvases and z-slice control ─────────────────────────
    root = QWidget()
    outer_layout = QVBoxLayout(root)
    controller.set_widget_parent(root)

    # Canvas row
    canvas_row = QWidget()
    canvas_layout = QHBoxLayout(canvas_row)
    canvas_layout.setContentsMargins(0, 0, 0, 0)

    widget_2d = controller.add_canvas(scene_id=scene_2d.id)
    widget_3d = controller.add_canvas(scene_id=scene_3d.id)

    canvas_layout.addWidget(widget_2d)
    canvas_layout.addWidget(widget_3d)
    outer_layout.addWidget(canvas_row)

    # Z-slice spinbox
    z_depth = store.shape[0]  # axis 0 = z
    controls = QWidget()
    ctrl_layout = QHBoxLayout(controls)
    ctrl_layout.setContentsMargins(0, 0, 0, 0)
    ctrl_layout.addWidget(QLabel("Z slice:"))

    z_spin = QSpinBox()
    z_spin.setRange(0, z_depth - 1)
    z_spin.setValue(0)
    ctrl_layout.addWidget(z_spin)
    ctrl_layout.addStretch()
    outer_layout.addWidget(controls)

    def _on_z_changed(z: int) -> None:
        scene_2d.dims.selection.slice_indices = {0: z}
        controller.reslice_scene(scene_2d.id)

    z_spin.valueChanged.connect(_on_z_changed)

    root.resize(1200, 700)
    root.show()

    # ── 4. Initial reslice + camera fit ────────────────────────────────
    # The async slicer needs a running event loop, so reslice is deferred
    # with QTimer.singleShot(0, ...) until QtAsyncio.run() has started.
    #
    # look_at_visual computes the bounding box from the current node
    # geometry, so it must run *after* data has been committed to the GPU
    # (otherwise the placeholder texture gives a near-zero bounding box).
    # We wrap the on_data_ready callbacks to trigger camera fit on the
    # first delivery — event-driven, not time-based.

    visuals_loaded: set[str] = set()

    def _wrap_callback(original, visual_id, label, view_dir, up):
        def _wrapped(batch):
            original(batch)
            if label not in visuals_loaded:
                visuals_loaded.add(label)
                controller.look_at_visual(visual_id, view_direction=view_dir, up=up)

        return _wrapped

    # Patch the render-layer visuals to fire camera fit on first data.
    scene_mgr_2d = controller._render_manager._scenes[scene_2d.id]
    gfx_vis_2d = scene_mgr_2d.get_visual(visual_2d.id)
    gfx_vis_2d.on_data_ready_2d = _wrap_callback(
        gfx_vis_2d.on_data_ready_2d,
        visual_2d.id,
        "2d",
        view_dir=(0, 0, -1),
        up=(0, 1, 0),
    )

    scene_mgr_3d = controller._render_manager._scenes[scene_3d.id]
    gfx_vis_3d = scene_mgr_3d.get_visual(visual_3d.id)
    gfx_vis_3d.on_data_ready = _wrap_callback(
        gfx_vis_3d.on_data_ready,
        visual_3d.id,
        "3d",
        view_dir=(-1, -1, -1),
        up=(0, 0, 1),
    )

    QTimer.singleShot(0, controller.reslice_all)

    QtAsyncio.run()


if __name__ == "__main__":
    main()
