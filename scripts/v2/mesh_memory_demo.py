#!/usr/bin/env python
"""Demo: in-memory mesh with toggleable 2D/3D view.

Creates a sphere isosurface via marching cubes.  In 2D mode the slab
filter shows only faces within ±0.5 of the current Z slice.

Run with:
    uv run python scripts/v2/mesh_memory_demo.py
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
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import marching_cubes

from cellier.v2.controller import CellierController
from cellier.v2.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.visuals._mesh_memory import MeshPhongAppearance

RNG = np.random.default_rng(0)


def _make_sphere_mesh():
    """Sphere of radius 40 in a 100^3 volume via marching cubes."""
    size = 100
    c = size // 2
    z, y, x = np.mgrid[:size, :size, :size]
    vol = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2
    verts, faces, normals, _ = marching_cubes(vol, level=40**2)
    # marching_cubes returns verts in (z, y, x) row order — correct for cellier.
    return (
        verts.astype(np.float32),
        faces.astype(np.int32),
        normals.astype(np.float32),
    )


def _random_vertex_colors(n: int) -> np.ndarray:
    colors = RNG.random((n, 4)).astype(np.float32)
    colors[:, 3] = 1.0
    return colors


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Build mesh ────────────────────────────────────────────────
    verts, faces, normals = _make_sphere_mesh()
    store = MeshMemoryStore(
        positions=verts,
        indices=faces,
        normals=normals,
        name="sphere",
    )
    z_min = int(verts[:, 0].min())
    z_max = int(verts[:, 0].max())
    z_mid = (z_min + z_max) // 2

    # ── 2. Controller and scene ──────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(
        dim="3d",
        coordinate_system=cs,
        name="sphere_scene",
        render_modes={"2d", "3d"},
        lighting="default",  # required for MeshPhongAppearance
    )

    appearance = MeshPhongAppearance(
        color=(0.4, 0.7, 1.0, 1.0),
        shininess=60.0,
        side="front",
    )
    visual = controller.add_mesh(
        data=store,
        scene_id=scene.id,
        appearance=appearance,
        name="sphere_mesh",
    )

    # ── 3. Build UI ──────────────────────────────────────────────────
    root = QWidget()
    root.setWindowTitle("Cellier v2 — Mesh Memory Demo")
    outer = QVBoxLayout(root)
    controller.set_widget_parent(root)

    canvas_widget = controller.add_canvas(scene_id=scene.id)
    canvas_id = next(iter(scene.canvases))
    outer.addWidget(canvas_widget)

    controls = QWidget()
    ctrl = QHBoxLayout(controls)
    ctrl.setContentsMargins(4, 4, 4, 4)

    z_label = QLabel("Z slice:")
    z_spin = QSpinBox()
    z_spin.setRange(z_min, z_max)
    z_spin.setValue(z_mid)
    z_label.setVisible(False)
    z_spin.setVisible(False)
    ctrl.addWidget(z_label)
    ctrl.addWidget(z_spin)
    ctrl.addStretch()

    color_btn = QPushButton("Randomize Colors")
    toggle_btn = QPushButton("Switch to 2D")
    ctrl.addWidget(color_btn)
    ctrl.addWidget(toggle_btn)
    outer.addWidget(controls)

    root.resize(900, 700)
    root.show()

    # ── 4. Interaction ───────────────────────────────────────────────
    _dim = ["3d"]

    def _toggle() -> None:
        if _dim[0] == "3d":
            _dim[0] = "2d"
            toggle_btn.setText("Switch to 3D")
            z_label.setVisible(True)
            z_spin.setVisible(True)
            scene.dims.selection.slice_indices = {0: z_spin.value()}
            scene.dims.selection.displayed_axes = (1, 2)
        else:
            _dim[0] = "3d"
            toggle_btn.setText("Switch to 2D")
            z_label.setVisible(False)
            z_spin.setVisible(False)
            scene.dims.selection.displayed_axes = (0, 1, 2)
        controller.reslice_scene(scene.id)

    toggle_btn.clicked.connect(_toggle)

    def _on_z(z: int) -> None:
        scene.dims.selection.slice_indices = {0: z}
        if _dim[0] == "2d":
            controller.reslice_scene(scene.id)

    z_spin.valueChanged.connect(_on_z)

    def _randomize() -> None:
        """Assign new random per-vertex colors and trigger a reslice.

        Updates store.colors directly (psygnal EventedModel emits the
        field-change signal).  Also ensures color_mode is "vertex" on the
        appearance so the material renders them.
        """
        store.colors = _random_vertex_colors(store.n_vertices)
        visual.appearance.color_mode = "vertex"
        controller.reslice_scene(scene.id)

    color_btn.clicked.connect(_randomize)

    # ── 5. Camera fit on first data delivery ─────────────────────────
    scene_mgr = controller._render_manager._scenes[scene.id]
    gfx_vis = scene_mgr.get_visual(visual.id)
    _fitted: set[str] = set()

    _orig_3d = gfx_vis.on_data_ready

    def _patch_3d(batch):
        _orig_3d(batch)
        if "3d" not in _fitted:
            _fitted.add("3d")
            controller.look_at_visual(
                visual.id, canvas_id, view_direction=(-1, -1, -1), up=(0, 0, 1)
            )

    gfx_vis.on_data_ready = _patch_3d

    _orig_2d = gfx_vis.on_data_ready_2d

    def _patch_2d(batch):
        _orig_2d(batch)
        if "2d" not in _fitted:
            _fitted.add("2d")
            controller.look_at_visual(
                visual.id, canvas_id, view_direction=(0, 0, -1), up=(0, 1, 0)
            )

    gfx_vis.on_data_ready_2d = _patch_2d

    # ── 6. Initial reslice ───────────────────────────────────────────
    QTimer.singleShot(0, controller.reslice_all)
    QtAsyncio.run()


if __name__ == "__main__":
    main()
