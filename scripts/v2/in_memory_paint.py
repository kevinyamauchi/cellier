"""Demo: SyncPaintController on a 256x256 in-memory float32 label array.

Click and drag on the image to paint.  Clicking outside the image (on the
background or other visuals) has no effect.  The toolbar exposes brush
value, brush radius, undo, commit, and abort.

Run with:
    uv run python scripts/v2/in_memory_paint.py
"""

from __future__ import annotations

import sys

import numpy as np
from PySide6 import QtAsyncio
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellier.v2.controller import CellierController
from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.data.points._points_memory_store import PointsMemoryStore
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._image_memory import ImageMemoryAppearance, ImageVisual
from cellier.v2.visuals._points_memory import PointsMarkerAppearance


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Data ─────────────────────────────────────────────────────────
    # Blank image — any painted voxel will be unambiguously visible
    # against the empty background.
    label_data = np.zeros((256, 256), dtype=np.float32)
    image_store = ImageMemoryStore(data=label_data, name="labels")

    # Magenta dots at known (y, x) data coordinates serve as reference
    # markers — click on a dot and the painted blob should land centered
    # on the same dot.
    ANCHOR_COORDS = [
        (50, 50),
        (50, 200),
        (200, 50),
        (200, 200),
    ]
    point_positions = np.array(ANCHOR_COORDS, dtype=np.float32)
    points_store = PointsMemoryStore(positions=point_positions, name="anchors")

    # ── 2. Controller and 2D scene ──────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("y", "x"))
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=cs,
        name="paint_scene",
        render_modes={"2d"},
    )

    appearance = ImageMemoryAppearance(color_map="grays", clim=(0.0, 1.0))
    # ``add_image`` would default the visual's transform to identity(ndim=3),
    # which mismatches the 2-D world coordinate emitted by the canvas and
    # breaks ``transform.imap_coordinates`` inside the paint controller.
    # Build the visual explicitly with a 2-D identity transform.
    image_visual_model = ImageVisual(
        name="labels",
        data_store_id=str(image_store.id),
        appearance=appearance,
        transform=AffineTransform.identity(ndim=2),
    )
    image_visual = controller.add_visual(
        scene.id, image_visual_model, data_store=image_store
    )

    # Bright magenta points at ANCHOR_COORDS — should overlay each image
    # bright block exactly if the conventions match.  depth_test off and
    # high render_order so the points always draw on top of the image.
    points_appearance = PointsMarkerAppearance(
        color=(1.0, 0.0, 1.0, 1.0),
        size=20.0,
        size_space="screen",
        opacity=0.5,
        depth_test=False,
        render_order=100,
    )
    controller.add_points(
        data=points_store,
        scene_id=scene.id,
        appearance=points_appearance,
        name="anchors",
    )

    # ── 3. Qt window ────────────────────────────────────────────────────
    root = QWidget()
    root.setWindowTitle("In-memory paint demo")
    layout = QVBoxLayout(root)

    toolbar = QWidget(root)
    toolbar_layout = QHBoxLayout(toolbar)
    toolbar_layout.setContentsMargins(4, 4, 4, 4)

    brush_value_label = QLabel("brush value")
    brush_value_spin = QDoubleSpinBox()
    brush_value_spin.setRange(0.0, 1.0)
    brush_value_spin.setSingleStep(0.05)
    brush_value_spin.setValue(1.0)

    brush_radius_label = QLabel("radius (vox)")
    brush_radius_spin = QSpinBox()
    brush_radius_spin.setRange(1, 30)
    brush_radius_spin.setValue(12)

    undo_btn = QPushButton("Undo")
    commit_btn = QPushButton("Commit")
    abort_btn = QPushButton("Abort")

    for w in (
        brush_value_label,
        brush_value_spin,
        brush_radius_label,
        brush_radius_spin,
        undo_btn,
        commit_btn,
        abort_btn,
    ):
        toolbar_layout.addWidget(w)
    toolbar_layout.addStretch(1)
    layout.addWidget(toolbar)

    controller.set_widget_parent(root)
    canvas_widget = controller.add_canvas(scene_id=scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    layout.addWidget(canvas_widget)
    root.resize(700, 750)

    # ── 4. Camera fit on first delivery ──────────────────────────────────
    scene_mgr = controller._render_manager._scenes[scene.id]
    gfx_vis = scene_mgr.get_visual(image_visual.id)
    _fitted = False
    _orig = gfx_vis.on_data_ready_2d

    def _patched(batch):
        nonlocal _fitted
        _orig(batch)
        if not _fitted:
            _fitted = True
            controller.look_at_visual(
                image_visual.id, canvas_id, view_direction=(0, 0, -1), up=(0, 1, 0)
            )

    gfx_vis.on_data_ready_2d = _patched
    QTimer.singleShot(0, controller.reslice_all)

    # ── 5. Paint controller ──────────────────────────────────────────────
    paint_ctrl = controller.add_paint_controller(
        visual_id=image_visual.id,
        canvas_id=canvas_id,
        brush_value=float(brush_value_spin.value()),
        brush_radius_voxels=float(brush_radius_spin.value()),
    )

    canvas_view = controller._render_manager._canvases[canvas_id]

    def _enabled() -> bool:
        return bool(canvas_view._controller.enabled)

    def _on_brush_value_changed(value: float) -> None:
        print(f"[PAINT-DBG demo] brush_value before={_enabled()}")
        paint_ctrl.brush_value = float(value)
        print(f"[PAINT-DBG demo] brush_value after={_enabled()}")

    def _on_brush_radius_changed(value: int) -> None:
        print(f"[PAINT-DBG demo] brush_radius before={_enabled()}")
        paint_ctrl.brush_radius_voxels = float(value)
        print(f"[PAINT-DBG demo] brush_radius after={_enabled()}")

    def _on_undo_clicked() -> None:
        print(f"[PAINT-DBG demo] undo before={_enabled()}")
        paint_ctrl.undo()
        print(f"[PAINT-DBG demo] undo after={_enabled()}")

    def _on_commit_clicked() -> None:
        print(f"[PAINT-DBG demo] commit before={_enabled()}")
        paint_ctrl.commit()
        print(f"[PAINT-DBG demo] commit after={_enabled()}")
        QMessageBox.information(root, "Commit", "Stroke history cleared.")
        commit_btn.setEnabled(False)
        abort_btn.setEnabled(False)
        undo_btn.setEnabled(False)

    def _on_abort_clicked() -> None:
        print(f"[PAINT-DBG demo] abort before={_enabled()}")
        paint_ctrl.abort()
        print(f"[PAINT-DBG demo] abort after={_enabled()}")
        QMessageBox.information(root, "Abort", "All strokes reverted.")
        commit_btn.setEnabled(False)
        abort_btn.setEnabled(False)
        undo_btn.setEnabled(False)

    brush_value_spin.valueChanged.connect(_on_brush_value_changed)
    brush_radius_spin.valueChanged.connect(_on_brush_radius_changed)
    undo_btn.clicked.connect(_on_undo_clicked)
    commit_btn.clicked.connect(_on_commit_clicked)
    abort_btn.clicked.connect(_on_abort_clicked)

    root.show()
    QtAsyncio.run(handle_sigint=True)


if __name__ == "__main__":
    main()
