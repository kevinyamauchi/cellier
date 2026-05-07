"""Demo: MultiscalePaintController on a 2-D OME-Zarr label store.

Mirror of ``scripts/v2/in_memory_paint.py`` but backed by an OME-Zarr
data store on disk and using :class:`MultiscalePaintController`.

What happens when you paint
---------------------------
- ``_apply_brush`` stages the new voxel values into a tensorstore
  transaction (RAM only, owned by the WriteBuffer).
- The paint controller marks the dirty level-0 bricks and writes the
  values directly into the visual's GPU paint cache + paint LUT via
  ``controller.patch_painted_tiles_2d``.
- The image shader composites the paint cache over the base sample,
  so painted voxels are visible on the very next frame — no slicer
  round-trip required.

Autosave (every 30 seconds, when dirty)
----------------------------------------
- The WriteBuffer flushes the current transaction to disk.
- The pyramid is rebuilt bottom-up (s0 → s1 → s2) for dirty bricks
  only using stride-2 decimation — all writes share the same transaction
  so the whole pyramid is atomic.
- A fresh WriteBuffer (new transaction) replaces the old one for
  continued staging.  Undo history is preserved.
- The GPU paint textures are cleared and the base cache is evicted +
  resliced from disk.  All GPU paint slots are freed for the next
  interval.

Commit
-------
Same as autosave, but also clears undo history and tears down the
session (re-enables camera, unsubscribes events).  After commit, the
buttons are disabled — paint is done.

Abort
------
Discards the transaction without rebuilding the pyramid or touching
disk.  GPU paint textures are cleared; the base cache already shows
pre-paint data so no reslice is needed.

Verifying pyramid rebuild
--------------------------
1. Paint a visible stroke.
2. Click Commit (or wait for autosave).
3. Scroll-wheel to zoom out beyond the level-1 LOD threshold.
   The stroke should still be visible in the downsampled tiles.
4. Zoom further to level-2 tiles; the stroke should remain visible.
5. If the stroke disappears at coarser zoom, pyramid rebuild is broken.

The demo OME-Zarr persists between runs — paint, commit, quit,
re-launch and the strokes reappear.  Delete
``multiscale_paint_labels_pyramid.ome.zarr`` to start fresh.

Run with::

    uv run python scripts/v2/multiscale_paint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr
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
from cellier.v2.data.image import OMEZarrImageDataStore
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.transform import AffineTransform
from cellier.v2.visuals._image import (
    ImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageVisual,
)

_DEMO_BLOCK_SIZE = 32
_DEMO_CHUNK = (64, 64)

# Three-level pyramid: s0=256×256, s1=128×128, s2=64×64.
_LEVEL_SHAPES = [(256, 256), (128, 128), (64, 64)]
_LEVEL_SCALES = [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]


def _ensure_demo_ome_zarr(target_dir: Path) -> Path:
    """Return the demo OME-Zarr at *target_dir*, creating it if missing.

    The store persists between runs so you can verify that committed paint
    survives on disk.  To start fresh, delete the directory before running.
    """
    if target_dir.exists():
        print(f"  Loading existing demo OME-Zarr from {target_dir}")
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(target_dir), mode="w")

    datasets = []
    for i, (shape, scale) in enumerate(zip(_LEVEL_SHAPES, _LEVEL_SCALES)):
        path = f"s{i}"
        arr = root.create_array(path, shape=shape, chunks=_DEMO_CHUNK, dtype=np.float32)
        arr[:] = np.zeros(shape, dtype=np.float32)
        datasets.append(
            {
                "path": path,
                "coordinateTransformations": [{"type": "scale", "scale": scale}],
            }
        )

    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": datasets,
                "name": "demo_labels",
            }
        ],
    }
    print(f"  Created demo OME-Zarr (3 levels) at {target_dir}")
    return target_dir


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Demo data on disk ────────────────────────────────────────────
    zarr_path = (
        Path(__file__).resolve().parent / "multiscale_paint_labels_pyramid.ome.zarr"
    )
    _ensure_demo_ome_zarr(zarr_path)

    zarr_uri = f"file://{zarr_path.resolve()}"
    data_store = OMEZarrImageDataStore.from_path(zarr_uri, name="labels")
    print(
        f"  Opened data store: levels={data_store.n_levels} "
        f"dtype={data_store.dtype}"
    )
    for i, shape in enumerate(data_store.level_shapes):
        print(f"    s{i}: {shape}")

    # ── 2. Controller and 2-D scene ─────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("y", "x"))
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=cs,
        name="paint_scene",
        render_modes={"2d"},
    )

    appearance = ImageAppearance(color_map="grays", clim=(0.0, 1.0))
    render_config = MultiscaleImageRenderConfig(
        block_size=_DEMO_BLOCK_SIZE,
        gpu_budget_bytes_2d=64 * 1024**2,
        interpolation="nearest",
    )
    image_visual_model = MultiscaleImageVisual(
        name="labels",
        data_store_id=str(data_store.id),
        level_transforms=data_store.level_transforms,
        appearance=appearance,
        render_config=render_config,
        transform=AffineTransform.identity(ndim=2),
    )
    image_visual = controller.add_visual(
        scene.id, image_visual_model, data_store=data_store
    )

    # ── 3. Qt window ─────────────────────────────────────────────────────
    root = QWidget()
    root.setWindowTitle("Multiscale paint demo (OME-Zarr, pyramid rebuild)")
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
    brush_radius_spin.setValue(8)

    undo_btn = QPushButton("Undo")
    commit_btn = QPushButton("Commit")
    abort_btn = QPushButton("Abort")

    autosave_title = QLabel("Autosave:")
    autosave_label = QLabel("never")

    for w in (
        brush_value_label,
        brush_value_spin,
        brush_radius_label,
        brush_radius_spin,
        undo_btn,
        commit_btn,
        abort_btn,
        autosave_title,
        autosave_label,
    ):
        toolbar_layout.addWidget(w)
    toolbar_layout.addStretch(1)
    layout.addWidget(toolbar)

    controller.set_widget_parent(root)
    canvas_widget = controller.add_canvas(scene_id=scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    layout.addWidget(canvas_widget)
    root.resize(700, 750)

    # ── 4. Camera fit on first delivery ─────────────────────────────────
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
        autosave_interval_s=30.0,
        downsample_mode="decimate",
    )
    print(f"  Paint controller: {paint_ctrl!r}")

    canvas_view = controller.get_canvas_view(canvas_id)

    def _enabled() -> bool:
        return bool(canvas_view._controller.enabled)

    def _on_brush_value_changed(value: float) -> None:
        paint_ctrl.brush_value = float(value)

    def _on_brush_radius_changed(value: int) -> None:
        paint_ctrl.brush_radius_voxels = float(value)

    def _on_undo_clicked() -> None:
        paint_ctrl.undo()

    def _on_commit_clicked() -> None:
        paint_ctrl.commit()
        status_timer.stop()
        QMessageBox.information(
            root,
            "Commit",
            f"Paint and pyramid flushed to disk.\n"
            f"All {data_store.n_levels} LOD levels updated.\n\n"
            f"Zoom out to verify painted values appear at coarser zoom levels.\n\n"
            f"Path: {zarr_path}",
        )
        commit_btn.setEnabled(False)
        abort_btn.setEnabled(False)
        undo_btn.setEnabled(False)

    def _on_abort_clicked() -> None:
        paint_ctrl.abort()
        status_timer.stop()
        QMessageBox.information(
            root, "Abort", "All staged paint discarded; on-disk zarr unchanged."
        )
        commit_btn.setEnabled(False)
        abort_btn.setEnabled(False)
        undo_btn.setEnabled(False)

    # ── 6. 5-second status timer ─────────────────────────────────────────
    def _print_status() -> None:
        last_save = (
            paint_ctrl.last_autosave_time.strftime("%H:%M:%S")
            if paint_ctrl.last_autosave_time
            else "never"
        )
        if paint_ctrl.last_autosave_time:
            autosave_label.setText(f"{last_save} ({paint_ctrl.autosave_count} saves)")

    status_timer = QTimer()
    status_timer.setInterval(5000)
    status_timer.timeout.connect(_print_status)
    status_timer.start()

    brush_value_spin.valueChanged.connect(_on_brush_value_changed)
    brush_radius_spin.valueChanged.connect(_on_brush_radius_changed)
    undo_btn.clicked.connect(_on_undo_clicked)
    commit_btn.clicked.connect(_on_commit_clicked)
    abort_btn.clicked.connect(_on_abort_clicked)

    root.show()
    QtAsyncio.run(handle_sigint=True)


if __name__ == "__main__":
    main()
