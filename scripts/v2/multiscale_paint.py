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

Click **Commit** to flush staged paints to the on-disk zarr (then
the visible base cache is repopulated from disk on the post-commit
reslice, and the GPU paint textures are cleared).  Click **Abort**
to discard staged paint; the GPU paint textures are cleared and the
on-disk zarr is untouched.

The OME-Zarr store lives next to this script (``multiscale_paint_labels.ome.zarr``)
and persists across runs so you can confirm commits actually reach
disk: paint, commit, quit, re-launch — your strokes should reappear.
Delete the directory to start from a blank canvas.

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

_DEMO_SHAPE = (256, 256)  # (y, x) — single-level 2-D float32 store
_DEMO_BLOCK_SIZE = 32
_DEMO_CHUNK = (64, 64)


def _ensure_demo_ome_zarr(target_dir: Path) -> Path:
    """Return the demo OME-Zarr at *target_dir*, creating it if missing.

    The store lives next to this script so commits persist between runs:
    if the directory already exists, it is loaded as-is so you can verify
    that previously committed paint survives on disk.  To start fresh,
    delete the directory before running.
    """
    if target_dir.exists():
        print(f"  Loading existing demo OME-Zarr from {target_dir}")
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(target_dir), mode="w")
    arr = root.create_array(
        "s0",
        shape=_DEMO_SHAPE,
        chunks=_DEMO_CHUNK,
        dtype=np.float32,
    )
    arr[:] = np.zeros(_DEMO_SHAPE, dtype=np.float32)

    # OME-NGFF v0.5 metadata.  Only y and x — 2-D.
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "s0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0]}
                        ],
                    }
                ],
                "name": "demo_labels",
            }
        ],
    }
    print(f"  Created demo OME-Zarr at {target_dir}")
    return target_dir


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Demo data on disk ────────────────────────────────────────────
    # The store lives next to this script and persists between runs so
    # you can verify that committed paint actually reaches disk: paint +
    # commit, quit, re-run — the painted voxels should reappear.  Delete
    # the directory by hand if you want to start fresh.
    zarr_path = Path(__file__).resolve().parent / "multiscale_paint_labels.ome.zarr"
    _ensure_demo_ome_zarr(zarr_path)

    # OMEZarrImageDataStore.from_path requires a file:// URI.
    zarr_uri = f"file://{zarr_path.resolve()}"
    data_store = OMEZarrImageDataStore.from_path(zarr_uri, name="labels")
    print(
        f"  Opened data store: levels={data_store.n_levels} "
        f"shape={data_store.level_shapes[0]} dtype={data_store.dtype}"
    )

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
        use_brick_shader=False,
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
    root.setWindowTitle("Multiscale paint demo (OME-Zarr)")
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
    )
    print(f"  Paint controller: {paint_ctrl!r}")

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
        QMessageBox.information(
            root,
            "Commit",
            f"Stroke history cleared and paint flushed to:\n{zarr_path}",
        )
        commit_btn.setEnabled(False)
        abort_btn.setEnabled(False)
        undo_btn.setEnabled(False)

    def _on_abort_clicked() -> None:
        print(f"[PAINT-DBG demo] abort before={_enabled()}")
        paint_ctrl.abort()
        print(f"[PAINT-DBG demo] abort after={_enabled()}")
        QMessageBox.information(
            root, "Abort", "All staged paint discarded; on-disk zarr unchanged."
        )
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
