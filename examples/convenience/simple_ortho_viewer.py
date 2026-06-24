"""Simple 4-panel orthoviewer using the cellier OrthoViewer convenience class.

Creates a (200, 200, 200) float32 image from scikit-image blobs and displays it
in a four-panel orthoviewer: three orthogonal slice planes (XY, XZ, YZ) and one
3D volume.  A single ``add_image`` call fans the visual out to all four panels.
The 3D panel renders the volume as an isosurface (``render_mode="iso"``).

Run::

    .venv/bin/python examples/convenience/simple_ortho_viewer.py

Controls
--------
2D panels : pan (left-drag), zoom (scroll); slice slider below each canvas
3D panel  : orbit (left-drag), zoom (scroll), pan (right-drag)
"""

from __future__ import annotations

import sys

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QWidget,
)
from skimage.data import binary_blobs

from cellier.convenience import (
    OrthoViewer,
    axis_ranges_from_ortho,
    launch,
)
from cellier.convenience.gui import build_ortho_grid_widget
from cellier.data.image._image_memory_store import ImageMemoryStore

# QApplication must exist before any QWidget is constructed.
app = QApplication.instance() or QApplication(sys.argv)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

blobs_3d = binary_blobs(length=200, n_dim=3, rng=0).astype(np.float32)

# ---------------------------------------------------------------------------
# Orthoviewer model: launches empty, then we add data.
# ---------------------------------------------------------------------------

viewer = OrthoViewer(axis_labels=("z", "y", "x"))

store = ImageMemoryStore(data=blobs_3d, name="blobs")
viewer.controller.add_data_store(store)

viewer.add_image(
    store,
    appearance={
        "color_map": "viridis",
        "clim": (0.0, 1.0),
        "render_mode": "iso",
        "iso_threshold": 0.5,
    },
    name="blobs",
)

# Center the 2D slice planes in the middle of the volume.
viewer.center_slices()

# ---------------------------------------------------------------------------
# Axis ranges and the 2x2 canvas grid
# ---------------------------------------------------------------------------

axis_ranges = axis_ranges_from_ortho(viewer)
canvas_widgets = build_ortho_grid_widget(viewer, axis_ranges)

# ---------------------------------------------------------------------------
# Qt window
# ---------------------------------------------------------------------------

app_window = QMainWindow()
app_window.setWindowTitle("Blob Orthoviewer")
app_window.resize(1200, 900)

central = QWidget()
root_layout = QHBoxLayout(central)
app_window.setCentralWidget(central)
root_layout.addWidget(canvas_widgets.widget, stretch=1)

# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------
# For the orthoviewer, on_ready fires once after *all four* panel scenes have
# loaded their data onto the GPU. launch() fits each panel on its first frame
# and re-fits on ready (fit="ready", the default), so no manual fit_camera or
# timer is needed. Here we just flag it in the title bar.


def _on_ready() -> None:
    app_window.setWindowTitle("Blob Orthoviewer (loaded)")


viewer.on_ready(_on_ready)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

launch(viewer, app_window, fit="ready")
