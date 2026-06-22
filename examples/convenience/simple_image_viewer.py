"""Simple 3-D blob viewer using the cellier Viewer convenience class.

Creates a (200, 200, 200) float32 image from scikit-image blobs, displays it
with the viridis colormap, and provides a button to toggle between 2D and 3D.

Run::

    .venv/bin/python examples/convenience/simple_image_viewer.py

Controls
--------
2D mode : pan (left-drag), zoom (scroll)
3D mode : orbit (left-drag), zoom (scroll), pan (right-drag)
Toggle  : "Switch to 3D / 2D" button in the side panel
Z slice : dims slider (2D mode only)

Note on render modes
--------------------
The in-memory ``add_image`` path uses ``InMemoryImageAppearance``, which now
exposes ``render_mode`` (``"mip"``, ``"iso"``, or ``"minip"``) and
``iso_threshold``.  This example renders the 3D view as an isosurface
(``render_mode="iso"``).
"""

from __future__ import annotations

import sys

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.data import binary_blobs

from cellier.convenience import (
    Viewer,
    axis_ranges_from_viewer,
    launch,
)
from cellier.convenience.gui import build_canvas_widget
from cellier.data.image._image_memory_store import ImageMemoryStore

# QApplication must exist before any QWidget is constructed
app = QApplication.instance() or QApplication(sys.argv)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

blobs_3d = binary_blobs(length=200, n_dim=3, rng=0).astype(np.float32)

# ---------------------------------------------------------------------------
# Viewer model
# ---------------------------------------------------------------------------

viewer = Viewer(axis_labels=("z", "y", "x"), dim="2d")

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

# ---------------------------------------------------------------------------
# Axis ranges and canvas widget
# ---------------------------------------------------------------------------

axis_ranges = axis_ranges_from_viewer(viewer)
canvas_widget = build_canvas_widget(viewer, axis_ranges)

# ---------------------------------------------------------------------------
# Qt window
# ---------------------------------------------------------------------------

app_window = QMainWindow()
app_window.setWindowTitle("Blob Viewer")
app_window.resize(1000, 700)

central = QWidget()
root_layout = QHBoxLayout(central)
app_window.setCentralWidget(central)

# Side panel
panel = QWidget()
panel.setFixedWidth(220)
panel_layout = QVBoxLayout(panel)
panel_layout.setContentsMargins(8, 8, 8, 8)

toggle_btn = QPushButton("Switch to 3D")
panel_layout.addWidget(toggle_btn)
panel_layout.addStretch()

root_layout.addWidget(panel)
root_layout.addWidget(canvas_widget.widget, stretch=1)

# ---------------------------------------------------------------------------
# 2D / 3D toggle
# ---------------------------------------------------------------------------

_mode = ["2d"]


def _on_toggle() -> None:
    # set_displayed_dimensions fits the camera to the new view itself, so no
    # explicit fit_camera call is needed here.
    if _mode[0] == "2d":
        viewer.set_displayed_dimensions(("z", "y", "x"))
        toggle_btn.setText("Switch to 2D")
        _mode[0] = "3d"
    else:
        viewer.set_displayed_dimensions(("y", "x"))
        toggle_btn.setText("Switch to 3D")
        _mode[0] = "2d"


toggle_btn.clicked.connect(_on_toggle)

# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------
# launch() fits the camera and loads the data on the canvas's first frame, then
# re-fits once everything is on the GPU (fit="ready", the default). on_ready
# fires once at that point -- a robust, timer-free hook that works for both
# in-memory and multiscale visuals. Here we just flag it in the title bar.


def _on_ready() -> None:
    app_window.setWindowTitle("Blob Viewer (loaded)")


viewer.on_ready(_on_ready)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

launch(viewer, app_window)
