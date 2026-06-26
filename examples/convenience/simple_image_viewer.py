"""Simple 3-D blob viewer using the cellier Viewer convenience class.

Creates a (200, 200, 200) float32 image from scikit-image blobs, displays it
with the viridis colormap, and provides a button to toggle between 2D and 3D.

Run::

    .venv/bin/python examples/convenience/simple_image_viewer.py

Controls
--------
2D mode : pan (left-drag), zoom (scroll)
3D mode : orbit (left-drag), zoom (scroll), pan (right-drag)
Toggle  : "Switch to 3D / 2D" button in the bottom dock
Z slice : dims slider (2D mode only)

Note on render modes
--------------------
The in-memory ``add_image`` path uses ``InMemoryImageAppearance``, which
exposes ``render_mode`` (``"mip"``, ``"iso"``, or ``"minip"``) and
``iso_threshold``.  This example renders the 3D view as an isosurface
(``render_mode="iso"``).
"""

from __future__ import annotations

import numpy as np
from skimage.data import binary_blobs

from cellier.convenience import (
    Layout,
    Viewer,
    axis_ranges_from_viewer,
    run,
)
from cellier.convenience.gui import build_canvas_widget
from cellier.data.image._image_memory_store import ImageMemoryStore

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
# Canvas + layout
# ---------------------------------------------------------------------------

axis_ranges = axis_ranges_from_viewer(viewer)
canvas_widget = build_canvas_widget(viewer, axis_ranges)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

run(viewer, Layout.single(canvas_widget, scene_controls="bottom"), fit="ready")
