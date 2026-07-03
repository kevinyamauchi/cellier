"""cellier v2 GUI widgets.

The Qt widgets live under :mod:`cellier.gui.qt`; the anywidget (notebook)
widgets live under :mod:`cellier.gui.anywidget`.  ``QtCanvasWidget`` and
``QtDimsSliders`` are re-exported here for back-compat.
"""

from cellier.gui.qt import QtCanvasWidget, QtDimsSliders

__all__ = ["QtCanvasWidget", "QtDimsSliders"]
