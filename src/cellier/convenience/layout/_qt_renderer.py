"""Qt renderer -- the view layer for Layout specs on the Qt backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.convenience.layout._spec import Layout


def render_qt(layout: Layout, viewer: object) -> object:
    """Render a Layout spec to a ``QMainWindow``.

    Builds the center widget from *layout.center* using Qt layout primitives,
    sets it as the central widget, and wraps each non-None dock spec in a
    ``QDockWidget``.

    Parameters
    ----------
    layout : Layout
        The layout spec to render.
    viewer :
        The viewer, reserved for future scene-level control building.

    Returns
    -------
    QMainWindow
    """
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt

    window = QtWidgets.QMainWindow()
    window.setCentralWidget(_render_center_qt(layout.center))

    dock_map = {
        "left": (layout.left_dock, Qt.DockWidgetArea.LeftDockWidgetArea),
        "right": (layout.right_dock, Qt.DockWidgetArea.RightDockWidgetArea),
        "top": (layout.top_dock, Qt.DockWidgetArea.TopDockWidgetArea),
        "bottom": (layout.bottom_dock, Qt.DockWidgetArea.BottomDockWidgetArea),
    }
    for name, (spec, area) in dock_map.items():
        widget = _render_dock_qt(spec, viewer)
        if widget is not None:
            widget = _wrap_dock_widget(widget, name)
            dock = QtWidgets.QDockWidget(name.capitalize(), window)
            dock.setWidget(widget)
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )
            window.addDockWidget(area, dock)

    from PySide6.QtWidgets import QApplication

    screen = QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        w = min(int(available.width() * 2 / 3), 1600)
        h = min(int(available.height() * 2 / 3), 1000)
        window.resize(w, h)

    return window


def _wrap_dock_widget(widget: object, position: str) -> object:
    """Center *widget* in a stretch container sized for *position*.

    Top/bottom docks: horizontal container (stretch | widget | stretch).
    Left/right docks: vertical container (stretch / widget / stretch).
    """
    from PySide6 import QtWidgets

    container = QtWidgets.QWidget()
    if position in ("top", "bottom"):
        box = QtWidgets.QHBoxLayout(container)
    else:
        box = QtWidgets.QVBoxLayout(container)
    box.setContentsMargins(4, 4, 4, 4)
    box.addStretch()
    box.addWidget(widget)
    box.addStretch()
    return container


def _render_center_qt(node: object) -> object:
    """Recursively render a center spec node to a Qt widget."""
    from PySide6 import QtWidgets

    from cellier.convenience.layout._spec import Grid, HStack, VStack

    if isinstance(node, HStack):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for item in node.items:
            layout.addWidget(_render_center_qt(item))
        return container

    if isinstance(node, VStack):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for item in node.items:
            layout.addWidget(_render_center_qt(item))
        return container

    if isinstance(node, Grid):
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        for row_idx, row in enumerate(node.cells):
            for col_idx, cell in enumerate(row):
                if cell is not None:
                    grid.addWidget(_render_center_qt(cell), row_idx, col_idx)
        return container

    # Leaf: QtCanvasWidget or OrthoCanvasWidgets -- both expose .widget.
    if hasattr(node, "widget"):
        return node.widget
    raise TypeError(
        f"Cannot render {type(node).__name__!r} as a Qt center widget. "
        "Expected a QtCanvasWidget, HStack, VStack, or Grid."
    )


def _render_scene_controls_qt(viewer: object) -> object | None:
    """Build a Qt scene controls widget (2D/3D dim toggle)."""
    from cellier.gui.qt._toggle import make_dim_toggle_qt

    return make_dim_toggle_qt(viewer)


def _render_dock_qt(spec: object, viewer: object) -> object | None:
    """Render one dock spec to a Qt widget, or return None."""
    if spec is None:
        return None

    from PySide6 import QtWidgets

    from cellier.convenience.layout._spec import (
        AppearanceControls,
        HStack,
        SceneControls,
        VStack,
    )

    if isinstance(spec, AppearanceControls):
        return None  # Qt appearance controls not yet implemented
    if isinstance(spec, SceneControls):
        return _render_scene_controls_qt(viewer)
    if isinstance(spec, (HStack, VStack)):
        container = QtWidgets.QWidget()
        box = (
            QtWidgets.QHBoxLayout(container)
            if isinstance(spec, HStack)
            else QtWidgets.QVBoxLayout(container)
        )
        box.setContentsMargins(4, 4, 4, 4)
        for item in spec.items:
            widget = _render_dock_qt(item, viewer)
            if widget is not None:
                box.addWidget(widget)
        return container
    return None
