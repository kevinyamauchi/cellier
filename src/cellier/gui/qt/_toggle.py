"""A 2D/3D toggle button as a Qt widget.

Mirrors the anywidget ``_DimToggle`` for the Qt backend.  Drives
``viewer.set_displayed_dimensions`` directly rather than the bus -- exactly as
the anywidget version does (design doc section 7, control #10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QPushButton

if TYPE_CHECKING:
    from cellier.convenience._viewer import Viewer


class QtDimToggle(QPushButton):
    """QPushButton that toggles a viewer between 2D and 3D.

    Mirrors the behaviour of the anywidget ``_DimToggle``.  Derives the two
    axis sets from the scene coordinate system at construction time.
    """

    def __init__(
        self,
        viewer: Viewer,
        axes_2d: tuple[str, ...],
        axes_3d: tuple[str, ...],
        *,
        start_3d: bool,
        parent=None,
    ) -> None:
        super().__init__("Switch to 2D" if start_3d else "Switch to 3D", parent)
        self._viewer = viewer
        self._axes_2d = axes_2d
        self._axes_3d = axes_3d
        self._is_3d = start_3d
        self.clicked.connect(self._on_click)

    def _on_click(self) -> None:
        if self._is_3d:
            self._viewer.set_displayed_dimensions(self._axes_2d)
            self._is_3d = False
            self.setText("Switch to 3D")
        else:
            self._viewer.set_displayed_dimensions(self._axes_3d)
            self._is_3d = True
            self.setText("Switch to 2D")


def make_dim_toggle_qt(viewer: Viewer, *, parent=None) -> QtDimToggle:
    """Return a 2D/3D toggle ``QPushButton`` bound to *viewer*.

    Derives axes from the scene's coordinate system: 3D displays the last three
    axis labels, 2D the last two -- matching the convention used by the example
    scripts and the anywidget :func:`make_dim_toggle`.

    Parameters
    ----------
    viewer : Viewer
        The viewer whose displayed dimensions the button toggles.
    parent : QWidget or None
        Optional Qt parent widget.

    Returns
    -------
    QtDimToggle
    """
    axis_labels = viewer.scene.dims.coordinate_system.axis_labels
    if len(axis_labels) < 3:
        raise ValueError(
            "make_dim_toggle_qt requires at least 3 axes to toggle 2D<->3D; "
            f"got {axis_labels!r}."
        )
    axes_3d = tuple(axis_labels[-3:])
    axes_2d = tuple(axis_labels[-2:])
    start_3d = len(viewer.scene.dims.selection.displayed_axes) == 3
    return QtDimToggle(viewer, axes_2d, axes_3d, start_3d=start_3d, parent=parent)
