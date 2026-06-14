"""Four-panel grid widget builder for the cellier OrthoViewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cellier.convenience._ortho_viewer import OrthoViewer
    from cellier.gui import QtCanvasWidget

# (row, col, panel_key, header) for the 2x2 layout.
_PANEL_LAYOUT: tuple[tuple[int, int, str, str], ...] = (
    (0, 0, "xy", "XY"),
    (0, 1, "xz", "XZ"),
    (1, 0, "yz", "YZ"),
    (1, 1, "vol", "3D"),
)


class OrthoCanvasWidgets:
    """The four panel canvas widgets plus the composed 2x2 grid container.

    Attributes
    ----------
    widget :
        The outer ``QWidget`` containing the 2x2 grid; embed in a Qt layout.
    canvases : dict[str, QtCanvasWidget]
        The per-panel ``QtCanvasWidget`` objects keyed ``"xy"``, ``"xz"``,
        ``"yz"``, ``"vol"``.
    """

    def __init__(self, widget, canvases: dict[str, QtCanvasWidget]) -> None:
        self.widget = widget
        self.canvases = canvases

    def close(self) -> None:
        """Unsubscribe every panel's dims sliders from the bus."""
        for canvas_widget in self.canvases.values():
            canvas_widget.close()


def build_ortho_grid_widget(
    ortho: OrthoViewer,
    axis_ranges: dict[int, tuple[float, float]],
    *,
    backend: Literal["qt"] = "qt",
    fov: float = 70.0,
    depth_range_3d: tuple[float, float] = (1.0, 8000.0),
    depth_range_2d: tuple[float, float] = (-500.0, 500.0),
) -> OrthoCanvasWidgets:
    """Build the 2x2 canvas grid for an :class:`OrthoViewer`.

    Creates (or reuses) a canvas per panel, wraps each in a ``QtCanvasWidget``
    with wired dims sliders, and arranges them in a labelled 2x2 grid: ``XY``
    and ``XZ`` on the top row, ``YZ`` and the ``3D`` volume on the bottom.

    Parameters
    ----------
    ortho : OrthoViewer
        The orthoviewer whose four scenes are attached.
    axis_ranges : dict[int, tuple[float, float]]
        Mapping of axis index to ``(world_min, world_max)`` for slider ranges,
        typically from
        :func:`cellier.convenience.axis_ranges_from_ortho`.
    backend : "qt"
        Rendering backend.  Only ``"qt"`` is currently supported.
    fov : float
        Vertical field of view in degrees for the 3D camera.  Default ``70``.
    depth_range_3d : tuple[float, float]
        ``(near, far)`` clip distances for the 3D camera.
    depth_range_2d : tuple[float, float]
        ``(near, far)`` clip distances for the 2D cameras.

    Returns
    -------
    OrthoCanvasWidgets

    Raises
    ------
    ValueError
        If *backend* is not ``"qt"``.
    """
    if backend != "qt":
        raise ValueError(f"Unknown backend {backend!r}. Only 'qt' is supported.")
    return _build_qt_ortho_grid(
        ortho,
        axis_ranges,
        fov=fov,
        depth_range_3d=depth_range_3d,
        depth_range_2d=depth_range_2d,
    )


def _build_qt_ortho_grid(
    ortho: OrthoViewer,
    axis_ranges: dict[int, tuple[float, float]],
    *,
    fov: float,
    depth_range_3d: tuple[float, float],
    depth_range_2d: tuple[float, float],
) -> OrthoCanvasWidgets:
    """Qt implementation of :func:`build_ortho_grid_widget`."""
    from PySide6 import QtCore, QtWidgets

    from cellier.convenience.gui._canvas import canvas_widget_for_scene

    canvases: dict[str, QtCanvasWidget] = {}
    for key, scene in ortho.scenes.items():
        canvases[key] = canvas_widget_for_scene(
            ortho.controller,
            scene,
            axis_ranges,
            fov=fov,
            depth_range_3d=depth_range_3d,
            depth_range_2d=depth_range_2d,
        )

    grid_widget = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(grid_widget)
    grid.setSpacing(4)
    grid.setContentsMargins(0, 0, 0, 0)

    for row, col, key, header in _PANEL_LAYOUT:
        cell = QtWidgets.QWidget()
        cell_layout = QtWidgets.QVBoxLayout(cell)
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(0)

        label = QtWidgets.QLabel(header)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-weight: bold; font-size: 11px; padding: 2px;")
        cell_layout.addWidget(label)
        cell_layout.addWidget(canvases[key].widget, stretch=1)
        grid.addWidget(cell, row, col)

    return OrthoCanvasWidgets(grid_widget, canvases)
