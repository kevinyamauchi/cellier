"""A 2D/3D toggle button as a small standalone anywidget.

Unlike the bus-wired controls, the toggle drives a convenience *method*
(``viewer.set_displayed_dimensions``) rather than a bus field, so it
legitimately holds the viewer reference -- exactly as the Qt example's button
handler does (design doc section 7, control #10).  The ``LayoutHost`` places it
alongside the panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anywidget
import traitlets

if TYPE_CHECKING:
    from cellier.convenience._viewer import Viewer

_STATIC = Path(__file__).parent / "static"


class _DimToggle(anywidget.AnyWidget):
    """Button anywidget that toggles a viewer between 2D and 3D."""

    _esm = _STATIC / "toggle.js"

    label = traitlets.Unicode("Switch to 3D").tag(sync=True)
    # Incremented by the JS click handler; observed on the Python side.
    _clicks = traitlets.Int(0).tag(sync=True)

    def __init__(
        self,
        viewer: Viewer,
        axes_2d: tuple[str, ...],
        axes_3d: tuple[str, ...],
        *,
        start_3d: bool,
    ) -> None:
        super().__init__(label="Switch to 2D" if start_3d else "Switch to 3D")
        self._viewer = viewer
        self._axes_2d = axes_2d
        self._axes_3d = axes_3d
        self._is_3d = start_3d
        self.observe(self._on_click, names="_clicks")

    def _on_click(self, _change) -> None:
        # set_displayed_dimensions fits the camera to the new view itself.
        if self._is_3d:
            self._viewer.set_displayed_dimensions(self._axes_2d)
            self._is_3d = False
            self.label = "Switch to 3D"
        else:
            self._viewer.set_displayed_dimensions(self._axes_3d)
            self._is_3d = True
            self.label = "Switch to 2D"


def make_dim_toggle(viewer: Viewer) -> _DimToggle:
    """Return a 2D/3D toggle button bound to *viewer*.

    The two spatial views are derived from the scene's coordinate system: the
    3D view displays the last three axis labels and the 2D view the last two,
    matching the convention used by the example scripts.

    Parameters
    ----------
    viewer : Viewer
        The viewer whose displayed dimensions the button toggles.

    Returns
    -------
    _DimToggle
        A small anywidget button; place it via ``display(..., controls=[...])``.
    """
    axis_labels = viewer.scene.dims.coordinate_system.axis_labels
    if len(axis_labels) < 3:
        raise ValueError(
            "make_dim_toggle requires at least 3 axes to toggle 2D<->3D; "
            f"got {axis_labels!r}."
        )
    axes_3d = tuple(axis_labels[-3:])
    axes_2d = tuple(axis_labels[-2:])
    start_3d = len(viewer.scene.dims.selection.displayed_axes) == 3
    return _DimToggle(viewer, axes_2d, axes_3d, start_3d=start_3d)
