"""Jupyter manager-rendered anywidget container.

Jupyter has no native multi-pane layout without a container that renders child
widget views.  To keep stock ipywidgets controls out, the container is itself a
tiny anywidget that asks the widget manager to mount its children (design doc
section 10).

This is the one place the anywidget path touches the ipywidgets layer
(``widget_manager.create_view``, ``ipywidgets.widget_serialization``,
``DOMWidget`` as the child type); marimo avoids it entirely via
``mo.vstack`` / ``mo.hstack``.
"""

from __future__ import annotations

from pathlib import Path

import anywidget
import ipywidgets
import traitlets

_STATIC = Path(__file__).parent / "static"


class AnywidgetBox(anywidget.AnyWidget):
    """A flexbox container that mounts child widget views via the manager.

    Parameters
    ----------
    children : list of ipywidgets.DOMWidget
        The child widgets to mount, in order.
    direction : "v" or "h"
        Stacking direction: ``"v"`` (column, default) or ``"h"`` (row).
    align : str
        Cross-axis alignment applied as CSS ``align-items`` (e.g. ``"center"``);
        empty string leaves the flexbox default (``stretch``).
    min_width : int
        When set (> 0), the box grows to fill available space but never
        narrower than this many pixels (``flex: 1 1 <min_width>px``); ``0``
        leaves the flexbox default (content-sized, no grow).
    gap : int
        Spacing between children in pixels.  Defaults to ``4``, tuned for
        macro layout blocks (canvas/dims/docks).  Pass a smaller value to
        tightly group sibling controls that used to live inside one widget
        (see ``compose_appearance_leaf``).
    padding : int
        Inner padding in pixels, on all four sides.  Defaults to ``0``.  Used
        by :class:`~cellier.convenience._hosts.JupyterHost` to keep the
        outermost box from touching the notebook cell / sidecar tab edges;
        nested boxes leave this at ``0`` so only the outer border shows.
    """

    _esm = _STATIC / "container.js"
    _css = _STATIC / "container.css"

    children = traitlets.List(traitlets.Instance(ipywidgets.DOMWidget)).tag(
        sync=True, **ipywidgets.widget_serialization
    )
    direction = traitlets.Unicode("v").tag(sync=True)
    align = traitlets.Unicode("").tag(sync=True)
    min_width = traitlets.Int(0).tag(sync=True)
    gap = traitlets.Int(4).tag(sync=True)
    padding = traitlets.Int(0).tag(sync=True)
