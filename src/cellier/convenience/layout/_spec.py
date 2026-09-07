"""Host-agnostic layout specification (the model layer).

These dataclasses describe the structure of a viewer layout without any
host-specific rendering logic.  The renderer for each host (anywidget or Qt)
reads the spec and produces the appropriate widget tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class HStack:
    """Horizontal stack of center items or control specs."""

    items: list


@dataclass
class VStack:
    """Vertical stack of center items or control specs."""

    items: list


@dataclass
class Grid:
    """2-D grid of center items.

    Parameters
    ----------
    cells : list[list]
        Row-major grid; ``None`` leaves a cell empty.
    """

    cells: list[list]


@dataclass
class AppearanceControls:
    """Dock spec: appearance control panel for the first configured visual."""


@dataclass
class ChannelControls:
    """Dock spec: per-channel controls for the first configured channel visual.

    Renders a channel-controls widget (Qt ``QtChannelList`` / anywidget
    ``AnywidgetChannelList``) for the multichannel visual configured via
    ``controls=`` on ``add_multichannel_image[_multiscale]``.  For an
    ``OrthoViewer`` the one widget drives every panel's sibling visual.
    """


@dataclass
class RenderControls:
    """Dock spec: one panel per renderer post-processing feature.

    Renders the Qt (``cellier.gui.qt.render``) or anywidget
    (``cellier.gui.anywidget.render``) panel for each named section, wired
    to the viewer's controller.  Unlike :class:`AppearanceControls` this
    needs no configured visual: render settings belong to the renderer, so
    the dock is available on any viewer.

    Parameters
    ----------
    sections : tuple[str, ...]
        Which panels to show, in order.  Any of ``"outline"``, ``"ambient_occlusion"``
        and ``"temporal"``.  Defaults to all three.
    """

    sections: tuple[str, ...] = ("ambient_occlusion", "outline", "temporal")


@dataclass
class Layout:
    """Full layout specification (the model).

    Describes what goes in the center region and each optional dock.  Pass to
    :func:`~cellier.convenience.display` (anywidget) or
    :func:`~cellier.convenience.launch` / :func:`~cellier.convenience.show`
    (Qt) to render.

    Parameters
    ----------
    center : canvas view or HStack or VStack or Grid
        Main content.  Typically a single ``AnywidgetCanvasView`` /
        ``QtCanvasWidget``, or a composed layout of multiple canvas views.
        The 2D/3D toggle is part of the dims control embedded in the canvas
        view, so it does not need a dock of its own.
    left_dock, right_dock, top_dock, bottom_dock :
        Content for each dock region.  Accepts :class:`AppearanceControls`,
        :class:`ChannelControls`, :class:`RenderControls`, or a stack of
        those.  ``None`` hides the dock.
    """

    center: object
    left_dock: object = None
    right_dock: object = None
    top_dock: object = None
    bottom_dock: object = None

    @classmethod
    def single(
        cls,
        canvas,
        *,
        appearance: Literal["left", "right", "top", "bottom"] | bool = False,
        channels: Literal["left", "right", "top", "bottom"] | bool = False,
        render: Literal["left", "right", "top", "bottom"] | bool = False,
    ) -> Layout:
        """Single-canvas preset.

        Parameters
        ----------
        canvas :
            Canvas view returned by ``build_canvas_widget``.
        appearance : dock name or False
            Where to place appearance controls.  ``False`` (default) omits them.
        channels : dock name or False
            Where to place per-channel controls.  ``False`` (default) omits
            them.
        render : dock name or False
            Where to place the renderer settings panels (outlines, ambient
            occlusion, temporal accumulation).  ``False`` (default) omits
            them.
        """
        docks: dict[str, object] = {}
        if appearance:
            docks[f"{appearance}_dock"] = AppearanceControls()
        if channels:
            docks[f"{channels}_dock"] = ChannelControls()
        if render:
            existing = docks.get(f"{render}_dock")
            panel = RenderControls()
            docks[f"{render}_dock"] = (
                VStack(items=[existing, panel]) if existing is not None else panel
            )
        return cls(center=canvas, **docks)
