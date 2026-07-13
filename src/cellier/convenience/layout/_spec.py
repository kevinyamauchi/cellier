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
class SceneControls:
    """Dock spec: scene-level controls (2D/3D dimension toggle)."""


@dataclass
class ChannelControls:
    """Dock spec: per-channel controls for the first configured channel visual.

    Renders a channel-controls widget (Qt ``QtChannelList`` / anywidget
    ``ChannelPanel``) for the multichannel visual configured via
    ``controls=`` on ``add_multichannel_image[_multiscale]``.  For an
    ``OrthoViewer`` the one widget drives every panel's sibling visual.
    """


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
    left_dock, right_dock, top_dock, bottom_dock :
        Content for each dock region.  Accepts :class:`AppearanceControls`,
        :class:`SceneControls`, or a stack of those.  ``None`` hides the dock.
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
        scene_controls: Literal["left", "right", "top", "bottom"] | bool = False,
        channels: Literal["left", "right", "top", "bottom"] | bool = False,
    ) -> Layout:
        """Single-canvas preset.

        Parameters
        ----------
        canvas :
            Canvas view returned by ``build_canvas_widget``.
        appearance : dock name or False
            Where to place appearance controls.  ``False`` (default) omits them.
        scene_controls : dock name or False
            Where to place scene controls (2D/3D toggle).  ``False`` omits them.
        channels : dock name or False
            Where to place per-channel controls.  ``False`` (default) omits
            them.
        """
        docks: dict[str, object] = {}
        if appearance:
            docks[f"{appearance}_dock"] = AppearanceControls()
        if scene_controls:
            docks[f"{scene_controls}_dock"] = SceneControls()
        if channels:
            docks[f"{channels}_dock"] = ChannelControls()
        return cls(center=canvas, **docks)
