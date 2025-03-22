from dataclasses import dataclass
from typing import Any

import numpy as np

from cellier.types import (
    CanvasId,
    MouseButton,
    MouseEventType,
    MouseModifiers,
    VisualId,
)


@dataclass(frozen=True)
class RendererCanvasMouseEvent:
    """The data from a mouse event.

    This is emitted by the RenderManager and used
    internally by Cellier. It serves to provide a standardized
    input into the MouseEventBus.


    Parameters
    ----------
    source_id CanvasId
        The id of the visual that was clicked on.
    type : MouseEventType
        The type of the event (press, release, move).
    coordinate : np.ndarray
        The coordinate of the click in the displayed world coordinate system.
    button : MouseButton
        The button that was clicked.
    modifiers : list[MouseModifiers]
        The keyboard modifiers that were pressed at the time of the click.
    pick_info : dict[str, Any]
        The picking information from the renderer.
    """

    source_id: CanvasId
    type: MouseEventType
    coordinate: np.ndarray
    button: "MouseButton"
    modifiers: list["MouseModifiers"]
    pick_info: dict[str, Any]


@dataclass(frozen=True)
class RendererVisualMouseEvent:
    """The data from a visual mouse event.

    This is emitted by the RenderManager and used
    internally by Cellier. It serves to provide a standardized
    input into the MouseEventBus.

    Parameters
    ----------
    source_id : VisualId
        The id of the visual that was clicked on.
    coordinate : np.ndarray
        The coordinate of the click in the displayed world coordinate system.
    type : MouseEventType
        The type of the event (press, release, move).
    button : MouseButton
        The button that was clicked.
    modifiers : list[MouseModifiers]
        The keyboard modifiers that were pressed at the time of the click.
    pick_info : dict[str, Any]
        The picking information from the
    """

    source_id: VisualId
    coordinate: np.ndarray
    type: MouseEventType
    button: "MouseButton"
    modifiers: list["MouseModifiers"]
    pick_info: dict[str, Any]
