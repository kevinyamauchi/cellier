"""Types used in the Cellier package."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeAlias, Union

import numpy as np
from pydantic import Field
from typing_extensions import Annotated

from cellier.models.visuals import LinesVisual, MultiscaleLabelsVisual, PointsVisual

# This is used for a discriminated union for typing the visual models
VisualType = Annotated[
    Union[LinesVisual, PointsVisual, MultiscaleLabelsVisual],
    Field(discriminator="visual_type"),
]

# The unique identifier for a DimsManager model
DimsId: TypeAlias = str

# The unique identifier for a Visual model
VisualId: TypeAlias = str

# The unique identifier for a Scene model
SceneId: TypeAlias = str

# The unique identifier for a Canvas model
CanvasId: TypeAlias = str

# The unique identifier for a Camera model
CameraId: TypeAlias = str


# The unique identifier for a data store
DataStoreId: TypeAlias = str


class MouseButton(Enum):
    """Mouse buttons for mouse click events."""

    NONE = "none"
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


class MouseModifiers(Enum):
    """Keyboard modifiers for mouse click events."""

    SHIFT = "shift"
    CTRL = "ctrl"
    ALT = "alt"
    META = "meta"


class MouseEventType(Enum):
    """Mouse event types."""

    PRESS = "press"
    RELEASE = "release"
    MOVE = "move"


@dataclass(frozen=True)
class MouseCallbackData:
    """Data from a mouse click on the canvas.

    This is the event received by mouse callback functions.
    """

    visual_id: VisualId
    type: MouseEventType
    button: MouseButton
    modifiers: list[MouseModifiers]
    coordinate: np.ndarray
    pick_info: dict[str, Any]


@dataclass(frozen=True)
class CameraControlsUpdateEvent:
    """Event data that is emitted when the state of a camera controls is updated.

    Parameters
    ----------
    id : CameraId
        The ID of the camera model that the controls are for.
    state : dict[str, Any]
        The state of the camera model to update.
        The key is the string name of the parameters and
        the value is the value to set.
    controls_update_callback : Callable | None
        The callback function to block when the camera model is updated.
        This is the callback function that is called when the camera model is updated.
        This is used to prevent the update from bouncing back to the GUI.
    """

    id: CameraId
    state: dict[str, Any]
    controls_update_callback: Callable | None = None


@dataclass(frozen=True)
class DimsControlsUpdateEvent:
    """Event data that is emitted when the state of a dims controls is updated.

    Parameters
    ----------
    id : DimsId
        The ID of the dims model that the controls are for.
    state : dict[str, Any]
        The state of the dims model to update.
        The key is the string name of the parameters and
        the value is the value to set.
    controls_update_callback : Callable | None
        The callback function to block when the dims model is updated.
        This is the callback function that is called when the dims model is updated.
        This is used to prevent the update from bouncing back to the GUI.
    """

    id: DimsId
    state: dict[str, Any]
    controls_update_callback: Callable | None = None
