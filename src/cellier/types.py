"""Types used in the Cellier package."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias, Union

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
