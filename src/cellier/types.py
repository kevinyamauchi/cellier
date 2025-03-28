"""Types used in the Cellier package."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, TypeAlias, Union
from uuid import uuid4

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


class CoordinateSpace(Enum):
    """Enum for the data space.

    WORLD: The world coordinate space.
    DATA: The data coordinate space.
    """

    WORLD = "world"
    DATA = "data"


class TilingMethod(Enum):
    """The method for computing how to chunk the request.

    NONE: No tiling, the entire request is sent as a single request.
    LOGICAL_PIXEL: The request tries to match the resolution
                   to the logical pixel size.
    """

    NONE = "none"
    LOGICAL_PIXEL = "logical_pixel"


@dataclass(frozen=True)
class AxisAlignedSample:
    """Data for an axis-aligned selected region.

    Parameters
    ----------
    space_type : CoordinateSpace
        The coordinate space of the data.
        This is either in the data coordinates or the world coordinates.
    ordered_dims : tuple[int, ...]
        The dimension indices in their displayed order.
    n_displayed_dims : int
        The number of displayed dimensions.
    index_selection : tuple[Union[int, slice], ...]
        The selections for each dimension.
    tiling_method : TilingMethod
        The method for computing how to chunk the request.
    """

    space_type: CoordinateSpace
    ordered_dims: tuple[int, ...]
    n_displayed_dims: Literal[2, 3]
    index_selection: tuple[Union[int, slice], ...]
    tiling_method: TilingMethod


@dataclass(frozen=True)
class AxisAlignedDataRequest:
    """Data for an axis-aligned data request.

    Parameters
    ----------
    scene_id : SceneId
        The unique identifier for which scene this visual belongs to.
    visual_id : VisualId
        The UID of the visual to be updated.
    min_corner_rendered : tuple[int, int] | tuple[int, int, int]
        The coordinates of the minimum corner of the data request
        in the rendered texture units.
    ordered_dims : tuple[int, ...]
        The dimension indices in their displayed order.
    n_displayed : int
        The number of displayed dimensions.
    resolution_level : int
        The resolution level to be rendered. 0 is the highest resolution
        and larger numbers are lower resolution.
    index_selection : tuple[slice, ...]
        The selections for each dimension.
    id : str
        The unique identifier for this request.
    """

    scene_id: SceneId
    visual_id: VisualId
    min_corner_rendered: tuple[int, int] | tuple[int, int, int]
    ordered_dims: tuple[int, ...]
    n_displayed_dims: int
    resolution_level: int
    index_selection: tuple[slice, ...]
    id: str = uuid4().hex


DataRequest = Union[AxisAlignedDataRequest]


@dataclass(frozen=True)
class DataResponse:
    """The data to be sent to the renderer.

    Parameters
    ----------
    id : str
        The unique identifier of the request.
    scene_id : SceneId
        The unique identifier for which scene this visual belongs to.
    visual_id : VisualId
        The UID of the visual to be updated.
    resolution_level : int
        The resolution level to be rendered. 0 is the highest resolution
        and larger numbers are lower resolution.
    """

    id: str
    scene_id: SceneId
    visual_id: VisualId
    resolution_level: int


@dataclass(frozen=True)
class ImageDataResponse(DataResponse):
    """Image data to be sent to the renderer.

    Parameters
    ----------
    id : str
        The unique identifier of the request.
    scene_id : SceneId
        The unique identifier for which scene this visual belongs to.
    visual_id : VisualId
        The UID of the visual to be updated.
    resolution_level : int
        The resolution level to be rendered. 0 is the highest resolution
        and larger numbers are lower resolution.
    min_corner_rendered : tuple[int, ...]
        The coordinates of the minimum corner of the data request
        in the rendered objects units.
    data : np.ndarray
        The image data to be rendered.
    """

    min_corner_rendered: tuple[int, int] | tuple[int, int, int]
    data: np.ndarray
