"""Types used in the Cellier package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeAlias, Union
from uuid import uuid4

from pydantic import Field
from typing_extensions import Annotated

from cellier.models.visuals import (
    LinesVisual,
    MultiscaleImageVisual,
    MultiscaleLabelsVisual,
    PointsVisual,
)

if TYPE_CHECKING:
    import numpy as np

    from cellier.transform import AffineTransform
    from cellier.utils.chunked_image._data_classes import (
        TextureConfiguration,
        ViewParameters,
    )

# This is used for a discriminated union for typing the visual models
VisualType = Annotated[
    Union[LinesVisual, PointsVisual, MultiscaleLabelsVisual, MultiscaleImageVisual],
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
class PlaneSelectedRegion:
    """Data for a plane selected region.

    Parameters
    ----------
    space_type : CoordinateSpace
        The coordinate space of the sample data.
        This is either in the data coordinates or the world coordinates.
    ordered_dims : tuple[int, ...]
        The dimension indices in their displayed order.
    n_displayed_dims : int
        The number of displayed dimensions.
    index_selection : tuple[int | slice, ...]
        The selections for each dimension.
        Has the same order as ordered dims.
    point : np.ndarray
        The coordinate of the origin of the plane.
    plane_transform: np.ndarray
        The original plane has a normal vector [1, 0, 0]
        and an up vector [0, 1, 0].
        They should be ordered normal, up, right.
    extents : tuple[None | int, None | int]
        The extents of the plane along the up and right axes.
    """

    space_type: CoordinateSpace
    ordered_dims: tuple[int, ...]
    n_displayed_dims: Literal[2, 3]
    index_selection: tuple[int | slice, ...]
    point: np.ndarray
    plane_transform: np.ndarray
    extents: tuple[None | int, None | int]


@dataclass(frozen=True)
class AxisAlignedSelectedRegion:
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
    index_selection : tuple[int | slice, ...]
        The selections for each dimension.
        Has the same order as ordered dims.
    """

    space_type: CoordinateSpace
    ordered_dims: tuple[int, ...]
    n_displayed_dims: Literal[2, 3]
    index_selection: tuple[int | slice, ...]


@dataclass(frozen=True)
class ChunkedSelectedRegion:
    """Selected region for a frustum-based chunked view.

    Parameters
    ----------
    view_parameters : ViewParameters
        Frustum corners, view direction, and near-plane center in world coordinates.
    visual_id : str
        ID of the visual that requested this region.
    scene_id : str
        ID of the scene containing the visual.
    """

    view_parameters: ViewParameters
    visual_id: str
    scene_id: str


SelectedRegion = Union[
    AxisAlignedSelectedRegion, PlaneSelectedRegion, ChunkedSelectedRegion
]


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
        Has the same order as ordered dims.
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


@dataclass(frozen=True)
class PlaneDataRequest:
    """Data for a plane data request.

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
    index_selection : tuple[int | slice, ...]
        The selections for each dimension.
        Has the same order as ordered dims.
    resolution_level : int
        The resolution level to be rendered. 0 is the highest resolution
        and larger numbers are lower resolution.
    point : np.ndarray
        The coordinate of the origin of the plane.
        Must be in the data coordinate system.
    plane_transform: np.ndarray
        The transformation to place the plane.
        The original plane has a normal vector [1, 0, 0]
        and an up vector [0, 1, 0].
        Must be in the data coordinate system.
    extents : tuple[int, int]
        The size of the plane in the up and right directions.
    id : str
        The unique identifier for this request.
    """

    scene_id: SceneId
    visual_id: VisualId
    min_corner_rendered: tuple[int, int]
    ordered_dims: tuple[int, ...]
    n_displayed_dims: int
    index_selection: tuple[int | slice, ...]
    resolution_level: int
    point: np.ndarray
    plane_transform: np.ndarray
    extents: tuple[int, int]
    id: str = uuid4().hex


DataRequest = Union[AxisAlignedDataRequest, PlaneDataRequest]


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
        The unique identifier of the request that generated this response.
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


@dataclass(frozen=True)
class PointsDataResponse(DataResponse):
    """Points data to be sent to the renderer.

    Parameters
    ----------
    id : str
        The unique identifier of the request that generated this response.
    scene_id : SceneId
        The unique identifier for which scene this visual belongs to.
    visual_id : VisualId
        The UID of the visual to be updated.
    resolution_level : int
        The resolution level to be rendered. 0 is the highest resolution
        and larger numbers are lower resolution.
    data : np.ndarray
        The image data to be rendered.
    """

    data: np.ndarray


@dataclass(frozen=True)
class LinesDataResponse(DataResponse):
    """Lines data to be sent to the renderer.

    Parameters
    ----------
    id : str
        The unique identifier of the request that generated this response.
    scene_id : SceneId
        The unique identifier for which scene this visual belongs to.
    visual_id : VisualId
        The UID of the visual to be updated.
    resolution_level : int
        The resolution level to be rendered. 0 is the highest resolution
        and larger numbers are lower resolution.
    data : np.ndarray
        The image data to be rendered.
    colors : np.ndarray | None
        The vertex colors for the lines to be rendered.
        Must be of shape (n_positions, 4) where n_positions is
        the number of vertices in the lines and the 4 values are RGBA colors.
    """

    data: np.ndarray
    colors: np.ndarray | None


@dataclass(frozen=True)
class ChunkRequest:
    """A request to load a single chunk from a chunked image store.

    Parameters
    ----------
    chunk_index : int
        Linear index into ScaleLevelModel (row-major z,y,x order).
    scale_index : int
        Which scale level to load from.
    priority : float
        Lower value = load first. Typically distance from near plane.
    visual_id : str
        ID of the visual that requested this chunk.
    scene_id : str
        ID of the scene containing the visual.
    """

    chunk_index: int
    scale_index: int
    priority: float
    visual_id: str
    scene_id: str


@dataclass
class ChunkData:
    """A loaded chunk's voxel data.

    Parameters
    ----------
    chunk_index : int
        Linear index matching the originating ChunkRequest.
    scale_index : int
        Scale level index matching the originating ChunkRequest.
    data : np.ndarray
        Voxel data as a numpy array. No GPU objects.
    """

    chunk_index: int
    scale_index: int
    data: np.ndarray


@dataclass(frozen=True)
class ChunkedDataRequest:
    """A batch of chunk requests produced by ChunkedImageStore.get_data_request().

    This object is returned by ``ChunkedImageStore.get_data_request()`` and
    passed back into ``ChunkedImageStore.get_data()`` by the slicer.

    Parameters
    ----------
    scene_id : str
    visual_id : str
    resolution_level : int
    chunk_requests : list[ChunkRequest]
        Individual chunks selected by ChunkCuller + ChunkSelector.
    texture_config : TextureConfiguration
        Texture geometry used during chunk selection.
    texture_to_world_transform : AffineTransform
        Maps texture-local coordinates to world coordinates; forwarded to the
        renderer in the corresponding ChunkedDataResponse.
    """

    scene_id: str
    visual_id: str
    resolution_level: int
    chunk_requests: list[ChunkRequest]
    texture_config: TextureConfiguration
    texture_to_world_transform: AffineTransform


@dataclass(frozen=True)
class ChunkedDataResponse(DataResponse):
    """Response from ChunkedImageStore.get_data(); consumed by RenderManager.

    Extends DataResponse so it can be emitted on slicer.events.new_slice
    without changing the signal type.

    Parameters
    ----------
    available_chunks : list[ChunkData]
        Chunks immediately available from the ChunkManager LRU cache.
    pending_count : int
        Number of chunks submitted for background loading (not yet available).
    texture_to_world_transform : AffineTransform
        Forwarded from the originating ChunkedDataRequest; maps texture-local
        coordinates to world coordinates for the renderer.
    """

    available_chunks: list[ChunkData]
    pending_count: int
    texture_to_world_transform: AffineTransform
