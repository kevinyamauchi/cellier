"""All event NamedTuples for the cellier v2 EventBus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np

    from cellier.v2._state import CameraState, DimsState
    from cellier.v2.transform import AffineTransform


class DimsChangedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    dims_state: DimsState
    displayed_axes_changed: bool


class CameraChangedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    camera_state: CameraState


class AppearanceChangedEvent(NamedTuple):
    source_id: UUID
    visual_id: UUID
    field_name: str
    new_value: Any
    requires_reslice: bool


class AABBChangedEvent(NamedTuple):
    source_id: UUID
    visual_id: UUID
    field_name: str
    new_value: Any


class VisualVisibilityChangedEvent(NamedTuple):
    source_id: UUID
    visual_id: UUID
    visible: bool


class DataStoreMetadataChangedEvent(NamedTuple):
    source_id: UUID
    data_store_id: UUID


class DataStoreContentsChangedEvent(NamedTuple):
    source_id: UUID
    data_store_id: UUID
    dirty_keys: Any


class ResliceStartedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    visual_ids: frozenset[UUID]


class ResliceCompletedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    brick_count: int


class ResliceCancelledEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class FrameRenderedEvent(NamedTuple):
    source_id: UUID
    canvas_id: UUID
    frame_time_ms: float


class VisualAddedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class VisualRemovedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class SceneAddedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID


class TransformChangedEvent(NamedTuple):
    """Fired when a visual's data-to-world transform is replaced."""

    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    transform: AffineTransform


class SceneRemovedEvent(NamedTuple):
    source_id: UUID
    scene_id: UUID


class CanvasPickInfo(NamedTuple):
    """Model-layer pick result attached to canvas mouse events.

    No ``gfx.*`` types appear here.  The render layer translates the
    hit world object to a model-layer UUID before emission.

    Attributes
    ----------
    hit_visual_id : UUID or None
        Model-layer ID of the visual whose active scene-graph node was
        hit, or None if the pointer landed on the background.
    """

    hit_visual_id: UUID | None


class ViewRay(NamedTuple):
    """A view ray from the camera through a screen-space point.

    Used by 3D canvas mouse events in place of a single world coordinate.

    Attributes
    ----------
    origin : np.ndarray
        Near-plane world position where the ray starts.  Shape (3,), float64.
    direction : np.ndarray
        Unit vector in world space giving the ray direction.  Shape (3,), float64.
    """

    origin: np.ndarray
    direction: np.ndarray


class CanvasMousePress2DEvent(NamedTuple):
    """Emitted when the primary pointer button is pressed on a 2D canvas."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo


class CanvasMouseMove2DEvent(NamedTuple):
    """Emitted on every pointer-move event on a 2D canvas (button up or down)."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo


class CanvasMouseRelease2DEvent(NamedTuple):
    """Emitted when the primary pointer button is released on a 2D canvas."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo


class CanvasMousePress3DEvent(NamedTuple):
    """Emitted when the primary pointer button is pressed on a 3D canvas."""

    source_id: UUID
    scene_id: UUID
    ray: ViewRay
    pick_info: CanvasPickInfo


class CanvasMouseMove3DEvent(NamedTuple):
    """Emitted on every pointer-move event on a 3D canvas (button up or down)."""

    source_id: UUID
    scene_id: UUID
    ray: ViewRay
    pick_info: CanvasPickInfo


class CanvasMouseRelease3DEvent(NamedTuple):
    """Emitted when the primary pointer button is released on a 3D canvas."""

    source_id: UUID
    scene_id: UUID
    ray: ViewRay
    pick_info: CanvasPickInfo


class _CanvasRawPointerEvent(NamedTuple):
    """Internal event emitted by RenderManager after render-layer translation.

    Not part of the public EventBus catalogue.
    Consumed only by CellierController._on_raw_pointer_event.

    Attributes
    ----------
    canvas_id : UUID
    scene_id : UUID
        Filled by RenderManager from _canvas_to_scene.
    action : str
        One of ``"press"``, ``"move"``, ``"release"``.
    camera_type : str
        ``"2d"`` or ``"3d"``.  Determines which of ``position_2d`` / ``ray``
        is populated.
    position_2d : np.ndarray or None
        2D world position for the displayed axes, already unprojected through
        the active orthographic camera.  Shape (2,), float64.
        Set iff ``camera_type == "2d"``.
    ray : ViewRay or None
        View ray unprojected through the active perspective camera.
        Set iff ``camera_type == "3d"``.
    hit_visual_id : UUID or None
        Visual ID translated from the pick-buffer world object via
        SceneManager.get_visual_id_for_node.  None on background hits.
    button : int
        Mouse button identifier from the pygfx event.
    modifiers : tuple[str, ...]
        Active keyboard modifiers from the pygfx event.
    """

    canvas_id: UUID
    scene_id: UUID
    action: str
    camera_type: str
    position_2d: np.ndarray | None
    ray: ViewRay | None
    hit_visual_id: UUID | None
    button: int
    modifiers: tuple


CellierEventTypes = (
    DimsChangedEvent
    | CameraChangedEvent
    | AppearanceChangedEvent
    | AABBChangedEvent
    | VisualVisibilityChangedEvent
    | DataStoreMetadataChangedEvent
    | DataStoreContentsChangedEvent
    | ResliceStartedEvent
    | ResliceCompletedEvent
    | ResliceCancelledEvent
    | FrameRenderedEvent
    | VisualAddedEvent
    | VisualRemovedEvent
    | TransformChangedEvent
    | SceneAddedEvent
    | SceneRemovedEvent
    | CanvasMousePress2DEvent
    | CanvasMouseMove2DEvent
    | CanvasMouseRelease2DEvent
    | CanvasMousePress3DEvent
    | CanvasMouseMove3DEvent
    | CanvasMouseRelease3DEvent
)
