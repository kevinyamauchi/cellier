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


class CanvasMousePressEvent(NamedTuple):
    """Emitted when the primary pointer button is pressed on a canvas."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo


class CanvasMouseMoveEvent(NamedTuple):
    """Emitted on every pointer-move event (button up or down)."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
    pick_info: CanvasPickInfo


class CanvasMouseReleaseEvent(NamedTuple):
    """Emitted when the primary pointer button is released on a canvas."""

    source_id: UUID
    scene_id: UUID
    world_coordinate: np.ndarray
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
    position_2d : np.ndarray
        2D world position for the displayed axes, already unprojected
        through the active camera.  Shape (2,), float64.
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
    position_2d: np.ndarray
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
    | CanvasMousePressEvent
    | CanvasMouseMoveEvent
    | CanvasMouseReleaseEvent
)
