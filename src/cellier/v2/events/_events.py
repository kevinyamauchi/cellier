"""All event NamedTuples for the cellier v2 EventBus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID

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


CellierEventTypes = (
    DimsChangedEvent
    | CameraChangedEvent
    | AppearanceChangedEvent
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
)
