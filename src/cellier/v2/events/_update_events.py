"""Incoming update event NamedTuples for the cellier v2 IncomingEventBus.

GUI widgets (and other callers) emit these onto ``CellierController.incoming_events``
to request model mutations.  The controller subscribes handler methods that
dispatch each event to the corresponding ``update_*`` method, threading
``source_id`` end-to-end so that echo filtering works on the outgoing bus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID


class AppearanceUpdateEvent(NamedTuple):
    """Request to set one appearance field on a visual.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``AppearanceChangedEvent``
        so the caller can echo-filter on ``on_visual_changed``.
    visual_id :
        Target visual.
    field :
        Attribute name on the appearance model, e.g. ``"clim"``.
    value :
        New value for the field.
    """

    source_id: UUID
    visual_id: UUID
    field: str
    value: Any


class DimsUpdateEvent(NamedTuple):
    """Request to update slice indices and/or displayed axes for a scene.

    Set whichever fields you want to change; leave the others as ``None``.
    When both are set, ``slice_indices`` is applied first so that the dims
    state is consistent when the ``displayed_axes`` mutation fires.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``DimsChangedEvent``.
    scene_id :
        Target scene.
    slice_indices :
        Mapping of axis index → slice position, or ``None`` to leave
        the current slice indices unchanged.
    displayed_axes :
        Tuple of axis indices to display, or ``None`` to leave the
        current displayed axes unchanged.
    """

    source_id: UUID
    scene_id: UUID
    slice_indices: dict[int, int] | None
    displayed_axes: tuple[int, ...] | None


class AABBUpdateEvent(NamedTuple):
    """Request to set one AABB parameter field on a visual.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``AABBChangedEvent``.
    visual_id :
        Target visual.
    field :
        Attribute name on the AABB model, e.g. ``"enabled"``.
    value :
        New value for the field.
    """

    source_id: UUID
    visual_id: UUID
    field: str
    value: Any


CellierUpdateEventTypes = AppearanceUpdateEvent | DimsUpdateEvent | AABBUpdateEvent
