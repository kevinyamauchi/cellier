# Cellier v2 EventBus — Design Document

## Overview

The v2 EventBus is the sole communication channel between the model layer (psygnal
`EventedModel` instances held by the controller) and the render layer (`RenderManager`,
`CanvasView`, `GFX*Visual`). No model class ever imports a render-layer class. No
render-layer class ever imports a model class. Every cross-layer communication passes
through a typed event on the bus.

The bus is an internal implementation detail. Application developers never hold a
reference to it. External state change notifications are surfaced through the
`controller.on_*` callback registration methods.

This document covers: the shared state NamedTuples, the event catalogue, the bus API,
bounce-back prevention, async handler conventions, how the controller bridges psygnal
signals to bus events, and the registration lifecycle.

---

## Module structure

```
src/cellier/v2/
├── _state.py          # DimsState, CameraState — shared by events and render layer
└── events/
    ├── __init__.py    # exports EventBus, all event types, SubscriptionHandle
    ├── _bus.py        # EventBus, SubscriptionHandle, make_weak_callback
    └── _events.py     # all event NamedTuples
```

Nothing outside `src/cellier/v2/` imports from this package. The controller is the only
production caller of `EventBus`.

---

## Design principles

### NamedTuples for all events and state snapshots

All events and shared state objects are `NamedTuple`s — immutable, fast, and consistent
with the project convention. Because Python `NamedTuple` subclasses cannot add fields,
there is no common base class with shared fields. Instead, every event NamedTuple
independently declares a `source_id: UUID` as its first field. The `CellierEventTypes`
Union alias at module level groups them for type annotations on the bus.

### One universal field: `source_id`

Every event carries a `source_id: UUID` identifying the object that emitted it. This
is the sole mechanism for preventing feedback loops (see §Bounce-back prevention). The
bus itself is source-agnostic; filtering on `source_id` is the subscriber's
responsibility.

### No model references inside the bus

The bus is constructed before any model or render object exists. It holds no reference
to `ViewerModel`, `DimsManager`, or any psygnal object. The controller owns all the
callbacks that bridge psygnal signals to bus events (see §How the controller bridges
psygnal to the bus).

### Typed dispatch with optional entity filtering

Subscribers declare the event type they want, optionally scoped to a specific entity
(scene, visual, canvas, data store). New event types require no changes to the bus
class itself.

### Synchronous bus, async-safe handlers

All bus dispatch is synchronous. Handlers that need to await work wrap it in
`asyncio.create_task()` and return immediately. This is a documented convention, not
a constraint enforced by the bus.

### Strong references by default; weak references opt-in

Subscriptions hold strong references to callbacks by default. The controller's explicit
`unsubscribe_all(owner_id)` lifecycle is the primary cleanup mechanism and is sufficient
for all render-layer objects whose lifetimes the controller manages directly.

Weak references (`weak=True`) are a safety net for subscribers that may be
garbage-collected outside the controller's teardown path — for example, transient
widgets closed by Qt independently. When a weak subscription's target is collected,
the entry is cleaned up lazily on the next emission of that event type.

### Symmetric registration lifecycle

Every entity that is registered can be fully deregistered. All callback references and
entity filters are cleaned up atomically on deregistration via `unsubscribe_all()`.
Weak subscriptions are additionally cleaned up automatically when their target is
collected, without requiring any explicit call.

---

## Shared state NamedTuples

These are not events — they are the immutable state payloads that events carry. They
live in `src/cellier/v2/_state.py`. This module sits above both the events package and
the render layer so neither needs to import the other to reach shared types.
`ReslicingRequest` in `src/cellier/v2/render/_requests.py` imports `DimsState` from
here. `CameraState` is new.

### `DimsState`

Move from `src/cellier/v2/render/_requests.py` to `src/cellier/v2/_state.py` and
update the import in `_requests.py` accordingly.

```python
class DimsState(NamedTuple):
    """Current dimension display state for a scene.

    Parameters
    ----------
    displayed_axes :
        Indices of axes currently rendered, e.g. ``(0, 1, 2)`` for 3D
        or ``(1, 2)`` for a 2D XY slice.
    slice_indices :
        Current integer index for each non-displayed axis, in the same
        order as the non-displayed axes.  Empty for a pure 3D view.
    """
    displayed_axes: tuple[int, ...]
    slice_indices: tuple[int, ...]
```

### `CameraState`

New. A numpy-free immutable snapshot suitable for event payloads, external callbacks,
and serialization. Distinct from `ReslicingRequest`, which carries numpy arrays for
the hot slicing path.

```python
from typing import Literal, NamedTuple


class CameraState(NamedTuple):
    """Immutable snapshot of the active camera's logical state.

    Carries all fields needed to reconstruct the camera view for display
    in external callbacks, camera sync across canvases, and
    save/restore.  Numpy-free — frustum corners for the slicer are
    computed separately inside ``CanvasView.capture_reslicing_request()``.

    For perspective cameras: ``fov`` is populated; ``extent`` is
    ``(0.0, 0.0)``.
    For orthographic cameras: ``extent`` is populated; ``fov`` is
    ``0.0``.
    ``zoom`` applies to both types.

    Parameters
    ----------
    camera_type :
        ``"perspective"`` or ``"orthographic"``.
    position :
        World-space camera position ``(x, y, z)``.
    rotation :
        Camera orientation as a unit quaternion ``(x, y, z, w)``,
        matching pygfx's convention.
    up :
        World-space up vector ``(x, y, z)``.
    fov :
        Vertical field of view in degrees.  Perspective cameras only;
        ``0.0`` for orthographic.
    zoom :
        Unitless zoom multiplier.  Applies to both camera types.
    extent :
        ``(width, height)`` in world units.  Orthographic cameras only;
        ``(0.0, 0.0)`` for perspective.
    depth_range :
        ``(near, far)`` clip distances.
    """
    camera_type: Literal["perspective", "orthographic"]
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    up: tuple[float, float, float]
    fov: float
    zoom: float
    extent: tuple[float, float]
    depth_range: tuple[float, float]
```

`CameraState` and `DimsState` are both defined in `src/cellier/v2/_state.py` and
re-exported from `cellier.v2.events` so subscribers have a single import location.

---

## Event catalogue

All events live in `src/cellier/v2/events/_events.py`. All are `NamedTuple`s. The
first field of every event is `source_id: UUID`.

```python
from __future__ import annotations

from typing import Literal, NamedTuple
from uuid import UUID

from cellier.v2._state import CameraState, DimsState
```

### Scene / dims events

```python
class DimsChangedEvent(NamedTuple):
    """Fired when a scene's DimsManager state changes.

    Triggers: dims slider move, programmatic ``dims_model.slice_indices =
    ...``, ``dims_model.displayed_axes = ...``.

    Primary consumers:
    - ``CanvasView``: switch active camera when ``displayed_axes_changed``
    - ``SliceCoordinator``: reslice all affected visuals
    - Dims slider widget: update slider position without re-emitting
    - External ``on_dims_changed`` callbacks

    Parameters
    ----------
    source_id :
        ID of the object that caused the change.  The dims slider widget
        sets this to its own ID so it can ignore the echo.
    scene_id :
        The scene whose ``DimsManager`` changed.
    dims_state :
        Full snapshot of the new dims state.
    displayed_axes_changed :
        ``True`` when ``displayed_axes`` changed, not just
        ``slice_indices``.  ``CanvasView`` uses this flag to decide
        whether to switch the active camera.
    """
    source_id: UUID
    scene_id: UUID
    dims_state: DimsState
    displayed_axes_changed: bool
```

```python
class CameraChangedEvent(NamedTuple):
    """Fired when the camera moves in a scene.

    Sources: user drag/zoom via a pygfx ``Controller`` in a
    ``CanvasView`` (``source_id`` = that canvas's ID), or a programmatic
    ``controller.look_at_visual()`` call (``source_id`` = controller ID).

    Primary consumers:
    - Other ``CanvasView`` instances attached to the same scene (sync)
    - ``SliceCoordinator``: reslice on camera move in continuous mode
    - External ``on_camera_changed`` callbacks

    Parameters
    ----------
    source_id :
        ID of the canvas that moved the camera.  A ``CanvasView`` that
        receives this event and finds ``source_id == self._id`` ignores
        it to break the sync feedback loop.
    scene_id :
        The scene whose active camera moved.
    camera_state :
        Full snapshot of the new camera state.
    """
    source_id: UUID
    scene_id: UUID
    camera_state: CameraState
```

### Visual / appearance events

```python
class AppearanceChangedEvent(NamedTuple):
    """Fired when any field on a visual's appearance model changes.

    Emitted by the controller whenever a psygnal field event fires on a
    ``BaseAppearance`` subclass.  ``visible`` changes emit a separate
    ``VisualVisibilityChangedEvent`` instead.

    Primary consumers:
    - ``GFX*Visual``: update material / shader parameters
    - ``SliceCoordinator``: reslice if ``requires_reslice``
    - External ``on_visual_changed`` callbacks

    Parameters
    ----------
    source_id :
        Always the controller's own ID (appearance changes originate
        from application code, never from the render layer).
    visual_id :
        The visual whose appearance changed.
    field_name :
        The name of the changed field, e.g. ``"color_map"``.
    new_value :
        The new value.  Typed as ``object``; consumers cast as needed.
    requires_reslice :
        ``True`` for fields that invalidate the current brick set
        (``lod_bias``, ``force_level``, ``frustum_cull``).
        ``False`` for pure GPU-side changes (``color_map``, ``clim``).
    """
    source_id: UUID
    visual_id: UUID
    field_name: str
    new_value: object
    requires_reslice: bool


class VisualVisibilityChangedEvent(NamedTuple):
    """Fired when ``appearance.visible`` changes.

    Separate from ``AppearanceChangedEvent`` so consumers that only need
    to show/hide a node do not have to inspect ``field_name``.

    Parameters
    ----------
    source_id :
        Always the controller's own ID.
    visual_id :
        The visual whose visibility changed.
    visible :
        The new visibility state.
    """
    source_id: UUID
    visual_id: UUID
    visible: bool
```

### Data store events

```python
class DataStoreMetadataChangedEvent(NamedTuple):
    """Fired when a data store's shape or chunk layout changes.

    This is the more disruptive mutation: ``VolumeGeometry`` must be
    rebuilt and the entire ``BrickCache`` evicted before reslicing.

    Parameters
    ----------
    source_id :
        Always the controller's own ID.
    data_store_id :
        The data store whose metadata changed.
    """
    source_id: UUID
    data_store_id: UUID


class DataStoreContentsChangedEvent(NamedTuple):
    """Fired when voxel values change but shape and chunk layout are unchanged.

    Allows surgical ``BrickCache`` eviction: only bricks whose keys
    appear in ``dirty_keys`` need to be evicted before reslicing.

    Parameters
    ----------
    source_id :
        Always the controller's own ID.
    data_store_id :
        The data store whose contents changed.
    dirty_keys :
        The set of ``BrickKey`` tuples that are now stale.
        ``None`` means the entire store is dirty — evict all.
    """
    source_id: UUID
    data_store_id: UUID
    dirty_keys: frozenset[tuple[int, ...]] | None
```

### Slicer lifecycle events

```python
class ResliceStartedEvent(NamedTuple):
    """Fired by ``SliceCoordinator`` before submitting async tasks.

    Useful for disabling the Update button and showing a loading spinner.

    Parameters
    ----------
    source_id :
        Always the ``SliceCoordinator``'s ID.
    scene_id :
        The scene being resliced.
    visual_ids :
        The visuals in this batch.
    """
    source_id: UUID
    scene_id: UUID
    visual_ids: frozenset[UUID]


class ResliceCompletedEvent(NamedTuple):
    """Fired by ``GFX*Visual`` after all bricks in a slice batch are committed.

    Fired once per visual per reslice cycle, not once per brick.

    Parameters
    ----------
    source_id :
        Always the completing visual's ID.
    scene_id :
        The scene that contains the visual.
    visual_id :
        The visual that finished loading.
    brick_count :
        Number of bricks committed in this batch.
    """
    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    brick_count: int


class ResliceCancelledEvent(NamedTuple):
    """Fired by ``SliceCoordinator`` when an in-flight task is cancelled.

    Happens when a second reslice is submitted before the first completes.

    Parameters
    ----------
    source_id :
        Always the ``SliceCoordinator``'s ID.
    scene_id :
        The scene whose task was cancelled.
    visual_id :
        The visual whose task was cancelled.
    """
    source_id: UUID
    scene_id: UUID
    visual_id: UUID
```

### Render events

```python
class FrameRenderedEvent(NamedTuple):
    """Fired by ``CanvasView`` at the end of each rendered frame.

    Useful for benchmarking, screenshot capture, and frame-rate display.

    Parameters
    ----------
    source_id :
        Always the ``CanvasView``'s ID (= ``canvas_id``).
    canvas_id :
        The canvas that completed a frame.
    frame_time_ms :
        Wall-clock time for this frame in milliseconds.
    """
    source_id: UUID
    canvas_id: UUID
    frame_time_ms: float
```

### Structural events

```python
class VisualAddedEvent(NamedTuple):
    """Fired after a visual is fully wired into a scene."""
    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class VisualRemovedEvent(NamedTuple):
    """Fired after a visual is fully removed from a scene."""
    source_id: UUID
    scene_id: UUID
    visual_id: UUID


class SceneAddedEvent(NamedTuple):
    """Fired after a scene is registered with the controller."""
    source_id: UUID
    scene_id: UUID


class SceneRemovedEvent(NamedTuple):
    """Fired after a scene and all its canvases and visuals are removed."""
    source_id: UUID
    scene_id: UUID
```

### Union alias

```python
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
    | SceneAddedEvent
    | SceneRemovedEvent
)
```

---

## EventBus API

```python
# src/cellier/v2/events/_bus.py

from __future__ import annotations

import logging
import types
import weakref
from collections import defaultdict
from typing import Callable, TypeVar
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)
E = TypeVar("E")


def make_weak_callback(callback: Callable) -> Callable:
    """Wrap a callable in a weak reference.

    Bound methods (``instance.handler``) are temporary objects — a
    direct ``weakref.ref`` to them dies immediately because nothing else
    holds them alive.  This function handles that case by storing a weak
    reference to the *instance* and a strong reference to the underlying
    *function*, then reconstructing the call at dispatch time.

    Plain functions and lambdas are wrapped with a direct ``weakref.ref``.
    Note: lambdas passed with ``weak=True`` will die immediately; callers
    should pass lambdas with ``weak=False`` (see ``EventBus.subscribe``).

    The returned wrapper exposes an ``is_dead()`` method that returns
    ``True`` once the referent has been garbage-collected.

    Parameters
    ----------
    callback :
        The callable to wrap.

    Returns
    -------
    Callable
        A wrapper that calls the original callback while it is alive and
        becomes a no-op once the referent is collected.  The wrapper also
        has an ``is_dead() -> bool`` attribute.
    """
    if isinstance(callback, types.MethodType):
        instance_ref = weakref.ref(callback.__self__)
        func = callback.__func__

        def _call(*args, **kwargs):
            instance = instance_ref()
            if instance is None:
                return None
            return func(instance, *args, **kwargs)

        _call.is_dead = lambda: instance_ref() is None  # type: ignore[attr-defined]
    else:
        fn_ref = weakref.ref(callback)

        def _call(*args, **kwargs):
            fn = fn_ref()
            if fn is None:
                return None
            return fn(*args, **kwargs)

        _call.is_dead = lambda: fn_ref() is None  # type: ignore[attr-defined]

    return _call


class SubscriptionHandle:
    """Opaque token returned by ``EventBus.subscribe()``.

    Pass to ``EventBus.unsubscribe()`` to remove the subscription.

    Parameters
    ----------
    id :
        Unique identifier for this subscription.
    """
    __slots__ = ("id",)

    def __init__(self) -> None:
        self.id: UUID = uuid4()


# Internal entry stored per subscription.
# Fields: (handle, entity_id_filter, callback_or_weak_wrapper, is_weak)
_Entry = tuple[SubscriptionHandle, UUID | None, Callable, bool]


class EventBus:
    """Central typed-dispatch event bus for Cellier v2.

    All inter-layer communication in Cellier passes through this object.
    The model layer emits events; the render layer subscribes to them.
    No model class imports render-layer code; no render-layer class
    imports model code.

    The bus is constructed by ``CellierController`` before any model or
    render object exists.  The controller performs all wiring.
    Application code never holds a reference to the bus.

    Notes
    -----
    Dispatch is fully synchronous.  Handlers that need to await async
    work must wrap it in ``asyncio.create_task()`` and return
    immediately.  See §Async handler convention in the design document.

    Subscriptions use strong references by default.  Pass ``weak=True``
    to ``subscribe()`` for callbacks that may be garbage-collected
    before the controller has a chance to call ``unsubscribe_all()``.
    Dead weak subscriptions are cleaned up lazily on the next emission
    of the same event type.
    """

    def __init__(self) -> None:
        # Primary store:
        #   _subscriptions[EventType] -> list of _Entry
        # entity_id_filter=None means "fire for all entities of this type".
        self._subscriptions: dict[type, list[_Entry]] = defaultdict(list)

        # Reverse index for O(1) unsubscribe:
        #   _handle_to_type[handle] -> event_type
        self._handle_to_type: dict[SubscriptionHandle, type] = {}

        # Owner index for bulk deregistration:
        #   _owner_to_handles[owner_id] -> list[SubscriptionHandle]
        self._owner_to_handles: dict[UUID, list[SubscriptionHandle]] = defaultdict(list)

    def emit(self, event: object) -> None:
        """Emit an event to all matching subscribers.

        Dispatch is breadth-first over the subscriber list for this
        event type.  Dead weak subscriptions are collected and removed
        before callbacks fire.  Exceptions from individual subscribers
        are logged but do not prevent other subscribers from receiving
        the event; the first exception is re-raised after all subscribers
        have run.

        Parameters
        ----------
        event :
            A ``CellierEventTypes`` NamedTuple instance.
        """
        event_type = type(event)
        subscribers = self._subscriptions.get(event_type)
        if not subscribers:
            return

        # Sweep dead weak entries before dispatch so they never fire.
        dead_handles: list[SubscriptionHandle] = []
        live: list[_Entry] = []
        for entry in subscribers:
            handle, filter_id, cb, is_weak = entry
            if is_weak and cb.is_dead():
                dead_handles.append(handle)
            else:
                live.append(entry)

        if dead_handles:
            self._subscriptions[event_type] = live
            for handle in dead_handles:
                self._handle_to_type.pop(handle, None)

        # Resolve the entity ID lazily — only when at least one live
        # subscriber has a filter set, to avoid the attribute lookup on
        # hot paths where all subscribers are unfiltered.
        entity_id: UUID | None = None
        entity_field = _ENTITY_FIELD.get(event_type)

        errors: list[Exception] = []
        for handle, filter_id, callback, _is_weak in live:
            if filter_id is not None:
                if entity_id is None and entity_field is not None:
                    entity_id = getattr(event, entity_field)
                if entity_id != filter_id:
                    continue
            try:
                callback(event)
            except Exception as exc:
                errors.append(exc)

        if errors:
            for exc in errors[1:]:
                logger.exception("EventBus subscriber raised", exc_info=exc)
            raise errors[0]

    def subscribe(
        self,
        event_type: type[E],
        callback: Callable[[E], None],
        *,
        entity_id: UUID | None = None,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Subscribe to events of a given type.

        Parameters
        ----------
        event_type :
            The event class to subscribe to, e.g. ``DimsChangedEvent``.
        callback :
            Called synchronously for each matching event.
        entity_id :
            When provided, the callback fires only for events whose
            canonical entity ID matches this value (e.g. a specific
            ``scene_id`` for scene events, ``visual_id`` for visual
            events).  ``None`` subscribes to all entities.
        owner_id :
            When provided, this subscription is registered under
            ``owner_id`` so that ``unsubscribe_all(owner_id)`` removes
            it atomically alongside all other subscriptions owned by
            the same object.  Pass the owning object's own ``.id``.
        weak :
            When ``True``, the bus stores a weak reference to the
            callback's owner.  The subscription is cleaned up
            automatically when the owner is garbage-collected.  Use for
            transient subscribers that may be destroyed outside the
            controller's teardown path.

            **Do not pass lambdas with** ``weak=True``: a lambda has no
            other referent to keep it alive and will be collected
            immediately, silently dropping the subscription.

        Returns
        -------
        SubscriptionHandle
            Opaque token.  Pass to ``unsubscribe()`` to remove this
            subscription individually.

        Raises
        ------
        ValueError
            If ``weak=True`` is combined with a lambda, which would
            result in the subscription dying immediately.
        """
        if weak and getattr(callback, "__name__", None) == "<lambda>":
            raise ValueError(
                "Cannot subscribe a lambda with weak=True: the lambda has no "
                "persistent referent and will be garbage-collected immediately, "
                "silently dropping the subscription.  Assign the lambda to a "
                "variable or use weak=False."
            )

        stored_cb = make_weak_callback(callback) if weak else callback
        handle = SubscriptionHandle()
        self._subscriptions[event_type].append(
            (handle, entity_id, stored_cb, weak)
        )
        self._handle_to_type[handle] = event_type
        if owner_id is not None:
            self._owner_to_handles[owner_id].append(handle)
        return handle

    def unsubscribe(self, handle: SubscriptionHandle) -> None:
        """Remove a single subscription.

        Parameters
        ----------
        handle :
            The handle returned by ``subscribe()``.  Passing an
            already-removed or unknown handle is a no-op.
        """
        event_type = self._handle_to_type.pop(handle, None)
        if event_type is None:
            return
        self._subscriptions[event_type] = [
            entry for entry in self._subscriptions[event_type]
            if entry[0] is not handle
        ]

    def unsubscribe_all(self, owner_id: UUID) -> None:
        """Remove every subscription owned by ``owner_id``.

        Called by the controller when an entity (scene, visual, canvas)
        is removed, ensuring no stale callbacks remain.  This is the
        primary cleanup path for render-layer objects.  Weak subscriptions
        are also cleaned up here if the owner is still alive but being
        explicitly removed; there is no need to wait for garbage
        collection.

        Parameters
        ----------
        owner_id :
            The ID of the object being removed.
        """
        handles = self._owner_to_handles.pop(owner_id, [])
        for handle in handles:
            self.unsubscribe(handle)
```

### Entity ID resolution

The bus resolves the "canonical entity ID" of an event for `entity_id` filtering using a
module-level mapping. Each event type declares which of its fields is the filterable
entity:

```python
# In _bus.py, module-level

_ENTITY_FIELD: dict[type, str] = {
    DimsChangedEvent:                "scene_id",
    CameraChangedEvent:              "scene_id",
    AppearanceChangedEvent:          "visual_id",
    VisualVisibilityChangedEvent:    "visual_id",
    DataStoreMetadataChangedEvent:   "data_store_id",
    DataStoreContentsChangedEvent:   "data_store_id",
    ResliceStartedEvent:             "scene_id",
    ResliceCompletedEvent:           "visual_id",
    ResliceCancelledEvent:           "visual_id",
    FrameRenderedEvent:              "canvas_id",
    VisualAddedEvent:                "scene_id",
    VisualRemovedEvent:              "scene_id",
    SceneAddedEvent:                 "scene_id",
    SceneRemovedEvent:               "scene_id",
}
```

---

## Bounce-back prevention

The v1 approach — temporarily disconnecting a callback before emitting and reconnecting
afterwards — requires the emitter to know the receiver's callback identity and fails
silently on lambdas or short-lived bound methods.

The v2 approach: **every subscriber that can also be a source checks `event.source_id`
at the top of its handler and returns immediately if it matches its own ID.**

### Example: dims slider ↔ model sync

```python
class DimsSliderWidget:
    def __init__(self, scene_id: UUID, bus: EventBus) -> None:
        self._id: UUID = uuid4()
        self._scene_id = scene_id
        bus.subscribe(
            DimsChangedEvent,
            self._on_dims_changed,
            entity_id=scene_id,
            owner_id=self._id,
        )

    def _on_dims_changed(self, event: DimsChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own slider move; ignore
        self._update_slider_position(event.dims_state)

    def _on_slider_moved(self, value: int) -> None:
        # Tag the event with our own ID so the handler above ignores it.
        self._bus.emit(DimsChangedEvent(
            source_id=self._id,
            scene_id=self._scene_id,
            dims_state=...,
            displayed_axes_changed=False,
        ))
```

### Example: camera sync across canvases

Two canvases showing the same scene mirror each other's camera. Canvas A emits
`CameraChangedEvent(source_id=canvas_a_id, ...)` on user drag. Canvas B applies the
state. Canvas A also receives the event but ignores it because `source_id == self._id`.

```python
class CanvasView:
    def __init__(self, canvas_id: UUID, scene_id: UUID, bus: EventBus) -> None:
        self._id = canvas_id
        self._scene_id = scene_id
        self._bus = bus
        self._applying_model_state = False
        bus.subscribe(
            CameraChangedEvent,
            self._on_camera_changed,
            entity_id=scene_id,
            owner_id=canvas_id,
        )

    def _on_camera_changed(self, event: CameraChangedEvent) -> None:
        if event.source_id == self._id:
            return  # we emitted this; do not re-apply
        self._applying_model_state = True
        try:
            self._camera.world.position = event.camera_state.position
            # ... additional setters for rotation, fov, depth_range, etc.
        finally:
            self._applying_model_state = False

    def _on_controller_event(self, event) -> None:
        # Fired by pygfx on every user interaction.
        if self._applying_model_state:
            return  # guard: we are applying a remote state; do not echo
        self._bus.emit(CameraChangedEvent(
            source_id=self._id,
            scene_id=self._scene_id,
            camera_state=self._capture_camera_state(),
        ))
```

---

## Async handler convention

The bus dispatches synchronously. Handlers that need to await async work must call
`asyncio.create_task()` and return immediately. Awaiting directly inside a handler will
deadlock the Qt event loop.

This convention is **mandatory** for any handler that touches the slicer pipeline. It
is documented in every such handler's docstring.

```python
class SliceCoordinator:

    def _on_dims_changed(self, event: DimsChangedEvent) -> None:
        """Synchronous entry point — schedules async reslice via create_task."""
        if event.scene_id not in self._scene_managers:
            logger.warning(
                "SliceCoordinator received DimsChangedEvent for unregistered "
                "scene %s — ignoring.", event.scene_id
            )
            return
        self._cancel_scene(event.scene_id)
        asyncio.create_task(
            self._submit_async(event.scene_id, event.dims_state)
        )

    async def _submit_async(
        self, scene_id: UUID, dims_state: DimsState
    ) -> None:
        requests = self._scene_managers[scene_id].build_slice_requests(...)
        for visual_id, chunk_requests in requests.items():
            await self._slicer.submit(chunk_requests, ...)
```

---

## How the controller bridges psygnal to the bus

The controller is the only place where psygnal models and the EventBus coexist. When a
psygnal field event fires on a model — e.g. `visual.appearance.color_map = "plasma"`
triggers `appearance.events.color_map.emit()` — the controller converts it into a typed
bus event and calls `bus.emit()`.

This bridging is done by ordinary private methods on the controller — plain psygnal
callbacks that read signal information and call `bus.emit()`. The bus has no psygnal
dependency and no knowledge of model types. There is nothing architecturally special
here; it is just the controller doing its job of connecting the two systems.

The only subtlety is Python's loop closure semantics. When the controller registers
callbacks for multiple entities in a loop (e.g. wiring a handler for each scene's
`DimsManager`), all closures in the loop would share the same loop variable binding
without a factory function. Every handler would end up using the last entity's ID.
A factory that takes the entity ID as an argument and returns a closure fixes this by
capturing the value at call time.

```python
# WRONG — all scene handlers capture the same `scene` reference from the loop
for scene in self._scenes.values():
    scene.dims.events.connect(lambda info: self._on_dims_psygnal(info, scene.id))

# CORRECT — factory captures scene_id by value at call time
def _make_dims_handler(self, scene_id: UUID) -> Callable:
    def _handler(info: EmissionInfo) -> None:
        self._on_dims_psygnal(info, scene_id)
    return _handler

for scene in self._scenes.values():
    scene.dims.events.connect(self._make_dims_handler(scene.id))
```

### DimsManager wiring

```python
def _wire_dims_model(self, scene: Scene) -> None:
    """Connect a scene's DimsManager psygnal events to the bus."""
    self._dims_cache[scene.id] = scene.dims.displayed_axes
    scene.dims.events.connect(self._make_dims_handler(scene.id))

def _make_dims_handler(self, scene_id: UUID) -> Callable:
    def _on_dims_psygnal(info: EmissionInfo) -> None:
        dims: DimsManager = Signal.sender()
        new_state = dims.to_state()
        prev_axes = self._dims_cache[scene_id]
        displayed_axes_changed = (prev_axes != new_state.displayed_axes)
        self._dims_cache[scene_id] = new_state.displayed_axes
        self._event_bus.emit(DimsChangedEvent(
            source_id=self._id,
            scene_id=scene_id,
            dims_state=new_state,
            displayed_axes_changed=displayed_axes_changed,
        ))
    return _on_dims_psygnal
```

### ImageAppearance wiring

```python
_RESLICE_FIELDS: frozenset[str] = frozenset({"lod_bias", "force_level", "frustum_cull"})

def _wire_appearance(self, visual: BaseVisual) -> None:
    """Connect a visual's appearance psygnal events to the bus."""
    visual.appearance.events.connect(self._make_appearance_handler(visual.id))

def _make_appearance_handler(self, visual_id: UUID) -> Callable:
    def _on_appearance_psygnal(info: EmissionInfo) -> None:
        field_name = info.signal.name
        new_value = getattr(Signal.sender(), field_name)
        if field_name == "visible":
            self._event_bus.emit(VisualVisibilityChangedEvent(
                source_id=self._id,
                visual_id=visual_id,
                visible=new_value,
            ))
        else:
            self._event_bus.emit(AppearanceChangedEvent(
                source_id=self._id,
                visual_id=visual_id,
                field_name=field_name,
                new_value=new_value,
                requires_reslice=(field_name in _RESLICE_FIELDS),
            ))
    return _on_appearance_psygnal
```

The pattern scales to any new model type. The factory always captures the entity's ID
by value. The psygnal callback always reads from `Signal.sender()` and `EmissionInfo`.
The result is always a typed bus event emitted with `self._id` as `source_id`, because
these changes originate from application code, not from the render layer.

---

## Registration and deregistration lifecycle

Every render-layer object passes its own ID as `owner_id` when subscribing. When the
controller removes an entity, it calls `bus.unsubscribe_all(entity_id)` to clean up
all subscriptions registered by that entity atomically.

```python
# Controller: wiring a new GFX visual
bus.subscribe(
    AppearanceChangedEvent,
    gfx_visual.on_appearance_changed,
    entity_id=visual_id,
    owner_id=visual_id,
)
bus.subscribe(
    DataStoreContentsChangedEvent,
    gfx_visual.on_data_store_contents_changed,
    entity_id=data_store_id,
    owner_id=visual_id,       # owned by the visual, not by the data store
)
bus.subscribe(
    VisualVisibilityChangedEvent,
    gfx_visual.on_visibility_changed,
    entity_id=visual_id,
    owner_id=visual_id,
)

# Controller: removing the visual
bus.unsubscribe_all(owner_id=visual_id)  # removes all three in one call
```

External callbacks registered by the application developer are owned by the controller:

```python
# Application code
controller.on_dims_changed(scene_id, my_callback)

# Controller internal
def on_dims_changed(self, scene_id: UUID, callback: Callable) -> None:
    handle = self._bus.subscribe(
        DimsChangedEvent,
        lambda e: callback(e.dims_state),
        entity_id=scene_id,
        owner_id=self._id,
    )
    self._external_handles.append(handle)
```

---

## Subscription matrix

Canonical specification for what the controller wires at construction time.

| Subscriber | Event type | `entity_id` filter | Action |
|---|---|---|---|
| `CanvasView` | `DimsChangedEvent` | `scene_id` | Switch active camera if `displayed_axes_changed` |
| `CanvasView` | `CameraChangedEvent` | `scene_id` | Apply camera state; skip if `source_id == self._id` |
| `SliceCoordinator` | `DimsChangedEvent` | *(none — all scenes)* | Schedule async reslice for the changed scene |
| `SliceCoordinator` | `AppearanceChangedEvent` | *(none — all visuals)* | Schedule async reslice for the visual if `requires_reslice` |
| `SliceCoordinator` | `DataStoreMetadataChangedEvent` | *(none)* | Evict all; reslice all visuals using this store |
| `SliceCoordinator` | `DataStoreContentsChangedEvent` | *(none)* | Evict dirty bricks; reslice affected visuals |
| `GFX*Visual` | `AppearanceChangedEvent` | `visual_id` | Apply material / shader parameter |
| `GFX*Visual` | `VisualVisibilityChangedEvent` | `visual_id` | Show / hide node |
| `GFX*Visual` | `DataStoreMetadataChangedEvent` | `data_store_id` | Rebuild `VolumeGeometry`; evict `BrickCache` |
| `GFX*Visual` | `DataStoreContentsChangedEvent` | `data_store_id` | Evict dirty bricks from `BrickCache` |
| `DimsSliderWidget` | `DimsChangedEvent` | `scene_id` | Update slider position; skip if `source_id == self._id` |
| Controller ext. callbacks | `DimsChangedEvent` | `scene_id` | Fire `on_dims_changed` user callback |
| Controller ext. callbacks | `CameraChangedEvent` | `scene_id` | Fire `on_camera_changed` user callback |
| Controller ext. callbacks | `AppearanceChangedEvent` | `visual_id` | Fire `on_visual_changed` user callback |

`SliceCoordinator` subscribes without `entity_id` filters because it manages all scenes
centrally. It resolves the correct `SceneManager` from `event.scene_id` or
`event.visual_id` internally.

---

## End-to-end example: appearance change propagation

This example traces a single line of application code — a colormap change — from the
developer's mutation through every layer of the system to the rendered frame. It covers
both the GPU-only path (`color_map`, `clim`) and the reslice path (`lod_bias`,
`force_level`, `frustum_cull`), using the multiscale volume viewer from
`scripts/v2/lut_texture_multiscale_frustum_async_cellier/example_cellier.py` as
the concrete context.

```
Application code
      ↓
psygnal EventedModel          ← color_map signal fires
      ↓
Controller adapter            ← emits AppearanceChangedEvent
      ↓
EventBus.emit()               ← dispatches to subscribers
      ↙ requires_reslice=False          requires_reslice=True ↘
GFXImageVisual               SliceCoordinator
on_appearance_changed()       asyncio.create_task()
      ↓                                ↓
pygfx renderer               AsyncSlicer
deferred GPU upload           concurrent brick reads
```

### The trigger

The developer holds a live `MultiscaleImageVisual` model reference returned by
`controller.add_image()`. A single property mutation is all that is needed:

```python
# Application code — nothing else required
self._visual.appearance.color_map = "plasma"
```

Everything below happens automatically. The developer never interacts with the EventBus,
the `GFXMultiscaleImageVisual`, or any render-layer object.

### Stage 1 — psygnal detects the mutation

`ImageAppearance` is a psygnal `EventedModel`. Setting `color_map` causes psygnal to
fire the signal for that field synchronously. The controller connected a callback to
this signal during `_wire_appearance()`:

```python
# Inside psygnal — not code we write; shown for clarity
appearance.events.color_map.emit(Colormap("plasma"))
# → calls the closure returned by controller._make_appearance_handler(visual_id)
```

### Stage 2 — Controller adapter converts to a bus event

The closure produced by `_make_appearance_handler` runs. It reads the changed field name
from `EmissionInfo` and the new value from `Signal.sender()`, determines
`requires_reslice=False` (because `"color_map"` is not in `_RESLICE_FIELDS`), and emits
a typed event:

```python
# Inside the controller closure — _on_appearance_psygnal
def _on_appearance_psygnal(info: EmissionInfo) -> None:
    field_name = info.signal.name          # "color_map"
    new_value = getattr(Signal.sender(), field_name)  # Colormap("plasma")
    self._event_bus.emit(AppearanceChangedEvent(
        source_id=self._id,                # controller's UUID
        visual_id=visual_id,               # captured by the factory
        field_name="color_map",
        new_value=Colormap("plasma"),
        requires_reslice=False,            # not in _RESLICE_FIELDS
    ))
```

### Stage 3 — EventBus dispatches to subscribers

`emit()` looks up `AppearanceChangedEvent` in `_subscriptions`. Two entries have
`entity_id == visual_id` and both fire:

1. `GFXMultiscaleImageVisual.on_appearance_changed` (registered with `entity_id=visual_id`)
2. `SliceCoordinator._on_appearance_changed` (registered with no `entity_id` filter)

An external `on_visual_changed` callback fires too, if the application registered one.

### Stage 4 — `GFXMultiscaleImageVisual.on_appearance_changed` (render layer)

This handler — **not yet implemented**, specified here — receives the event and applies
the change directly to the pygfx material. The conversion from `cmap.Colormap` to
`gfx.TextureMap` uses the existing `cmap_to_gfx_colormap()` utility in
`src/cellier/v2/render/pygfx_utils.py`. The `if node is not None` guards handle visuals
built with only `render_modes={"2d"}` or only `render_modes={"3d"}`:

```python
# In GFXMultiscaleImageVisual
_RESLICE_FIELDS: frozenset[str] = frozenset({"lod_bias", "force_level", "frustum_cull"})

def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
    """Handle AppearanceChangedEvent — update GPU-side material parameters.

    Notes
    -----
    This handler must NOT block or await.  Async work is never initiated
    here; that is the SliceCoordinator's responsibility.
    """
    if event.field_name == "color_map":
        new_map = cmap_to_gfx_colormap(event.new_value)
        if self.material_3d is not None:
            self.material_3d.map = new_map
        if self.material_2d is not None:
            self.material_2d.map = new_map

    elif event.field_name == "clim":
        if self.material_3d is not None:
            self.material_3d.clim = event.new_value
        if self.material_2d is not None:
            self.material_2d.clim = event.new_value

    elif event.field_name == "visible":
        pass  # handled by on_visibility_changed via VisualVisibilityChangedEvent

    elif event.field_name in _RESLICE_FIELDS:
        # lod_bias, force_level, frustum_cull: no GPU state to update here.
        # The SliceCoordinator schedules the async reslice; the new bricks
        # that arrive will reflect the updated planning parameters.
        pass
```

Setting `material_3d.map` marks the material dirty in pygfx. No GPU work happens yet.

### Stage 5 — `SliceCoordinator._on_appearance_changed`

The coordinator also receives the event. It checks `requires_reslice` and returns
immediately for `color_map` changes — no async work is scheduled:

```python
# In SliceCoordinator
def _on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
    if event.visual_id not in self._visual_to_scene:
        logger.warning(
            "SliceCoordinator received AppearanceChangedEvent for "
            "unregistered visual %s — ignoring.", event.visual_id
        )
        return
    if not event.requires_reslice:
        return  # color_map and clim: GPU-only change, no reslice needed
    scene_id = self._visual_to_scene[event.visual_id]
    self._cancel_visual(scene_id, event.visual_id)
    asyncio.create_task(
        self._submit_async(
            scene_id,
            target_visual_ids=frozenset({event.visual_id}),
        )
    )
```

### Stage 6 — pygfx deferred GPU upload

On the next call to `renderer.render()` — which happens in `CanvasView._draw_frame()`
on the rendercanvas repaint callback — pygfx detects the dirty material and uploads the
new colormap texture to the GPU:

```python
# In CanvasView._draw_frame — already exists, no changes needed
def _draw_frame(self) -> None:
    self._renderer.render(self._scene, self._camera, flush=True)
    # ↑ pygfx sees material_3d.map is dirty, uploads the new TextureMap.
    # The rendered frame uses the new colormap from this point forward.
```

### The reslice path: `lod_bias`, `force_level`, `frustum_cull`

When a field that requires reslicing changes, the path diverges at Stage 2:

```python
# lod_bias, force_level, frustum_cull → requires_reslice=True
self._visual.appearance.lod_bias = 2.0
```

The `AppearanceChangedEvent` is emitted with `requires_reslice=True`. The
`GFXMultiscaleImageVisual` handler does nothing (the `elif event.field_name in
_RESLICE_FIELDS: pass` branch). The `SliceCoordinator` handler cancels any in-flight
task for this visual and schedules a new async reslice via `asyncio.create_task()`. The
new `lod_bias` value is read from the `ImageAppearance` model at planning time, inside
`GFXMultiscaleImageVisual.build_slice_request()` — the event payload itself does not
need to carry planning parameters.

### Key observations

**Conversion responsibility.** `AppearanceChangedEvent.new_value` carries a
`cmap.Colormap`. The render layer converts it to `gfx.TextureMap` using
`cmap_to_gfx_colormap()`. The bus and controller never import pygfx types.

**`requires_reslice=False` costs nothing async.** The `SliceCoordinator` returns
immediately. No tasks are created, no bricks are invalidated, and the existing brick
cache remains fully valid. A colormap change is pure GPU state.

**Both subscribers always receive the event.** The EventBus does not route to one
or the other based on `requires_reslice`. Both `GFXMultiscaleImageVisual` and
`SliceCoordinator` always receive `AppearanceChangedEvent` for the visual they're
interested in. The `requires_reslice` flag is checked inside each subscriber, not by
the bus.

**The `if node is not None` guard is load-bearing.** A visual constructed with
`render_modes={"2d"}` has `material_3d = None`. The handler must check before every
material assignment. This is not defensive programming — it is the correct behaviour
for 2D-only visuals.

---

## What does not flow through the bus

- **Data payloads.** Brick data from `AsyncSlicer` is delivered directly to
  `GFX*Visual.on_data_ready()` via the `callback` argument to `slicer.submit()`.
  Only the lifecycle signal (`ResliceCompletedEvent`) goes through the bus.
- **Structural add/remove operations.** `add_visual()`, `remove_visual()`, etc. are
  synchronous and handled atomically by the controller. `VisualAddedEvent` and
  `VisualRemovedEvent` are emitted *after* the operation completes, for external
  observers such as a layer list widget.
- **Frame render loop.** The pygfx render loop runs via rendercanvas. `CanvasView`
  emits `FrameRenderedEvent` from its draw callback, but the loop itself is not
  bus-driven.

---

## Module exports

```python
# src/cellier/v2/events/__init__.py

from cellier.v2.events._bus import EventBus, SubscriptionHandle, make_weak_callback
from cellier.v2.events._events import (
    AppearanceChangedEvent,
    CameraChangedEvent,
    CellierEventTypes,
    DataStoreContentsChangedEvent,
    DataStoreMetadataChangedEvent,
    DimsChangedEvent,
    FrameRenderedEvent,
    ResliceCancelledEvent,
    ResliceCompletedEvent,
    ResliceStartedEvent,
    SceneAddedEvent,
    SceneRemovedEvent,
    VisualAddedEvent,
    VisualRemovedEvent,
    VisualVisibilityChangedEvent,
)
# Re-export shared state types — single import location for all subscribers
from cellier.v2._state import CameraState, DimsState

__all__ = [
    "CameraState",
    "CellierEventTypes",
    "DimsState",
    "EventBus",
    "make_weak_callback",
    "SubscriptionHandle",
    "AppearanceChangedEvent",
    "CameraChangedEvent",
    "DataStoreContentsChangedEvent",
    "DataStoreMetadataChangedEvent",
    "DimsChangedEvent",
    "FrameRenderedEvent",
    "ResliceCancelledEvent",
    "ResliceCompletedEvent",
    "ResliceStartedEvent",
    "SceneAddedEvent",
    "SceneRemovedEvent",
    "VisualAddedEvent",
    "VisualRemovedEvent",
    "VisualVisibilityChangedEvent",
]
```

---

## Pending

- **`MouseEvent` family.** To be designed once the interaction model for picking,
  selection, and annotation is specified. Will follow the same NamedTuple / `source_id`
  pattern. Likely types: `CanvasMouseClickEvent`, `VisualPickedEvent`,
  `SelectionChangedEvent`.
