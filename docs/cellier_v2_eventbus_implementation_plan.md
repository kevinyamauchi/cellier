# Cellier v2 EventBus — Implementation Plan

## Context and scope

This plan covers the incremental implementation of the v2 EventBus as specified in
`cellier_v2_event_bus_design.md`. It starts from the current Phase 1 codebase
(`src/cellier/v2/controller.py` complete, no `events/` package, no `_state.py`) and
proceeds in seven independently-verifiable steps.

---

## Current state summary

| What exists | Notes |
|---|---|
| `src/cellier/v2/controller.py` | Phase 1 complete; no `_id`, no `_event_bus`, no psygnal wiring |
| `src/cellier/v2/render/_requests.py` | `DimsState` currently lives here |
| `src/cellier/v2/render/__init__.py` | Re-exports `DimsState` from `_requests.py` |
| `src/cellier/v2/visuals/_image.py` | `ImageAppearance` with `lod_bias`, `force_level`, `frustum_cull` — **confirmed present** |
| `src/cellier/v2/scene/dims.py` | `DimsManager` with direct `displayed_axes` and `slice_indices` fields; **no `to_state()` method** |
| No `src/cellier/v2/_state.py` | Does not exist yet |
| No `src/cellier/v2/events/` | Package does not exist yet |

---

## Design decisions resolved before implementation

### D1 — `DimsManager.to_state()` does not exist

Inspecting `src/cellier/v2/scene/dims.py`: the v2 `DimsManager` is a flat psygnal
`EventedModel` with `displayed_axes: tuple[int, ...]` and `slice_indices: tuple[int, ...]`
as direct fields. There is no `to_state()` method.

**Decision:** Do not add `to_state()` to `DimsManager`. Build `DimsState` directly in
the controller's psygnal bridge:

```python
new_state = DimsState(
    displayed_axes=dims.displayed_axes,
    slice_indices=dims.slice_indices,
)
```

This keeps `DimsManager` unchanged and the bridge self-contained.

---

### D2 — psygnal catch-all is confirmed working; use `info.args[0]` for the value

psygnal `EventedModel` offers two ways to connect:

**Per-field** — connect to each field's dedicated signal individually:
```python
appearance.events.color_map.connect(handler_a)
appearance.events.clim.connect(handler_b)
```
You must enumerate every field you care about.

**Catch-all** — connect once; the handler receives an `EmissionInfo` for any field:
```python
appearance.events.connect(handler)
# info.signal.name  → name of the field that changed
# info.args[0]      → the new value
```

The catch-all is already used in production in the v1 codebase
(`src/cellier/models/visuals/base.py`):

```python
def _on_appearance_change(self, event: EmissionInfo):
    property_name = event.signal.name
    property_value = event.args[0]
```

**Decision:** Use the catch-all style. One handler covers all appearance fields,
including any future additions.

**Correction from design doc:** the design doc uses `getattr(Signal.sender(), field_name)`
for the new value. The v1 codebase confirms `info.args[0]` is the correct idiom.
Use `info.args[0]`.

---

### D3 — Camera event wiring is deferred (dormant)

`CameraChangedEvent` is defined and exported, but no emitter or subscriber is wired
in this phase. `CanvasView` already has the `_applying_model_state` guard in place.

---

### D4 — Button-triggered update is the only reslice trigger

`SliceCoordinator` bus subscriptions for `DimsChangedEvent` and `AppearanceChangedEvent`
are **not wired** in this phase. The Update button via `RenderManager.trigger_update()`
remains the sole reslice path. Coordinator handler stubs are added to the class body
but not subscribed to the bus.

---

### D5 — `ImageAppearance` render fields confirmed present

`src/cellier/v2/visuals/_image.py` already contains `lod_bias: float = 1.0`,
`force_level: int | None = None`, and `frustum_cull: bool = True`. No changes needed.

---

## Implementation steps

### Step 1 — Create `src/cellier/v2/_state.py`

**Files changed:** `_state.py` (new), `render/_requests.py`, `render/__init__.py`

Move `DimsState` from `render/_requests.py` to `_state.py` and add `CameraState`.
Update `_requests.py` to re-import from the new location; all consumers of
`cellier.v2.render.DimsState` continue to work unchanged.

**`src/cellier/v2/_state.py`** (new file):

```python
"""Shared immutable state snapshots — used by events and render layer."""

from __future__ import annotations

from typing import Literal, NamedTuple


class DimsState(NamedTuple):
    """Current dimension display state for a scene."""
    displayed_axes: tuple[int, ...]
    slice_indices: tuple[int, ...]


class CameraState(NamedTuple):
    """Immutable snapshot of the active camera's logical state."""
    camera_type: Literal["perspective", "orthographic"]
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    up: tuple[float, float, float]
    fov: float
    zoom: float
    extent: tuple[float, float]
    depth_range: tuple[float, float]
```

**`src/cellier/v2/render/_requests.py`** — replace the inline `DimsState` definition
with:

```python
from cellier.v2._state import DimsState  # moved; re-exported for backwards compat
```

**Verification:**

```python
from cellier.v2._state import CameraState, DimsState
from cellier.v2.render import DimsState as DS
assert DimsState is DS
```

---

### Step 2 — Create the `events/` package with `_events.py`

**Files changed:** `events/__init__.py` (new), `events/_events.py` (new)

Create `src/cellier/v2/events/` and populate `_events.py` with all 14 event
NamedTuples and the `CellierEventTypes` union alias as specified in the design doc.
`__init__.py` exports all event types and re-exports `CameraState` and `DimsState`
from `_state.py`. No bus implementation yet — pure data-structure creation.

**All 14 event types:**

| Class | Key fields beyond `source_id` |
|---|---|
| `DimsChangedEvent` | `scene_id`, `dims_state`, `displayed_axes_changed` |
| `CameraChangedEvent` | `scene_id`, `camera_state` |
| `AppearanceChangedEvent` | `visual_id`, `field_name`, `new_value`, `requires_reslice` |
| `VisualVisibilityChangedEvent` | `visual_id`, `visible` |
| `DataStoreMetadataChangedEvent` | `data_store_id` |
| `DataStoreContentsChangedEvent` | `data_store_id`, `dirty_keys` |
| `ResliceStartedEvent` | `scene_id`, `visual_ids` |
| `ResliceCompletedEvent` | `scene_id`, `visual_id`, `brick_count` |
| `ResliceCancelledEvent` | `scene_id`, `visual_id` |
| `FrameRenderedEvent` | `canvas_id`, `frame_time_ms` |
| `VisualAddedEvent` | `scene_id`, `visual_id` |
| `VisualRemovedEvent` | `scene_id`, `visual_id` |
| `SceneAddedEvent` | `scene_id` |
| `SceneRemovedEvent` | `scene_id` |

**Verification:**

```python
from cellier.v2.events import DimsChangedEvent, DimsState
from uuid import uuid4
e = DimsChangedEvent(
    source_id=uuid4(),
    scene_id=uuid4(),
    dims_state=DimsState(displayed_axes=(0, 1, 2), slice_indices=()),
    displayed_axes_changed=False,
)
assert isinstance(e, tuple)
assert e.dims_state.displayed_axes == (0, 1, 2)
```

---

### Step 3 — Create `events/_bus.py` and wire `_ENTITY_FIELD`

**Files changed:** `events/_bus.py` (new), `events/__init__.py` (update to add bus exports)

Implement `make_weak_callback`, `SubscriptionHandle`, and `EventBus` exactly as
specified in the design doc. Add the `_ENTITY_FIELD` module-level dict mapping all 14
event types to their canonical entity field names:

```python
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

**New test file:** `tests/v2/events/test_event_bus.py`

One test per bus invariant, no Qt/pygfx/async:

| Test | Invariant |
|---|---|
| `test_emit_no_subscribers_is_noop` | No error when no subscribers registered |
| `test_subscribe_no_filter_fires_always` | `entity_id=None` fires for any entity |
| `test_subscribe_with_filter_fires_only_for_matching_entity` | `entity_id=X` skips non-X |
| `test_unsubscribe_removes_entry` | Subsequent emit does not call removed handler |
| `test_unsubscribe_all_removes_all_owned` | All subscriptions under `owner_id` removed |
| `test_weak_subscription_dies_with_referent` | Dead weak entry cleaned up on next emit |
| `test_lambda_with_weak_raises` | `subscribe(lambda, weak=True)` raises `ValueError` |
| `test_subscriber_exception_propagates_after_others_run` | First exc re-raised; others still fire |

**Verification:** `uv run pytest tests/v2/events/`

---

### Step 4 — Add `_id`, `_event_bus`, and psygnal bridges to `CellierController`

**Files changed:** `src/cellier/v2/controller.py`

#### 4a — New instance attributes in `__init__`

```python
from cellier.v2._state import DimsState
from cellier.v2.events import (
    AppearanceChangedEvent,
    DimsChangedEvent,
    EventBus,
    SceneAddedEvent,
    SubscriptionHandle,
    VisualAddedEvent,
    VisualVisibilityChangedEvent,
)

# in __init__:
self._id: UUID = uuid4()
self._event_bus: EventBus = EventBus()
self._dims_cache: dict[UUID, tuple[int, ...]] = {}
self._external_handles: list[SubscriptionHandle] = []
```

#### 4b — DimsManager bridge

Call `self._wire_dims_model(scene)` at the end of `add_scene()`.

```python
def _wire_dims_model(self, scene: Scene) -> None:
    self._dims_cache[scene.id] = scene.dims.displayed_axes
    scene.dims.events.connect(self._make_dims_handler(scene.id))

def _make_dims_handler(self, scene_id: UUID) -> Callable:
    def _on_dims_psygnal(info: EmissionInfo) -> None:
        dims = self._model.scenes[scene_id].dims
        new_state = DimsState(
            displayed_axes=dims.displayed_axes,
            slice_indices=dims.slice_indices,
        )
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

> **Why look up dims via `self._model` rather than `Signal.sender()`?**
> `Signal.sender()` is a psygnal global that returns the model instance that fired.
> It works, but looking up through the model is more explicit and avoids a dependency
> on that API. The `scene_id` is captured by the factory closure, so there is no
> loop-variable binding bug.

#### 4c — ImageAppearance bridge

Call `self._wire_appearance(visual_model)` at the end of `add_image()`.

```python
_RESLICE_FIELDS: frozenset[str] = frozenset({"lod_bias", "force_level", "frustum_cull"})

def _wire_appearance(self, visual: MultiscaleImageVisual) -> None:
    visual.appearance.events.connect(self._make_appearance_handler(visual.id))

def _make_appearance_handler(self, visual_id: UUID) -> Callable:
    def _on_appearance_psygnal(info: EmissionInfo) -> None:
        field_name = info.signal.name   # which field changed
        new_value = info.args[0]        # new value — confirmed v1 pattern
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

#### 4d — Structural event emission

At the end of each `add_*` method, after all wiring is complete:

```python
# end of add_scene():
self._event_bus.emit(SceneAddedEvent(source_id=self._id, scene_id=scene.id))

# end of add_image():
self._event_bus.emit(VisualAddedEvent(
    source_id=self._id,
    scene_id=scene_id,
    visual_id=visual_model.id,
))
```

#### 4e — External callback registration methods

Implement the three `on_*` methods deferred from Phase 1:

```python
def on_dims_changed(
    self, scene_id: UUID, callback: Callable[[DimsState], None]
) -> None:
    handle = self._event_bus.subscribe(
        DimsChangedEvent,
        lambda e: callback(e.dims_state),
        entity_id=scene_id,
        owner_id=self._id,
    )
    self._external_handles.append(handle)

def on_camera_changed(
    self, scene_id: UUID, callback: Callable[[CameraState], None]
) -> None:
    handle = self._event_bus.subscribe(
        CameraChangedEvent,
        lambda e: callback(e.camera_state),
        entity_id=scene_id,
        owner_id=self._id,
    )
    self._external_handles.append(handle)

def on_visual_changed(
    self, visual_id: UUID, callback: Callable
) -> None:
    handle = self._event_bus.subscribe(
        AppearanceChangedEvent,
        lambda e: callback(self.get_visual_model(e.visual_id)),
        entity_id=visual_id,
        owner_id=self._id,
    )
    self._external_handles.append(handle)
```

**Verification (no Qt):**

```python
fired = []
controller.on_dims_changed(scene.id, lambda ds: fired.append(ds))
controller.get_dims_model(scene.id).slice_indices = (5,)
assert len(fired) == 1 and fired[0].slice_indices == (5,)
```

---

### Step 5 — Wire render-layer subscriptions

**Files changed:** `src/cellier/v2/render/visuals/_image.py` (add handler methods),
`src/cellier/v2/controller.py` (subscribe calls in `add_image()`),
`src/cellier/v2/render/canvas_view.py` (add `self._id` if not already present),
`src/cellier/v2/render/slice_coordinator.py` (add dormant handler stubs)

**Architecture:** The controller performs all `bus.subscribe()` calls. Render-layer
objects expose named handler methods but never import or hold the bus themselves.

#### 5a — `GFXMultiscaleImageVisual` handler methods

```python
def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
    """Apply GPU-only appearance changes.
    Reslice-field changes (lod_bias, force_level, frustum_cull) are
    no-ops here; new values are read from the model at the next
    trigger_update() call.
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
    # lod_bias / force_level / frustum_cull: no GPU state to update here.

def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
    if self.node_3d is not None:
        self.node_3d.visible = event.visible
    if self.node_2d is not None:
        self.node_2d.visible = event.visible

def on_data_store_contents_changed(self, event: DataStoreContentsChangedEvent) -> None:
    pass  # stub — brick eviction deferred to a future phase

def on_data_store_metadata_changed(self, event: DataStoreMetadataChangedEvent) -> None:
    pass  # stub — geometry rebuild deferred to a future phase
```

#### 5b — Subscribe calls in `controller.add_image()`

```python
self._event_bus.subscribe(
    AppearanceChangedEvent,
    gfx_visual.on_appearance_changed,
    entity_id=visual_id,
    owner_id=visual_id,
)
self._event_bus.subscribe(
    VisualVisibilityChangedEvent,
    gfx_visual.on_visibility_changed,
    entity_id=visual_id,
    owner_id=visual_id,
)
self._event_bus.subscribe(
    DataStoreContentsChangedEvent,
    gfx_visual.on_data_store_contents_changed,
    entity_id=data_store_id,
    owner_id=visual_id,   # owned by the visual, not the store
)
```

#### 5c — `CanvasView` — add `self._id`

Confirm or add to `CanvasView.__init__`:

```python
self._id: UUID = uuid4()
# self._applying_model_state already present from render architecture plan
```

Camera subscription is not wired (D3). No other changes to `CanvasView`.

#### 5d — `SliceCoordinator` dormant stubs

```python
def _on_dims_changed(self, event: DimsChangedEvent) -> None:
    """Dormant stub — not yet subscribed to the bus.
    Will schedule async reslice when bus-driven triggers are enabled
    in a future phase."""
    pass

def _on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
    """Dormant stub — not yet subscribed to the bus.
    Will schedule async reslice for requires_reslice=True fields
    in a future phase."""
    pass
```

**Verification:**

- `uv run pytest tests/v2/render/` — all existing tests still pass
- `uv run example_cellier.py` — Update button still works
- Programmatically mutate `visual.appearance.color_map`; confirm material updates
  on the next rendered frame without pressing Update

---

### Step 6 — Teardown: `unsubscribe_all` in remove stubs

**Files changed:** `src/cellier/v2/controller.py`

Add bus cleanup as the first action in the `remove_*` stubs. The stubs still raise
`NotImplementedError` for the render-layer teardown, but cleanup is safe to do now:

```python
def remove_visual(self, visual_id: UUID) -> None:
    self._event_bus.unsubscribe_all(visual_id)
    self._visual_to_scene.pop(visual_id, None)
    raise NotImplementedError(
        "Visual render-layer teardown not yet implemented. "
        "Bus subscriptions have been cleaned up."
    )

def remove_scene(self, scene_id: UUID) -> None:
    for canvas_id in self._scene_to_canvases.pop(scene_id, []):
        self._event_bus.unsubscribe_all(canvas_id)
        self._canvas_to_scene.pop(canvas_id, None)
    self._event_bus.unsubscribe_all(scene_id)
    raise NotImplementedError(
        "Scene render-layer teardown not yet implemented. "
        "Bus subscriptions have been cleaned up."
    )
```

Note: `SliceCoordinator` subscriptions are owned by the coordinator's own ID (not by
scene/visual IDs), so they survive individual visual/scene removal correctly.

**Verification:**

```python
visual = controller.add_image(...)
fired = []
controller._event_bus.subscribe(
    AppearanceChangedEvent, lambda e: fired.append(e), entity_id=visual.id
)
controller._event_bus.unsubscribe_all(visual.id)
visual.appearance.color_map = "plasma"
assert fired == []
```

---

### Step 7 — Tests and polish

**Files changed:** `tests/v2/test_controller.py` (extend)

New tests to add (all headless — no Qt, no GPU, no async):

| Test | What it covers |
|---|---|
| `test_dims_bridge_emits_event` | Mutate `DimsManager.slice_indices`; assert callback fires with correct `DimsState` |
| `test_dims_bridge_displayed_axes_flag_true` | Mutate `DimsManager.displayed_axes`; assert `displayed_axes_changed=True` |
| `test_dims_bridge_displayed_axes_flag_false` | Mutate `slice_indices` only; assert `displayed_axes_changed=False` |
| `test_appearance_bridge_color_map` | Mutate `color_map`; assert `AppearanceChangedEvent` with `requires_reslice=False` |
| `test_appearance_bridge_lod_bias` | Mutate `lod_bias`; assert `requires_reslice=True` |
| `test_appearance_bridge_force_level` | Mutate `force_level`; assert `requires_reslice=True` |
| `test_appearance_bridge_visible` | Mutate `visible=False`; assert `VisualVisibilityChangedEvent` fired, no `AppearanceChangedEvent` |
| `test_scene_added_event_emitted` | `add_scene()` emits `SceneAddedEvent` |
| `test_visual_added_event_emitted` | `add_image()` emits `VisualAddedEvent` |
| `test_on_dims_changed_callback` | `on_dims_changed()` fires user callback with `DimsState` |
| `test_unsubscribe_all_cleans_up` | Subscribe; call `unsubscribe_all`; emit; assert no callback |

**Verification:** `uv run pytest tests/v2/` — all tests pass

---

## Implementation order summary

| Step | Files changed | Verifiable |
|---|---|---|
| 1 | `_state.py` (new), `render/_requests.py`, `render/__init__.py` | REPL import check |
| 2 | `events/_events.py` (new), `events/__init__.py` (new) | REPL NamedTuple instantiation |
| 3 | `events/_bus.py` (new), `events/__init__.py` (update) | `pytest tests/v2/events/` |
| 4 | `controller.py` — `_id`, bus, bridges, `on_*`, structural events | `pytest tests/v2/test_controller.py` (no Qt) |
| 5 | `visuals/_image.py` (GFX handlers), `controller.py` (subscribe calls), `canvas_view.py` (`_id`), `slice_coordinator.py` (stubs) | Manual `example_cellier.py`; appearance mutation test |
| 6 | `controller.py` — teardown stubs | `pytest tests/v2/test_controller.py` |
| 7 | `tests/v2/test_controller.py` (extend) | `pytest tests/v2/` all pass |

Steps 1–3 are pure Python with no external dependencies and can be done in one sitting.
Steps 4–7 are each independently testable before proceeding to the next.

---

## What is NOT in scope for this phase

| Item | Reason |
|---|---|
| Camera event wiring | Defined dormant (D3); wiring deferred |
| Bus-driven reslice | Button-only trigger chosen for this phase (D4) |
| `MouseEvent` family | Interaction model not yet specified |
| `DimsSliderWidget` | No v2 dims slider widget exists yet |
| `FrameRenderedEvent` emission | No consumer yet |
| `DataStoreContentsChangedEvent` / `DataStoreMetadataChangedEvent` emit | Requires psygnal-evented data store mutation API, not yet designed |
| Full camera capture in `to_model()` / `to_file()` | Deferred since Phase 1 |
