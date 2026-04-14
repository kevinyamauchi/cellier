# Cellier v2 Widget Design

This document describes the pattern for building cellier-aware GUI widgets —
controls that stay in sync with the viewer model in both directions without
event loops.  The target audience is a developer who wants to implement a new
widget (a slider, combo box, checkbox, etc.) that reads from and writes to the
cellier model layer.

---

## The problem: bidirectional sync without loops

A naive approach to keeping a spinbox in sync with a model value is:

1. When the spinbox changes → write to the model.
2. When the model changes → write to the spinbox.

The problem is that step 2 fires the spinbox's change signal, which triggers
step 1 again, which fires a model change, which triggers step 2 again — an
infinite loop.

Three mechanisms work together to break the loop:

### 1. Source-ID echo filtering (bus level)

Every event on the cellier `EventBus` carries a `source_id: UUID` identifying
who caused the change.  When a widget mutates the model it passes its own UUID
as `source_id`.  The controller stamps that UUID on the emitted event.  The
widget's bus handler checks this field at the top:

```python
def _on_dims_changed(self, event) -> None:
    if event.source_id == self._id:
        return  # this change came from us; ignore the echo
    ...
```

This handles the inter-layer round-trip: widget → model → bus → widget.

### 2. Signal blocking (toolkit level)

When the bus handler *does* need to update the widget (because the change came
from somewhere else), setting the widget's value programmatically would fire
the widget's own change signal.  Block it first:

```python
def _set_value(self, value: int) -> None:
    self._spinbox.blockSignals(True)
    self._spinbox.setValue(value)
    self._spinbox.blockSignals(False)
```

This handles the intra-widget round-trip: bus handler sets widget → widget
fires signal → widget handler would write to model again.

### 3. One UUID per widget (not per window)

The UUID must belong to the smallest independent actor — the individual widget,
not its parent window.  If two widgets (e.g. a spinbox and a slider) both
control the same model field and share one UUID, a change from the spinbox
would suppress the update to the slider.  With separate UUIDs each widget only
filters its own echoes; it correctly observes changes from all other sources.

---

## Required elements

Every cellier-aware widget wrapper must have:

| Element | Purpose |
|---|---|
| `self._id = uuid4()` | The widget's identity on the bus |
| `self._controller` | Reference to `CellierController` |
| `controller.on_*(…, owner_id=self._id)` | Subscribe to model events |
| `_on_model_changed(event)` | Model → widget handler; starts with `source_id` check |
| `_on_widget_changed(value)` | Widget → model handler; calls `controller.update_*(…, source_id=self._id)` |
| `_set_value(value)` | Apply value to the widget with signal suppression |
| `widget` property | Expose the underlying toolkit element for layout insertion |
| `close()` | Call `controller.unsubscribe_owner(self._id)` for cleanup |

---

## Design principle: composition over inheritance

Widget wrappers *compose* a toolkit widget rather than *inheriting* from it.
The wrapper is a cellier object that happens to contain a `QSpinBox` (or
anywidget element, or whatever the backend provides).  Callers insert
`wrapper.widget` into their layout.

Reasons to prefer composition:

- **Backend agnosticism** — the cellier layer (UUID, subscription, echo
  filter, controller call) is independent of the toolkit.  Swapping Qt for
  anywidget only requires replacing the two toolkit seams (see below).
- **Testability** — the wrapper can be tested without a running Qt event loop.
- **Clear layering** — the wrapper is a cellier concept, not a Qt concept.
  Inheriting from `QSpinBox` would blur that boundary.

---

## Backend seams

Two methods isolate all toolkit-specific code.  When porting to a different
backend, only these two change; everything else is reused verbatim.

### Seam 1: `widget` property

Returns the toolkit element that callers insert into their layout.

```python
# Qt
@property
def widget(self):
    return self._spinbox

# anywidget (hypothetical)
@property
def widget(self):
    return self._int_slider  # ipywidgets IntSlider or similar
```

### Seam 2: `_set_value(value)`

Pushes a new value to the widget without triggering the widget's own change
callback.

```python
# Qt
def _set_value(self, value: int) -> None:
    self._spinbox.blockSignals(True)
    self._spinbox.setValue(value)
    self._spinbox.blockSignals(False)

# anywidget (hypothetical)
def _set_value(self, value: int) -> None:
    self._int_slider.unobserve(self._on_widget_changed, names=["value"])
    self._int_slider.value = value
    self._int_slider.observe(self._on_widget_changed, names=["value"])
```

---

## Lifecycle: subscribing and unsubscribing

Subscribe in `__init__` using `owner_id=self._id`:

```python
controller.on_dims_changed(scene_id, self._on_dims_changed, owner_id=self._id)
```

The `owner_id` parameter groups this subscription under the widget's UUID.
Call `close()` when the widget is destroyed:

```python
def close(self) -> None:
    self._controller.unsubscribe_owner(self._id)
```

For transient widgets that may be garbage-collected before an explicit
`close()` call, pass `weak=True` to the registration method.  The bus will
automatically drop the subscription when the widget is collected.  Use this as
a safety net for floating panels; it is not a substitute for explicit cleanup
in long-lived widgets.

---

## Worked example: `DimsAxisSpinBox`

`DimsAxisSpinBox` controls a single axis of a scene's `slice_indices` — for
example the Z position in a 2D slice view of a ZYX volume.

### What it controls

`scene.dims.selection.slice_indices` is a `dict[int, int]` mapping axis index
to slice position.  This widget controls one entry in that dict.

### What it subscribes to

`DimsChangedEvent` — fired whenever any field on a scene's `DimsManager`
changes, including `slice_indices`.  The event carries:

- `source_id` — who caused the change
- `dims_state.selection.slice_indices` — the new slice position dict

### What controller method it calls

`controller.update_slice_indices(scene_id, {axis: value}, source_id=self._id)`

This mutates `scene.dims.selection.slice_indices` and stamps the event with
the widget's own UUID.

### Full implementation

```python
from uuid import uuid4


class DimsAxisSpinBox:
    def __init__(
        self,
        controller,
        scene_id,
        axis: int,
        *,
        min_val: int = 0,
        max_val: int = 65535,
        initial_value: int = 0,
        parent=None,
    ) -> None:
        from PySide6.QtWidgets import QSpinBox

        # ── Cellier layer ────────────────────────────────────────────────
        self._id = uuid4()
        self._controller = controller
        self._scene_id = scene_id
        self._axis = axis

        # ── Qt seam 1: widget creation and signal wiring ─────────────────
        self._spinbox = QSpinBox(parent=parent)
        self._spinbox.setRange(min_val, max_val)
        self._spinbox.setValue(initial_value)
        self._spinbox.valueChanged.connect(self._on_widget_changed)

        controller.on_dims_changed(scene_id, self._on_dims_changed, owner_id=self._id)

    @property
    def widget(self):
        return self._spinbox  # Qt seam 1

    @property
    def value(self) -> int:
        return self._spinbox.value()

    def close(self) -> None:
        self._controller.unsubscribe_owner(self._id)

    # ── Model → widget ───────────────────────────────────────────────────

    def _on_dims_changed(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        value = event.dims_state.selection.slice_indices.get(self._axis)
        if value is None:
            return  # axis not sliced in this mode (e.g. 3D); nothing to do
        self._set_value(value)

    # ── Widget → model ───────────────────────────────────────────────────

    def _on_widget_changed(self, value: int) -> None:
        self._controller.update_slice_indices(
            self._scene_id, {self._axis: value}, source_id=self._id
        )

    # ── Qt seam 2: push value without re-firing valueChanged ─────────────

    def _set_value(self, value: int) -> None:
        self._spinbox.blockSignals(True)
        self._spinbox.setValue(value)
        self._spinbox.blockSignals(False)
```

### How a button that sets Z=1000 interacts with this widget

```python
# Somewhere else in the UI — no source_id, no DimsAxisSpinBox involved
scene.dims.selection.slice_indices = {0: 1000}
```

Trace through the system:

1. psygnal fires on `slice_indices`
2. Controller bridge emits `DimsChangedEvent(source_id=controller._id, …)`
3. Bus dispatches to `DimsAxisSpinBox._on_dims_changed`
4. `event.source_id == controller._id` ≠ `self._id` → handler proceeds
5. `value = event.dims_state.selection.slice_indices.get(0)` → `1000`
6. `_set_value(1000)` → spinbox updates to 1000 with signals blocked

The user sees the spinbox jump to 1000.  No loop, no extra reslice.

### How the spinbox interacts with a hypothetical Z label

```python
class ZLabel:
    def __init__(self, controller, scene_id):
        self._id = uuid4()
        self._label = QLabel("z: 0")
        controller.on_dims_changed(scene_id, self._on_dims_changed, owner_id=self._id)

    def _on_dims_changed(self, event) -> None:
        # ZLabel never emits, so no echo filter needed.
        # But it's still good practice to add one in case that changes.
        value = event.dims_state.selection.slice_indices.get(0)
        if value is not None:
            self._label.setText(f"z: {value}")
```

When the spinbox changes Z from 50 to 75:

1. `DimsAxisSpinBox._on_widget_changed(75)` fires
2. `controller.update_slice_indices(…, source_id=spinbox._id)`
3. Bus emits `DimsChangedEvent(source_id=spinbox._id)`
4. `DimsAxisSpinBox._on_dims_changed`: `source_id == self._id` → skips ✓
5. `ZLabel._on_dims_changed`: `source_id` ≠ `label._id` → updates to "z: 75" ✓

Each widget only suppresses its own echo.

---

## Checklist for implementing a new widget

- [ ] `self._id = uuid4()` in `__init__`
- [ ] Subscribe with `owner_id=self._id`
- [ ] Model → widget handler starts with `if event.source_id == self._id: return`
- [ ] Widget → model handler passes `source_id=self._id` to the controller mutation method
- [ ] `_set_value` suppresses the widget's own change signal while applying the value
- [ ] `widget` property exposes the toolkit element
- [ ] `close()` calls `controller.unsubscribe_owner(self._id)`
- [ ] Use `weak=True` on the subscription for transient (closeable) panels

---

## Available controller mutation methods

| Method | Emits | Use for |
|---|---|---|
| `controller.update_slice_indices(scene_id, indices, *, source_id)` | `DimsChangedEvent` | Slice position spinboxes, sliders |
| `controller.update_appearance_field(visual_id, field, value, *, source_id)` | `AppearanceChangedEvent` | Clim spinboxes, colormap pickers, threshold sliders |

## Available controller subscription methods

| Method | Event | `entity_id` filter |
|---|---|---|
| `controller.on_dims_changed(scene_id, cb, *, owner_id, weak)` | `DimsChangedEvent` | `scene_id` |
| `controller.on_visual_changed(visual_id, cb, *, owner_id, weak)` | `AppearanceChangedEvent` | `visual_id` |
| `controller.on_camera_changed(scene_id, cb, *, owner_id, weak)` | `CameraChangedEvent` | `scene_id` |
| `controller.on_reslice_started(scene_id, cb, *, owner_id, weak)` | `ResliceStartedEvent` | `scene_id` |
| `controller.on_reslice_completed(visual_id, cb, *, owner_id, weak)` | `ResliceCompletedEvent` | `visual_id` |

All methods return a `SubscriptionHandle` for individual removal if needed.
`controller.unsubscribe_owner(owner_id)` removes all subscriptions registered
under a given UUID in one call.
