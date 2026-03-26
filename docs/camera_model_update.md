# Camera Model Update and Automatic Redraw

## Overview

When the user interacts with the camera (orbit, pan, zoom), cellier v2
automatically detects the movement, updates the model-layer camera state, and
triggers a data reslice after the camera has been still for a configurable
threshold.  This document describes how that pipeline works, how feedback
loops are avoided, and how the 2D and 3D camera type mismatch is handled.

## Architecture

Three layers are involved:

| Layer | Object | Camera type |
|-------|--------|-------------|
| **pygfx (render)** | `gfx.PerspectiveCamera` or `gfx.OrthographicCamera` | Determined by scene dimensionality |
| **Render bridge** | `CanvasView` | Owns the pygfx camera and renderer |
| **Model** | `PerspectiveCamera` or `OrthographicCamera` (psygnal `EventedModel`) | Currently always `PerspectiveCamera` from `add_canvas` |

The render-layer pygfx camera is the source of truth during user interaction.
The model-layer camera is updated to reflect render-layer state (for
serialization and programmatic access), but does not drive the pygfx camera
during normal interaction.

## Step-by-step flow

### 1. Camera change detection (`CanvasView._draw_frame`)

Every frame, `_draw_frame` captures the current pygfx camera state as a
`CameraState` NamedTuple and compares it against a cached copy from the
previous frame:

```
current_state = self._capture_camera_state()
if current_state != self._last_camera_state and not self._applying_model_state:
    self._camera_dirty = True
    self._last_camera_state = current_state
```

`CameraState` is a NamedTuple of plain Python floats and tuples, so the `!=`
comparison is a straightforward value comparison across all fields (position,
rotation, zoom, fov, extent, depth_range).  This polling approach uses only
pygfx's public API — it reads `cam.world.position`, `cam.world.rotation`,
`cam.fov`, `cam.zoom`, etc. — and avoids depending on internal pygfx event
systems.

The pygfx controller's `tick()` method runs via a `before_render` event
**before** `_draw_frame`, so by the time we compare, the camera has already
been updated for this frame if the user was interacting.

### 2. Event emission (`CanvasView._draw_frame`)

When the dirty flag is set and an `EventBus` is wired, `_draw_frame` emits
a `CameraChangedEvent` carrying the `CameraState` snapshot:

```
if self._camera_dirty and self._event_bus is not None:
    self._camera_dirty = False
    self._event_bus.emit(CameraChangedEvent(
        source_id=self._canvas_id,
        scene_id=self._scene_id,
        camera_state=current_state,
    ))
```

This emission is synchronous — all subscribers run before the frame renders.

### 3. Model writeback (`CellierController._on_camera_changed`)

The controller subscribes to `CameraChangedEvent` on the `EventBus`.  When
the event fires, the handler:

1. Writes the camera state back into the model-layer camera
   (`_update_camera_model`).
2. If `camera_reslice_enabled` is `True`, schedules a settle task.

The model writeback keeps the model layer current for serialization (e.g.
`to_file`) even though it does not drive the pygfx camera.

### 4. Settle timer (`CellierController._settle_after`)

The settle task is an `asyncio.Task` that sleeps for the configurable
threshold (default 300 ms).  If a new `CameraChangedEvent` arrives before
the sleep completes, the existing task is cancelled and a new one is
scheduled.  This ensures only one reslice fires after the camera stops
moving:

```
existing = self._settle_tasks.get(canvas_id)
if existing is not None and not existing.done():
    existing.cancel()

self._settle_tasks[canvas_id] = asyncio.create_task(
    self._settle_after(canvas_id, event.scene_id)
)
```

When the sleep completes without cancellation, `_settle_after` collects all
visuals with `requires_camera_reslice=True` and calls
`render_manager.reslice_scene` with only those visual IDs.

### 5. Reslice

`RenderManager.reslice_scene` captures a `ReslicingRequest` from the canvas
(which reads the current pygfx camera state) and submits it to the
`SliceCoordinator` for async data fetching.

## Avoiding feedback loops

There are two potential feedback loops to prevent:

### Loop 1: Programmatic camera move → detect as user interaction

When the model layer drives a programmatic camera update (e.g.
`apply_camera_state`), the pygfx camera properties change.  Without a guard,
`_draw_frame` would detect this as a user interaction and emit a
`CameraChangedEvent`, which would update the model again, and so on.

**Prevention:** `CanvasView._applying_model_state` is a boolean flag set to
`True` during programmatic updates.  `_draw_frame` checks this flag and skips
change detection when it is set.  After the programmatic update completes,
`_last_camera_state` is refreshed so the next frame does not see a stale
comparison:

```python
def apply_camera_state(self, request):
    self._applying_model_state = True
    try:
        self._camera.world.position = tuple(request.camera_pos)
    finally:
        self._applying_model_state = False
    self._last_camera_state = self._capture_camera_state()
```

### Loop 2: Model writeback → psygnal events → bus emission

When `_update_camera_model` writes fields on the model-layer camera
(a psygnal `EventedModel`), psygnal fires per-field signals.  These could
theoretically trigger bus emissions if anyone subscribed to them.  However,
the camera model's psygnal field events are **not wired** to the `EventBus`
emission path.  `CameraChangedEvent` is only emitted from `CanvasView` (the
render layer), never from model-layer field changes.  So writing to the model
does not cause a re-entrant `CameraChangedEvent`.

## Handling 2D vs 3D camera types

### Render layer

The render layer creates the correct pygfx camera for each scene:

- **2D scenes:** `gfx.OrthographicCamera` with a `PanZoomController`
- **3D scenes:** `gfx.PerspectiveCamera` with an `OrbitController`

`_capture_camera_state` reads from whichever pygfx camera exists and
produces a `CameraState` with `camera_type="orthographic"` or
`"perspective"` accordingly.  When `depth_range` is `None` (the pygfx
`OrthographicCamera` default, meaning auto-calculated), it is captured as
`(0.0, 0.0)`.

### Model layer

`CellierController.add_canvas` currently always creates a `PerspectiveCamera`
model, even for 2D scenes.  This means a 2D scene emits an orthographic
`CameraState`, but the model-layer camera receiving the writeback is a
`PerspectiveCamera`.

`_update_camera_model` handles this by branching on the **actual model type**
(`isinstance`), not on `camera_state.camera_type`:

```python
# Common fields present on both camera model types.
camera_model.position = np.array(camera_state.position, dtype=np.float32)
camera_model.rotation = np.array(camera_state.rotation, dtype=np.float32)
camera_model.zoom = camera_state.zoom
camera_model.near_clipping_plane = camera_state.depth_range[0]
camera_model.far_clipping_plane = camera_state.depth_range[1]

# Type-specific fields.
if isinstance(camera_model, PerspectiveCamera):
    camera_model.up_direction = np.array(camera_state.up, dtype=np.float32)
    camera_model.fov = camera_state.fov
elif isinstance(camera_model, OrthographicCamera):
    camera_model.width = camera_state.extent[0]
    camera_model.height = camera_state.extent[1]
```

This ensures that only fields present on the actual model are written,
regardless of which pygfx camera type produced the `CameraState`.

## Visual opt-in: `requires_camera_reslice`

Not all visuals need to refetch data when the camera moves.  The
`requires_camera_reslice` field on `BaseVisual` (default `False`, frozen)
controls whether a visual is included in camera-settle reslicing.
`MultiscaleImageVisual` overrides this to `True` because its LOD and brick
planning depend on the camera frustum and distance.

The settle task filters visuals using this field:

```python
target_ids = frozenset(
    v.id for v in scene.visuals if v.requires_camera_reslice
)
```

## Debug logging

Camera activity is logged under the `camera` category
(`cellier.render.camera`).  Enable it with:

```python
from cellier.v2.logging import enable_debug_logging
enable_debug_logging(categories=("camera",))
```

| Level | What you see |
|-------|--------------|
| `DEBUG` | Per-frame camera change detection, event emission, settle schedule/cancel |
| `INFO` | Settle reslice trigger (scene ID, visual count) |

## Configuration

| Parameter | Default | Where |
|-----------|---------|-------|
| `camera_settle_threshold_s` | 0.3 | `CellierController.__init__` |
| `camera_reslice_enabled` | `True` | `CellierController.__init__`, togglable at runtime via property |

Setting `camera_reslice_enabled = False` cancels all in-flight settle tasks
immediately and suppresses future scheduling.  The model writeback still
occurs so camera state stays current for serialization.
