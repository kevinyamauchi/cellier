# Cellier v2 Render Architecture — Implementation Plan

## Overview

This document specifies the implementation of the render-layer architecture for Cellier v2,
integrated as a test into the existing async example script at
`scripts/v2/lut_texture_multiscale_frustum_async_cellier/example.py`.

The goal is to replace the ad-hoc app-level wiring in that script with the proper
`RenderManager` / `SceneManager` / `SliceCoordinator` components defined here, while keeping
the Update button as the manual trigger (no EventBus yet). The result should be a working
example that exercises the full render pipeline through the new architecture and that can later
be wired to the EventBus without structural changes.

All new code lives under `src/cellier/v2/render/`. The existing script is not modified until
Step 7, when the example is migrated to use the new components.

---

## Code style

Follow the project styleguide throughout:

- numpy-style docstrings on all public classes and methods
- `NamedTuple` for all immutable data-transfer objects (faster than frozen dataclasses)
- Type annotations using built-in generics (`list[int]` not `List[int]`)
- Ruff-compliant formatting (line length 88, target Python 3.10)
- No comment separator blocks (`# ----` or `# ====`)
- All scripts carry the `uv` metadata block at the top

---

## Architecture overview

```
model + data layer
  VisualModel   CameraModel   DataStore
       ↕              ↕           │ (future)
            EventBus              │
       (not wired yet)            │
                                  │
render layer  ┌───────────────────────────────────────────────┐
              │  RenderManager                                 │
              │                                               │
              │  SliceCoordinator ──► AsyncSlicer             │
              │         │               (shared)              │
              │         ▼                                      │
              │  dict[scene_id, SceneManager]                 │
              │    SceneManager(dim=3d, scene_A)              │
              │    SceneManager(dim=2d, scene_B)  ...         │
              │         │                                      │
              │         ▼                                      │
              │  GFXMultiscaleImageVisual  (+ others)         │
              │    BlockCache · LUT · VolumeGeometry          │
              │                                               │
              │  dict[canvas_id, CanvasView]                  │
              │    canvas_to_scene: dict[UUID, UUID]          │
              └───────────────────────────────────────────────┘
```

---

## Data structures

These are defined in `src/cellier/v2/render/_requests.py`. All are `NamedTuple`s.

### `DimsState`

Captures which dimensions are currently displayed and the current slice index for
non-displayed dimensions. Used by `SceneManager` to select the correct slice plane in 2D mode.

```python
class DimsState(NamedTuple):
    displayed_axes: tuple[int, ...]   # e.g. (0, 1, 2) for a 3D view
    slice_indices: tuple[int, ...]    # current index per non-displayed axis
```

### `ReslicingRequest`

The single frozen snapshot that drives a complete reslicing cycle. Constructed by
`CanvasView.capture_reslicing_request()` and consumed by `SliceCoordinator`. All array
fields must be `.copy()`d by the caller — the `NamedTuple` does not enforce this.

```python
class ReslicingRequest(NamedTuple):
    camera_pos: np.ndarray            # shape (3,), world-space, copy
    frustum_corners: np.ndarray       # shape (2, 4, 3), world-space, copy
    fov_y_rad: float
    screen_height_px: float           # baked in at snapshot time
    dims_state: DimsState
    request_id: UUID                  # unique per trigger; used for cancellation
    scene_id: UUID                    # which scene this camera belongs to
    target_visual_ids: frozenset[UUID] | None  # None = all visuals in scene
```

`target_visual_ids = None` means "reslice all visuals in this scene" (camera-moved case).
A non-None set means "reslice only these visuals" (data-updated case).

---

## Module structure

New files to create, all under `src/cellier/v2/render/`:

```
src/cellier/v2/render/
├── _requests.py          # DimsState, ReslicingRequest  (NEW)
├── canvas_view.py        # CanvasView                   (NEW)
├── scene_manager.py      # SceneManager                 (NEW)
├── slice_coordinator.py  # SliceCoordinator             (NEW)
├── render_manager.py     # RenderManager                (NEW)
├── visuals/
│   └── _image.py         # GFXMultiscaleImageVisual     (EXISTS — no changes)
├── block_cache/          # (EXISTS — no changes)
├── lut_indirection/      # (EXISTS — no changes)
└── shaders/              # (EXISTS — no changes)
```

The example script is migrated in Step 7:

```
scripts/v2/lut_texture_multiscale_frustum_async_cellier/
└── example.py            # migrated in Step 7
```

---

## Component specifications

### `_requests.py`

Contains only `DimsState` and `ReslicingRequest`. No imports from the rest of the render
layer. Safe to import anywhere without circular dependencies.

---

### `CanvasView`

Owns one rendered canvas: a `QRenderWidget`, a `WgpuRenderer`, a `PerspectiveCamera`, and an
`OrbitController`. Responsible for:

- Holding a reference to its `scene_id` (set by `RenderManager` at registration)
- Exposing `capture_reslicing_request(scene_id)` — builds a `ReslicingRequest` from the
  current camera state with all arrays copied
- Exposing `get_logical_size() -> tuple[float, float]` — delegates to `_canvas`
- Exposing `request_draw()` — calls `_canvas.request_draw(self._draw_frame)`
- Running the render loop callback `_draw_frame()` which calls `renderer.render(scene, camera)`

The `scene` rendered by `_draw_frame` is obtained from the owning `RenderManager` via the
`scene_id` mapping. `CanvasView` does not hold a direct reference to the scene graph; it
receives a `get_scene_fn: Callable[[UUID], gfx.Scene]` at construction time.

**Camera sync loop prevention** (for future EventBus wiring — not needed yet):

```python
class CanvasView:
    _applying_model_state: bool = False

    def _on_controller_event(self, _event) -> None:
        """Fires on every orbit/zoom gesture."""
        if self._applying_model_state:
            return   # suppress re-entrant feedback during model→camera sync
        # future: emit CameraMovedEvent(source="controller") to EventBus

    def apply_camera_state(self, request: ReslicingRequest) -> None:
        """Apply a camera snapshot from the model (programmatic move)."""
        self._applying_model_state = True
        try:
            self._camera.world.position = request.camera_pos
            # ... other setters ...
        finally:
            self._applying_model_state = False
```

The `_applying_model_state` flag suppresses `_on_controller_event` during synchronous
pygfx camera setter calls, which can fire the OrbitController callback before returning.
The `source` tag on the future event distinguishes controller-originated from
model-originated changes in multi-canvas setups (a canvas must update its camera when
another canvas moves in the same scene, but must not re-emit an event for its own move).

**Public interface:**

```python
class CanvasView:
    def __init__(
        self,
        canvas_id: UUID,
        scene_id: UUID,
        get_scene_fn: Callable[[UUID], gfx.Scene],
        parent: QWidget | None = None,
        fov: float = 70.0,
        depth_range: tuple[float, float] = (1.0, 8000.0),
    ) -> None: ...

    @property
    def canvas_id(self) -> UUID: ...

    @property
    def scene_id(self) -> UUID: ...

    @property
    def widget(self) -> QRenderWidget:
        """The Qt widget to embed in the application layout."""
        ...

    def capture_reslicing_request(
        self,
        dims_state: DimsState,
        target_visual_ids: frozenset[UUID] | None = None,
    ) -> ReslicingRequest:
        """Snapshot the current camera state into a ReslicingRequest.

        All array fields are copied. screen_height_px is read from the
        canvas at call time and baked into the returned request.
        """
        ...

    def show_object(self, scene: gfx.Scene) -> None:
        """Fit the camera to the scene bounding box."""
        ...

    def request_draw(self) -> None: ...
```

---

### `SceneManager`

Owns one pygfx `gfx.Scene` and the registry of `GFXVisual` objects attached to it.
Dimensionality (`"2d"` or `"3d"`) is set at construction time and determines how
`build_slice_requests()` interprets the `ReslicingRequest`.

Responsibilities:
- Registering and unregistering visuals (`add_visual`, `remove_visual`)
- Adding visual nodes to the scene graph
- Building `ChunkRequest` lists by iterating registered visuals and calling
  `visual.build_slice_request()`
- Filtering by `request.target_visual_ids` when non-None

**Public interface:**

```python
class SceneManager:
    def __init__(
        self,
        scene_id: UUID,
        dim: str,   # "2d" or "3d"
    ) -> None: ...

    @property
    def scene_id(self) -> UUID: ...

    @property
    def dim(self) -> str: ...

    @property
    def scene(self) -> gfx.Scene:
        """The pygfx Scene object. Passed to CanvasView via get_scene_fn."""
        ...

    def add_visual(self, visual: GFXMultiscaleImageVisual) -> None:
        """Register a visual and add its node to the scene graph.

        Adds node_3d if dim == "3d", node_2d if dim == "2d".
        Raises ValueError if the visual does not support this dimensionality.
        """
        ...

    def remove_visual(self, visual_id: UUID) -> None: ...

    def get_visual(self, visual_id: UUID) -> GFXMultiscaleImageVisual:
        """Return the registered visual for visual_id.

        Raises KeyError if visual_id is not registered in this scene.
        Used by SliceCoordinator to retrieve the on_data_ready callback.
        """
        ...

    def build_slice_requests(
        self, request: ReslicingRequest
    ) -> dict[UUID, list[ChunkRequest]]:
        """Collect ChunkRequests from all (or targeted) registered visuals.

        Parameters
        ----------
        request :
            The reslicing request. If request.target_visual_ids is not None,
            only visuals whose ID is in that set are processed.

        Returns
        -------
        dict[UUID, list[ChunkRequest]]
            Mapping of visual_model_id to that visual's ChunkRequests,
            ordered nearest-first within each list. Visuals with no missing
            bricks (all cached) are omitted from the dict.
        """
        ...
```

---

### `SliceCoordinator`

Thin orchestrator owned by `RenderManager`. Given a `ReslicingRequest`, it:

1. Looks up the target `SceneManager` via `request.scene_id`
2. Calls `scene_manager.build_slice_requests(request)` to run the synchronous
   planning phase (LOD selection, frustum culling, cache diff)
3. Cancels any in-flight `AsyncSlicer` task for this scene
4. Submits the new `ChunkRequest` list to `AsyncSlicer` with the correct
   `fetch_fn` and per-visual `on_data_ready` callbacks

The `SliceCoordinator` does not hold references to visuals directly. It delegates all
visual-level logic to `SceneManager`. Because `SceneManager.build_slice_requests()` returns
a `dict[UUID, list[ChunkRequest]]` keyed by `visual_model_id`, the coordinator receives
requests already partitioned per visual — no post-hoc splitting is needed.

One `AsyncSlicer` task maps to one `(scene_id, visual_id)` pair. The coordinator
maintains a `dict[tuple[UUID, UUID], UUID]` (keyed by `(scene_id, visual_id)`) tracking
active slice IDs so that per-visual cancellation (`reslice_visual`) can cancel only the
affected task while leaving other visuals in the same scene running.

**Public interface:**

```python
class SliceCoordinator:
    def __init__(
        self,
        scenes: dict[UUID, SceneManager],
        slicer: AsyncSlicer,
        data_stores: dict[UUID, MultiscaleZarrDataStore],
    ) -> None: ...

    def submit(self, request: ReslicingRequest) -> None:
        """Execute the full reslicing cycle for the scene in request.scene_id.

        Cancels in-flight tasks for visuals that will be re-submitted.
        No-op for visuals not targeted by request.target_visual_ids.
        """
        ...

    def cancel_scene(self, scene_id: UUID) -> None:
        """Cancel all in-flight tasks for a scene."""
        ...

    def cancel_visual(self, scene_id: UUID, visual_id: UUID) -> None:
        """Cancel the in-flight task for one visual."""
        ...
```

**Callback wiring detail:**

`SliceCoordinator.submit()` calls `scene_manager.build_slice_requests(request)` once to get
the full `dict[UUID, list[ChunkRequest]]`, then iterates it:

```python
requests_by_visual = scene_manager.build_slice_requests(request)
for visual_id, chunk_requests in requests_by_visual.items():
    visual = scene_manager.get_visual(visual_id)
    data_store = self._data_stores[visual_id]
    self.cancel_visual(request.scene_id, visual_id)
    slice_id = slicer.submit(
        chunk_requests,
        fetch_fn=data_store.get_data,
        callback=visual.on_data_ready,
        consumer_id=str(visual_id),
    )
    self._active_slice_ids[(request.scene_id, visual_id)] = slice_id
```

This keeps `GFXMultiscaleImageVisual.on_data_ready` as the direct batch callback with no
EventBus routing — data payloads never touch the EventBus.

---

### `RenderManager`

The single top-level render-layer object. Owns:

- `dict[UUID, SceneManager]` — keyed by `scene_id`
- `dict[UUID, CanvasView]` — keyed by `canvas_id`
- `dict[UUID, UUID]` — `canvas_to_scene` mapping
- `AsyncSlicer` — shared across all scenes
- `SliceCoordinator` — drives all reslicing

Exposes three reslicing entry points:

```python
class RenderManager:

    def trigger_update(self, dims_state: DimsState) -> None:
        """Reslice all visuals across all scenes.

        Builds one ReslicingRequest per scene using the camera state of
        the canvas attached to that scene, then submits each to the
        SliceCoordinator.

        This is the method called by the Update button in the example.
        """
        ...

    def reslice_scene(
        self,
        scene_id: UUID,
        dims_state: DimsState,
    ) -> None:
        """Reslice all visuals in one scene."""
        ...

    def reslice_visual(
        self,
        visual_id: UUID,
        dims_state: DimsState,
    ) -> None:
        """Reslice one visual.

        Looks up which SceneManager owns visual_id, builds a
        ReslicingRequest with target_visual_ids={visual_id}, and submits.
        The camera snapshot still uses the full current camera state —
        LOD thresholds and frustum culling depend on it.
        """
        ...
```

Registration methods:

```python
    def add_scene(
        self,
        scene_id: UUID,
        dim: str,
    ) -> SceneManager: ...

    def add_canvas(
        self,
        canvas_id: UUID,
        scene_id: UUID,
        parent: QWidget | None = None,
        **canvas_view_kwargs,
    ) -> CanvasView:
        """Create a CanvasView, register it, and return it.

        The caller embeds canvas_view.widget in their Qt layout.
        """
        ...

    def add_visual(
        self,
        scene_id: UUID,
        visual: GFXMultiscaleImageVisual,
        data_store: MultiscaleZarrDataStore,
    ) -> None:
        """Register a visual with a scene and its associated data store."""
        ...

    def get_scene(self, scene_id: UUID) -> gfx.Scene:
        """Return the pygfx Scene for scene_id.

        Passed as a callable to CanvasView at construction time so CanvasView
        can render the scene without holding a direct SceneManager reference.
        """
        ...
```

---

## Implementation steps

Each step produces a passing state that can be verified independently before proceeding.

### Step 1 — `_requests.py`: `DimsState` and `ReslicingRequest`

Create `src/cellier/v2/render/_requests.py` with both `NamedTuple`s exactly as specified
above. Add them to `src/cellier/v2/render/__init__.py`.

**Verification:** `from cellier.v2.render import DimsState, ReslicingRequest` works in a
Python REPL. Construct one of each with dummy values.

---

### Step 2 — `CanvasView`

Create `src/cellier/v2/render/canvas_view.py`.

Extract the scene/camera/controller setup from `example.py`'s `_setup_scene()` method into
`CanvasView.__init__`. The `_draw_frame` method calls
`self._renderer.render(self._get_scene_fn(self._scene_id), self._camera)`.

Implement `capture_reslicing_request()`:

```python
def capture_reslicing_request(
    self,
    dims_state: DimsState,
    target_visual_ids: frozenset[UUID] | None = None,
) -> ReslicingRequest:
    _, screen_h = self._canvas.get_logical_size()
    return ReslicingRequest(
        camera_pos=np.array(self._camera.world.position, dtype=np.float64),
        frustum_corners=np.asarray(self._camera.frustum, dtype=np.float64).copy(),
        fov_y_rad=float(np.radians(self._camera.fov)),
        screen_height_px=float(screen_h),
        dims_state=dims_state,
        request_id=uuid4(),
        scene_id=self._scene_id,
        target_visual_ids=target_visual_ids,
    )
```

Note: `np.asarray(...).copy()` is required — `camera.frustum` returns a view into an
internal pygfx buffer that will be mutated on the next frame.

**Verification:** Instantiate a `CanvasView` in isolation (no `SceneManager` yet), call
`show_object()` and `capture_reslicing_request()`. Check all fields are populated and arrays
are copies (modify the returned array and verify the camera is unaffected).

---

### Step 3 — `SceneManager`

Create `src/cellier/v2/render/scene_manager.py`.

`add_visual` appends to `self._visuals: dict[UUID, GFXMultiscaleImageVisual]` and calls
`self._scene.add(visual.node_3d)` or `visual.node_2d` depending on `self._dim`.

`build_slice_requests` iterates `self._visuals.values()`, filters by
`request.target_visual_ids` if not `None`, and calls each visual's
`build_slice_request(camera_pos, frustum_planes, thresholds, force_level)`.

The LOD threshold computation lives here, not in the visual:

```python
def _compute_thresholds(self, request: ReslicingRequest, n_levels: int) -> list[float]:
    focal_half_height = (request.screen_height_px / 2.0) / np.tan(request.fov_y_rad / 2.0)
    return [
        (2 ** (k - 1)) * focal_half_height
        for k in range(1, n_levels)
    ]
```

Frustum planes are computed from `request.frustum_corners` using the existing
`cellier.v2.render._frustum.frustum_planes_from_corners()`.

`build_slice_requests` returns a `dict[UUID, list[ChunkRequest]]` keyed by
`visual_model_id`. Visuals with no missing bricks (all cached) are omitted from the dict
entirely — the caller treats a missing key the same as an empty list.
The visual's `build_slice_request()` call is unchanged from how the example script calls it
today; `SceneManager` is simply the new home for that call.

**Verification:** Create a `SceneManager(dim="3d")`, add a `GFXMultiscaleImageVisual`, call
`build_slice_requests()` with a hand-constructed `ReslicingRequest`. Confirm the result is a
`dict` keyed by the visual's `visual_model_id` and that the value is a `list[ChunkRequest]`.

---

### Step 4 — `SliceCoordinator`

Create `src/cellier/v2/render/slice_coordinator.py`.

`submit(request)`:

1. Look up `scene_manager = self._scenes[request.scene_id]`
2. Call `scene_manager.build_slice_requests(request)` — returns
   `dict[UUID, list[ChunkRequest]]` keyed by `visual_model_id`, already filtered by
   `request.target_visual_ids` and with empty-cache visuals omitted
3. For each `(visual_id, requests)` pair in that dict: cancel any existing active slice
   task for `(scene_id, visual_id)`
4. Call `self._slicer.submit(requests, fetch_fn=..., callback=...)` for each visual
5. Store the returned `slice_id` in `self._active_slice_ids[(scene_id, visual_id)]`

`data_stores` is a `dict[UUID, MultiscaleZarrDataStore]` keyed by `visual_model_id` — passed
at construction so `SliceCoordinator` can resolve the correct store per visual without
reaching into the model layer.

**Verification:** Construct a `SliceCoordinator` with real `SceneManager`, `AsyncSlicer`, and
`data_stores`. Call `submit()` twice in quick succession (second should cancel first). Verify
via `AsyncSlicer._tasks` that only one task is running after the second submit.

---

### Step 5 — `RenderManager`

Create `src/cellier/v2/render/render_manager.py`.

`__init__` creates an `AsyncSlicer(batch_size=8)` and a `SliceCoordinator`. All other state
is empty dicts.

`add_canvas` creates a `CanvasView` with `get_scene_fn=self.get_scene` and stores it. The
canvas's `_canvas.request_draw(canvas_view._draw_frame)` is set inside `CanvasView.__init__`.

`trigger_update(dims_state)` iterates `self._canvas_to_scene`, gets the `CanvasView` for
each canvas, calls `canvas.capture_reslicing_request(dims_state)`, and calls
`self._slice_coordinator.submit(request)`.

`reslice_visual(visual_id, dims_state)` must find which scene owns the visual. Maintain a
reverse map `self._visual_to_scene: dict[UUID, UUID]` that is populated in `add_visual`.

Update `src/cellier/v2/render/__init__.py` to export `RenderManager`.

**Verification:** Full construction test — create a `RenderManager`, add a scene, add a
canvas pointing at that scene, add a visual with its data store. Call `trigger_update()` with
a dummy `DimsState`. No assertions on render output; just verify no exceptions are raised and
the `AsyncSlicer` has a running task.

---

### Step 6 — Unit tests

Add `tests/v2/render/test_render_manager.py`. Cover:

- `ReslicingRequest` construction and field copying
- `CanvasView.capture_reslicing_request()` — array independence from camera state
- `SceneManager.build_slice_requests()` — `target_visual_ids` filtering
- `SliceCoordinator.submit()` — cancellation of prior task before re-submit
- `RenderManager.reslice_visual()` — reverse map lookup, only one visual's task affected

Use `pytest` with `uv run pytest`. Mock the `AsyncSlicer` with a simple stub that records
submitted requests.

---

### Step 7 — Migrate `example.py`

The existing script is preserved as-is. A new script is created alongside it:

```
scripts/v2/lut_texture_multiscale_frustum_async_cellier/example_v2.py
```

This new script replaces the ad-hoc app-level wiring with `RenderManager`. The diff in
application logic is:

**Before (current `example.py`):**
```python
# In MainWindow.__init__:
self._visual = GFXMultiscaleImageVisual.from_cellier_model(...)
self._data_store = MultiscaleZarrDataStore(...)
self._slicer = AsyncSlicer(batch_size=COMMIT_BATCH_SIZE)
self._setup_scene()   # manual pygfx scene/camera/controller setup

# In _on_update_clicked():
cam_pos = get_camera_position_world(self._camera)
corners = get_frustum_corners_world(self._camera)
# ... manual threshold computation ...
chunk_requests = self._visual.build_slice_request(cam_pos, ...)
self._slicer.submit(chunk_requests, fetch_fn=..., callback=...)
```

**After (`example_v2.py`):**
```python
# In MainWindow.__init__:
scene_id = uuid4()
canvas_id = uuid4()

self._render_manager = RenderManager()
self._render_manager.add_scene(scene_id, dim="3d")
canvas_view = self._render_manager.add_canvas(canvas_id, scene_id, parent=self)
self._render_manager.add_visual(scene_id, visual, data_store)

# Embed the canvas widget
root_layout.addWidget(canvas_view.widget)

# Set camera initial position
canvas_view.show_object(self._render_manager.get_scene(scene_id))

# In _on_update_clicked():
dims_state = DimsState(displayed_axes=(0, 1, 2), slice_indices=())
self._render_manager.trigger_update(dims_state)
```

The Update button still triggers the full pipeline. All debug printing (LOD stats, frustum
planes, brick counts) is preserved by reading from `visual._last_plan_stats` after
`trigger_update()` returns (the planning phase is synchronous; only the fetch is async).

The `QTimer` debounce, frustum wireframe, LOD bias spinbox, force-level radios, and
far-plane spinbox are all retained in the application layer — they are app-level concerns,
not architecture concerns.

**Verification:** Run `uv run example_v2.py`. The viewer should behave identically to
`example.py`: camera orbits freely, Update button loads bricks, cancel/restart works, debug
print output is unchanged.

---

## Key invariants to preserve

These are easy to break accidentally; check them after each step.

**Array copying in `capture_reslicing_request`:**
`camera.frustum` and `camera.world.position` return views / references into pygfx internals.
Always `.copy()` them. Failure mode: the `ReslicingRequest` array values change silently as
the camera moves, causing incorrect LOD selection on the next frame.

**`screen_height_px` is baked in at snapshot time:**
Do not fetch it lazily in `SceneManager` or `SliceCoordinator`. By the time those methods
run, the window may have been resized. The `CanvasView` must read it from
`self._canvas.get_logical_size()` inside `capture_reslicing_request()`.

**LOD threshold computation belongs in `SceneManager`:**
Not in `CanvasView` (which knows nothing about visuals) and not in
`GFXMultiscaleImageVisual` (which must not know about camera state). `SceneManager` is the
correct location because it knows both the visual's `n_levels` and the request's camera
fields.

**`CancelledError` is always re-raised in `AsyncSlicer`:**
The existing `AsyncSlicer._run` already does this. Do not add any `except CancelledError:
pass` blocks in new code.

**`GFXMultiscaleImageVisual` is not modified:**
All existing methods (`build_slice_request`, `on_data_ready`, `cancel_pending`) are used
as-is. The architecture wraps around the visual; it does not reach into it.

**`SceneManager` holds no reference to `DataStore`:**
`DataStore` references flow through `SliceCoordinator` only
(`self._data_stores: dict[UUID, MultiscaleZarrDataStore]`). `SceneManager` knows only about
`GFXVisual` objects and the pygfx scene graph.

---

## Camera sync loop prevention (future EventBus wiring)

Not needed for the current implementation but must be in place before EventBus events are
connected. Implement the guard in `CanvasView` now even though it is dormant:

```python
class CanvasView:
    def __init__(self, ...) -> None:
        ...
        self._applying_model_state: bool = False
        # Wire the controller event — currently a no-op placeholder
        self._controller.add_event_handler(
            self._on_controller_event, "pointer_move", "wheel"
        )

    def _on_controller_event(self, event) -> None:
        if self._applying_model_state:
            return
        # future: self._event_bus.emit(CameraMovedEvent(source="controller", ...))

    def apply_camera_state(self, request: ReslicingRequest) -> None:
        """Apply a camera snapshot from the model layer (programmatic move).

        The _applying_model_state guard prevents the resulting camera setter
        calls from firing _on_controller_event, which would otherwise cause
        a feedback loop: model change → apply to pygfx → controller event →
        model change → ...
        """
        self._applying_model_state = True
        try:
            self._camera.world.position = tuple(request.camera_pos)
            # additional setters as needed
        finally:
            self._applying_model_state = False
```

When EventBus integration is added, the `CameraMovedEvent` payload must include:

- `source: Literal["controller", "model"]` — so each `CanvasView` can ignore events it
  originated itself
- `canvas_id: UUID` — so in multi-canvas setups, canvas B applies a camera-moved event
  from canvas A (same scene), but canvas A does not re-apply its own event

---

## Future: EventBus wiring (out of scope for this plan)

When the EventBus is connected, the only changes needed are:

1. `CanvasView._on_controller_event` emits `CameraMovedEvent` instead of being a no-op
2. `RenderManager` subscribes to `CameraMovedEvent` and calls `reslice_scene(scene_id, ...)`
3. `RenderManager` subscribes to `DataStoreMutatedEvent` and calls
   `reslice_visual(visual_id, ...)`

The `RenderManager.trigger_update()` method used by the Update button remains in place —
it becomes an explicit override path for testing and debugging.
