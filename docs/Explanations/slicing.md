# Slicing

This document describes how slicing works in Cellier. 

## Core concepts & data types
The slicing pipeline is broken up into three main steps:

1. Plan: determine which data needs to be loaded based on the current view.
2. Fetch: request the required data from the data stores. The requests are performed asynchronously.
3. Commit: as the requested data arrives from the data stores, upload it to the appropriate buffer/texture on the GPU.
 
The table below describes the core objects and data types used to orchestrate slicing. 

| **Component** | **Explanation** |
|---|---|
| **`ReslicingRequest`** | Immutable per-canvas snapshot of camera + dims at trigger time. The unit of work. One per canvas so each camera drives its own LOD/culling. Carries `request_id` (one per trigger) — *not* the ID used for fetch cancellation; see `slice_request_id` below. |
| **`DimsState` / `AxisAlignedSelectionState`** | Which axes are *displayed* (→ `slice(None)`) vs *sliced* (→ integer index). The selection that defines the plane/slab. |
| **`ChunkRequest`** | A single padded brick/region read: `scale_index` + `axis_selections` (per-axis `int` for sliced, `(start,stop)` for displayed). Coords may be out-of-bounds; `get_data` clamps + zero-pads. Carries `slice_request_id`, shared by all chunks from one planning event. |
| **`AsyncSlicer`** | Generic cancellable batch-fetch service. One asyncio.Task per `slice_request_id` (the shared `ChunkRequest` ID, *not* `ReslicingRequest.request_id`); data source injected per-submit as `fetch_fn`. `submit` returns the `slice_request_id`, which `SliceCoordinator` stores keyed by `(scene, canvas, visual)` and later passes to `cancel`. |
| **`SliceCoordinator`** | Orchestrator: per-`(scene,canvas,visual)` cancellation, planning dispatch, reslice-start/complete events. |
| **GPU brick/tile cache** (multiscale) | Fixed-slot texture atlas. `write_brick` / `write_tile` upload into a reserved slot. |
| **`TileManager2D/3D`** (`stage`/`commit`) | Residency bookkeeping: `tilemap` of resident bricks. When the cache is full, it uses LRU eviction.|
| **LUT indirection texture** | Maps virtual brick-grid coordinates → physical atlas slot + level; the shader walks it to sample resident bricks. Rebuilt coarsest→finest each commit, with a two-phase sweep that keeps off-slice bricks as LOD placeholders while the new slice streams in. |

## Slicing flow

From the perspective of slicing, there are two main flavors of visuals: in-memory and multiscale. The in-memory visuals load the whole scene at a single scale at one time. The multiscale visuals have to determine the level of detail to load and only load the rendered region of the scene. While they flow through the same components, there are some differences in how they are sliced. See an explanation of how slicing works using the `InMemoryImageVisual` and `MultiscaleImageVisual` as an example.

### In-memory visual

**Plan** (synchronous)

4. **`CellierController.reslice_scene`**: snapshots the dims state and per-visual render config.
5. **`RenderManager.reslice_scene`**: issues one reslicing request per canvas.
6. **`CanvasView.capture_reslicing_request`**: freezes the camera and dims into an immutable `ReslicingRequest`.
7. **`SliceCoordinator.submit`**: cancels any in-flight task for the visual, then plans.
8. **`SceneManager.build_slice_requests`**: dispatches to the 2D or 3D planner by displayed-axis count.
9. **`GFXImageMemoryVisual.build_slice_request`**: maps the world slice position into data space and emits one `ChunkRequest` for the whole slice.

**Fetch** (async, off render thread)

10. **`AsyncSlicer.submit`**: makes the request for a single read on one async task.
11. **`ImageMemoryStore.get_data`**: returns the requested slice / sub-volume array.

**Commit** (main thread)

12. **`GFXImageMemoryVisual.on_data_ready`**: uploads the whole array to the GPU.
13. **`Renderer.render`**: draws the next frame.

### Multiscale visual

**Plan** (synchronous)

7. **`CellierController.reslice_scene`**: snapshots dims state and per-visual `VisualRenderConfig` (lod_bias, force_level, frustum_cull).
8. **`RenderManager.reslice_scene`**: issues one reslicing request per canvas.
9. **`CanvasView.capture_reslicing_request`**: freezes the camera (position, frustum, screen size, world extent) and dims into a `ReslicingRequest`.
10. **`SliceCoordinator.submit`**: cancels in-flight tasks for cancellable visuals, then plans.
11. **`SceneManager.build_slice_requests`**: dispatches to the 2D or 3D planner.
12. **`GFXMultiscaleImageVisual.build_slice_request`**: selects LOD levels, sorts bricks nearest-first, culls to the frustum/viewport, caps to the cache budget, and emits one `ChunkRequest` per missing brick.
13. **`TileManager2D/3D.stage`**: splits required bricks into cache hits vs misses and reserves an atlas slot for each miss.
14. **`LutIndirectionManager2D/3D.rebuild`**: rewrites the GPU lookup table so already-resident bricks render immediately.

**Fetch** (async, off render thread)

15. **`AsyncSlicer.submit`**: runs one async task for the visual that fetches the missing bricks in concurrent batches (`asyncio.gather`, default `batch_size=8`), yielding to the renderer between batches.
16. **`MultiscaleZarrDataStore.get_data`**: reads each requested brick.

**Commit** (main thread, per batch)

17. **`GFXMultiscaleImageVisual.on_data_ready`**: writes each arriving brick into its reserved cache slot and commits it via `TileManager2D/3D.commit`.
18. **`LutIndirectionManager2D/3D.rebuild`**: rebuilds the lookup table so the new bricks become visible.
19. **`Renderer.render`**: draws the next frame.

## Triggering slicing

Re-slicing is triggered by changes to the dims model or the camera model. Only multiscale visuals replace based on changes to the camera model: this is driven by the `requires_camera_reslice` flag on the visual model, which defaults to `True` only on the multiscale image and label visuals. Camera-triggered re-slicing is also gated by `config.camera.reslice_enabled` — when that is disabled, camera motion never triggers a reslice. To maintain performance while changing the camera state interactively (e.g., rotating the camera), the controller waits for the camera to be stationary for a specified amount of time (i.e., the settle time) before triggering re-slicing. Both triggering mechanisms are described below.

### Change to the dims model

1. **`CellierController.update_slice_indices`**: writes the new slice position onto the dims model (a psygnal field).
2. **`CellierController._make_dims_handler`**: the controller event bridge emits a `DimsChangedEvent` on the outgoing bus.
3. **`CellierController._on_dims_changed_bus`**: second bus subscriber — calls `reslice_scene` for the whole scene.

### Change to the camera model

1. **`CellierController._on_camera_changed`**: on each `CameraChangedEvent`, updates the camera model and debounces — cancels any pending settle task and schedules a fresh one.
2. **`CellierController._settle_after`**: after `settle_threshold_s` with no further movement, gathers visuals with `requires_camera_reslice` and calls `RenderManager.reslice_scene` directly with `target_visual_ids`. 

## Canceling requests

During interactive use the dims and camera models change faster than data loads complete, so a new reslice usually *supersedes* an in-flight one. Canceling the superseded load stops wasted reads from the data store and prevents stale data from committing to the GPU after the view has already moved on.

### What cancels in-flight requests

There are four actions/events that cancel an in-flight fetch:

- **New slice request**: if a new slice request is made before the previous one completed, the in-flight request is canceled.
- **Trigger debounce**: `CellierController._on_camera_changed` cancels the pending `_settle_after` task on each camera event. This cancels the *trigger* before any request exists, rather than an in-flight fetch.
- **Safety-net resubmit**: `AsyncSlicer.submit` cancels any lingering task that shares the same `slice_request_id` before starting a new one.
- **Scene teardown**: `SliceCoordinator.cancel_scene` cancels every in-flight task for a scene.

### What gets canceled: the `cancellable` flag

Whether a visual's in-flight load is canceled is gated by its `cancellable` property:

- The image and label visuals -- both in-memory (`GFXImageMemoryVisual`, `GFXLabelMemoryVisual`) and multiscale (`GFXMultiscaleImageVisual`, `GFXMultiscaleLabelVisual`) -- default to `cancellable = True`, so a superseding reslice cancels their in-flight reads.
- The static-geometry in-memory visuals (mesh, points, lines) are `cancellable = False`. Their tasks always run to completion so every intermediate slice position reaches the GPU. This is because these tend to be very fast to slice and thus there isn’t a need to cancel.

`SliceCoordinator.submit` checks this flag for each visual it is about to re-submit and only cancels the ones marked cancellable.

### The cancellation path

1. **`SliceCoordinator.submit`** (or `cancel_scene`): decides which visuals to cancel. On submit, the visuals about to be re-loaded are canceled first, subject to the `cancellable` flag above.
2. **`SliceCoordinator.cancel_visual`**: pops the `slice_request_id` for the `(scene, canvas, visual)` key out of `_active_slice_ids` and calls `AsyncSlicer.cancel`.
3. **`AsyncSlicer.cancel`**: calls `task.cancel()`. Inside `_run`, the `CancelledError` is re-raised so asyncio marks the task canceled; the in-progress batch is discarded (its `callback` never fires), `on_complete` is skipped so no spurious `ResliceCompletedEvent` is emitted, and the `finally` block drops the task from `_tasks`.
4. **`visual.cancel_pending_2d` / `cancel_pending`**: `cancel_visual` then calls these (selected by the visual's `render_modes`) to release reserved-but-uncommitted GPU atlas slots via `release_all_in_flight`. In-memory visuals reserve no slots, so these are no-ops.

### GPU state after a cancel

For multiscale visuals, canceling leaves already-committed bricks/tiles and the current LUT untouched; only the slots reserved for not-yet-arrived bricks are freed back to the pool. In 2D, old committed tiles are deliberately kept as a visible fallback while the new slice loads (`invalidate_2d_cache` cancels in-flight reads but keeps committed tiles, and because `BlockKey2D` encodes the slice coordinate, tiles from different slice positions cannot collide).
