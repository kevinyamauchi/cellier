# CellierController — Phase 1 Implementation Plan

## Overview

This document is a step-by-step implementation plan for Phase 1 of `CellierController`.
It is written for Claude Code and covers three concerns in order:

1. **Refactor the render layer** so that per-visual render config (`lod_bias`,
   `force_level`, `frustum_cull`) flows through the call stack as a parameter rather
   than being stored as mutable state on `SceneManager`.  This is a prerequisite that
   eliminates the bridge pattern the controller would otherwise require.

2. **Implement `CellierController`** with full implementations for the methods needed by
   the example and `NotImplementedError` stubs for everything else.

3. **Write `example_cellier.py`**, a new script in
   `scripts/v2/lut_texture_multiscale_frustum_async_cellier/` that replaces the
   manual render-layer wiring in `example_v2.py` with `CellierController`.

Each step ends with a **verification** that must pass before proceeding to the next.
Run all tests with `uv run pytest`.

---

## Background: why the refactor comes first

`SceneManager` currently stores `lod_bias`, `force_level`, and `frustum_cull` on a
`SceneRenderConfig` singleton.  `example_v2.py` mutates that singleton in
`_on_update_clicked` before calling `trigger_update`.  The design doc specifies that
these settings live on per-visual `ImageAppearance` models.

Without the refactor, `CellierController` would need to sync
`visual.appearance → scene_manager.config` before each reslice call — a bridge that
only works correctly for one visual per scene and perpetuates the wrong data owner.

The fix is to pass per-visual configs as a parameter through the call stack:

```
controller.reslice_scene(scene_id)
  └─ builds visual_configs: dict[UUID, VisualRenderConfig] from model appearances
       └─ RenderManager.reslice_scene(scene_id, dims_state, visual_configs)
            └─ SliceCoordinator.submit(request, visual_configs)
                 └─ SceneManager.build_slice_requests(request, visual_configs)
                      └─ per visual: reads visual_configs[visual_id]
```

`SceneManager` becomes stateless with respect to appearance.  The controller is the
natural translation point between model-layer appearances and the render-layer dict.

---

## Step 1 — Rename `SceneRenderConfig` → `VisualRenderConfig`

**File:** `src/cellier/v2/render/_scene_config.py`

Rename the dataclass from `SceneRenderConfig` to `VisualRenderConfig`.  The three
fields (`lod_bias`, `force_level`, `frustum_cull`) and their defaults are unchanged.

```python
# src/cellier/v2/render/_scene_config.py
@dataclass
class VisualRenderConfig:
    lod_bias: float = 1.0
    force_level: int | None = None
    frustum_cull: bool = True
```

Update `src/cellier/v2/render/__init__.py` to export `VisualRenderConfig` in place of
`SceneRenderConfig`.

**Verification:** `from cellier.v2.render import VisualRenderConfig` in a REPL; confirm
default values.

---

## Step 2 — Refactor `SceneManager`

**File:** `src/cellier/v2/render/scene_manager.py`

### 2a — Remove config state

Delete the `config: SceneRenderConfig | None = None` constructor parameter, the
`self._config` attribute, and the `config` property.  The `SceneManager` constructor
becomes:

```python
def __init__(self, scene_id: UUID, dim: str) -> None:
```

### 2b — Update `build_slice_requests` signature

Add `visual_configs: dict[UUID, VisualRenderConfig]` as a second parameter.  When a
visual's ID is absent from the dict, fall back to a default `VisualRenderConfig()`.

```python
def build_slice_requests(
    self,
    request: ReslicingRequest,
    visual_configs: dict[UUID, VisualRenderConfig],
) -> dict[UUID, list[ChunkRequest]]:
```

Inside the method body, replace all reads of `self._config.*` with reads from the
per-visual config:

```python
for visual_id, visual in self._visuals.items():
    if (
        request.target_visual_ids is not None
        and visual_id not in request.target_visual_ids
    ):
        continue

    cfg = visual_configs.get(visual_id, VisualRenderConfig())

    frustum_planes = (
        frustum_planes_from_corners(request.frustum_corners)
        if cfg.frustum_cull
        else None
    )

    n_levels = visual._volume_geometry.n_levels
    thresholds = self._compute_thresholds(request, n_levels, cfg.lod_bias)

    chunk_requests = visual.build_slice_request(
        camera_pos=request.camera_pos,
        frustum_planes=frustum_planes,
        thresholds=thresholds,
        force_level=cfg.force_level,
    )
    if chunk_requests:
        result[visual_id] = chunk_requests
```

Note: frustum plane computation now lives inside the per-visual loop because each
visual may have a different `frustum_cull` value.  This is a minor change from the
current implementation where planes are computed once for the whole scene, but it is
correct and the cost is negligible (one fast numpy call per visual).

### 2c — Update `_compute_thresholds`

Accept `lod_bias: float` as a direct parameter instead of reading from `self._config`:

```python
def _compute_thresholds(
    self, request: ReslicingRequest, n_levels: int, lod_bias: float
) -> list[float]:
    focal_half_height = (request.screen_height_px / 2.0) / np.tan(
        request.fov_y_rad / 2.0
    )
    return [
        (2 ** (k - 1)) * focal_half_height * lod_bias
        for k in range(1, n_levels)
    ]
```

**Verification:** `uv run pytest tests/v2/render/` — all existing scene manager tests
pass after updating them (see Step 5).

---

## Step 3 — Refactor `SliceCoordinator`

**File:** `src/cellier/v2/render/slice_coordinator.py`

Add `visual_configs: dict[UUID, VisualRenderConfig]` to `submit`:

```python
def submit(
    self,
    request: ReslicingRequest,
    visual_configs: dict[UUID, VisualRenderConfig],
) -> None:
```

Pass it straight through to `build_slice_requests`:

```python
requests_by_visual = scene_manager.build_slice_requests(request, visual_configs)
```

No other logic changes.

**Verification:** No new tests needed here; this is pure pass-through.

---

## Step 4 — Refactor `RenderManager`

**File:** `src/cellier/v2/render/render_manager.py`

### 4a — Add `visual_configs` to all three reslicing entry points

```python
def trigger_update(
    self,
    dims_state: DimsState,
    visual_configs: dict[UUID, VisualRenderConfig] | None = None,
) -> None:
    if visual_configs is None:
        visual_configs = {}
    for canvas_id, _scene_id in self._canvas_to_scene.items():
        canvas = self._canvases[canvas_id]
        request = canvas.capture_reslicing_request(dims_state)
        self._slice_coordinator.submit(request, visual_configs)

def reslice_scene(
    self,
    scene_id: UUID,
    dims_state: DimsState,
    visual_configs: dict[UUID, VisualRenderConfig] | None = None,
) -> None:
    if visual_configs is None:
        visual_configs = {}
    canvas = self._find_canvas_for_scene(scene_id)
    if canvas is None:
        return
    request = canvas.capture_reslicing_request(dims_state)
    self._slice_coordinator.submit(request, visual_configs)

def reslice_visual(
    self,
    visual_id: UUID,
    dims_state: DimsState,
    visual_config: VisualRenderConfig | None = None,
) -> None:
    cfg = visual_config if visual_config is not None else VisualRenderConfig()
    scene_id = self._visual_to_scene[visual_id]
    canvas = self._find_canvas_for_scene(scene_id)
    if canvas is None:
        return
    request = canvas.capture_reslicing_request(
        dims_state, target_visual_ids=frozenset({visual_id})
    )
    self._slice_coordinator.submit(request, {visual_id: cfg})
```

### 4b — Delete `get_scene_config`

Remove the `get_scene_config(self, scene_id)` method entirely.  The only caller was
`example_v2.py`; that caller is replaced by `example_cellier.py` (Step 7).

### 4c — Add `CanvasView.set_depth_range`

**File:** `src/cellier/v2/render/canvas_view.py`

The controller needs a way to adjust the camera far plane without exposing
`CanvasView._camera` publicly.  Add this method:

```python
def set_depth_range(self, depth_range: tuple[float, float]) -> None:
    """Set the camera near/far clip distances.

    Parameters
    ----------
    depth_range : tuple[float, float]
        ``(near, far)`` clip distances in world units.
    """
    self._camera.depth_range = depth_range
```

**Verification:** `uv run pytest tests/v2/render/` — all render-layer tests pass after
updating them (see Step 5).

---

## Step 5 — Update render-layer tests

**File:** `tests/v2/render/test_render_manager.py`

The test file contains tests that construct `SceneManager` with a `config=` kwarg and
tests that call `build_slice_requests` with one argument.  Update each:

- Replace `SceneRenderConfig` imports and usages with `VisualRenderConfig`.
- Replace `SceneManager(scene_id=..., dim=..., config=cfg)` with
  `SceneManager(scene_id=..., dim=...)`.  Pass the config via `visual_configs` to
  `build_slice_requests` instead.
- Update all `sm.build_slice_requests(req)` calls to
  `sm.build_slice_requests(req, {visual_id: cfg})`.
- Remove any tests that called `RenderManager.get_scene_config()` — these are now
  obsolete.

The test logic for `lod_bias_scales_thresholds`, `force_level_forwarded_to_visual`,
`frustum_cull_false_passes_none_planes`, and `frustum_cull_true_passes_planes` is all
preserved; only the delivery mechanism changes from stored state to passed argument.

**Verification:** `uv run pytest tests/v2/render/` — all tests pass.

---

## Step 6 — Implement `CellierController`

**File:** `src/cellier/v2/controller.py`  (new file)

### Module location

```
src/cellier/v2/
├── controller.py          ← NEW
├── viewer_model.py
├── scene/
├── data/
├── visuals/
└── render/
```

Export from `src/cellier/v2/__init__.py`:
```python
from cellier.v2.controller import CellierController
```

### Full imports block

```python
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from cellier.v2.data._base_data_store import BaseDataStore
from cellier.v2.render._scene_config import VisualRenderConfig
from cellier.v2.render._requests import DimsState
from cellier.v2.render.render_manager import RenderManager
from cellier.v2.render.visuals._image import GFXMultiscaleImageVisual
from cellier.v2.scene.canvas import Canvas
from cellier.v2.scene.cameras import OrbitCameraController, PerspectiveCamera
from cellier.v2.scene.dims import CoordinateSystem, DimsManager
from cellier.v2.scene.scene import Scene
from cellier.v2.viewer_model import DataManager, ViewerModel
from cellier.v2.visuals._image import MultiscaleImageVisual

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget
    from cellier.v2.visuals._image import ImageAppearance
```

### Class skeleton

```python
class CellierController:
    """Single entry-point for building Cellier v2 visualization tools.

    Wraps a ViewerModel (model layer) and a RenderManager (render layer),
    exposing a clean public API that hides all internal types.
    """
```

---

### 6a — Construction

```python
def __init__(self, widget_parent: QWidget | None = None) -> None:
    self._widget_parent = widget_parent
    self._model = ViewerModel(data=DataManager())
    self._render_manager = RenderManager()
    # Reverse map: visual_model_id → scene_id (model layer mirror of render layer map)
    self._visual_to_scene: dict[UUID, UUID] = {}
    # Reverse map: canvas_id → scene_id (needed for set_camera_depth_range)
    self._canvas_to_scene: dict[UUID, UUID] = {}
    # Forward map: scene_id → list[canvas_id]
    self._scene_to_canvases: dict[UUID, list[UUID]] = {}

def set_widget_parent(self, parent: QWidget) -> None:
    """Set the Qt parent for subsequently created canvas widgets."""
    self._widget_parent = parent
```

---

### 6b — Stubs: construction class methods

```python
@classmethod
def from_model(
    cls,
    model: ViewerModel,
    widget_parent: QWidget | None = None,
) -> CellierController:
    """Construct a controller from a pre-built ViewerModel.

    Not implemented in Phase 1.  Materializing the full render layer from
    an existing model requires reading level_shapes from every data store
    and is deferred until needed.
    """
    raise NotImplementedError(
        "CellierController.from_model is not implemented in Phase 1."
    )

@classmethod
def from_file(
    cls,
    path: str | pathlib.Path,
    widget_parent: QWidget | None = None,
) -> CellierController:
    """Deserialize a ViewerModel from disk and construct a controller.

    Not implemented in Phase 1.
    """
    raise NotImplementedError(
        "CellierController.from_file is not implemented in Phase 1."
    )
```

---

### 6c — Serialization

```python
def to_file(self, path: str | pathlib.Path) -> None:
    """Serialize the current model state to a JSON file.

    Camera positions in the model are NOT synchronized from the render
    layer before saving in Phase 1 — the model cameras remain at their
    construction-time defaults.  Full camera capture is deferred to a
    later phase.
    """
    self._model.to_file(path)

def to_model(self) -> ViewerModel:
    """Return a copy of the current model state.

    Camera positions are NOT synchronized from the render layer in
    Phase 1 (same caveat as to_file).
    """
    return self._model.model_copy(deep=True)
```

---

### 6d — `add_scene`

```python
def add_scene(
    self,
    dim: str,
    coordinate_system: CoordinateSystem,
    name: str,
) -> Scene:
    """Create and register a new empty scene.

    Parameters
    ----------
    dim : str
        ``"2d"`` or ``"3d"``.
    coordinate_system : CoordinateSystem
        World coordinate system for the scene.
    name : str
        Human-readable name.

    Returns
    -------
    Scene
        The live model object.  Use ``scene.id`` in subsequent calls.
    """
    if dim == "3d":
        displayed_axes = (0, 1, 2)
        slice_indices: tuple[int, ...] = ()
    elif dim == "2d":
        displayed_axes = (0, 1)
        slice_indices = (0,)
    else:
        raise ValueError(f"dim must be '2d' or '3d', got {dim!r}")

    dims = DimsManager(
        coordinate_system=coordinate_system,
        displayed_axes=displayed_axes,
        slice_indices=slice_indices,
    )
    scene = Scene(name=name, dims=dims)
    self._model.scenes[scene.id] = scene
    self._render_manager.add_scene(scene.id, dim=dim)
    self._scene_to_canvases[scene.id] = []
    return scene
```

---

### 6e — `add_data_store`

```python
def add_data_store(self, data_store: BaseDataStore) -> BaseDataStore:
    """Register a data store and return it.

    Parameters
    ----------
    data_store : BaseDataStore
        The store to register.  Pass ``data_store.id`` to ``add_visual``
        or ``add_image`` to bind it to a visual.

    Returns
    -------
    BaseDataStore
        The same object passed in.
    """
    self._model.data.stores[data_store.id] = data_store
    return data_store
```

---

### 6f — `add_image` (high-level)

This is the primary entry point for adding image data.  It builds all three objects
(data store registration, model visual, GFX visual) in one atomic call.

```python
def add_image(
    self,
    data: BaseDataStore,
    scene_id: UUID,
    appearance: ImageAppearance,
    name: str,
    block_size: int = 32,
    gpu_budget_bytes: int = 1 * 1024**3,
    threshold: float = 0.2,
    interpolation: str = "linear",
) -> MultiscaleImageVisual:
    """Add a multiscale image visual to a scene.

    Registers the data store (if not already registered), creates the
    MultiscaleImageVisual model, builds the GFXMultiscaleImageVisual
    render object, and wires everything together atomically.

    Parameters
    ----------
    data : BaseDataStore
        A MultiscaleZarrDataStore (or any BaseDataStore subclass that
        exposes ``level_shapes`` and ``n_levels``).
    scene_id : UUID
        ID of an existing scene (returned by ``add_scene``).
    appearance : ImageAppearance
        Appearance parameters for the visual.
    name : str
        Human-readable name for the visual.
    block_size : int
        Brick side length in voxels.  Default 32.
    gpu_budget_bytes : int
        GPU memory budget for the brick cache.  Default 1 GiB.
    threshold : float
        Isosurface threshold for 3D raycast rendering.  Default 0.2.
    interpolation : str
        Sampler filter ``"linear"`` or ``"nearest"``.  Default ``"linear"``.

    Returns
    -------
    MultiscaleImageVisual
        The live model object.  Mutate ``visual.appearance`` fields
        directly; they are read at the next ``reslice_*`` call.
    """
    # Register the data store if it hasn't been already.
    if data.id not in self._model.data.stores:
        self._model.data.stores[data.id] = data

    scene = self._model.scenes[scene_id]
    dim = scene.dims.displayed_axes  # (0,1,2) → "3d"

    # Determine the downscale factors from the data store.
    # MultiscaleZarrDataStore exposes n_levels; downscale_factors is
    # [1, 2, 4, ...] by convention (each level 2× coarser).
    downscale_factors = [2**i for i in range(data.n_levels)]

    # Build the model-layer visual.
    visual_model = MultiscaleImageVisual(
        name=name,
        data_store_id=str(data.id),
        downscale_factors=downscale_factors,
        appearance=appearance,
    )

    # Add model visual to scene.
    scene.visuals.append(visual_model)

    # Determine render mode from scene dimensionality.
    render_modes = {"3d"} if len(dim) == 3 else {"2d"}

    # Build the GFX render-layer visual.
    level_shapes = list(data.level_shapes)
    gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
        model=visual_model,
        level_shapes=level_shapes,
        render_modes=render_modes,
        block_size=block_size,
        gpu_budget_bytes=gpu_budget_bytes,
        threshold=threshold,
        interpolation=interpolation,
    )

    # Register with the render manager.
    self._render_manager.add_visual(scene_id, gfx_visual, data)
    self._visual_to_scene[visual_model.id] = scene_id

    return visual_model
```

---

### 6g — `add_visual` (fine-grained)

```python
def add_visual(
    self,
    scene_id: UUID,
    visual_model: MultiscaleImageVisual,
    data_store_id: UUID,
    block_size: int = 32,
    gpu_budget_bytes: int = 1 * 1024**3,
    threshold: float = 0.2,
    interpolation: str = "linear",
) -> MultiscaleImageVisual:
    """Register a pre-built visual model with a scene.

    Use this when you have already constructed a MultiscaleImageVisual
    model yourself.  For the common case, prefer add_image().

    Parameters
    ----------
    scene_id : UUID
        ID of an existing scene.
    visual_model : MultiscaleImageVisual
        The pre-built model.
    data_store_id : UUID
        ID of a data store already registered via add_data_store().

    Returns
    -------
    MultiscaleImageVisual
        The same model passed in.
    """
    data_store = self._model.data.stores[data_store_id]
    scene = self._model.scenes[scene_id]
    scene.visuals.append(visual_model)

    render_modes = {"3d"} if len(scene.dims.displayed_axes) == 3 else {"2d"}
    level_shapes = list(data_store.level_shapes)
    gfx_visual = GFXMultiscaleImageVisual.from_cellier_model(
        model=visual_model,
        level_shapes=level_shapes,
        render_modes=render_modes,
        block_size=block_size,
        gpu_budget_bytes=gpu_budget_bytes,
        threshold=threshold,
        interpolation=interpolation,
    )
    self._render_manager.add_visual(scene_id, gfx_visual, data_store)
    self._visual_to_scene[visual_model.id] = scene_id
    return visual_model
```

---

### 6h — `add_canvas`

```python
def add_canvas(
    self,
    scene_id: UUID,
    available_dims: str = "3d",
    fov: float = 70.0,
    depth_range: tuple[float, float] = (1.0, 8000.0),
) -> QWidget:
    """Create a canvas attached to a scene and return its embeddable widget.

    Parameters
    ----------
    scene_id : UUID
        ID of an existing scene.
    available_dims : str
        ``"3d"`` or ``"2d"``.  Only ``"3d"`` is fully supported in Phase 1.
    fov : float
        Vertical field of view in degrees.
    depth_range : tuple[float, float]
        ``(near, far)`` clip distances.

    Returns
    -------
    QWidget
        The render widget.  Embed with ``layout.addWidget(widget)``.
    """
    canvas_id = uuid4()

    # Build the model-layer Canvas (cameras dict, serializable state).
    camera_model = PerspectiveCamera(
        fov=fov,
        near_clipping_plane=depth_range[0],
        far_clipping_plane=depth_range[1],
        controller=OrbitCameraController(enabled=True),
    )
    canvas_model = Canvas(cameras={available_dims: camera_model})
    self._model.scenes[scene_id].canvases[canvas_model.id] = canvas_model

    # Build and register the render-layer CanvasView.
    canvas_view = self._render_manager.add_canvas(
        canvas_id,
        scene_id,
        parent=self._widget_parent,
        fov=fov,
        depth_range=depth_range,
    )

    # Keep reverse maps for controller operations.
    self._canvas_to_scene[canvas_id] = scene_id
    self._scene_to_canvases[scene_id].append(canvas_id)

    # Fit the camera to the scene's current bounding box.
    gfx_scene = self._render_manager.get_scene(scene_id)
    canvas_view.show_object(gfx_scene)

    return canvas_view.widget
```

---

### 6i — Scene and visual lookup

```python
def get_scene(self, scene_id: UUID) -> Scene:
    """Return the live Scene model for scene_id."""
    return self._model.scenes[scene_id]

def get_scene_by_name(self, name: str) -> Scene:
    """Return the live Scene model for the given name.

    Raises KeyError if no scene with that name exists.
    """
    for scene in self._model.scenes.values():
        if scene.name == name:
            return scene
    raise KeyError(f"No scene named {name!r}")

def get_visual_model(self, visual_id: UUID) -> MultiscaleImageVisual:
    """Return the live visual model for visual_id.

    Searches all scenes.  Raises KeyError if not found.
    """
    for scene in self._model.scenes.values():
        for visual in scene.visuals:
            if visual.id == visual_id:
                return visual
    raise KeyError(f"No visual with id {visual_id}")
```

---

### 6j — Reslicing

These are the core controller methods.  Their job is to:

1. Read appearance fields from each model-layer visual.
2. Build a `dict[UUID, VisualRenderConfig]` mapping visual ID to config.
3. Derive a `DimsState` from the scene's `DimsManager`.
4. Forward both to the appropriate `RenderManager` method.

```python
def _build_visual_configs_for_scene(
    self, scene_id: UUID
) -> dict[UUID, VisualRenderConfig]:
    """Build a VisualRenderConfig dict from all visuals in a scene."""
    scene = self._model.scenes[scene_id]
    configs: dict[UUID, VisualRenderConfig] = {}
    for visual in scene.visuals:
        if isinstance(visual, MultiscaleImageVisual):
            configs[visual.id] = VisualRenderConfig(
                lod_bias=visual.appearance.lod_bias,
                force_level=visual.appearance.force_level,
                frustum_cull=visual.appearance.frustum_cull,
            )
    return configs

def _dims_state_for_scene(self, scene_id: UUID) -> DimsState:
    """Derive a DimsState from the scene's DimsManager."""
    dims = self._model.scenes[scene_id].dims
    return DimsState(
        displayed_axes=dims.displayed_axes,
        slice_indices=dims.slice_indices,
    )

def reslice_all(self) -> None:
    """Trigger a data load for all visuals across all scenes."""
    for scene_id in self._model.scenes:
        self.reslice_scene(scene_id)

def reslice_scene(self, scene_id: UUID) -> None:
    """Trigger a data load for all visuals in one scene."""
    dims_state = self._dims_state_for_scene(scene_id)
    visual_configs = self._build_visual_configs_for_scene(scene_id)
    self._render_manager.reslice_scene(scene_id, dims_state, visual_configs)

def reslice_visual(self, visual_id: UUID) -> None:
    """Trigger a data load for one visual."""
    scene_id = self._visual_to_scene[visual_id]
    dims_state = self._dims_state_for_scene(scene_id)
    visual = self.get_visual_model(visual_id)
    if isinstance(visual, MultiscaleImageVisual):
        cfg = VisualRenderConfig(
            lod_bias=visual.appearance.lod_bias,
            force_level=visual.appearance.force_level,
            frustum_cull=visual.appearance.frustum_cull,
        )
    else:
        cfg = VisualRenderConfig()
    self._render_manager.reslice_visual(visual_id, dims_state, cfg)
```

---

### 6k — Camera operations

```python
def look_at_visual(
    self,
    visual_id: UUID,
    view_direction: tuple[float, float, float] = (-1, -1, -1),
    up: tuple[float, float, float] = (0, 0, 1),
) -> None:
    """Fit the camera to a visual's bounding box.

    Parameters
    ----------
    visual_id : UUID
        ID of the target visual.
    view_direction : tuple[float, float, float]
        Camera look direction vector (need not be normalized).
    up : tuple[float, float, float]
        Camera up vector.
    """
    scene_id = self._visual_to_scene[visual_id]
    canvas_ids = self._scene_to_canvases.get(scene_id, [])
    gfx_scene = self._render_manager.get_scene(scene_id)
    for canvas_id in canvas_ids:
        canvas_view = self._render_manager._canvases[canvas_id]
        canvas_view._camera.show_object(
            gfx_scene, view_dir=view_direction, up=up
        )

def set_camera_depth_range(
    self,
    scene_id: UUID,
    depth_range: tuple[float, float],
) -> None:
    """Set the near/far clip distances for all canvases attached to a scene.

    Parameters
    ----------
    scene_id : UUID
        ID of the target scene.
    depth_range : tuple[float, float]
        ``(near, far)`` clip distances in world units.
    """
    canvas_ids = self._scene_to_canvases.get(scene_id, [])
    for canvas_id in canvas_ids:
        canvas_view = self._render_manager._canvases[canvas_id]
        canvas_view.set_depth_range(depth_range)
```

---

### 6l — Stubs for future features

```python
def add_points(self, *args, **kwargs):
    raise NotImplementedError("add_points is not implemented in Phase 1.")

def add_labels(self, *args, **kwargs):
    raise NotImplementedError("add_labels is not implemented in Phase 1.")

def add_mesh(self, *args, **kwargs):
    raise NotImplementedError("add_mesh is not implemented in Phase 1.")

def remove_scene(self, scene_id: UUID) -> None:
    raise NotImplementedError("remove_scene is not implemented in Phase 1.")

def remove_visual(self, visual_id: UUID) -> None:
    raise NotImplementedError("remove_visual is not implemented in Phase 1.")

def remove_data_store(self, data_store_id: UUID) -> None:
    raise NotImplementedError("remove_data_store is not implemented in Phase 1.")

def get_dims_widget(self, scene_id: UUID, parent: QWidget | None = None):
    """Return a dims slider widget for a scene.

    Not implemented in Phase 1.  A dims slider widget requires a Qt widget
    that is wired to the scene's DimsManager via the EventBus, which is not
    yet implemented in v2.
    """
    raise NotImplementedError("get_dims_widget is not implemented in Phase 1.")

def on_dims_changed(self, scene_id: UUID, callback) -> None:
    """Register a callback to be called when dims change.

    Not implemented in Phase 1.  Requires the EventBus.
    """
    raise NotImplementedError("on_dims_changed is not implemented in Phase 1.")
```

---

## Step 7 — Unit tests for `CellierController`

**File:** `tests/v2/test_controller.py`  (new file)

These tests use the `small_zarr_store` fixture from `tests/v2/conftest.py` (a 2-level
zarr store at `s0` (8³) and `s1` (4³)).  All tests are headless — no Qt event loop,
no GPU.

### 7a — `test_add_scene_registers_in_model`

```python
def test_add_scene_registers_in_model():
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    assert scene.id in controller._model.scenes
    assert controller._model.scenes[scene.id].name == "main"
```

### 7b — `test_add_data_store_registers_in_model`

```python
def test_add_data_store_registers_in_model(small_zarr_store):
    controller = CellierController()
    store = MultiscaleZarrDataStore(zarr_path=str(small_zarr_store), scale_names=["s0", "s1"])
    result = controller.add_data_store(store)
    assert result is store
    assert store.id in controller._model.data.stores
```

### 7c — `test_add_image_populates_scene_and_render_layer`

```python
def test_add_image_populates_scene_and_render_layer(small_zarr_store):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(zarr_path=str(small_zarr_store), scale_names=["s0", "s1"])
    appearance = ImageAppearance(color_map="viridis", clim=(0.0, 1.0),
                                 lod_bias=1.0, force_level=None, frustum_cull=True)
    visual = controller.add_image(data=store, scene_id=scene.id,
                                  appearance=appearance, name="vol")
    assert isinstance(visual, MultiscaleImageVisual)
    # Model layer
    assert visual in controller._model.scenes[scene.id].visuals
    assert store.id in controller._model.data.stores
    # Render layer
    assert visual.id in controller._render_manager._visual_to_scene
```

### 7d — `test_get_scene_by_name`

```python
def test_get_scene_by_name():
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="my_scene")
    found = controller.get_scene_by_name("my_scene")
    assert found.id == scene.id
    with pytest.raises(KeyError):
        controller.get_scene_by_name("nonexistent")
```

### 7e — `test_reslice_scene_reads_appearance_fields`

This test verifies that `reslice_scene` correctly translates `ImageAppearance` fields
into `VisualRenderConfig` without a bridge.  It mocks `RenderManager.reslice_scene` to
capture the `visual_configs` argument.

```python
def test_reslice_scene_reads_appearance_fields(small_zarr_store):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(zarr_path=str(small_zarr_store), scale_names=["s0", "s1"])
    appearance = ImageAppearance(color_map="viridis", clim=(0.0, 1.0),
                                 lod_bias=2.5, force_level=1, frustum_cull=False)
    visual = controller.add_image(data=store, scene_id=scene.id,
                                  appearance=appearance, name="vol")

    captured = {}
    original = controller._render_manager.reslice_scene
    def capturing_reslice(s_id, dims_state, visual_configs=None):
        captured.update(visual_configs or {})
    controller._render_manager.reslice_scene = capturing_reslice

    controller.reslice_scene(scene.id)

    assert visual.id in captured
    cfg = captured[visual.id]
    assert cfg.lod_bias == 2.5
    assert cfg.force_level == 1
    assert cfg.frustum_cull is False
```

### 7f — `test_to_file_roundtrip`

```python
def test_to_file_roundtrip(tmp_path, small_zarr_store):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = MultiscaleZarrDataStore(zarr_path=str(small_zarr_store), scale_names=["s0", "s1"])
    appearance = ImageAppearance(color_map="viridis", clim=(0.0, 1.0))
    controller.add_image(data=store, scene_id=scene.id, appearance=appearance, name="vol")

    path = tmp_path / "session.json"
    controller.to_file(path)

    restored = ViewerModel.from_file(path)
    assert len(restored.scenes) == 1
    assert len(list(restored.scenes.values())[0].visuals) == 1
```

### 7g — `test_stubs_raise`

```python
def test_stubs_raise():
    controller = CellierController()
    with pytest.raises(NotImplementedError):
        controller.from_model(ViewerModel(data=DataManager()))
    with pytest.raises(NotImplementedError):
        controller.remove_scene(uuid4())
    with pytest.raises(NotImplementedError):
        controller.get_dims_widget(uuid4())
    with pytest.raises(NotImplementedError):
        controller.on_dims_changed(uuid4(), lambda x: None)
```

**Verification:** `uv run pytest tests/v2/test_controller.py` — all pass.

---

## Step 8 — Write `example_cellier.py`

**File:** `scripts/v2/lut_texture_multiscale_frustum_async_cellier/example_cellier.py`

This is a new file alongside the existing `example.py` (Phase 3 async, direct wiring)
and `example_v2.py` (Phase 3 async, `RenderManager`-based wiring).  The visual
behaviour is identical to both; only the construction and update wiring changes.

`example_v2.py` is preserved as-is; `example_cellier.py` does not modify it.

### What changes vs `example_v2.py`

| Aspect | `example_v2.py` | `example_cellier.py` |
|---|---|---|
| Imports | `RenderManager`, `DimsState`, `GFXMultiscaleImageVisual`, `frustum_planes_from_corners`, `MultiscaleImageVisual` | `CellierController`, `CoordinateSystem` |
| Construction wiring | 7 lines: `uuid4()` × 2, `add_scene`, `add_canvas`, `add_visual`, `get_scene`, `show_object` | 3 lines: `add_scene`, `add_image`, `add_canvas` |
| Stored references | `self._scene_id: UUID`, `self._render_manager`, `self._canvas_view` | `self._scene: Scene`, `self._visual: MultiscaleImageVisual`, `self._controller` |
| `_on_update_clicked` | Calls `get_scene_config()`, mutates `config.*`, constructs `DimsState`, calls `trigger_update` | Mutates `self._visual.appearance.*` directly, calls `self._controller.reslice_scene(self._scene.id)` |
| `_on_far_plane_changed` | Accesses `self._canvas_view._camera.depth_range` directly | Calls `self._controller.set_camera_depth_range(scene_id=self._scene.id, depth_range=(1.0, value))` |
| `_on_show_frustum_toggled` | Unchanged | Unchanged |
| AABB wireframe | Added directly to scene via `render_manager.get_scene(scene_id)` | Added directly to scene via `controller._render_manager.get_scene(scene.id)` (acceptable in the script) |
| Frustum wireframe rebuild | Unchanged (accesses `_canvas_view._camera` directly for snapshot) | Gets camera snapshot via `controller._render_manager._canvases` (acceptable in the script) |
| `async_main` / `main` | Unchanged | Only change: construct `MainWindow(data_store)` not `MainWindow(visual, data_store)`; the controller builds `visual` internally |

### Imports block

```python
from cellier.v2.controller import CellierController
from cellier.v2.data.image import MultiscaleZarrDataStore
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.visuals._image import ImageAppearance
```

Remove from imports:
```python
# These are no longer needed in application code:
# from cellier.v2.render import DimsState, RenderManager
# from cellier.v2.render._frustum import frustum_planes_from_corners
# from cellier.v2.render.visuals import GFXMultiscaleImageVisual
# from cellier.v2.visuals import MultiscaleImageVisual
```

### `MainWindow.__init__` construction block

Replace the seven-line block:

```python
# OLD (example_v2.py)
scene_id = uuid4()
canvas_id = uuid4()
self._scene_id = scene_id
self._render_manager = RenderManager()
self._render_manager.add_scene(scene_id, dim="3d")
self._canvas_view = self._render_manager.add_canvas(canvas_id, scene_id, parent=self)
self._render_manager.add_visual(scene_id, visual, data_store)
...
self._canvas_view.show_object(self._render_manager.get_scene(scene_id))
```

With:

```python
# NEW (example_cellier.py)
self._controller = CellierController(widget_parent=self)
cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
self._scene = self._controller.add_scene(dim="3d", coordinate_system=cs, name="main")
self._visual = self._controller.add_image(
    data=data_store,
    scene_id=self._scene.id,
    appearance=ImageAppearance(
        color_map="viridis",
        clim=(0.0, 1.0),
        lod_bias=LOD_BIAS,
        force_level=None,
        frustum_cull=True,
    ),
    name="volume",
    threshold=0.2,
)
canvas_widget = self._controller.add_canvas(self._scene.id)
```

The AABB wireframe is added directly to the pygfx scene (acceptable in a script):

```python
gfx_scene = self._controller._render_manager.get_scene(self._scene.id)
gfx_scene.add(_make_box_wireframe(aabb_min, aabb_max, AABB_COLOR))
```

The canvas widget is embedded in the layout:

```python
root.addWidget(canvas_widget, stretch=1)
```

Note: `show_object` is called automatically inside `add_canvas`.  There is no
`self._canvas_view` stored; the widget reference is not needed after layout embedding.

### `_on_update_clicked`

```python
async def _on_update_clicked(self) -> None:
    self._update_count += 1
    t_total = time.perf_counter()

    # Write current UI values into the visual appearance model.
    # CellierController reads these at reslice time — no separate config object.
    self._visual.appearance.lod_bias = self._lod_bias_sb.value()
    self._visual.appearance.force_level = self._force_level
    self._visual.appearance.frustum_cull = self._frustum_cull_cb.isChecked()

    # Trigger: snapshot camera, plan LOD + frustum cull, submit async loads.
    self._controller.reslice_scene(self._scene.id)

    # Timing / stats readout from the render layer (planning is synchronous).
    # Access the GFX visual via the render manager for debug stats only.
    gfx_visual = self._controller._render_manager._scenes[
        self._scene.id
    ]._visuals[self._visual.id]
    stats = gfx_visual._last_plan_stats
    t_planning_ms = (time.perf_counter() - t_total) * 1000

    # Frustum wireframe snapshot — same as example_v2.py.
    canvas_view = next(iter(self._controller._render_manager._canvases.values()))
    corners = np.asarray(canvas_view._camera.frustum, dtype=np.float64).copy()
    self._rebuild_frustum_wireframe(corners)

    # ... rest of debug print and task submission unchanged from example_v2.py ...
```

### `_on_far_plane_changed`

```python
def _on_far_plane_changed(self, value: float) -> None:
    self._controller.set_camera_depth_range(
        scene_id=self._scene.id,
        depth_range=(1.0, value),
    )
```

### `MainWindow` constructor signature

`MainWindow` no longer receives a pre-built `GFXMultiscaleImageVisual`.  The data store
is passed instead; the controller builds the visual internally.

```python
class MainWindow(QMainWindow):
    def __init__(self, data_store: MultiscaleZarrDataStore) -> None:
```

`async_main` is updated accordingly:

```python
async def async_main(data_store: MultiscaleZarrDataStore) -> None:
    window = MainWindow(data_store)
    window.resize(1100, 700)
    window.show()
    await asyncio.get_event_loop().create_future()
```

`main()` removes the manual `GFXMultiscaleImageVisual.from_cellier_model` call and
passes `data_store` directly to `async_main`.

**Verification:** `uv run example_cellier.py` from the script directory.  The viewer
must behave identically to `example_v2.py`: orbiting works, Update loads bricks,
frustum cull checkbox / force-level radios / LOD bias spinbox / far-plane spinbox all
function, debug print output is unchanged.

---

## Implementation order summary

| Step | Files changed | Verification |
|---|---|---|
| 1 | `_scene_config.py`, `render/__init__.py` | REPL import |
| 2 | `scene_manager.py` | `pytest tests/v2/render/` |
| 3 | `slice_coordinator.py` | (covered by render tests) |
| 4 | `render_manager.py`, `canvas_view.py` | `pytest tests/v2/render/` |
| 5 | `tests/v2/render/test_render_manager.py` | `pytest tests/v2/render/` — all pass |
| 6 | `src/cellier/v2/controller.py` (new) | REPL import |
| 7 | `tests/v2/test_controller.py` (new) | `pytest tests/v2/test_controller.py` |
| 8 | `scripts/.../example_cellier.py` (new) | Manual run |

Each step is independently verifiable.  Steps 1–5 touch only the existing render layer.
Steps 6–7 add the new controller without touching the render layer.  Step 8 is a script,
not library code, and has no automated test.

---

## What is NOT implemented in Phase 1

The following `CellierController` methods raise `NotImplementedError` with an
explanatory message.  They are intentionally stubbed, not forgotten.

| Method | Reason deferred |
|---|---|
| `from_model(model)` | Materializing the render layer from a model requires reading `level_shapes` from every store; not needed for the example |
| `from_file(path)` | Depends on `from_model` |
| `remove_scene` / `remove_visual` / `remove_data_store` | Render-layer teardown paths not implemented |
| `add_points` / `add_labels` / `add_mesh` | No v2 render-layer visuals yet |
| `get_dims_widget` | Requires a v2 dims slider widget backed by the EventBus |
| `on_dims_changed` | Requires the EventBus |

`to_model()` is implemented but carries an explicit docstring warning: camera positions
in the returned model are NOT synchronized from the render layer (they remain at
construction-time defaults).  Full camera capture is deferred to a later phase.
