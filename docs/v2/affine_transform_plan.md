# Plan: `AffineTransform` for Cellier v2

The v2 transform lives at `src/cellier/v2/transform/` — a clean copy, independent
of the v1 `src/cellier/transform/` module. This avoids breaking any v1 code while
allowing v2 to own the design going forward.

The changes fall into three groups: structural (base class), correctness fixes, and
API additions. Each step produces a verifiable working state.

---

## File layout

```
src/cellier/v2/transform/
    __init__.py          # exports AffineTransform
    _base.py             # BaseTransform (ABC, frozen BaseModel)
    _affine.py           # AffineTransform

tests/v2/transform/
    __init__.py
    test_affine.py
```

---

## Step 1 — `BaseTransform`: drop EventedModel, add frozen config

**File:** `src/cellier/v2/transform/_base.py`

The v1 `BaseTransform` inherits from both `EventedModel` (psygnal) and `ABC`. In v2
the transform is a plain frozen pydantic model: the signal for a transform change
travels on the visual's `EventedModel` field, not on the transform itself.

Changes:
- Inherit from `ABC` and pydantic `BaseModel` only; remove `EventedModel`.
- Add `model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)` on
  `BaseTransform` so every subclass inherits it without redeclaring.
- Keep all four abstract methods: `map_coordinates`, `imap_coordinates`,
  `map_normal_vector`, `imap_normal_vector`.

**Verify:** `AffineTransform` construction succeeds; attempting
`transform.matrix = np.eye(4)` raises `ValidationError` (frozen).

---

## Step 2 — `AffineTransform`: config, validator, serializer cleanup

**File:** `src/cellier/v2/transform/_affine.py`

- Remove `model_config` from `AffineTransform` (now inherited).
- Fix `field_validator`:
  - Change `v: str` → `v: Any`.
  - Drop the unused `info: ValidationInfo` parameter and the `ValidationInfo` import.
  - Add a shape assertion: `if np.asarray(v).shape != (4, 4): raise ValueError(...)`.
- Fix `field_serializer`: remove `@classmethod`; use `self` (or `_`).

**Verify:** constructing with a `(3, 3)` matrix raises a clear `ValueError`;
JSON round-trip test passes.

---

## Step 3 — Make `_to_vec4` private; fix error message

**File:** `src/cellier/v2/transform/_affine.py`

- Rename `to_vec4` → `_to_vec4`.
- Rename internal variable `ndim` → `n_components`.
- Rewrite error:
  `f"coordinates must have 3 or 4 components per point, got {n_components}"`.

No behaviour change.

**Verify:** tests still pass; `from cellier.v2.transform import _to_vec4` is not
possible (underscore prefix convention).

---

## Step 4 — Add `inverse_matrix` cached property

**File:** `src/cellier/v2/transform/_affine.py`

```python
@cached_property
def inverse_matrix(self) -> np.ndarray:
    """Cached inverse of the affine matrix."""
    return np.linalg.inv(self.matrix).astype(np.float32)
```

Replace both bare `np.linalg.inv(self.matrix)` calls in `imap_coordinates` and
`map_normal_vector` with `self.inverse_matrix`.

`cached_property` works correctly on frozen pydantic models: pydantic's `__setattr__`
freeze applies only to declared *fields*, not to Python descriptor-set attributes.

**Verify:** call `imap_coordinates` and `map_normal_vector` on the same instance;
assert the cached value equals `np.linalg.inv(transform.matrix)`.

---

## Step 5 — Clarify normal-vector transform convention in docstrings

**File:** `src/cellier/v2/transform/_affine.py`

The relationship between `map_normal_vector` and `imap_normal_vector` is not
obvious:

- `map_normal_vector` transforms a normal from data space to world space.
  Normals transform by the *transpose-inverse* of the point transform matrix,
  so it uses `M⁻¹` (i.e. `self.inverse_matrix`).
- `imap_normal_vector` is the inverse operation: world normal → data normal.
  That is `(M⁻¹)⁻¹ = M`, so it uses `self.matrix`.

Both methods already compute this correctly. The fix is docstring-only: add an
`"Notes"` section to each method stating the convention, and link the two docstrings
to each other.

**Verify:** add a test with a known non-uniform scale transform where the expected
normal direction after transformation can be calculated by hand.

---

## Step 6 — Add missing constructors

**File:** `src/cellier/v2/transform/_affine.py`

Add two classmethods alongside the existing `from_translation` and
`from_scale_and_translation`:

**`identity()`**

```python
@classmethod
def identity(cls) -> Self:
    """Return the identity transform."""
    return cls(matrix=np.eye(4, dtype=np.float32))
```

**`from_scale(scale)`**

```python
@classmethod
def from_scale(cls, scale: tuple[float, float, float]) -> Self:
    """Return a scale-only transform with no translation."""
    return cls.from_scale_and_translation(scale, (0.0, 0.0, 0.0))
```

The `default_transform` helper used for `BaseVisual.transform` should call
`AffineTransform.identity()` instead of `AffineTransform(matrix=np.eye(4))`.

**Verify:** `identity().map_coordinates(p) == p`; `from_scale((2,2,2)).map_coordinates(p) == 2*p`;
`identity()` round-trips through `model_dump_json` / `model_validate_json`.

---

## Step 7 — Add `compose` and `__matmul__`

**File:** `src/cellier/v2/transform/_affine.py`

```python
def compose(self, other: "AffineTransform") -> "AffineTransform":
    """Return a new transform equivalent to applying self then other.

    compose(A, B).map_coordinates(p) == B.map_coordinates(A.map_coordinates(p))
    """
    return AffineTransform(matrix=other.matrix @ self.matrix)

def __matmul__(self, other: "AffineTransform") -> "AffineTransform":
    """Alias for compose: (A @ B) applies A first, then B."""
    return self.compose(other)
```

The `@` operator reads left-to-right in application order:
`scale_to_data @ data_to_world` means "first scale_to_data, then data_to_world".

Document the convention in both the method docstring and the class-level docstring
with a one-line example.

**Verify:**
- `A.compose(B).map_coordinates(p) ≈ B.map_coordinates(A.map_coordinates(p))`.
- `(scale_to_data @ data_to_world)` ≠ `(data_to_world @ scale_to_data)` for
  non-commuting transforms.
- The ad-hoc composition in `_chunk_selection.py` is replaced with the new method
  as a follow-on cleanup (separate commit, not part of this plan).

---

## Step 8 — Tests

**File:** `tests/v2/transform/test_affine.py`

One test function per behaviour (no test classes, per project convention):

| Test function | Covers |
|---|---|
| `test_identity_maps_point_to_itself` | `identity()` forward and inverse |
| `test_identity_json_roundtrip` | serialization of `identity()` |
| `test_from_scale_diagonal` | correct matrix; `imap` recovers input |
| `test_from_translation_constructor` | correct matrix; round-trip |
| `test_from_scale_and_translation_constructor` | correct matrix; round-trip |
| `test_map_coordinates_roundtrip` | `imap(map(p)) ≈ p` for batch |
| `test_invalid_matrix_shape_raises` | `(3,3)` input raises `ValueError` |
| `test_frozen_raises_on_mutation` | `transform.matrix = …` raises |
| `test_inverse_matrix_cached` | value matches `np.linalg.inv`; computed once |
| `test_normal_vector_transform_known_result` | non-uniform scale, analytic answer |
| `test_normal_vector_roundtrip` | `imap_normal(map_normal(n)) ≈ n` |
| `test_compose_order` | `(A @ B)(p) == B(A(p))` |
| `test_compose_noncommutative` | `A @ B ≠ B @ A` for scale+translate |
| `test_to_vec4_private` | `_to_vec4` not in public `__all__` |

---

## What is explicitly out of scope (transform class)

- 2D coordinates (2-component points). `_to_vec4` continues to require 3 or 4
  components. If 2D slicing ever needs 2D affine math, that is a separate decision.
- `BaseTransform` abstract methods for `map_normal_vector` / `imap_normal_vector`
  are kept as-is; they remain required on all subclasses.
- No changes to the v1 `src/cellier/transform/` module.

---

# Part 2: Connecting `AffineTransform` to the v2 System

There are two distinct responsibilities that must be wired:

**GPU-side positioning** — the rendered node must sit in the correct world-space
position. pygfx handles this natively via `node.local.matrix`. The matrix is written
once at construction and updated on every transform change.

**CPU-side slicing** — the camera quantities in a `ReslicingRequest` are in world
space. The planning methods inside each GFX visual work in data space (brick grid
indices derived from voxel coordinates). Each visual is responsible for transforming
incoming world-space camera geometry into its own data space before planning.

### Design decisions

**Pattern 5: GFX visual stores `_transform` and applies it internally.**
The GFX visual holds `self._transform: AffineTransform`, initialised at construction
and updated in `on_transform_changed`. Because `on_transform_changed` must exist
anyway to update `node.local.matrix`, storing the transform there is free. The
`SceneManager` passes world-space quantities unchanged and gains no new dependencies.
`VisualRenderConfig` stays lean — three fields, no geometry.

**Option A: `build_slice_request` receives `frustum_corners_world`, not pre-computed
planes.**
The visual receives raw world-space frustum corners, applies `imap_coordinates` to
them, then calls `frustum_planes_from_corners` on the data-space result. This is the
correct order: transform first, then compute derived geometry. The alternative
(transforming pre-computed plane equations) is mathematically equivalent but requires
the transpose-inverse of plane normals — a less obvious operation that is harder to
test and more fragile to modify. The `SceneManager` no longer calls
`frustum_planes_from_corners` directly.

**`_world` suffix on incoming parameters.**
All world-space parameters on `build_slice_request` and `build_slice_request_2d` gain
a `_world` suffix. This makes it unambiguous at every call site that the visual is
responsible for the world→data transform, and that the caller is passing world-space
geometry without knowing the visual's data-space layout.

The steps below address each layer in dependency order. Each step is independently
verifiable before the next begins.

---

## Step 9 — Add `transform` field to v2 `BaseVisual`

**File:** `src/cellier/v2/visuals/_base_visual.py`

Add one field to `BaseVisual`:

```python
from cellier.v2.transform import AffineTransform

class BaseVisual(EventedModel):
    ...
    transform: AffineTransform = Field(default_factory=AffineTransform.identity)
```

Because `AffineTransform` is a frozen pydantic `BaseModel`, psygnal treats it as a
scalar field: replacing `visual.transform` with a new `AffineTransform` fires
`visual.events.transform(new_transform)`. In-place mutation is impossible (frozen), so
there is no risk of silent, unsignalled changes.

`AffineTransform` serialises to a nested list via `field_serializer`, so the existing
JSON round-trip for `Scene` continues to work without additional changes.

**Verify:**
- `MultiscaleImageVisual()` (no `transform` argument) has `transform.matrix == np.eye(4)`.
- Setting `visual.transform = AffineTransform.from_scale((2, 2, 2))` fires
  `visual.events.transform`.
- `Scene` JSON round-trip test still passes.

---

## Step 10 — Add `TransformChangedEvent` to the EventBus catalogue

**File:** `src/cellier/v2/events/_events.py`

```python
if TYPE_CHECKING:
    from cellier.v2.transform import AffineTransform

class TransformChangedEvent(NamedTuple):
    """Fired when a visual's data-to-world transform is replaced.

    Primary consumers:
    - ``GFX*Visual``: update ``node.local.matrix`` and ``_transform`` field.
    - ``CellierController``: trigger a reslice (data space has changed).
    """
    source_id: UUID
    scene_id: UUID
    visual_id: UUID
    transform: "AffineTransform"
```

Also update `CellierEventTypes` to include `TransformChangedEvent`, and re-export it
from `src/cellier/v2/events/__init__.py`.

**Verify:** `TransformChangedEvent` appears in `__all__`; it constructs as a plain
NamedTuple.

---

## Step 11 — Controller: wire the transform signal

**File:** `src/cellier/v2/controller.py`

`VisualRenderConfig` is unchanged — it keeps its three existing fields and gains no
`transform` field. The transform travels via the EventBus, not via the reslice config.

Three additions to the controller:

**11a — `_wire_transform`**

Analogous to `_wire_appearance`. Connects `visual.events.transform` to a closure
that emits `TransformChangedEvent` on the bus, then triggers a reslice:

```python
def _wire_transform(self, visual: BaseVisual, scene_id: UUID) -> None:
    visual.events.transform.connect(
        self._make_transform_handler(visual.id, scene_id)
    )

def _make_transform_handler(
    self, visual_id: UUID, scene_id: UUID
) -> Callable:
    def _on_transform(new_transform: AffineTransform) -> None:
        self._event_bus.emit(
            TransformChangedEvent(
                source_id=self._id,
                scene_id=scene_id,
                visual_id=visual_id,
                transform=new_transform,
            )
        )
        # Transform change invalidates the current brick set — always reslice.
        self.reslice_scene(scene_id)

    return _on_transform
```

**11b — Call `_wire_transform` from `add_image` / `add_visual`**

Both `_add_image_multiscale` and `_add_image_memory` call `_wire_appearance` after
constructing the visual model. Add `_wire_transform(visual_model, scene_id)` directly
below each of those calls.

**11c — Subscribe GFX visual to `TransformChangedEvent`**

After `self._render_manager.add_visual(...)`:

```python
self._event_bus.subscribe(
    TransformChangedEvent,
    gfx_visual.on_transform_changed,
    entity_id=visual_model.id,
    owner_id=visual_model.id,
)
```

**Verify:** replacing `visual.transform` triggers `on_transform_changed` on the GFX
visual (mock test), and `reslice_scene` is called once per transform replacement.

---

## Step 12 — GFX visuals: store transform, update GPU, apply in planning

**Files:**
- `src/cellier/v2/render/visuals/_image.py` (`GFXMultiscaleImageVisual`)
- `src/cellier/v2/render/visuals/_image_memory.py` (`GFXImageMemoryVisual`)

This step has three parts that all land together.

**12a — Store `_transform` and initialise `node.local.matrix` at construction**

`from_cellier_model` gains a `transform: AffineTransform` parameter, defaulting to
`AffineTransform.identity()` so existing call sites compile without changes. Two things
happen with it:

```python
# In __init__ / from_cellier_model
self._transform: AffineTransform = transform

if self.node_3d is not None:
    self.node_3d.local.matrix = transform.matrix
if self.node_2d is not None:
    self.node_2d.local.matrix = transform.matrix
```

pygfx reads `node.local.matrix` as the local-to-parent transform and concatenates it
with parent transforms when building the world matrix for `u_wobject.world_transform`
in the shader. No shader changes are required.

**12b — `on_transform_changed`: update both `_transform` and the GPU node**

The two updates happen in a single handler, guaranteeing they are always in sync:

```python
def on_transform_changed(self, event: TransformChangedEvent) -> None:
    """Update the stored transform and pygfx node when the model's transform changes."""
    self._transform = event.transform
    matrix = event.transform.matrix
    if self.node_3d is not None:
        self.node_3d.local.matrix = matrix
    if self.node_2d is not None:
        self.node_2d.local.matrix = matrix
```

**12c — Apply `_transform` inside `build_slice_request` and `build_slice_request_2d`**

The planning methods receive world-space quantities and immediately convert them to
data space before any planning logic runs. The parameter names gain `_world` suffixes
to make the coordinate system explicit at every call site.

*3D signature change:*

```python
def build_slice_request(
    self,
    camera_pos_world: np.ndarray,
    frustum_corners_world: np.ndarray | None,   # was: frustum_planes
    thresholds: list[float],
    dims_state: DimsState,
    force_level: int | None = None,
) -> list[ChunkRequest]:
    camera_pos_data = self._transform.imap_coordinates(
        camera_pos_world.reshape(1, -1)
    ).flatten()

    if frustum_corners_world is not None:
        corners_data = self._transform.imap_coordinates(
            frustum_corners_world.reshape(-1, 3)
        ).reshape(frustum_corners_world.shape)
        frustum_planes_data = frustum_planes_from_corners(corners_data)
    else:
        frustum_planes_data = None

    # remainder of planning unchanged, uses camera_pos_data and frustum_planes_data
    ...
```

`frustum_planes_from_corners` moves from `SceneManager` into the visual. The import
of `frustum_planes_from_corners` is removed from `scene_manager.py` and added to
`_image.py`.

*2D signature change:*

```python
def build_slice_request_2d(
    self,
    camera_pos_world: np.ndarray,
    viewport_width_px: float,
    world_width: float,
    view_min_world: np.ndarray | None,   # was: view_min
    view_max_world: np.ndarray | None,   # was: view_max
    dims_state: DimsState,
    lod_bias: float = 1.0,
    force_level: int | None = None,
    use_culling: bool = True,
) -> list[ChunkRequest]:
    camera_pos_data = self._transform.imap_coordinates(
        camera_pos_world.reshape(1, -1)
    ).flatten()

    if use_culling and view_min_world is not None and view_max_world is not None:
        # Inverse-transform the four corners of the viewport rectangle.
        # Taking the AABB of the result is conservative (over-selects near
        # corners for rotated transforms) but never under-selects.
        cx = float(camera_pos_world[0])
        cy = float(camera_pos_world[1])
        half_w = world_width / 2.0
        half_h = world_height / 2.0
        corners_world = np.array(
            [
                [cx - half_w, cy - half_h, 0.0],
                [cx + half_w, cy - half_h, 0.0],
                [cx + half_w, cy + half_h, 0.0],
                [cx - half_w, cy + half_h, 0.0],
            ],
            dtype=np.float32,
        )
        corners_data = self._transform.imap_coordinates(corners_world)
        view_min_data = corners_data[:, :2].min(axis=0)
        view_max_data = corners_data[:, :2].max(axis=0)
    else:
        view_min_data = None
        view_max_data = None

    # remainder of planning unchanged, uses camera_pos_data and view_min/max_data
    ...
```

Note that `view_min_world` / `view_max_world` are no longer used directly inside the
method body — the AABB is recomputed from corners. They are kept as parameters so the
`SceneManager` calling convention is unchanged (it still passes a bounding box, not
corners). Internally the corners are reconstructed from `camera_pos_world` and
`world_width`. This avoids changing the `SceneManager` 2D call site.

**Known limitation — LOD accuracy under non-uniform scale**

`world_width` (used in 2D LOD selection) and the distance thresholds (used in 3D LOD
selection) are calibrated in world units. When the transform includes non-uniform
scale, the data-space voxel density at a given camera distance differs from the
world-space value. LOD selection may be off by one level for strongly anisotropic
transforms. Culling (AABB and frustum-plane tests) is always correct in data space.
Correct LOD under non-uniform scale requires propagating the effective scale factor
into threshold computation and is deferred.

**Verify:**
- Construct a `GFXMultiscaleImageVisual` with a translation transform; assert
  `node_3d.local.matrix` matches the expected matrix.
- Emit a `TransformChangedEvent`; assert `node_3d.local.matrix` and `_transform`
  both update.
- Call `build_slice_request` with a 2× uniform scale transform and non-identity
  `camera_pos_world`; assert the `camera_pos` reaching the brick planner is halved.
- Call `build_slice_request` with a 45° rotation; assert `frustum_planes_from_corners`
  receives data-space corners that differ from the world-space input.
- Identity transform: all existing GFX visual tests pass unchanged.

---

## Step 13 — `SceneManager`: pass world-space quantities and remove plane computation

**File:** `src/cellier/v2/render/scene_manager.py`

With the transform logic now inside the GFX visual, `SceneManager` becomes simpler,
not more complex. Two changes:

**13a — 3D path: pass `frustum_corners_world` instead of `frustum_planes`**

Remove the `frustum_planes_from_corners` call. Pass `request.frustum_corners` (or
`None` when `frustum_cull=False`) directly to the visual:

```python
frustum_corners_world = request.frustum_corners if cfg.frustum_cull else None

chunk_requests = visual.build_slice_request(
    camera_pos_world=request.camera_pos,
    frustum_corners_world=frustum_corners_world,
    thresholds=thresholds,
    dims_state=request.dims_state,
    force_level=cfg.force_level,
)
```

The import of `frustum_planes_from_corners` is removed from `scene_manager.py`.

**13b — 2D path: rename arguments to `_world` suffix**

The AABB computation and the call to `build_slice_request_2d` update parameter names
to match the new signature:

```python
chunk_requests = visual.build_slice_request_2d(
    camera_pos_world=request.camera_pos,
    viewport_width_px=viewport_width_px,
    world_width=world_width,
    view_min_world=view_min if cfg.frustum_cull else None,
    view_max_world=view_max if cfg.frustum_cull else None,
    dims_state=request.dims_state,
    lod_bias=cfg.lod_bias,
    force_level=cfg.force_level,
    use_culling=cfg.frustum_cull,
)
```

The world-space AABB (`view_min`, `view_max`) is still computed in `SceneManager`
before the loop — it is the same value for all visuals and the computation is cheap.
The visual ignores the received `view_min_world` / `view_max_world` directly; it uses
them only to decide whether culling is active and reconstructs data-space bounds from
corners internally (see Step 12c).

**Verify:**
- All existing `SceneManager` tests pass (identity transform, no behaviour change).
- `frustum_planes_from_corners` is no longer imported or called in `scene_manager.py`.
- A mock visual's `build_slice_request` receives `frustum_corners_world` not
  `frustum_planes`.

---

## Step 14 — Tests

**New test functions** (added to or alongside existing test files):

| File | Test function | Covers |
|---|---|---|
| `tests/v2/visuals/test_base_visual.py` | `test_transform_defaults_to_identity` | field default |
| `tests/v2/visuals/test_base_visual.py` | `test_transform_field_fires_psygnal` | psygnal signal on replace |
| `tests/v2/visuals/test_scene.py` | `test_scene_roundtrip_with_non_identity_transform` | JSON round-trip |
| `tests/v2/events/test_events.py` | `test_transform_changed_event_in_catalogue` | `CellierEventTypes` membership |
| `tests/v2/render/test_gfx_image_visual.py` | `test_node_matrix_set_on_construction` | GPU-side init |
| `tests/v2/render/test_gfx_image_visual.py` | `test_on_transform_changed_updates_node_and_field` | GPU + stored field sync |
| `tests/v2/render/test_gfx_image_visual.py` | `test_build_slice_request_camera_pos_transformed` | 3D `imap` on camera pos |
| `tests/v2/render/test_gfx_image_visual.py` | `test_build_slice_request_frustum_corners_transformed` | 3D corners → data-space planes |
| `tests/v2/render/test_gfx_image_visual.py` | `test_build_slice_request_2d_viewport_transformed` | 2D AABB in data space |
| `tests/v2/render/test_gfx_image_visual.py` | `test_identity_transform_is_noop_3d` | regression |
| `tests/v2/render/test_gfx_image_visual.py` | `test_identity_transform_is_noop_2d` | regression |
| `tests/v2/render/test_scene_manager.py` | `test_3d_passes_corners_not_planes` | interface change |
| `tests/v2/render/test_scene_manager.py` | `test_frustum_cull_false_passes_none_corners` | `None` passthrough |
| `tests/v2/render/test_scene_manager.py` | `test_frustum_planes_from_corners_not_called` | import removed |
| `tests/v2/controller/test_controller.py` | `test_wire_transform_emits_event` | controller bridge |
| `tests/v2/controller/test_controller.py` | `test_transform_change_triggers_reslice` | reslice on change |

---

## Execution order and dependencies

```
Step 9  (BaseVisual.transform field)
    ↓
Step 10 (TransformChangedEvent)
    ↓
Step 11 (Controller wiring)
    ↓
Step 12 (GFX visual: store + GPU + planning)  ←── highest-risk step
    ↓
Step 13 (SceneManager: interface change)
    ↓
Step 14 (Tests — written alongside each step above)
```

Step 12 is the highest-risk step: it changes a hot path (the planning methods) and
removes `frustum_planes_from_corners` from `SceneManager` at the same time. It should
be validated against the full integration example
(`scripts/v2/integration_2d_3d/example_combined_camera_redraw.py`) with a
non-identity transform before Step 13 is merged.

---

## What is explicitly out of scope (connection)

- **Undo/redo.** Transform replacement fires a psygnal event, which is sufficient for
  a future undo stack, but the undo stack itself is not part of this plan.
- **Animated transforms.** Smooth interpolation between two poses would be implemented
  at the application layer by rapidly replacing `visual.transform`; no internal
  infrastructure is needed.
- **LOD correction for non-uniform scale.** Documented as a known limitation in
  Step 12; deferred.
- **Normal-vector transform in slicing.** The current slicing pipeline does not use
  normal vectors. If oblique-plane slicing is added later, `imap_normal_vector` will
  be needed inside `build_slice_request`; it is not needed now.
- **Multi-visual shared transforms** (e.g. a group node). Out of scope; each visual
  owns its own independent `AffineTransform`.
