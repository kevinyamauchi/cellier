# In-Memory Image Viewing — Cellier v2 Implementation Plan

**Goal:** Add `ImageMemoryStore` (data layer) and `ImageVisual` / `GFXImageMemoryVisual`
(model + render layers) so that a numpy array can be displayed as a `gfx.Image` (2D) or
`gfx.Volume` (3D) visual in a Cellier v2 scene.

**Scope:** Seven files to create, four to modify, one demo script.  
**Do not touch:** Any existing v2 file that is not listed in the modifications section.

---

## 0. File map

| Action | Path |
|---|---|
| **Create** | `src/cellier/v2/data/image/_image_memory_store.py` |
| **Modify** | `src/cellier/v2/data/image/__init__.py` |
| **Modify** | `src/cellier/v2/data/_types.py` |
| **Create** | `src/cellier/v2/visuals/_image_memory.py` |
| **Modify** | `src/cellier/v2/visuals/_types.py` |
| **Modify** | `src/cellier/v2/visuals/__init__.py` |
| **Create** | `src/cellier/v2/render/visuals/_image_memory.py` |
| **Modify** | `src/cellier/v2/render/visuals/__init__.py` |
| **Modify** | `src/cellier/v2/controller.py` |
| **Create** | `scripts/v2/image_memory_demo.py` |
| **Create** | `tests/v2/data/test_image_memory_store.py` |
| **Create** | `tests/v2/render/test_gfx_image_memory_visual.py` |

---

## 1. `src/cellier/v2/data/image/_image_memory_store.py`  *(create)*

This is a `BaseDataStore` subclass that holds the numpy array and serves slices to the
`AsyncSlicer`.

```python
# src/cellier/v2/data/image/_image_memory_store.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator

from cellier.v2.data._base_data_store import BaseDataStore

if TYPE_CHECKING:
    from cellier.v2.data.image._image_requests import ChunkRequest


class ImageMemoryStore(BaseDataStore):
    """In-memory image data store backed by a numpy array.

    Serves axis-aligned slices or full sub-volumes to the AsyncSlicer.
    All reads are synchronous (the array is in CPU RAM); the method is
    still declared ``async`` to satisfy the AsyncSlicer contract.

    Parameters
    ----------
    data : np.ndarray
        The image data. Any dtype; coerced to float32 on construction.
        Shape convention follows numpy axis order — e.g. (D, H, W) for
        3-D, (H, W) for 2-D, (T, C, D, H, W) for 5-D.
    name : str
        Human-readable label. Default ``"image_memory_store"``.
    """

    store_type: Literal["image_memory"] = "image_memory"
    name: str = "image_memory_store"
    data: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Pydantic validators / serializers for numpy
    # ------------------------------------------------------------------

    @field_validator("data", mode="before")
    @classmethod
    def _coerce_float32(cls, v: Any) -> np.ndarray:
        """Coerce the input to a contiguous float32 C-array."""
        arr = np.asarray(v, dtype=np.float32)
        return np.ascontiguousarray(arr)

    @field_serializer("data")
    def _serialize_data(self, array: np.ndarray, _info: Any) -> list:
        """Serialise the array as a nested Python list for JSON round-trips."""
        return array.tolist()

    # ------------------------------------------------------------------
    # Read-only properties (used by CellierController.add_image)
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of dimensions in the stored array."""
        return self.data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the stored array in numpy axis order."""
        return tuple(self.data.shape)

    @property
    def n_levels(self) -> int:
        """Always 1 — single-resolution, no multiscale pyramid."""
        return 1

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        """List with one entry (level 0 = the full array)."""
        return [self.shape]

    # ------------------------------------------------------------------
    # Async data access (called by AsyncSlicer)
    # ------------------------------------------------------------------

    async def get_data(self, request: ChunkRequest) -> np.ndarray:
        """Return the requested sub-region as a float32 array.

        Interprets ``request.axis_selections`` generically:

        - ``int`` entry  → sliced axis; the integer index is applied and the
          axis is dropped from the output.
        - ``(start, stop)`` tuple → displayed axis; a slice is applied and
          the axis is kept in the output.

        Out-of-bounds coordinates are clamped to array extents and
        zero-padded on the output side so the returned shape always matches
        what the caller requested.

        Parameters
        ----------
        request : ChunkRequest
            Built by ``GFXImageMemoryVisual.build_slice_request[_2d]``.
            ``request.scale_index`` is always 0 (ignored).
            ``request.axis_selections`` has one entry per data axis.

        Returns
        -------
        np.ndarray
            float32 array with one dimension per displayed (tuple) axis.
        """
        store_shape = self.data.shape

        # ── 1. Compute the output shape ─────────────────────────────────
        out_shape: list[int] = []
        for ax, sel in enumerate(request.axis_selections):
            if isinstance(sel, tuple):
                start, stop = sel
                out_shape.append(stop - start)
            # int → axis dropped from output

        out = np.zeros(out_shape, dtype=np.float32)

        # ── 2. Build clamped source indices and destination slices ───────
        src: list[int | slice] = []
        dst: list[slice] = []
        all_valid = True

        for ax, sel in enumerate(request.axis_selections):
            dim_size = store_shape[ax]
            if isinstance(sel, tuple):
                start, stop = sel
                c_start = max(0, start)
                c_stop = min(dim_size, stop)
                if c_stop <= c_start:
                    all_valid = False
                    break
                src.append(slice(c_start, c_stop))
                dst_start = c_start - start
                dst_stop = dst_start + (c_stop - c_start)
                dst.append(slice(dst_start, dst_stop))
            else:
                # Scalar — clamp to valid range, keeps axis out of output.
                idx = int(np.clip(sel, 0, dim_size - 1))
                src.append(idx)

        if all_valid:
            out[tuple(dst)] = self.data[tuple(src)]

        return out
```

**Key notes for the implementer:**

- `store_type = "image_memory"` is the pydantic discriminator; it must match the entry
  added to `DataStoreType` in step 3.
- `get_data` is `async` even though the body is synchronous. This satisfies the
  `AsyncSlicer` contract. For very large arrays an `await asyncio.sleep(0)` could be
  inserted before the read to yield the event loop, but for typical in-memory sizes it
  is unnecessary.
- Do **not** add a `PrivateAttr` for the backing array; the array lives directly in the
  pydantic `data` field. This is intentional — the field validator handles coercion.

---

## 2. `src/cellier/v2/data/image/__init__.py`  *(modify)*

Add `ImageMemoryStore` to the package's public API. The existing file already exports
`ChunkRequest` and `MultiscaleZarrDataStore`. Append:

```python
from cellier.v2.data.image._image_memory_store import ImageMemoryStore

__all__ = [
    "ChunkRequest",
    "MultiscaleZarrDataStore",
    "ImageMemoryStore",          # ← add
]
```

---

## 3. `src/cellier/v2/data/_types.py`  *(modify)*

The existing file contains:

```python
DataStoreType = Annotated[
    Union[MultiscaleZarrDataStore,],
    Field(discriminator="store_type"),
]
```

Add `ImageMemoryStore`:

```python
from cellier.v2.data.image._image_memory_store import ImageMemoryStore   # ← add

DataStoreType = Annotated[
    Union[MultiscaleZarrDataStore, ImageMemoryStore],                      # ← add
    Field(discriminator="store_type"),
]
```

Keep the existing `MultiscaleZarrDataStore` import unchanged.

---

## 4. `src/cellier/v2/visuals/_image_memory.py`  *(create)*

Two classes: an `Appearance` model and the `Visual` model.

```python
# src/cellier/v2/visuals/_image_memory.py
from __future__ import annotations

from typing import Literal

from cmap import Colormap
from pydantic import Field

from cellier.v2.visuals._base_visual import BaseAppearance, BaseVisual


class ImageMemoryAppearance(BaseAppearance):
    """Appearance parameters for an in-memory image visual.

    Parameters
    ----------
    color_map : cmap.Colormap
        Colourmap applied after contrast normalisation. Accepts any
        cmap-registered name string (e.g. ``"viridis"``, ``"bids:magma"``).
    clim : tuple[float, float]
        Contrast limits ``(min, max)`` used to normalise pixel values
        before colour-mapping. Default ``(0.0, 1.0)``.
    interpolation : str
        Texture sampler filter. ``"linear"`` (default) or ``"nearest"``.
    visible : bool
        Inherited from ``BaseAppearance``. Default ``True``.
    """

    color_map: Colormap
    clim: tuple[float, float] = (0.0, 1.0)
    interpolation: Literal["linear", "nearest"] = "linear"


class ImageVisual(BaseVisual):
    """Model-layer visual for a single-resolution in-memory image.

    Wraps a pygfx ``gfx.Image`` (2D scene) or ``gfx.Volume`` (3D scene).
    The associated data store must be an ``ImageMemoryStore``.

    Camera movement does **not** trigger a reslice because data is not
    view-dependent — the full slice is always loaded.

    Parameters
    ----------
    visual_type : Literal["image_memory"]
        Discriminator field; always ``"image_memory"``.
    name : str
        Human-readable label.
    data_store_id : str
        UUID (as a string) of the ``ImageMemoryStore`` this visual reads from.
    appearance : ImageMemoryAppearance
        Appearance parameters.
    requires_camera_reslice : bool
        Always ``False``; frozen. Camera movement does not trigger reslicing.
    """

    visual_type: Literal["image_memory"] = "image_memory"
    data_store_id: str
    appearance: ImageMemoryAppearance

    requires_camera_reslice: bool = Field(default=False, frozen=True)
```

**Key notes:**

- `data_store_id` is typed `str` (matching the pattern in `MultiscaleImageVisual`) even
  though the underlying UUID is a `UUID4`. The controller stores it as `str(data.id)`.
- `ImageMemoryAppearance` intentionally omits `lod_bias`, `force_level`, and
  `frustum_cull` — those only make sense for multiscale out-of-core data.

---

## 5. `src/cellier/v2/visuals/_types.py`  *(modify)*

The existing file contains:

```python
VisualType = Annotated[
    Union[MultiscaleImageVisual,],
    Field(discriminator="visual_type"),
]
```

Add `ImageVisual`:

```python
from cellier.v2.visuals._image_memory import ImageVisual   # ← add

VisualType = Annotated[
    Union[MultiscaleImageVisual, ImageVisual],               # ← add
    Field(discriminator="visual_type"),
]
```

---

## 6. `src/cellier/v2/visuals/__init__.py`  *(modify)*

The existing file exports `ImageAppearance`, `MultiscaleImageVisual`, `VisualType`.
Append:

```python
from cellier.v2.visuals._image_memory import ImageMemoryAppearance, ImageVisual

__all__ = [
    "ImageAppearance",
    "MultiscaleImageVisual",
    "VisualType",
    "ImageMemoryAppearance",   # ← add
    "ImageVisual",             # ← add
]
```

---

## 7. `src/cellier/v2/render/visuals/_image_memory.py`  *(create)*

This is the render-layer class. It is a plain Python class (not a pydantic model) that
owns the pygfx nodes and implements the planning + commit interface expected by
`SceneManager` and `SliceCoordinator`.

```python
# src/cellier/v2/render/visuals/_image_memory.py
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.data.image._image_requests import ChunkRequest

if TYPE_CHECKING:
    from cellier.v2._state import DimsState
    from cellier.v2.events._events import (
        AppearanceChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.v2.visuals._image_memory import ImageVisual


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_colormap(color_map) -> gfx.TextureMap:
    """Convert a cmap Colormap to a pygfx TextureMap (1-D LUT texture)."""
    lut = (color_map.lut(256) * 255).astype(np.uint8)
    tex = gfx.Texture(lut, dim=1)
    return gfx.TextureMap(tex)


def _build_axis_selections(
    dims_state: DimsState,
    store_shape: tuple[int, ...],
) -> tuple[int | tuple[int, int], ...]:
    """Build axis_selections for a ChunkRequest from a DimsState.

    Displayed axes receive ``(0, store_shape[axis])`` — the full extent.
    Sliced axes receive their integer slice index.

    Parameters
    ----------
    dims_state : DimsState
        Immutable snapshot from the controller.
    store_shape : tuple[int, ...]
        Shape of the backing numpy array (from ``ImageMemoryStore.shape``).

    Returns
    -------
    tuple[int | tuple[int, int], ...]
        One entry per data axis, in data axis order.
    """
    sel = dims_state.selection  # AxisAlignedSelectionState
    ndim = len(store_shape)
    result: list[int | tuple[int, int]] = []
    for ax in range(ndim):
        if ax in sel.displayed_axes:
            result.append((0, store_shape[ax]))
        else:
            result.append(sel.slice_indices[ax])
    return tuple(result)


# ---------------------------------------------------------------------------
# GFXImageMemoryVisual
# ---------------------------------------------------------------------------


class GFXImageMemoryVisual:
    """Render-layer visual for one ``ImageVisual`` backed by ``ImageMemoryStore``.

    Owns a single pygfx node — either ``gfx.Image`` (render_mode=="2d") or
    ``gfx.Volume`` (render_mode=="3d") — and a placeholder 1×1 or 2×2×2
    texture that is replaced on the first ``on_data_ready[_2d]`` call.

    There is no brick cache, no LUT indirection, and no LOD selection. Every
    reslice produces exactly one ``ChunkRequest`` for the full slice / volume.
    The node geometry is replaced on every commit (see step 7 of the plan for
    the trade-off discussion).

    Parameters
    ----------
    visual_model : ImageVisual
        Associated model-layer visual. Provides the initial appearance.
    data_store : ImageMemoryStore
        The backing data store. Used to query shape in planning methods.
    render_mode : str
        ``"2d"`` builds a ``gfx.Image`` node; ``"3d"`` builds a
        ``gfx.Volume`` node.

    Interface consumed by SceneManager / SliceCoordinator
    -------------------------------------------------------
    ``visual_model_id``         — UUID linking back to the model layer
    ``node_2d`` / ``node_3d``   — pygfx WorldObject (one is None)
    ``n_levels``                — always 1
    ``build_slice_request_2d``  — planning path for 2D scenes
    ``build_slice_request``     — planning path for 3D scenes
    ``on_data_ready_2d``        — commit path for 2D scenes
    ``on_data_ready``           — commit path for 3D scenes
    ``on_appearance_changed``   — react to AppearanceChangedEvent
    ``on_visibility_changed``   — toggle node visibility
    """

    def __init__(
        self,
        visual_model: ImageVisual,
        data_store: ImageMemoryStore,
        render_mode: str,
    ) -> None:
        if render_mode not in ("2d", "3d"):
            raise ValueError(f"render_mode must be '2d' or '3d', got {render_mode!r}")

        self.visual_model_id: UUID = visual_model.id
        self._render_mode = render_mode
        self._data_store = data_store

        appearance = visual_model.appearance
        colormap = _make_colormap(appearance.color_map)

        if render_mode == "2d":
            # Placeholder 1×1 texture — replaced on first on_data_ready_2d call.
            # pygfx dim=2 textures must be (W, H, C) with C=1 for single-channel.
            placeholder = np.zeros((1, 1, 1), dtype=np.float32)
            tex = gfx.Texture(placeholder, dim=2, format="1xf4")
            self.node_2d: gfx.WorldObject | None = gfx.Image(
                gfx.Geometry(grid=tex),
                gfx.ImageBasicMaterial(
                    clim=appearance.clim,
                    map=colormap,
                    interpolation=appearance.interpolation,
                    pick_write=visual_model.pick_write,
                ),
            )
            self.node_3d: gfx.WorldObject | None = None

        else:  # render_mode == "3d"
            # Placeholder 2×2×2 texture — replaced on first on_data_ready call.
            # pygfx dim=3 textures must be (W, H, D) for gfx.Volume.
            placeholder = np.zeros((2, 2, 2), dtype=np.float32)
            tex = gfx.Texture(placeholder, dim=3, format="1xf4")
            self.node_3d = gfx.Volume(
                gfx.Geometry(grid=tex),
                gfx.VolumeBasicMaterial(
                    clim=appearance.clim,
                    map=colormap,
                    interpolation=appearance.interpolation,
                    pick_write=visual_model.pick_write,
                ),
            )
            self.node_2d = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Always 1 — single-resolution in-memory store."""
        return 1

    # ------------------------------------------------------------------
    # Planning — build ChunkRequests (synchronous, < 1 ms)
    # ------------------------------------------------------------------

    def build_slice_request_2d(
        self,
        camera_pos: np.ndarray,
        viewport_width_px: float,
        world_width: float,
        view_min: np.ndarray | None,
        view_max: np.ndarray | None,
        dims_state: DimsState,
        lod_bias: float = 1.0,
        force_level: int | None = None,
        use_culling: bool = True,
    ) -> list[ChunkRequest]:
        """Return a single ChunkRequest for the full 2-D slice.

        All camera/viewport parameters are accepted for interface
        compatibility but are unused — no frustum culling or LOD selection
        is performed for in-memory data.

        The full extent of each displayed axis is always requested:
        ``(0, store_shape[axis])``.

        Parameters
        ----------
        dims_state : DimsState
            Current dimension state. Determines which axes are displayed
            and which are sliced (and at what index).

        Returns
        -------
        list[ChunkRequest]
            Always contains exactly one element.
        """
        axis_selections = _build_axis_selections(dims_state, self._data_store.shape)
        return [
            ChunkRequest(
                chunk_request_id=uuid4(),
                slice_request_id=uuid4(),
                scale_index=0,
                axis_selections=axis_selections,
            )
        ]

    def build_slice_request(
        self,
        camera_pos: np.ndarray,
        frustum_planes: np.ndarray | None,
        thresholds: list[float] | None,
        dims_state: DimsState | None = None,
        force_level: int | None = None,
    ) -> list[ChunkRequest]:
        """Return a single ChunkRequest for the full 3-D sub-volume.

        All camera/frustum parameters are accepted for interface
        compatibility but are unused — no frustum culling or LOD selection
        is performed for in-memory data.

        The full extent of each displayed axis is always requested:
        ``(0, store_shape[axis])``.

        Parameters
        ----------
        dims_state : DimsState or None
            Current dimension state. If ``None`` (e.g. during headless
            tests), all axes are treated as displayed.

        Returns
        -------
        list[ChunkRequest]
            Always contains exactly one element.
        """
        if dims_state is None:
            # Fallback: treat all axes as displayed, no slicing.
            ndim = self._data_store.ndim
            axis_selections = tuple(
                (0, self._data_store.shape[ax]) for ax in range(ndim)
            )
        else:
            axis_selections = _build_axis_selections(dims_state, self._data_store.shape)

        return [
            ChunkRequest(
                chunk_request_id=uuid4(),
                slice_request_id=uuid4(),
                scale_index=0,
                axis_selections=axis_selections,
            )
        ]

    # ------------------------------------------------------------------
    # Commit — receive data from AsyncSlicer and upload to GPU
    # ------------------------------------------------------------------

    def on_data_ready(
        self, batch: list[tuple[ChunkRequest, np.ndarray]]
    ) -> None:
        """Upload a 3-D array to the pygfx Volume node.

        Called on the main thread by ``SliceCoordinator`` after the
        ``AsyncSlicer`` completes the read. Replaces the node geometry
        entirely (no texture reuse; see plan step 7 for trade-off notes).

        pygfx defers the actual GPU memcpy until ``renderer.render()`` is
        called — this method only updates the CPU-side backing array.

        Parameters
        ----------
        batch : list of (ChunkRequest, np.ndarray)
            Always contains exactly one element for this visual type.
            ``data.shape`` is ``(D, H, W)`` (numpy axis order, displayed
            axes only).
        """
        if not batch or self.node_3d is None:
            return

        _request, data = batch[0]

        # pygfx Volume expects (W, H, D) — transpose from numpy (D, H, W).
        data_wgpu = np.ascontiguousarray(data.T)
        tex = gfx.Texture(data_wgpu, dim=3, format="1xf4")
        self.node_3d.geometry = gfx.Geometry(grid=tex)

    def on_data_ready_2d(
        self, batch: list[tuple[ChunkRequest, np.ndarray]]
    ) -> None:
        """Upload a 2-D slice to the pygfx Image node.

        Called on the main thread by ``SliceCoordinator`` after the
        ``AsyncSlicer`` completes the read.

        Parameters
        ----------
        batch : list of (ChunkRequest, np.ndarray)
            Always contains exactly one element for this visual type.
            ``data.shape`` is ``(H, W)`` (numpy axis order, displayed
            axes only).
        """
        if not batch or self.node_2d is None:
            return

        _request, data = batch[0]

        # pygfx Image expects (W, H, 1) — transpose H↔W and add channel dim.
        data_wgpu = np.ascontiguousarray(data.T[:, :, np.newaxis])
        tex = gfx.Texture(data_wgpu, dim=2, format="1xf4")
        self.node_2d.geometry = gfx.Geometry(grid=tex)

    # ------------------------------------------------------------------
    # Appearance and visibility event handlers
    # ------------------------------------------------------------------

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Apply a pure GPU-side appearance update (no reslice needed).

        Handles ``clim``, ``color_map``, and ``interpolation``.
        Unrecognised field names are silently ignored.

        Parameters
        ----------
        event : AppearanceChangedEvent
            Carries ``field_name`` and ``new_value``.
        """
        node = self.node_2d if self._render_mode == "2d" else self.node_3d
        if node is None:
            return
        material = node.material

        if event.field_name == "clim":
            material.clim = event.new_value
        elif event.field_name == "color_map":
            material.map = _make_colormap(event.new_value)
        elif event.field_name == "interpolation":
            material.interpolation = event.new_value
        # "visible" is handled by on_visibility_changed; ignore here.

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        """Toggle node visibility.

        Parameters
        ----------
        event : VisualVisibilityChangedEvent
            Carries ``visible`` bool.
        """
        node = self.node_2d if self._render_mode == "2d" else self.node_3d
        if node is not None:
            node.visible = event.visible
```

**Key notes for the implementer:**

- The `format="1xf4"` texture format is float32 single-channel.  If the store ever
  provides uint8 data it would need `"1xu8"` or `"rgba8unorm"` — but the store always
  coerces to float32 so this is fixed.
- `data.T` transposes numpy `(H, W)` → `(W, H)` for 2D and `(D, H, W)` → `(W, H, D)`
  for 3D.  This is the required pygfx axis order.  See the known pygfx gotcha in the
  project memory for the dim=2 format requirements.
- The `[:, :, np.newaxis]` adds the required channel dimension for `gfx.Texture(dim=2)`.
- `AppearanceChangedEvent.field_name` and `.new_value` — confirm these attribute names
  match the actual `AppearanceChangedEvent` NamedTuple in
  `src/cellier/v2/events/_events.py` before finalising. The existing controller code
  uses `field_name` in `_make_appearance_handler`; use the same names here.

---

## 8. `src/cellier/v2/render/visuals/__init__.py`  *(modify)*

The existing file exports `GFXMultiscaleImageVisual` and `VolumeGeometry`. Append:

```python
from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

__all__ = [
    "GFXMultiscaleImageVisual",
    "VolumeGeometry",
    "GFXImageMemoryVisual",   # ← add
]
```

---

## 9. `src/cellier/v2/controller.py`  *(modify)*

Two changes: new imports and replacing the existing `add_image` method with an
overloaded version that dispatches on the type of `data`.

### 9.1 Add imports at the top of the file

```python
# Add to the typing imports at the top of the file:
from typing import overload

# Near the existing render-layer imports:
from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual

# Near the existing visual model imports:
from cellier.v2.visuals._image_memory import ImageMemoryAppearance, ImageVisual

# Near the existing data store imports:
from cellier.v2.data.image._image_memory_store import ImageMemoryStore
```

### 9.2 Replace the existing `add_image` with an overloaded version

Remove the existing `add_image` method body and replace it with the three-part
overload pattern below. The existing multiscale logic moves verbatim into the
`isinstance(data, MultiscaleZarrDataStore)` branch; the memory logic is new.

```python
@overload
def add_image(
    self,
    data: ImageMemoryStore,
    scene_id: UUID,
    appearance: ImageMemoryAppearance,
    name: str = ...,
) -> ImageVisual: ...

@overload
def add_image(
    self,
    data: MultiscaleZarrDataStore,
    scene_id: UUID,
    appearance: ImageAppearance,
    name: str = ...,
    block_size: int = ...,
    gpu_budget_bytes: int = ...,
    threshold: float = ...,
    interpolation: str = ...,
) -> MultiscaleImageVisual: ...

def add_image(
    self,
    data,
    scene_id: UUID,
    appearance,
    name: str = "image",
    block_size: int = 32,
    gpu_budget_bytes: int = 1 * 1024**3,
    threshold: float = 0.2,
    interpolation: str = "linear",
) -> MultiscaleImageVisual | ImageVisual:
    """Add an image visual to a scene.

    Dispatches to the appropriate rendering path based on the type of
    ``data``:

    - ``ImageMemoryStore`` → ``GFXImageMemoryVisual`` backed by
      ``gfx.Image`` (2D) or ``gfx.Volume`` (3D). No brick cache; the full
      slice is uploaded on every reslice.
    - ``MultiscaleZarrDataStore`` → ``GFXMultiscaleImageVisual`` backed by
      a brick cache + LUT indirection system. Supports LOD, frustum
      culling, and out-of-core streaming.

    The scene's ``dims.selection.displayed_axes`` determines 2D vs 3D
    rendering in both cases (length 2 → 2D, length 3 → 3D).

    Parameters
    ----------
    data : ImageMemoryStore | MultiscaleZarrDataStore
        The data source. Registered in the ``DataManager`` automatically
        if not already present.
    scene_id : UUID
        ID of an existing scene (returned by ``add_scene``).
    appearance : ImageMemoryAppearance | ImageAppearance
        Appearance parameters. Must match the type of ``data``:
        ``ImageMemoryAppearance`` for ``ImageMemoryStore``,
        ``ImageAppearance`` for ``MultiscaleZarrDataStore``.
    name : str
        Human-readable label for the visual. Default ``"image"``.
    block_size : int
        Brick side length in voxels. Only used for the multiscale path.
        Default 32.
    gpu_budget_bytes : int
        GPU memory budget for the brick cache. Only used for the
        multiscale path. Default 1 GiB.
    threshold : float
        Isosurface threshold for 3D raycast rendering. Only used for the
        multiscale path. Default 0.2.
    interpolation : str
        Sampler filter ``"linear"`` or ``"nearest"``. Only used for the
        multiscale path. Default ``"linear"``.

    Returns
    -------
    ImageVisual
        When ``data`` is an ``ImageMemoryStore``.
    MultiscaleImageVisual
        When ``data`` is a ``MultiscaleZarrDataStore``.
    """
    if isinstance(data, ImageMemoryStore):
        return self._add_image_memory(data, scene_id, appearance, name)
    else:
        return self._add_image_multiscale(
            data, scene_id, appearance, name,
            block_size, gpu_budget_bytes, threshold, interpolation,
        )
```

### 9.3 Add `_add_image_memory` private method

This is the extracted body of the memory path, called by the dispatch above.

```python
def _add_image_memory(
    self,
    data: ImageMemoryStore,
    scene_id: UUID,
    appearance: ImageMemoryAppearance,
    name: str,
) -> ImageVisual:
    # ── 1. Register the data store if needed ────────────────────────────
    if data.id not in self._model.data.stores:
        self._model.data.stores[data.id] = data

    # ── 2. Determine render mode from scene dimensionality ───────────────
    scene = self._model.scenes[scene_id]
    displayed_axes = scene.dims.selection.displayed_axes
    render_mode = "3d" if len(displayed_axes) == 3 else "2d"

    # ── 3. Build model-layer objects ─────────────────────────────────────
    visual_model = ImageVisual(
        name=name,
        data_store_id=str(data.id),
        appearance=appearance,
    )
    scene.visuals.append(visual_model)

    # ── 4. Build render-layer object ─────────────────────────────────────
    gfx_visual = GFXImageMemoryVisual(
        visual_model=visual_model,
        data_store=data,
        render_mode=render_mode,
    )

    # ── 5. Register with RenderManager and controller maps ───────────────
    self._render_manager.add_visual(scene_id, gfx_visual, data)
    self._visual_to_scene[visual_model.id] = scene_id

    # ── 6. Wire appearance bridge and EventBus subscriptions ─────────────
    self._wire_appearance(visual_model)
    self._event_bus.subscribe(
        AppearanceChangedEvent,
        gfx_visual.on_appearance_changed,
        entity_id=visual_model.id,
        owner_id=visual_model.id,
    )
    self._event_bus.subscribe(
        VisualVisibilityChangedEvent,
        gfx_visual.on_visibility_changed,
        entity_id=visual_model.id,
        owner_id=visual_model.id,
    )
    self._event_bus.emit(
        VisualAddedEvent(
            source_id=self._id,
            scene_id=scene_id,
            visual_id=visual_model.id,
        )
    )
    return visual_model
```

### 9.4 Rename `_add_image_multiscale` private method

Rename the existing `add_image` body to `_add_image_multiscale`. The body is
unchanged — only the name changes and `self` is added as the first parameter (it was
already there). This keeps the multiscale logic isolated and testable independently.

```python
def _add_image_multiscale(
    self,
    data: MultiscaleZarrDataStore,
    scene_id: UUID,
    appearance: ImageAppearance,
    name: str,
    block_size: int,
    gpu_budget_bytes: int,
    threshold: float,
    interpolation: str,
) -> MultiscaleImageVisual:
    # ... existing add_image body verbatim ...
```

**Note on `_wire_appearance`:** `_wire_appearance` is called with both
`MultiscaleImageVisual` (multiscale path) and `ImageVisual` (memory path). Verify it is
generic over any `BaseVisual` — it should only inspect the field name from the psygnal
`EmissionInfo`, not the visual type. The `_RESLICE_FIELDS` frozenset
(`{"lod_bias", "force_level", "frustum_cull"}`) will never match any field on
`ImageMemoryAppearance`, so no spurious reslice events will be emitted for memory
visuals. If `_wire_appearance` does contain any `isinstance` check on the visual type,
remove it.

---

## 10. `scripts/v2/image_memory_demo.py`  *(create)*

A standalone runnable demo that creates a 3-D numpy array and opens it in both a 2-D
and a 3-D canvas side by side.

```python
#!/usr/bin/env python
"""Demo: in-memory 3D image viewed as 2D slice and 3D volume.

Run with:
    uv run python scripts/v2/image_memory_demo.py
"""

from __future__ import annotations

import sys

import numpy as np
import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtWidgets import QApplication, QHBoxLayout, QWidget
from skimage.data import binary_blobs

from cellier.v2.controller import CellierController
from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.scene.dims import CoordinateSystem
from cellier.v2.visuals._image_memory import ImageMemoryAppearance


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Create data ──────────────────────────────────────────────────
    volume = binary_blobs(length=128, volume_fraction=0.1, n_dim=3).astype(np.float32)
    store = ImageMemoryStore(data=volume, name="demo_volume")

    # ── 2. Set up the controller ────────────────────────────────────────
    controller = CellierController()

    # Coordinate system: 3 spatial axes
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))

    # 2D scene: add_scene infers displayed_axes=(1,2), slice_indices={0:0}
    scene_2d = controller.add_scene(dim="2d", coordinate_system=cs, name="slice")
    # 3D scene: add_scene infers displayed_axes=(0,1,2), slice_indices={}
    scene_3d = controller.add_scene(dim="3d", coordinate_system=cs, name="volume")

    appearance = ImageMemoryAppearance(
        color_map="viridis",
        clim=(0.0, 1.0),
    )

    # Add the same store to both scenes (store is registered only once)
    visual_2d = controller.add_image(
        data=store,
        scene_id=scene_2d.id,
        appearance=appearance,
        name="slice_view",
    )
    visual_3d = controller.add_image(
        data=store,
        scene_id=scene_3d.id,
        appearance=appearance,
        name="volume_view",
    )

    # ── 3. Create canvases ──────────────────────────────────────────────
    root = QWidget()
    layout = QHBoxLayout(root)

    canvas_2d = controller._render_manager.add_canvas(
        canvas_id=scene_2d.canvases and list(scene_2d.canvases.keys())[0]
        if hasattr(scene_2d, "canvases") and scene_2d.canvases
        else __import__("uuid", fromlist=["uuid4"]).uuid4(),
        scene_id=scene_2d.id,
        parent=root,
    )
    canvas_3d = controller._render_manager.add_canvas(
        canvas_id=__import__("uuid", fromlist=["uuid4"]).uuid4(),
        scene_id=scene_3d.id,
        parent=root,
    )

    layout.addWidget(canvas_2d.widget)
    layout.addWidget(canvas_3d.widget)
    root.resize(1200, 600)
    root.show()

    # ── 4. Initial reslice ──────────────────────────────────────────────
    controller.reslice_all()

    QtAsyncio.run()


if __name__ == "__main__":
    main()
```

**Note:** The canvas creation pattern above is provisional — match the exact
`add_canvas` call signature used in existing v2 tests/demos (see
`tests/v2/test_controller.py` and `src/cellier/v2/controller.py` for how canvas IDs are
managed). The `add_canvas` call needs a UUID for `canvas_id`; generate one fresh with
`uuid4()` and keep a reference if you need it later. The important pattern is:

```python
from uuid import uuid4
canvas_id_2d = uuid4()
canvas_id_3d = uuid4()
canvas_2d = controller._render_manager.add_canvas(canvas_id_2d, scene_2d.id, parent=root)
canvas_3d = controller._render_manager.add_canvas(canvas_id_3d, scene_3d.id, parent=root)
```

---

## 11. Tests  *(create)*

### 11.1 `tests/v2/data/test_image_memory_store.py`

These tests are headless (no Qt, no GPU, no async runtime required beyond a simple
`asyncio.run()`).

```python
"""Tests for ImageMemoryStore."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.data.image._image_requests import ChunkRequest
from uuid import uuid4


def _req(*axis_selections) -> ChunkRequest:
    return ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=0,
        axis_selections=axis_selections,
    )


# ── Construction ────────────────────────────────────────────────────────────

def test_coerces_to_float32():
    data = np.ones((4, 4, 4), dtype=np.uint8)
    store = ImageMemoryStore(data=data)
    assert store.data.dtype == np.float32


def test_shape_and_ndim():
    data = np.zeros((10, 20, 30))
    store = ImageMemoryStore(data=data)
    assert store.shape == (10, 20, 30)
    assert store.ndim == 3
    assert store.n_levels == 1
    assert store.level_shapes == [(10, 20, 30)]


# ── Serialisation round-trip ────────────────────────────────────────────────

def test_json_roundtrip():
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    store = ImageMemoryStore(data=data)
    restored = ImageMemoryStore.model_validate_json(store.model_dump_json())
    np.testing.assert_array_equal(store.data, restored.data)


# ── get_data: 2D slice from 3D volume ──────────────────────────────────────

def test_get_data_2d_slice_yx():
    """Slice z=1 from (Z=3, Y=4, X=5) array; displayed=(Y,X)."""
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    store = ImageMemoryStore(data=data)
    req = _req(1, (0, 4), (0, 5))  # z=1 fixed, full Y and X
    result = asyncio.run(store.get_data(req))
    assert result.shape == (4, 5)
    np.testing.assert_array_equal(result, data[1, :, :])


def test_get_data_2d_slice_zx():
    """Slice y=2 from (Z=3, Y=4, X=5) array; displayed=(Z,X)."""
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    store = ImageMemoryStore(data=data)
    req = _req((0, 3), 2, (0, 5))  # full Z and X, y=2 fixed
    result = asyncio.run(store.get_data(req))
    assert result.shape == (3, 5)
    np.testing.assert_array_equal(result, data[:, 2, :])


# ── get_data: full 3D volume ────────────────────────────────────────────────

def test_get_data_3d_full():
    """Request the full (Z, Y, X) volume."""
    data = np.random.rand(5, 6, 7).astype(np.float32)
    store = ImageMemoryStore(data=data)
    req = _req((0, 5), (0, 6), (0, 7))
    result = asyncio.run(store.get_data(req))
    assert result.shape == (5, 6, 7)
    np.testing.assert_array_equal(result, data)


# ── get_data: 5D dataset, 2D scene ─────────────────────────────────────────

def test_get_data_5d_to_2d():
    """5D (T=2, C=3, Z=10, Y=20, X=30), display Y-X at t=0, c=1, z=5."""
    shape = (2, 3, 10, 20, 30)
    data = np.random.rand(*shape).astype(np.float32)
    store = ImageMemoryStore(data=data)
    req = _req(0, 1, 5, (0, 20), (0, 30))
    result = asyncio.run(store.get_data(req))
    assert result.shape == (20, 30)
    np.testing.assert_array_equal(result, data[0, 1, 5, :, :])


# ── get_data: 5D dataset, 3D scene ─────────────────────────────────────────

def test_get_data_5d_to_3d():
    """5D (T=2, C=3, Z=10, Y=20, X=30), display Z-Y-X at t=0, c=1."""
    shape = (2, 3, 10, 20, 30)
    data = np.random.rand(*shape).astype(np.float32)
    store = ImageMemoryStore(data=data)
    req = _req(0, 1, (0, 10), (0, 20), (0, 30))
    result = asyncio.run(store.get_data(req))
    assert result.shape == (10, 20, 30)
    np.testing.assert_array_equal(result, data[0, 1, :, :, :])


# ── get_data: out-of-bounds clamping ───────────────────────────────────────

def test_get_data_clamps_negative_start():
    """Negative start is clamped; result is zero-padded on the left."""
    data = np.ones((10, 10), dtype=np.float32)
    store = ImageMemoryStore(data=data)
    req = _req((-2, 8), (0, 10))   # Y starts at -2 → first 2 rows zero-padded
    result = asyncio.run(store.get_data(req))
    assert result.shape == (10, 10)
    np.testing.assert_array_equal(result[:2, :], 0.0)    # padded rows
    np.testing.assert_array_equal(result[2:, :], 1.0)    # real data
```

### 11.2 `tests/v2/render/test_gfx_image_memory_visual.py`

These tests verify planning and commit logic without a GPU. The `gfx.Texture` and
`gfx.Geometry` constructors do touch pygfx internals — if those require a WGPU device
in the test environment, mock `gfx.Texture` and `gfx.Geometry` with `MagicMock` as
done in the existing render manager tests.

```python
"""Tests for GFXImageMemoryVisual."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from cellier.v2._state import AxisAlignedSelectionState, DimsState
from cellier.v2.data.image._image_memory_store import ImageMemoryStore
from cellier.v2.data.image._image_requests import ChunkRequest
from cellier.v2.visuals._image_memory import ImageMemoryAppearance, ImageVisual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(shape=(10, 20, 30)) -> ImageMemoryStore:
    return ImageMemoryStore(data=np.zeros(shape, dtype=np.float32))


def _make_appearance() -> ImageMemoryAppearance:
    return ImageMemoryAppearance(color_map="viridis", clim=(0.0, 1.0))


def _make_visual_model(store: ImageMemoryStore) -> ImageVisual:
    return ImageVisual(
        name="test",
        data_store_id=str(store.id),
        appearance=_make_appearance(),
    )


def _make_dims_state_2d() -> DimsState:
    """3D store, 2D scene: display (Y, X), slice Z at 5."""
    return DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(1, 2),
            slice_indices={0: 5},
        ),
    )


def _make_dims_state_3d() -> DimsState:
    """3D store, 3D scene: display all axes."""
    return DimsState(
        axis_labels=("z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(0, 1, 2),
            slice_indices={},
        ),
    )


# ---------------------------------------------------------------------------
# Tests: planning
# ---------------------------------------------------------------------------

@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_build_slice_request_2d_returns_one_request(mock_gfx):
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="2d")

    dims = _make_dims_state_2d()
    requests = visual.build_slice_request_2d(
        camera_pos=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min=None,
        view_max=None,
        dims_state=dims,
    )

    assert len(requests) == 1
    req = requests[0]
    assert isinstance(req, ChunkRequest)
    assert req.scale_index == 0
    # Axis 0 (z) is sliced at index 5
    assert req.axis_selections[0] == 5
    # Axes 1 and 2 (y, x) are displayed — full extent
    assert req.axis_selections[1] == (0, 20)
    assert req.axis_selections[2] == (0, 30)
    # Both UUIDs are freshly generated
    assert req.chunk_request_id is not None
    assert req.slice_request_id is not None


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_build_slice_request_3d_returns_one_request(mock_gfx):
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="3d")

    dims = _make_dims_state_3d()
    requests = visual.build_slice_request(
        camera_pos=np.zeros(3),
        frustum_planes=None,
        thresholds=None,
        dims_state=dims,
    )

    assert len(requests) == 1
    req = requests[0]
    assert req.axis_selections == ((0, 10), (0, 20), (0, 30))


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_build_slice_request_3d_handles_none_dims_state(mock_gfx):
    """With dims_state=None all axes are treated as displayed."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    store = _make_store(shape=(5, 6, 7))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="3d")

    requests = visual.build_slice_request(
        camera_pos=np.zeros(3),
        frustum_planes=None,
        thresholds=None,
        dims_state=None,
    )
    assert requests[0].axis_selections == ((0, 5), (0, 6), (0, 7))


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_5d_dims_state_2d_scene(mock_gfx):
    """5D store, 2D scene: display (Y, X), slice T=0, C=1, Z=5."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    store = _make_store(shape=(2, 3, 10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="2d")

    dims = DimsState(
        axis_labels=("t", "c", "z", "y", "x"),
        selection=AxisAlignedSelectionState(
            displayed_axes=(3, 4),
            slice_indices={0: 0, 1: 1, 2: 5},
        ),
    )
    requests = visual.build_slice_request_2d(
        camera_pos=np.zeros(3),
        viewport_width_px=800.0,
        world_width=30.0,
        view_min=None,
        view_max=None,
        dims_state=dims,
    )
    req = requests[0]
    assert req.axis_selections == (0, 1, 5, (0, 20), (0, 30))


# ---------------------------------------------------------------------------
# Tests: commit (on_data_ready / on_data_ready_2d)
# ---------------------------------------------------------------------------

@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_on_data_ready_2d_transposes_correctly(mock_gfx):
    """Data (H=20, W=30) must be transposed to (W=30, H=20, 1) for pygfx."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="2d")

    data = np.random.rand(20, 30).astype(np.float32)
    req = ChunkRequest(
        chunk_request_id=uuid4(), slice_request_id=uuid4(),
        scale_index=0, axis_selections=(5, (0, 20), (0, 30)),
    )

    captured_arrays = []
    mock_gfx.Texture.side_effect = lambda arr, **kw: captured_arrays.append(arr)
    mock_gfx.Geometry.return_value = MagicMock()

    visual.on_data_ready_2d([(req, data)])

    assert len(captured_arrays) == 1
    arr = captured_arrays[0]
    assert arr.shape == (30, 20, 1)
    np.testing.assert_array_equal(arr[:, :, 0], data.T)


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_on_data_ready_3d_transposes_correctly(mock_gfx):
    """Data (D=10, H=20, W=30) must be transposed to (W=30, H=20, D=10) for pygfx."""
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    store = _make_store(shape=(10, 20, 30))
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="3d")

    data = np.random.rand(10, 20, 30).astype(np.float32)
    req = ChunkRequest(
        chunk_request_id=uuid4(), slice_request_id=uuid4(),
        scale_index=0, axis_selections=((0, 10), (0, 20), (0, 30)),
    )

    captured_arrays = []
    mock_gfx.Texture.side_effect = lambda arr, **kw: captured_arrays.append(arr)
    mock_gfx.Geometry.return_value = MagicMock()

    visual.on_data_ready([(req, data)])

    assert len(captured_arrays) == 1
    arr = captured_arrays[0]
    assert arr.shape == (30, 20, 10)
    np.testing.assert_array_equal(arr, data.T)


@patch("cellier.v2.render.visuals._image_memory.gfx")
def test_on_data_ready_noop_on_empty_batch(mock_gfx):
    from cellier.v2.render.visuals._image_memory import GFXImageMemoryVisual
    store = _make_store()
    model = _make_visual_model(store)
    visual = GFXImageMemoryVisual(model, store, render_mode="3d")

    # Should not raise
    visual.on_data_ready([])
    visual.on_data_ready_2d([])
```

---

## 12. Implementation order

Work through the steps in this order so each step compiles and passes tests before
the next begins:

1. **Step 1** — Create `_image_memory_store.py`. Run
   `uv run pytest tests/v2/data/test_image_memory_store.py` — all store tests pass.

2. **Steps 2–3** — Add `ImageMemoryStore` to `data/image/__init__.py` and `_types.py`.
   Verify `from cellier.v2.data import DataStoreType` imports without error.

3. **Steps 4–6** — Create `_image_memory.py` visual models. Add to `_types.py` and
   `__init__.py`. Verify `from cellier.v2.visuals import ImageVisual` imports without error.

4. **Steps 7–8** — Create `GFXImageMemoryVisual`. Add to render `__init__.py`. Run
   `uv run pytest tests/v2/render/test_gfx_image_memory_visual.py` — all planning and
   commit tests pass.

5. **Step 9** — Add the overloaded `add_image` (replacing the existing method), plus
   `_add_image_memory` and `_add_image_multiscale` private helpers. Run
   `uv run pytest tests/v2/` — existing `add_image` call-sites still pass unchanged;
   the new memory path is exercised by the integration tests.

6. **Step 10** — Write and manually run the demo script.
   `uv run python scripts/v2/image_memory_demo.py`
   Verify: two windows open side by side, 2D slice visible in one, 3D volume in the
   other. Adjust the z-slider (if wired) and confirm the 2D view updates.

7. **Step 11** — Update `scripts/v2/integration_2d_3d/` (see section 14). Fix the
   import path in `example_combined.py` and `example_combined_camera_redraw.py`, add
   the discoverable comment above each `add_image` call site, and run the smoke-test
   for all three scripts to confirm the `add_image` refactor introduced no regressions.

---

## 14. `scripts/v2/integration_2d_3d` — example verification and updates

### 14.1 What needs to change and what does not

All three existing examples already call `controller.add_image()` with a
`MultiscaleZarrDataStore` and an `ImageAppearance`. After the section 9 refactor, that
call routes through the overload dispatch and lands in `_add_image_multiscale`, which
contains the old body verbatim. **No call-site code changes are needed in any of the
three files.**

What does need to be done:

1. **Verify** each script still launches and renders after the refactor.
2. **Update imports** in `example_combined.py` and `example_combined_camera_redraw.py`
   if they import `ImageAppearance` from the private path
   (`from cellier.v2.visuals._image import ImageAppearance`) — prefer the public path
   (`from cellier.v2.visuals import ImageAppearance`) for consistency with the rest of
   the codebase.
3. **Add a short inline comment** at each `add_image` call site noting that
   `ImageMemoryStore` is also accepted, pointing readers to `image_memory_demo.py` for
   an in-memory example. This makes the unified API discoverable from the existing
   examples without bloating them.

### 14.2 Per-file checklist

**`scripts/v2/integration_2d_3d/example_2d.py`**

| Item | Action |
|---|---|
| `from cellier.v2.visuals import ImageAppearance` | Already uses the public path — no change |
| `controller.add_image(data=data_store, ...)` (one call, 2D scene) | No change; verify it dispatches correctly after refactor |
| Run smoke-test | `uv run python scripts/v2/integration_2d_3d/example_2d.py` — window opens, tiles load |

Add the following comment immediately above the `add_image` call:

```python
# For in-memory numpy arrays, pass ImageMemoryStore + ImageMemoryAppearance instead.
# See scripts/v2/image_memory_demo.py for an example.
self._visual = self._controller.add_image(
    data=data_store,
    ...
)
```

---

**`scripts/v2/integration_2d_3d/example_combined.py`**

| Item | Action |
|---|---|
| `from cellier.v2.visuals._image import ImageAppearance` | Change to `from cellier.v2.visuals import ImageAppearance` |
| `controller.add_image(data=data_store, ...)` (2D scene call) | No change |
| `controller.add_image(data=data_store, ...)` (3D scene call) | No change |
| Run smoke-test | `uv run python scripts/v2/integration_2d_3d/example_combined.py --make-files && uv run python scripts/v2/integration_2d_3d/example_combined.py` |

Add the same discoverable comment above both `add_image` call sites (see above).

---

**`scripts/v2/integration_2d_3d/example_combined_camera_redraw.py`**

| Item | Action |
|---|---|
| `from cellier.v2.visuals._image import ImageAppearance` | Change to `from cellier.v2.visuals import ImageAppearance` |
| `controller.add_image(data=data_store, ...)` (2D scene call) | No change |
| `controller.add_image(data=data_store, ...)` (3D scene call) | No change |
| Run smoke-test | `uv run python scripts/v2/integration_2d_3d/example_combined_camera_redraw.py` |

Add the same discoverable comment above both `add_image` call sites.

### 14.3 Add this step to the implementation order

This work belongs at the end of the implementation sequence, after the controller
refactor and demo script are working:

> **Step 11** — Open each script in `scripts/v2/integration_2d_3d/`, fix the one import
> path (`_image` → public `visuals`), add the discoverable comment above each
> `add_image` call, and run the smoke-test for each. Confirm all three still launch and
> render correctly.

---

## 13. Known pitfalls and cross-checks

| Item | What to verify |
|---|---|
| `AppearanceChangedEvent` field names | Confirm `.field_name` and `.new_value` (not `.field` / `.value`) match the actual NamedTuple in `events/_events.py`. The controller `_make_appearance_handler` is the ground truth. |
| `_wire_appearance` genericity | Confirm it works with `ImageVisual` not just `MultiscaleImageVisual`. If it hard-codes type checks, generalise it. |
| `gfx.Texture(dim=2, format="1xf4")` requires `(W, H, C)` | The `C=1` channel axis is mandatory. Forgetting `[:, :, np.newaxis]` in `on_data_ready_2d` causes a silent wrong shape. |
| `gfx.Volume` expects `(W, H, D)` not `(D, H, W)` | The `.T` transpose in `on_data_ready` must be applied. |
| `RenderManager.add_visual` type annotation | The current type hint on `add_visual` says `GFXMultiscaleImageVisual`. Either widen to `Any` or create a protocol/ABC. Do not change the existing method body — just the annotation if needed. |
| `SceneManager.add_visual` node lookup | `SceneManager.add_visual` accesses `visual.node_3d` (3D scene) or `visual.node_2d` (2D scene). `GFXImageMemoryVisual` sets both attributes at construction; the unused one is `None`. Confirm the `SceneManager` raises if the expected node is `None` — the existing code does this, so both nodes must be set correctly. |
| UUID `chunk_request_id` / `slice_request_id` uniqueness | Each call to `build_slice_request[_2d]` generates fresh `uuid4()` values. Do not reuse a single pre-generated UUID across calls — the `SliceCoordinator` uses these to match arriving data to the correct pending request. |
