"""MultiscaleZarrDataStore — async-capable data store for multiscale zarr volumes.

Inherits from ``BaseDataStore`` (psygnal ``EventedModel`` / pydantic ``BaseModel``).
Opens all tensorstore handles synchronously in ``model_post_init`` so they are
ready before ``QtAsyncio.run()`` starts the event loop.

Placement note
--------------
Place this file adjacent to (or in the same package as) ``chunk_request.py``
and adjust the import below to match::

    from <your_package>.chunk_request import ChunkRequest

Likewise adjust the ``BaseDataStore`` import to wherever you have placed the
base class definition.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import tensorstore as ts
from pydantic import ConfigDict, PrivateAttr, model_validator

from cellier.v2.data._base_data_store import BaseDataStore
from cellier.v2.transform import AffineTransform

if TYPE_CHECKING:
    from cellier.v2.data.image._image_requests import ChunkRequest

# ---------------------------------------------------------------------------
# Zarr format detection helpers (private)
# ---------------------------------------------------------------------------


def _detect_zarr_driver(level_path: pathlib.Path) -> str:
    """Return the tensorstore driver string for a zarr level directory.

    Detects format by the metadata sentinel file:

    * ``.zarray``   → zarr v2 → driver ``"zarr"``
    * ``zarr.json`` → zarr v3 → driver ``"zarr3"``

    Raises ``FileNotFoundError`` if neither file is present.
    """
    p = pathlib.Path(level_path)
    if (p / ".zarray").exists():
        return "zarr"
    if (p / "zarr.json").exists():
        return "zarr3"
    raise FileNotFoundError(
        f"Cannot determine zarr format for '{p}': "
        f"neither '.zarray' (zarr v2) nor 'zarr.json' (zarr v3) found.\n"
        f"Re-run with --make-files to regenerate the store."
    )


def _open_ts_stores(
    zarr_path: pathlib.Path,
    scale_names: list[str],
) -> list[ts.TensorStore]:
    """Open one tensorstore per scale level (read-only, synchronous).

    Must be called — or triggered via ``model_post_init`` — **before**
    ``QtAsyncio.run()`` starts the event loop.

    Parameters
    ----------
    zarr_path :
        Root directory of the multiscale zarr store.
    scale_names :
        Subdirectory names in order finest → coarsest, e.g.
        ``["s0", "s1", "s2"]``.

    Returns
    -------
    stores :
        One open ``ts.TensorStore`` per scale level.  Chunk data is
        not loaded until ``await store[...].read()`` is called.
    """
    stores: list[ts.TensorStore] = []
    for name in scale_names:
        level_path = pathlib.Path(zarr_path) / name
        driver = _detect_zarr_driver(level_path)
        spec: dict[str, Any] = {
            "driver": driver,
            "kvstore": {
                "driver": "file",
                "path": str(level_path),
            },
        }
        store = ts.open(spec).result()
        fmt = "v2" if driver == "zarr" else "v3"
        shape = tuple(int(d) for d in store.domain.shape)
        print(f"  {name}: opened as zarr {fmt}  shape={shape}")
        stores.append(store)
    return stores


# ---------------------------------------------------------------------------
# MultiscaleZarrDataStore
# ---------------------------------------------------------------------------


class MultiscaleZarrDataStore(BaseDataStore):
    """Data store for a multiscale zarr volume read via tensorstore.

    Public fields are validated and serialisable (pydantic).
    Tensorstore handles are opened synchronously in ``model_post_init``
    and stored as private attributes so they are not serialised.

    Parameters
    ----------
    store_type : Literal["multiscale_zarr"]
        Discriminator field. Always ``"multiscale_zarr"``.
    zarr_path :
        Path to the root directory of the multiscale zarr store.
        Pass as a string; ``pathlib.Path`` is accepted and coerced.
    scale_names :
        Ordered list of subdirectory names, finest → coarsest,
        e.g. ``["s0", "s1", "s2"]``.
    level_transforms :
        Per-level affine transforms mapping level-k voxel coords to
        level-0 voxel coords. ``level_transforms[0]`` must be the
        identity. Length must match ``scale_names``.
    name :
        Human-readable name for the store (inherited from
        ``BaseDataStore``; defaults to ``"multiscale zarr data store"``).

    Attributes (read-only properties)
    ------------------------------------
    n_levels :
        Number of scale levels (length of ``scale_names``).
    level_shapes :
        List of shape tuples, one per level.
    """

    # ── Public pydantic fields ──────────────────────────────────────────
    store_type: Literal["multiscale_zarr"] = "multiscale_zarr"
    zarr_path: str
    scale_names: list[str]
    level_transforms: list[AffineTransform]
    name: str = "multiscale zarr data store"

    # ── Private tensorstore handles (not serialised) ────────────────────
    _ts_stores: list[ts.TensorStore] = PrivateAttr(default_factory=list)

    # Allow non-pydantic types in private attrs.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Validation ─────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_level_transforms(self) -> MultiscaleZarrDataStore:
        """Check that level_transforms length matches scale_names."""
        if len(self.level_transforms) != len(self.scale_names):
            raise ValueError(
                f"level_transforms has {len(self.level_transforms)} entries "
                f"but scale_names has {len(self.scale_names)} entries; "
                f"they must match."
            )
        return self

    # ── Lifecycle ───────────────────────────────────────────────────────

    def model_post_init(self, __context: Any) -> None:
        """Open all tensorstore handles.

        Called automatically by pydantic after ``__init__``.
        Must run before ``QtAsyncio.run()`` starts the event loop.
        """
        self._ts_stores = _open_ts_stores(
            pathlib.Path(self.zarr_path),
            self.scale_names,
        )

    # ── Convenience constructor ─────────────────────────────────────────

    @classmethod
    def from_scale_and_translation(
        cls,
        *,
        zarr_path: str,
        scale_names: list[str],
        level_scales: list[tuple[float, ...]],
        level_translations: list[tuple[float, ...]],
        name: str = "multiscale zarr data store",
    ) -> MultiscaleZarrDataStore:
        """Construct from per-level scale and translation vectors.

        Parameters
        ----------
        zarr_path :
            Path to the root directory of the multiscale zarr store.
        scale_names :
            Ordered list of subdirectory names, finest → coarsest.
        level_scales :
            Per-level scale vectors. ``level_scales[0]`` should be all 1s.
        level_translations :
            Per-level translation vectors. ``level_translations[0]``
            should be all 0s.
        name :
            Human-readable name for the store.
        """
        if len(level_scales) != len(level_translations):
            raise ValueError(
                f"level_scales has {len(level_scales)} entries but "
                f"level_translations has {len(level_translations)} entries; "
                f"they must match."
            )
        transforms = [
            AffineTransform.from_scale_and_translation(scale=sc, translation=tr)
            for sc, tr in zip(level_scales, level_translations)
        ]
        return cls(
            zarr_path=zarr_path,
            scale_names=scale_names,
            level_transforms=transforms,
            name=name,
        )

    # ── Read-only properties ────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        """Number of scale levels."""
        return len(self._ts_stores)

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        """Shape for each scale level, finest first."""
        return [tuple(int(d) for d in store.domain.shape) for store in self._ts_stores]

    # ── Async data access ───────────────────────────────────────────────

    async def get_data(self, request: ChunkRequest) -> np.ndarray:
        """Read a single padded brick, returning a zero-padded float32 array.

        Interprets ``request.axis_selections`` generically: displayed axes
        (tuple ranges) become slice dimensions in the output; sliced axes
        (int values) become point selections.

        Parameters
        ----------
        request :
            Padded brick specification.  Coordinates may be negative or
            exceed store bounds; clamping is handled internally.

        Returns
        -------
        out :
            ``float32`` array.  Shape has one dimension per displayed axis
            (those with tuple selections).  Out-of-bounds regions are
            filled with zero.
        """
        return await self._get_data_and(request)

    async def _get_data_and(self, request: ChunkRequest) -> np.ndarray:
        """Read a padded brick using generic nD axis_selections."""
        store = self._ts_stores[request.scale_index]
        store_shape = tuple(int(d) for d in store.domain.shape)

        # Output shape: one dimension per displayed (tuple) axis.
        out_shape = tuple(
            stop - start
            for sel in request.axis_selections
            if isinstance(sel, tuple)
            for start, stop in [sel]
        )
        out = np.zeros(out_shape, dtype=np.float32)

        # Build the clamped store index and track destination offsets.
        store_idx: list[int | slice] = []
        dest_starts: list[int] = []
        valid = True

        for axis_i, sel in enumerate(request.axis_selections):
            size = store_shape[axis_i]
            if isinstance(sel, int):
                # Point selection: clamp to valid range.
                store_idx.append(max(0, min(sel, size - 1)))
            else:
                start, stop = sel
                c_start = max(start, 0)
                c_stop = min(stop, size)
                if c_stop <= c_start:
                    valid = False
                    break
                store_idx.append(slice(c_start, c_stop))
                dest_starts.append(c_start - start)

        if valid:
            region = np.asarray(
                await store[tuple(store_idx)].read(),
                dtype=np.float32,
            )
            # Compute the destination slice in out for each displayed axis.
            dest_idx = tuple(slice(d, d + s) for d, s in zip(dest_starts, region.shape))
            out[dest_idx] = region

        return out
