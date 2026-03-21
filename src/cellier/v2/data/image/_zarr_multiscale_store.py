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
from pydantic import ConfigDict, PrivateAttr

from cellier.v2.data._base_data_store import BaseDataStore

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
    name :
        Human-readable name for the store (inherited from
        ``BaseDataStore``; defaults to ``"multiscale zarr data store"``).

    Attributes (read-only properties)
    ------------------------------------
    n_levels :
        Number of scale levels (length of ``scale_names``).
    level_shapes :
        List of ``(depth, height, width)`` tuples, one per level.
    """

    # ── Public pydantic fields ──────────────────────────────────────────
    store_type: Literal["multiscale_zarr"] = "multiscale_zarr"
    zarr_path: str
    scale_names: list[str]
    name: str = "multiscale zarr data store"

    # ── Private tensorstore handles (not serialised) ────────────────────
    _ts_stores: list[ts.TensorStore] = PrivateAttr(default_factory=list)

    # Allow non-pydantic types in private attrs.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def model_post_init(self, __context: Any) -> None:
        """Open all tensorstore handles synchronously after model init.

        Called automatically by pydantic after ``__init__``.
        Must run before ``QtAsyncio.run()`` starts the event loop.
        """
        self._ts_stores = _open_ts_stores(
            pathlib.Path(self.zarr_path),
            self.scale_names,
        )

    # ── Read-only properties ────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        """Number of scale levels."""
        return len(self._ts_stores)

    @property
    def level_shapes(self) -> list[tuple[int, int, int]]:
        """``(depth, height, width)`` for each scale level, finest first."""
        return [
            tuple(int(d) for d in store.domain.shape)  # type: ignore[return-value]
            for store in self._ts_stores
        ]

    # ── Async data access ───────────────────────────────────────────────

    async def get_data(self, request: ChunkRequest) -> np.ndarray:
        """Read a single padded brick, returning a zero-padded float32 array.

        Boundary bricks (where the padded region extends outside the store)
        are handled by allocating a zeroed output of the full requested size
        and copying the valid clamped region into the correct destination
        offset.

        The ``await`` on ``store[...].read()`` suspends the coroutine and
        yields to the Qt event loop while tensorstore fetches the chunk(s)
        from disk.

        Parameters
        ----------
        request :
            Padded brick specification.  Coordinates may be negative or
            exceed store bounds; clamping is handled internally.

        Returns
        -------
        out :
            ``float32`` array of shape
            ``(z_stop-z_start, y_stop-y_start, x_stop-x_start)``.
            Out-of-bounds regions are filled with zero.
        """
        store = self._ts_stores[request.scale_index]

        dz = request.z_stop - request.z_start
        dy = request.y_stop - request.y_start
        dx = request.x_stop - request.x_start
        out = np.zeros((dz, dy, dx), dtype=np.float32)

        sd, sh, sw = (int(d) for d in store.domain.shape)
        sz0 = max(request.z_start, 0)
        sz1 = min(request.z_stop, sd)
        sy0 = max(request.y_start, 0)
        sy1 = min(request.y_stop, sh)
        sx0 = max(request.x_start, 0)
        sx1 = min(request.x_stop, sw)

        if sz1 > sz0 and sy1 > sy0 and sx1 > sx0:
            dest_z0 = sz0 - request.z_start
            dest_y0 = sy0 - request.y_start
            dest_x0 = sx0 - request.x_start

            region = np.asarray(
                await store[sz0:sz1, sy0:sy1, sx0:sx1].read(),
                dtype=np.float32,
            )
            out[
                dest_z0 : dest_z0 + (sz1 - sz0),
                dest_y0 : dest_y0 + (sy1 - sy0),
                dest_x0 : dest_x0 + (sx1 - sx0),
            ] = region

        return out
